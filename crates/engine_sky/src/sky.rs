use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParamItem;
use engine_core::math::{max, min};
use engine_core::{Error, FrameBudget, FrameTime, PrimaryView, StoreId};
use engine_places_rendering::sky::EyeMedium;
use engine_places_rendering::sun::{Sun, SunPath};
use engine_places_rendering::wind::ViewWind;
use engine_render::{
    AfterRead, Allocation, Attach, AttachPoint, Bind, Buffer, ComputePipeline, Copied, CopyRegion,
    Count, DepthUse, Encode, GpuPart, Install, Lifetime, Own, Pass, Pipeline, Prepare,
    RenderPipeline, Sampler, SamplerDesc, Shape, Texture, wgpu,
};
use glam::{Mat4, Vec3};

use crate::clouds::{
    CloudFrame, CloudPass, CloudPlan, CloudSteps, cloud_regions, shadow_lighting,
};
use crate::environment::{Environment, EnvironmentFresh, EnvironmentLight, EnvironmentPlan};
use crate::gpu_layouts::{SKY_TABLE_GROUP, SKY_VIEW_HEIGHT, SKY_VIEW_WIDTH, SkyAir, SkyPiece};
use crate::haze::{Haze, HazeFrame, HazePlan, HazeSteps};
use crate::settings::{Clouds, Haze as HazeSettings, NightSky, Sky, haze_runs};
use crate::tables::sky::{MULTIPLE_SIZE, SkyTables, TRANSMITTANCE_SIZE};
use crate::tables::{StaticTable, StaticTables};
use crate::views::SkyLightLevel;

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const VIEW_SIZE: (u32, u32) = (SKY_VIEW_WIDTH, SKY_VIEW_HEIGHT);
const NO_VIEW_SIZE: (u32, u32) = (1, 1);
const ENVIRONMENT_HEIGHT_STEP: f32 = 200.0;
const DERIVED_BYTES: u64 = 6 * 16;
const DERIVED_LIGHT: u64 = DERIVED_HORIZON * 16 + 12;

pub(crate) const DERIVED_SUN: u64 = 0;
pub(crate) const DERIVED_ZENITH: u64 = 1;
pub(crate) const DERIVED_HORIZON: u64 = 2;
pub(crate) const DERIVED_GROUND: u64 = 3;
pub(crate) const DERIVED_AIR: u64 = 4;
pub(crate) const DERIVED_GLOW: u64 = 5;

pub(crate) fn inverse_or_identity(matrix: Mat4) -> Mat4 {
    let inverse = matrix.inverse();
    if inverse.is_finite() {
        inverse
    } else {
        Mat4::IDENTITY
    }
}

fn extent((width, height): (u32, u32)) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    }
}

fn piece_regions() -> [CopyRegion; 6] {
    use std::mem::offset_of;
    let at = |offset: usize| offset as u64;
    let region = |index: u64, to: u64, size: u64| CopyRegion::Bytes {
        from: index * 16,
        to,
        size,
    };
    [
        region(DERIVED_SUN, at(offset_of!(SkyPiece, sun_light)), 12),
        region(DERIVED_ZENITH, at(offset_of!(SkyPiece, zenith)), 12),
        region(DERIVED_HORIZON, at(offset_of!(SkyPiece, horizon)), 16),
        region(DERIVED_GROUND, at(offset_of!(SkyPiece, ground)), 12),
        region(DERIVED_AIR, at(offset_of!(SkyPiece, aerial_color)), 12),
        region(DERIVED_GLOW, at(offset_of!(SkyPiece, aerial)) + 12, 4),
    ]
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Air {
    #[default]
    Off,
    Waiting,
    Shown(u64),
}

#[derive(Default)]
pub(crate) struct AirPlan {
    air: Air,
    sun: Option<[u32; 3]>,
    generation: u64,
    environment_height: i32,
}

impl AirPlan {
    fn shown(&self) -> Option<u64> {
        match self.air {
            Air::Shown(tables) => Some(tables),
            Air::Off | Air::Waiting => None,
        }
    }

    pub(crate) fn shows(&self) -> bool {
        self.shown().is_some()
    }

    fn environment_key(&self) -> [f32; 4] {
        if self.shows() {
            [
                1.0,
                self.generation as f32,
                self.environment_height as f32,
                0.0,
            ]
        } else {
            [0.0; 4]
        }
    }

    fn prepare(
        &mut self,
        (settings, tables): (&Sky, Option<&StaticTable<SkyTables>>),
        (sun, eye, moon): (Vec3, Vec3, [f32; 4]),
        air: &mut SkyAir,
    ) {
        let next = match (settings.enabled, tables) {
            (false, _) => Air::Off,
            (true, Some(tables)) => Air::Shown(tables.id),
            (true, None) => Air::Waiting,
        };
        let uploaded = next != self.air;
        self.air = next;
        if !self.shows() {
            self.sun = None;
            return;
        }
        let height = max(eye.y + settings.altitude, 0.5);
        write_air(air, settings, (sun, height, moon));
        let sun_bits = sun.to_array().map(f32::to_bits);
        let environment_height = (height / ENVIRONMENT_HEIGHT_STEP).floor() as i32;
        if uploaded || self.sun != Some(sun_bits) || environment_height != self.environment_height {
            self.generation += 1;
        }
        self.environment_height = environment_height;
        self.sun = Some(sun_bits);
    }
}

fn write_air(
    air: &mut SkyAir,
    settings: &Sky,
    (sun, height, moon): (Vec3, f32, [f32; 4]),
) {
    air.rayleigh = [
        settings.rayleigh[0],
        settings.rayleigh[1],
        settings.rayleigh[2],
        settings.rayleigh_height,
    ];
    air.mie = [
        settings.mie,
        settings.mie_absorption,
        settings.mie_height,
        settings.mie_g,
    ];
    air.ozone = [
        settings.ozone[0],
        settings.ozone[1],
        settings.ozone[2],
        settings.sun_intensity,
    ];
    air.ground = [
        settings.ground_albedo[0],
        settings.ground_albedo[1],
        settings.ground_albedo[2],
        settings.sun_size.to_radians(),
    ];
    air.radii = [
        settings.planet_radius,
        settings.planet_radius + settings.atmosphere_height,
        settings.adaptation,
        height * 0.001,
    ];
    air.sun = sun.extend(0.0).to_array();
    air.moon = moon;
}

fn night_lighting(night: &NightSky, path: &SunPath, sun: Vec3) -> [[f32; 4]; 3] {
    let moon = path.moon.filter(|_| night.moon > 0.0);
    let lit = moon.map_or(0.0, |moon| 0.5 - 0.5 * moon.dot(sun));
    [
        [
            night.stars,
            moon.map_or(0.0, |_| night.moon),
            night.cirrus,
            lit,
        ],
        path.stars.to_array(),
        moon.unwrap_or(Vec3::Y).extend(night.moon_size).to_array(),
    ]
}

fn sky_piece(sky: &Sky, shows: bool, sun: Vec3, kept: &SkyPiece) -> SkyPiece {
    let (aerial, sun_light, light) = if shows {
        let altitude = sky.altitude * 0.001;
        let rayleigh = (sky.rayleigh[0] + sky.rayleigh[1] + sky.rayleigh[2]) / 3.0
            * (-altitude / sky.rayleigh_height).exp();
        let mie = (sky.mie + sky.mie_absorption) * (-altitude / sky.mie_height).exp();
        let extinction = rayleigh + mie;
        let height = if extinction > 0.0 {
            (rayleigh * sky.rayleigh_height + mie * sky.mie_height) / extinction
        } else {
            sky.rayleigh_height
        };
        (
            [0.0, extinction * 0.001 * sky.aerial, 0.001 / height, 0.0],
            [0.0; 4],
            0.0,
        )
    } else {
        let white = sky.sun_intensity;
        (
            [0.0; 4],
            [white, white, white, 0.0],
            white * max(sun.y, 0.0),
        )
    };
    SkyPiece {
        zenith: [0.0, 0.0, 0.0, sky.ambient],
        horizon: [0.0, 0.0, 0.0, light],
        ground: [0.0; 4],
        aerial,
        aerial_color: [0.0; 4],
        cloud_shadow: [0.0; 4],
        cloud_layer: [0.0; 4],
        night: [0.0; 4],
        star_turn: [0.0, 0.0, 0.0, 1.0],
        moon: [0.0; 4],
        night_shape: [0.0; 4],
        haze_march: [0.0; 4],
        sun_light,
        ..*kept
    }
}

fn apply_night(piece: &mut SkyPiece, night: &NightSky, path: &SunPath, sun: Vec3) {
    [piece.night, piece.star_turn, piece.moon] = night_lighting(night, path, sun);
    piece.night_shape = [
        night.cirrus_height,
        night.cirrus_drift[0],
        night.cirrus_drift[1],
        night.star_density,
    ];
}

fn apply_haze(
    piece: &mut SkyPiece,
    haze: &HazeSettings,
    view_projection: Mat4,
    eye: Vec3,
    marches: bool,
) {
    if !haze.enabled {
        return;
    }
    let visibility = if haze.visibility > 0.0 {
        haze.visibility
    } else {
        let far = inverse_or_identity(view_projection).project_point3(Vec3::ZERO);
        2.0 * far.distance(eye)
    };
    let extinction = 3.912 / max(visibility, 1.0);
    let falloff = piece.aerial[2];
    let at_zero = extinction * min(falloff * eye.y, 60.0).exp();
    piece.aerial[1] = max(piece.aerial[1], at_zero) * haze.strength;
    piece.aerial_color[3] = 1.0;
    if marches {
        piece.haze_march = [haze.reach, haze.forward, haze.spread, 1.0 / haze.size];
    }
}

#[derive(Resource, Default)]
pub(crate) struct SkyPlan {
    piece: SkyPiece,
    air: SkyAir,
    tables: AirPlan,
    clouds: CloudPlan,
    environment: EnvironmentPlan,
    haze: HazePlan,
    frame: Option<CloudFrame>,
}

impl SkyPlan {
    pub(crate) fn environment(&self) -> Option<StoreId> {
        self.environment.beyond()
    }

    pub(crate) fn tables(&self) -> u32 {
        if self.tables.shows() { 3 } else { 0 }
    }

    pub(crate) fn cloud_rows(&self) -> (u32, u32) {
        (self.clouds.rows, self.clouds.view_pixels)
    }

    pub(crate) fn hazes(&self) -> bool {
        self.haze.runs()
    }
}

type SkySections<'w> = (
    Res<'w, Sky>,
    Res<'w, Clouds>,
    Res<'w, HazeSettings>,
    Res<'w, NightSky>,
);

type SkyLights<'w> = (
    Res<'w, Sun>,
    Res<'w, SunPath>,
    Res<'w, ViewWind>,
);

type SkyFrame<'w> = (
    Res<'w, StaticTables>,
    Res<'w, EyeMedium>,
    Res<'w, PrimaryView>,
    Res<'w, FrameTime>,
);

type SkySteps<'w> = (
    ResMut<'w, CloudSteps>,
    ResMut<'w, HazeSteps>,
    ResMut<'w, EnvironmentFresh>,
    ResMut<'w, FrameBudget>,
);

pub(crate) fn plan_sky(
    (sky, clouds, haze, night): SkySections<'_>,
    (sun, path, wind): SkyLights<'_>,
    (tables, medium, view, time): SkyFrame<'_>,
    (mut cloud_steps, mut haze_steps, mut fresh, mut budget): SkySteps<'_>,
    mut plan: ResMut<SkyPlan>,
) {
    let plan = &mut *plan;
    plan.clouds.acknowledge(&mut cloud_steps);
    let debug = view.debug();
    let eye = view.eye();
    let view_projection = view.view_projection();
    let seconds = time.elapsed;
    let sun_direction = Vec3::from_slice(&sun.direction[..3]);
    let [night_light, _, moon] = night_lighting(&night, &path, sun_direction);
    plan.tables.prepare(
        (&*sky, tables.sky()),
        (
            sun_direction,
            eye,
            [moon[0], moon[1], moon[2], night_light[1] * night_light[3]],
        ),
        &mut plan.air,
    );
    let shows = plan.tables.shows();
    let clouds = Clouds {
        enabled: clouds.enabled && shows,
        ..*clouds
    };
    let hazes = !debug && haze_runs(&haze);
    plan.piece = sky_piece(&sky, shows, sun_direction, &plan.piece);
    apply_night(&mut plan.piece, &night, &path, sun_direction);
    apply_haze(&mut plan.piece, &haze, view_projection, eye, hazes);
    (plan.piece.cloud_shadow, plan.piece.cloud_layer) = shadow_lighting(&clouds, eye);
    plan.piece.wind = [wind.velocity[0], wind.velocity[1], 0.0, 0.0];
    plan.clouds.prepare(
        (&clouds, tables.cloud_noise(), (eye, medium.surface), seconds),
        (&*sun, &plan.piece),
        (!debug, &mut budget),
    );
    plan.clouds.prepare_view(
        (&clouds, eye, seconds),
        (view_projection, view.display_size(), view.render_size()),
        (&*sun, &plan.piece),
    );
    [
        plan.piece.domes_older,
        plan.piece.domes_newer,
        plan.piece.domes_older_mirror,
        plan.piece.domes_newer_mirror,
        plan.piece.domes_eye,
        plan.piece.domes_view,
    ] = plan.clouds.dome_piece();
    plan.piece.domes_layers = plan.clouds.dome_layers();
    let light: EnvironmentLight = [
        sun.direction,
        plan.piece.sun_light,
        plan.piece.zenith,
        plan.piece.horizon,
        plan.piece.ground,
    ];
    let keys = (plan.clouds.environment_key(), plan.tables.environment_key());
    plan.environment.prepare(&light, keys, fresh.take());
    let haze_frame = HazeFrame {
        view_projection,
        eye,
        display: view.display_size(),
        render: view.render_size(),
        history: view.history().valid(),
        frame: time.frame,
    };
    plan.piece.haze = plan.haze.prepare((hazes, haze_steps.missed()), &haze_frame);
    plan.frame = if debug { None } else { plan.clouds.frame() };
}

pub(crate) struct AirTables {
    transmittance: Texture<SkyGpu>,
    multiple: Texture<SkyGpu>,
    view: Texture<SkyGpu>,
    derived: Buffer<SkyGpu>,
    sampler: Sampler<SkyGpu>,
    pass: Pass<SkyGpu>,
    building: ComputePipeline<SkyGpu>,
    deriving: ComputePipeline<SkyGpu>,
    copies: Vec<Pass<SkyGpu>>,
    uploaded: Option<u64>,
}

impl AirTables {
    fn install(cx: &mut Install<'_, SkyGpu>) -> Result<Self, Error> {
        use crate::shaders::sky::{light, tables};
        let transmittance = cx.texture("sky_transmittance", FORMAT, Lifetime::Kept);
        let multiple = cx.texture("sky_multiple", FORMAT, Lifetime::Kept);
        let view = cx.texture("sky_view", FORMAT, Lifetime::Kept);
        let derived = cx.buffer("sky_derived", Lifetime::Kept);
        let sampler = cx.sampler(SamplerDesc {
            filter: wgpu::FilterMode::Linear,
            mipmap: wgpu::MipmapFilterMode::Nearest,
            address: [wgpu::AddressMode::ClampToEdge; 3],
            compare: None,
            anisotropy: 1,
        });
        cx.frame(light::SKY_VIEW_TABLE, Bind::Texture(view))?;
        cx.frame(light::SKY_TABLE_SAMPLER, Bind::Sampler(sampler))?;
        let pass = cx.pass("sky").label("engine/sky").span("sky").id();
        let building = cx.compute_pipeline(
            pass,
            tables::BUILD_SKY_VIEW,
            &[
                (tables::TRANSMITTANCE_TABLE, Bind::Texture(transmittance)),
                (tables::MULTIPLE_TABLE, Bind::Texture(multiple)),
                (tables::TABLE_SAMPLER, Bind::Sampler(sampler)),
                (tables::TABLE_OUT, Bind::Texture(view)),
            ],
        )?;
        let deriving = cx.compute_pipeline(
            pass,
            tables::DERIVE_LIGHT,
            &[
                (tables::TRANSMITTANCE_TABLE, Bind::Texture(transmittance)),
                (tables::VIEW_TABLE, Bind::Texture(view)),
                (tables::TABLE_SAMPLER, Bind::Sampler(sampler)),
                (tables::DERIVED, Bind::Buffer(derived)),
            ],
        )?;
        Ok(Self {
            transmittance,
            multiple,
            view,
            derived,
            sampler,
            pass,
            building,
            deriving,
            copies: Vec::new(),
            uploaded: None,
        })
    }

    fn connect(
        &mut self,
        cx: &mut Install<'_, SkyGpu>,
        piece: Buffer<SkyGpu>,
        clouds: [Buffer<SkyGpu>; 2],
        environment: Pass<SkyGpu>,
    ) {
        let [march, view] = clouds;
        self.copies = vec![
            cx.copy(
                "sky_derived_piece",
                Copied::Buffer(self.derived),
                Copied::Buffer(piece),
            ),
            cx.copy(
                "sky_derived_clouds",
                Copied::Buffer(self.derived),
                Copied::Buffer(march),
            ),
            cx.copy(
                "sky_derived_view_clouds",
                Copied::Buffer(self.derived),
                Copied::Buffer(view),
            ),
        ];
        for &derived in &self.copies {
            cx.after(environment, derived);
        }
    }

    fn bindings(&self) -> (Texture<SkyGpu>, Sampler<SkyGpu>) {
        (self.transmittance, self.sampler)
    }

    fn upload(&self, cx: &mut Prepare<'_, SkyGpu>, tables: &SkyTables) {
        for (texture, texels, (width, height)) in [
            (
                self.transmittance,
                &tables.transmittance,
                TRANSMITTANCE_SIZE,
            ),
            (self.multiple, &tables.multiple, MULTIPLE_SIZE),
        ] {
            cx.write_texture(
                texture,
                wgpu::Origin3d::ZERO,
                bytemuck::cast_slice(texels),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(width * 8),
                    rows_per_image: Some(height),
                },
                extent((width, height)),
            );
        }
    }

    fn prepare(
        &mut self,
        cx: &mut Prepare<'_, SkyGpu>,
        (plan, tables): (&AirPlan, Option<&StaticTable<SkyTables>>),
    ) {
        let shown = plan.shown();
        let transmittance = cx.allocate(self.transmittance, extent(TRANSMITTANCE_SIZE), 1);
        let multiple = cx.allocate(self.multiple, extent(MULTIPLE_SIZE), 1);
        if transmittance == Allocation::Fresh || multiple == Allocation::Fresh {
            self.uploaded = None;
        }
        let view = if shown.is_some() {
            VIEW_SIZE
        } else {
            NO_VIEW_SIZE
        };
        cx.allocate(self.view, extent(view), 1);
        cx.allocate_buffer(self.derived, DERIVED_BYTES);
        if shown.is_none() {
            return;
        }
        if let Some(tables) = tables.filter(|tables| Some(tables.id) == shown)
            && self.uploaded != shown
        {
            self.upload(cx, &tables.table);
            self.uploaded = shown;
        }
        cx.run(self.pass);
        cx.read_back::<SkyLightLevel>(
            0,
            self.derived,
            DERIVED_LIGHT..DERIVED_LIGHT + 4,
            AfterRead::Keep,
        );
        let regions = [
            Vec::from(piece_regions()),
            Vec::from(cloud_regions()),
            Vec::from(cloud_regions()),
        ];
        for (&pass, regions) in self.copies.iter().zip(regions) {
            cx.copy(pass, &regions);
        }
    }

    fn encode(&self, cx: &mut Encode<'_, SkyGpu>) -> u32 {
        cx.dispatch(
            &self.building,
            Count::Groups([
                VIEW_SIZE.0.div_ceil(SKY_TABLE_GROUP),
                VIEW_SIZE.1.div_ceil(SKY_TABLE_GROUP),
                1,
            ]),
        );
        cx.dispatch(&self.deriving, Count::Groups([1, 1, 1]));
        0
    }
}

pub(crate) struct SkyGpu {
    tables: AirTables,
    clouds: CloudPass,
    environment: Environment,
    haze: Haze,
    dome_pass: Pass<Self>,
    dome: Pipeline<Self, Own>,
    piece_buffer: Buffer<Self>,
    air_buffer: Buffer<Self>,
}

fn dome_pipeline(
    cx: &mut Install<'_, SkyGpu>,
    pass: Pass<SkyGpu>,
    transmittance: Texture<SkyGpu>,
) -> Result<Pipeline<SkyGpu, Own>, Error> {
    use crate::shaders::sky::draw;
    cx.render_pipeline(
        pass,
        &RenderPipeline {
            vertex: draw::VS_SKY,
            fragment: Some(draw::FS_SKY),
            debug: None,
            vertices: &[],
            primitive: wgpu::PrimitiveState::default(),
            depth: DepthUse::Compare {
                compare: wgpu::CompareFunction::GreaterEqual,
                write: false,
                bias: wgpu::DepthBiasState::default(),
            },
            blend: None,
            writes: wgpu::ColorWrites::ALL,
        },
        &[(draw::SKY_TRANSMITTANCE_TABLE, Bind::Texture(transmittance))],
    )
}

impl GpuPart for SkyGpu {
    type Reads = (Res<'static, SkyPlan>, Res<'static, StaticTables>);
    const NAME: &'static str = "sky";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use crate::shaders::sky::piece;
        let piece_buffer = cx.buffer("sky_piece", Lifetime::Kept);
        let air_buffer = cx.buffer("sky_air", Lifetime::Kept);
        cx.frame(piece::SKY_PIECE, Bind::Buffer(piece_buffer))?;
        cx.frame(piece::SKY_AIR, Bind::Buffer(air_buffer))?;
        let dome_pass = cx
            .pass("sky_dome")
            .at(AttachPoint::Sky)
            .label("engine/sky-dome")
            .span("sky_dome")
            .id();
        let mut tables = AirTables::install(cx)?;
        let environment = Environment::install(cx)?;
        let clouds = CloudPass::install(cx, tables.bindings(), (environment.pass(), dome_pass))?;
        tables.connect(cx, piece_buffer, clouds.uniforms(), environment.pass());
        let haze = Haze::install(cx)?;
        let dome = dome_pipeline(cx, dome_pass, tables.transmittance)?;
        Ok(Self {
            tables,
            clouds,
            environment,
            haze,
            dome_pass,
            dome,
            piece_buffer,
            air_buffer,
        })
    }

    fn prepare(&mut self, reads: SystemParamItem<'_, '_, Self::Reads>, cx: &mut Prepare<'_, Self>) {
        let (plan, tables) = reads;
        if plan.tables.shows() {
            cx.run(self.dome_pass);
        }
        self.tables.prepare(cx, (&plan.tables, tables.sky()));
        self.clouds
            .prepare(cx, (&plan.clouds, tables.cloud_noise()), plan.frame);
        self.environment.prepare(cx, &plan.environment);
        self.haze.prepare(cx, &plan.haze);
        cx.allocate_buffer(self.piece_buffer, size_of::<SkyPiece>() as u64);
        cx.write(self.piece_buffer, 0, &[plan.piece]);
        cx.allocate_buffer(self.air_buffer, size_of::<SkyAir>() as u64);
        cx.write(self.air_buffer, 0, &[plan.air]);
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        if self.clouds.owns(pass) {
            return self.clouds.encode(pass, cx);
        }
        if self.haze.owns(pass) {
            return self.haze.encode(pass, cx);
        }
        if pass == self.environment.pass() {
            return self.environment.encode(cx);
        }
        if pass == self.tables.pass {
            return self.tables.encode(cx);
        }
        if pass == self.dome_pass {
            let mut draw = cx.render(Attach::Declared);
            draw.draw(
                &self.dome,
                Shape::Pulled {
                    vertices: 0..3,
                    instances: 0..1,
                },
            );
        }
        0
    }
}
