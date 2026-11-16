use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParamItem;
use engine_core::math::max;
use engine_core::{Error, Filter, FrameBudget, FrameTime, StoreId, Switched, Views};
use engine_places_rendering::sky::EyeMedium;
use engine_places_rendering::sun::{Sun, SunPath};
use engine_render::{
    AfterRead, Allocation, Attach, AttachPoint, Bind, Buffer, ComputePipeline, Copied, CopyRegion,
    Count, DepthUse, Encode, GpuPart, Install, Lifetime, Own, Pass, Pipeline, Prepare,
    RenderPipeline, Sampler, Shape, Texture, wgpu,
};
use glam::{UVec2, Vec3};

use crate::clouds::{CloudFrame, CloudPass, CloudPlan, CloudSteps, cloud_regions};
use crate::environment::{Environment, EnvironmentInputs, EnvironmentPlan, EnvironmentSteps};
use crate::gpu_layouts::{
    SKY_TABLE_GROUP, SKY_VIEW_HEIGHT, SKY_VIEW_WIDTH, SkyAir, SkyDomes, SkyPiece,
};
use crate::settings::{Clouds, NightSky, Sky};
use crate::tables::sky::{MULTIPLE_SIZE, SkyTables, TRANSMITTANCE_SIZE};
use crate::tables::{StaticTable, StaticTables};
use crate::views::SkyLightLevel;

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const VIEW_SIZE: (u32, u32) = (SKY_VIEW_WIDTH, SKY_VIEW_HEIGHT);
const NO_VIEW_SIZE: (u32, u32) = (1, 1);
pub(crate) const DERIVED_ROWS: usize = 9;
pub(crate) const DERIVED_BYTES: u64 = DERIVED_ROWS as u64 * 16;

pub(crate) const DERIVED_SUN: u64 = 0;
pub(crate) const DERIVED_ZENITH: u64 = 1;
pub(crate) const DERIVED_HORIZON: u64 = 2;
pub(crate) const DERIVED_GROUND: u64 = 3;
pub(crate) const DERIVED_SKY: u64 = 4;
pub(crate) const DERIVED_SKY_UP: u64 = 6;
pub(crate) const DERIVED_CLEAR: u64 = 8;

fn extent((width, height): (u32, u32)) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    }
}

fn piece_regions() -> [CopyRegion; 5] {
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
        region(DERIVED_HORIZON, at(offset_of!(SkyPiece, horizon)), 12),
        region(DERIVED_GROUND, at(offset_of!(SkyPiece, ground)), 12),
        region(DERIVED_SKY, at(offset_of!(SkyPiece, sky_flat)), 64),
    ]
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Air {
    #[default]
    Waiting,
    Shown(u64),
}

#[derive(Default)]
pub(crate) struct AirPlan {
    air: Air,
}

impl AirPlan {
    fn shown(&self) -> Option<u64> {
        match self.air {
            Air::Shown(tables) => Some(tables),
            Air::Waiting => None,
        }
    }

    pub(crate) fn shows(&self) -> bool {
        self.shown().is_some()
    }

    fn prepare(
        &mut self,
        (settings, tables): (&Sky, Option<&StaticTable<SkyTables>>),
        (sun, eye, moon): (Vec3, Vec3, [f32; 4]),
        air: &mut SkyAir,
    ) {
        self.air = match tables {
            Some(tables) => Air::Shown(tables.id),
            None => Air::Waiting,
        };
        if self.shows() {
            let height = max(eye.y + settings.altitude, 0.5);
            write_air(air, settings, (sun, height, moon));
        }
    }
}

fn write_air(air: &mut SkyAir, settings: &Sky, (sun, height, moon): (Vec3, f32, [f32; 4])) {
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
        0.0,
        height * 0.001,
    ];
    air.sun = sun.extend(0.0).to_array();
    air.moon = moon;
}

fn night_lighting(night: &NightSky, path: &SunPath, sun: Vec3) -> [[f32; 4]; 3] {
    let moon = path.moon;
    let lit = moon.map_or(0.0, |moon| 0.5 - 0.5 * moon.dot(sun));
    let disc = moon.map_or([0.0, 1.0, 0.0, 0.0], |moon| {
        moon.extend(night.moon_size).to_array()
    });
    [
        [night.stars, moon.map_or(0.0, |_| night.moon), 0.0, lit],
        path.stars.to_array(),
        disc,
    ]
}

fn sky_piece(sky: &Sky, shows: bool, kept: &SkyPiece) -> SkyPiece {
    let sun_light = if shows {
        [0.0; 4]
    } else {
        let white = sky.sun_intensity;
        [white, white, white, 0.0]
    };
    SkyPiece {
        zenith: [0.0, 0.0, 0.0, sky.ambient],
        horizon: [0.0; 4],
        ground: [0.0; 4],
        night: [0.0; 4],
        star_turn: [0.0, 0.0, 0.0, 1.0],
        moon: [0.0; 4],
        night_shape: [0.0; 4],
        sun_light,
        sky_flat: [0.0; 4],
        sky_x: [0.0; 4],
        sky_y: [0.0; 4],
        sky_z: [0.0; 4],
        ..*kept
    }
}

fn dome_light(piece: &SkyPiece, sky: &Sky) -> SkyDomes {
    let shown = piece.domes_layers[2] != 0 && piece.domes_older[3] >= 0.5;
    SkyDomes {
        layers: [
            piece.domes_layers[0],
            piece.domes_layers[1],
            u32::from(shown),
            0,
        ],
        size: [
            piece.domes_size[0],
            piece.domes_size[1],
            piece.domes_newer[3],
            0.0,
        ],
        taps: [sky.light_taps[0], sky.light_taps[1], 0, 0],
    }
}

fn apply_night(piece: &mut SkyPiece, night: &NightSky, path: &SunPath, sun: Vec3) {
    [piece.night, piece.star_turn, piece.moon] = night_lighting(night, path, sun);
    piece.night_shape = [0.0, 0.0, 0.0, night.star_density];
}

#[derive(Resource, Default)]
pub(crate) struct SkyPlan {
    piece: SkyPiece,
    air: SkyAir,
    domes: SkyDomes,
    tables: AirPlan,
    clouds: CloudPlan,
    environment: EnvironmentPlan,
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
}

type SkySections<'w> = (
    Res<'w, Sky>,
    (Res<'w, Clouds>, Switched<'w, Clouds>),
    Res<'w, NightSky>,
);

type SkyLights<'w> = (Res<'w, Sun>, Res<'w, SunPath>);

type SkyFrame<'w> = (
    Res<'w, StaticTables>,
    Res<'w, EyeMedium>,
    Res<'w, Views>,
    Res<'w, FrameTime>,
);

type SkySteps<'w> = (
    ResMut<'w, CloudSteps>,
    ResMut<'w, EnvironmentSteps>,
    ResMut<'w, FrameBudget>,
);

pub(crate) fn plan_sky(
    (sky, (clouds, clouds_on), night): SkySections<'_>,
    (sun, path): SkyLights<'_>,
    (tables, medium, views, time): SkyFrame<'_>,
    (mut cloud_steps, mut environment_steps, mut budget): SkySteps<'_>,
    mut plan: ResMut<SkyPlan>,
) {
    let plan = &mut *plan;
    plan.clouds.acknowledge(&mut cloud_steps);
    let main = views.main();
    let eye = main.map_or(Vec3::ZERO, |main| main.eye);
    let (display, render) = main.map_or((UVec2::ZERO, UVec2::ZERO), |main| {
        (main.size(), main.render)
    });
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
    let clouds = *clouds;
    let clouded = clouds_on.on() && shows;
    plan.piece = sky_piece(&sky, shows, &plan.piece);
    apply_night(&mut plan.piece, &night, &path, sun_direction);
    plan.clouds.prepare(
        (
            (&clouds, clouded),
            tables.cloud_noise(),
            (eye, medium.surface),
            seconds,
        ),
        (&*sun, &plan.piece),
        &mut budget,
    );
    plan.clouds.prepare_view(
        (&clouds, eye, seconds),
        (display, render),
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
    plan.piece.domes_size = plan.clouds.dome_size();
    plan.domes = dome_light(&plan.piece, &sky);
    let inputs = EnvironmentInputs {
        light: [
            sun.direction,
            plan.piece.sun_light,
            plan.piece.zenith,
            plan.piece.horizon,
            plan.piece.ground,
        ],
        clouds: plan.clouds.environment_key(),
        tables: plan.tables.shown(),
        air: plan.air,
    };
    plan.environment.prepare(inputs, &mut environment_steps);
    plan.frame = plan.clouds.frame();
}

pub(crate) struct AirTables {
    transmittance: Texture<SkyGpu>,
    multiple: Texture<SkyGpu>,
    derived: Buffer<SkyGpu>,
    domes: Buffer<SkyGpu>,
    sampler: Sampler<SkyGpu>,
    pass: Pass<SkyGpu>,
    building: ComputePipeline<SkyGpu>,
    deriving: ComputePipeline<SkyGpu>,
    copies: Vec<Pass<SkyGpu>>,
    uploaded: Option<u64>,
}

impl AirTables {
    fn install(cx: &mut Install<'_, SkyGpu>, view: Texture<SkyGpu>) -> Result<Self, Error> {
        use crate::shaders::sky::{light, tables};
        let transmittance = cx.texture("sky_transmittance", FORMAT, Lifetime::Kept);
        let multiple = cx.texture("sky_multiple", FORMAT, Lifetime::Kept);
        let derived = cx.buffer("sky_derived", Lifetime::Kept);
        let domes = cx.buffer("sky_dome_light", Lifetime::Kept);
        let sampler = cx.sampler(engine_core::Sampler {
            mipmap_filter: Some(Filter::Nearest),
            ..engine_core::Sampler::LINEAR_CLAMP
        });
        cx.frame(light::SKY_LAYERS, Bind::Texture(view))?;
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
                (tables::SKY_DOMES, Bind::Buffer(domes)),
            ],
        )?;
        Ok(Self {
            transmittance,
            multiple,
            derived,
            domes,
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
        domes: SkyDomes,
    ) {
        let shown = plan.shown();
        let transmittance = cx.allocate(self.transmittance, extent(TRANSMITTANCE_SIZE), 1);
        let multiple = cx.allocate(self.multiple, extent(MULTIPLE_SIZE), 1);
        if transmittance == Allocation::Fresh || multiple == Allocation::Fresh {
            self.uploaded = None;
        }
        let (Allocation::Fresh | Allocation::Kept) =
            cx.allocate_buffer(self.derived, DERIVED_BYTES);
        let (Allocation::Fresh | Allocation::Kept) =
            cx.allocate_buffer(self.domes, size_of::<SkyDomes>() as u64);
        cx.write(self.domes, 0, &[domes]);
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
        cx.read_back::<SkyLightLevel>(0, self.derived, 0..DERIVED_BYTES, AfterRead::Keep);
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
    layers: Texture<Self>,
    tables: AirTables,
    clouds: CloudPass,
    environment: Environment,
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
            .at(AttachPoint::LateOpaque)
            .label("engine/sky-dome")
            .span("sky_dome")
            .id();
        let layers = cx.texture("sky_layers", FORMAT, Lifetime::Kept);
        let mut tables = AirTables::install(cx, layers)?;
        let environment = Environment::install(cx)?;
        let clouds = CloudPass::install(
            cx,
            tables.bindings(),
            (environment.pass(), dome_pass),
            layers,
        )?;
        tables.connect(cx, piece_buffer, clouds.uniforms(), environment.pass());
        let dome = dome_pipeline(cx, dome_pass, tables.transmittance)?;
        Ok(Self {
            layers,
            tables,
            clouds,
            environment,
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
        let view = if plan.tables.shows() {
            VIEW_SIZE
        } else {
            NO_VIEW_SIZE
        };
        let fresh =
            cx.allocate(self.layers, plan.clouds.layers_extent(view), 1) == Allocation::Fresh;
        self.tables
            .prepare(cx, (&plan.tables, tables.sky()), plan.domes);
        self.clouds.prepare(
            cx,
            (&plan.clouds, tables.cloud_noise()),
            (plan.frame, fresh),
        );
        self.environment.prepare(cx, &plan.environment);
        let (Allocation::Fresh | Allocation::Kept) =
            cx.allocate_buffer(self.piece_buffer, size_of::<SkyPiece>() as u64);
        cx.write(self.piece_buffer, 0, &[plan.piece]);
        let (Allocation::Fresh | Allocation::Kept) =
            cx.allocate_buffer(self.air_buffer, size_of::<SkyAir>() as u64);
        cx.write(self.air_buffer, 0, &[plan.air]);
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        if self.clouds.owns(pass) {
            return self.clouds.encode(pass, cx);
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
                    material: None,
                },
            );
        }
        0
    }
}
