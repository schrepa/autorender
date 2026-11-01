use bevy_ecs::prelude::*;
use engine_core::math::{max, min};
use engine_core::{Camera, Error, FrameTime, View};
use engine_places::probes::{ProbeView, ProbeViews};
use engine_places::sun::{Sun, SunPath};
use engine_places::tonemap::Display;
use engine_render::{
    AfterRead, Bind, Buffer, ComputePipeline, Copied, CopyRegion, Count, DebugShown, DepthUse,
    Draw, Encode, GpuPart, Install, Lifetime, MainView, Pass, Pipeline, Prepare, RenderPipeline,
    Sampler, SamplerDesc, Shape, Texture, wgpu,
};
use glam::{Mat4, Vec3};

use crate::clouds::{CloudFrame, CloudPass, shadow_lighting};
use crate::environment::{Environment, SkyLight};
use crate::gpu_layouts::{SKY_TABLE_GROUP, SkyAir, SkyPiece};
use crate::haze::{Haze, HazeFrame};
use crate::settings::{
    Clouds, Haze as HazeSettings, NightSky, Sky, SkyLighting, SkyLightingValues, haze_runs,
};
use crate::tables::sky::{MULTIPLE_SIZE, SkyTables, TRANSMITTANCE_SIZE};
use crate::tables::{StaticTable, StaticTables};
use crate::under_eye::SurfaceUnderEye;
use crate::views::SkyCounts;

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const VIEW_SIZE: (u32, u32) = (192, 108);
const ENVIRONMENT_HEIGHT_STEP: f32 = 200.0;
const DERIVED_BYTES: u64 = 7 * 16;
const COUNTS: u32 = 0;
const COUNT_BYTES: u64 = 16;

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

type AirInputs = (Vec3, Vec3, f32, f64, [f32; 4]);

pub(crate) struct AirTables {
    transmittance: Texture<SkyGpu>,
    multiple: Texture<SkyGpu>,
    view: Texture<SkyGpu>,
    derived: Buffer<SkyGpu>,
    sampler: Sampler<SkyGpu>,
    view_pass: Pass<SkyGpu>,
    derive_pass: Pass<SkyGpu>,
    building: ComputePipeline<SkyGpu>,
    deriving: ComputePipeline<SkyGpu>,
    copies: Vec<Pass<SkyGpu>>,
    tables: Option<u64>,
    view_key: Option<(u64, [u32; 20], [u32; 3], u32)>,
    last_seconds: Option<f64>,
    generation: u64,
    environment_height: i32,
    active: bool,
    pub(crate) built: u32,
}

impl AirTables {
    fn install(cx: &mut Install<'_, SkyGpu>) -> Result<Self, Error> {
        use crate::shaders::sky::{light, tables};
        let transmittance = cx.texture("sky_transmittance", FORMAT, Lifetime::Kept);
        let multiple = cx.texture("sky_multiple", FORMAT, Lifetime::Kept);
        let view = cx.texture("sky_view", FORMAT, Lifetime::Kept);
        let derived = cx.buffer("sky_derived", Lifetime::Kept);
        let sampler = cx.sampler(SamplerDesc::LINEAR_CLAMP);
        cx.frame(light::SKY_VIEW_TABLE, Bind::Texture(view))?;
        cx.frame(light::SKY_TABLE_SAMPLER, Bind::Sampler(sampler))?;
        let view_pass = cx.pass("sky_view").id();
        let derive_pass = cx.pass("sky_derive").after(view_pass).id();
        let building = cx.compute_pipeline(
            view_pass,
            tables::BUILD_SKY_VIEW,
            &[
                (tables::TRANSMITTANCE_TABLE, Bind::Texture(transmittance)),
                (tables::MULTIPLE_TABLE, Bind::Texture(multiple)),
                (tables::TABLE_SAMPLER, Bind::Sampler(sampler)),
                (tables::TABLE_OUT, Bind::Texture(view)),
            ],
        )?;
        let deriving = cx.compute_pipeline(
            derive_pass,
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
            view_pass,
            derive_pass,
            building,
            deriving,
            copies: Vec::new(),
            tables: None,
            view_key: None,
            last_seconds: None,
            generation: 0,
            environment_height: 0,
            active: false,
            built: 0,
        })
    }

    fn connect(&mut self, cx: &mut Install<'_, SkyGpu>, piece: Buffer<SkyGpu>, clouds: [Buffer<SkyGpu>; 2]) {
        let [march, view] = clouds;
        self.copies = vec![
            cx.copy("sky_derived_piece", Copied::Buffer(self.derived), Copied::Buffer(piece)),
            cx.copy("sky_derived_clouds", Copied::Buffer(self.derived), Copied::Buffer(march)),
            cx.copy("sky_derived_view_clouds", Copied::Buffer(self.derived), Copied::Buffer(view)),
        ];
    }

    fn bindings(&self) -> (Texture<SkyGpu>, Sampler<SkyGpu>) {
        (self.transmittance, self.sampler)
    }

    fn environment_key(&self) -> [f32; 4] {
        if self.active {
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

    fn upload(&self, cx: &mut Prepare<'_, SkyGpu>, tables: &SkyTables) {
        for (texture, texels, (width, height)) in [
            (self.transmittance, &tables.transmittance, TRANSMITTANCE_SIZE),
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

    fn allocate(&self, cx: &mut Prepare<'_, SkyGpu>) {
        cx.allocate(self.transmittance, extent(TRANSMITTANCE_SIZE), 1);
        cx.allocate(self.multiple, extent(MULTIPLE_SIZE), 1);
        cx.allocate(self.view, extent(VIEW_SIZE), 1);
        cx.allocate_buffer(self.derived, DERIVED_BYTES);
    }

    fn prepare(
        &mut self,
        cx: &mut Prepare<'_, SkyGpu>,
        (settings, tables): (&Sky, Option<&StaticTable<SkyTables>>),
        (sun, eye, exposure, seconds, moon): AirInputs,
        air: &mut SkyAir,
        cloud_regions: impl Fn() -> [CopyRegion; 3],
    ) -> bool {
        self.built = 0;
        self.allocate(cx);
        let uploaded = match tables {
            Some(tables) if settings.enabled && self.tables != Some(tables.id) => {
                self.upload(cx, &tables.table);
                self.tables = Some(tables.id);
                self.built += 2;
                true
            }
            _ => false,
        };
        self.active = settings.enabled && self.tables.is_some();
        let Some(uploaded_tables) = self.tables.filter(|_| self.active) else {
            self.view_key = None;
            return false;
        };
        let height = max(eye.y + settings.altitude, 0.5);
        let elapsed = self
            .last_seconds
            .replace(seconds)
            .map_or(0.0, |last| max((seconds - last) as f32, 0.0));
        write_air(air, settings, (sun, exposure, height, elapsed, moon));
        let (ozone, ground, radii) = (air.ozone, air.ground, air.radii);
        let keyed: [u32; 20] = bytemuck::cast([
            air.rayleigh,
            air.mie,
            [ozone[0], ozone[1], ozone[2], 0.0],
            [ground[0], ground[1], ground[2], 0.0],
            [radii[0], radii[1], 0.0, 0.0],
        ]);
        let view_key = (
            uploaded_tables,
            keyed,
            sun.to_array().map(f32::to_bits),
            height.to_bits(),
        );
        let build_view = uploaded || self.view_key != Some(view_key);
        if build_view {
            cx.run(self.view_pass);
            self.built += 1;
        }
        cx.run(self.derive_pass);
        let regions = [Vec::from(piece_regions()), Vec::from(cloud_regions()), Vec::from(cloud_regions())];
        for (&pass, regions) in self.copies.iter().zip(regions) {
            cx.copy(pass, &regions);
            cx.run(pass);
        }
        if build_view {
            let environment_height = (height / ENVIRONMENT_HEIGHT_STEP).floor() as i32;
            let sun_moved = self.view_key.is_none_or(|key| key.2 != view_key.2);
            if uploaded || sun_moved || environment_height != self.environment_height {
                self.generation += 1;
            }
            self.environment_height = environment_height;
            self.view_key = Some(view_key);
        }
        true
    }

    fn owns(&self, pass: Pass<SkyGpu>) -> bool {
        pass == self.view_pass || pass == self.derive_pass
    }

    fn encode(&self, pass: Pass<SkyGpu>, cx: &mut Encode<'_, SkyGpu>) -> u32 {
        if pass == self.view_pass {
            let groups = |(width, height): (u32, u32)| {
                [
                    width.div_ceil(SKY_TABLE_GROUP),
                    height.div_ceil(SKY_TABLE_GROUP),
                    1,
                ]
            };
            cx.dispatch(&self.building, Count::Groups(groups(VIEW_SIZE)));
        } else {
            cx.dispatch(&self.deriving, Count::Groups([1, 1, 1]));
        }
        0
    }
}

fn write_air(air: &mut SkyAir, settings: &Sky, (sun, exposure, height, elapsed, moon): (Vec3, f32, f32, f32, [f32; 4])) {
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
    air.sun = sun.extend(exposure).to_array();
    air.frame = [elapsed, 0.0, 0.0, 0.0];
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

fn sky_piece(sky: &Sky, light: &SkyLightingValues, exposure: f32, kept: &SkyPiece) -> SkyPiece {
    let (fog, aerial, physical) = if sky.enabled {
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
            [0.0; 4],
            [0.0, extinction * 0.001 * sky.aerial, 0.001 / height, 0.0],
            1.0,
        )
    } else {
        (
            [
                light.fog_start,
                light.fog_density,
                light.fog_falloff,
                light.fog_max,
            ],
            [
                light.aerial_start,
                light.aerial_density,
                light.aerial_falloff,
                light.aerial_glow,
            ],
            0.0,
        )
    };
    SkyPiece {
        zenith: light.zenith.extend(light.ambient).to_array(),
        horizon: light.horizon.extend(exposure).to_array(),
        ground: light.ground.extend(0.0).to_array(),
        fog,
        aerial,
        aerial_color: light.aerial_color.extend(0.0).to_array(),
        cloud_shadow: [0.0; 4],
        cloud_layer: [0.0; 4],
        night: [0.0; 4],
        star_turn: [0.0, 0.0, 0.0, 1.0],
        moon: [0.0; 4],
        night_shape: [0.0; 4],
        haze_march: [0.0; 4],
        sun_light: [0.0; 4],
        state: [physical, 0.0, 0.0, 0.0],
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
    piece.aerial[0] = 0.0;
    piece.aerial[1] = max(piece.aerial[1], at_zero) * haze.strength;
    piece.aerial_color[3] = 1.0;
    if marches {
        piece.haze_march = [haze.reach, haze.forward, haze.spread, 1.0 / haze.size];
    }
}

pub(crate) struct SkyGpu {
    tables: AirTables,
    clouds: CloudPass,
    environment: Environment,
    haze: Haze,
    dome: Pipeline<Self, MainView>,
    probe_dome: Pipeline<Self, ProbeViews>,
    piece_buffer: Buffer<Self>,
    air_buffer: Buffer<Self>,
    counts: Buffer<Self>,
    piece: SkyPiece,
    air: SkyAir,
    frame: Option<CloudFrame>,
}

impl SkyGpu {
    pub(crate) fn draw_dome(&self, _: &(), draw: &mut Draw<'_, Self, MainView>) -> u32 {
        draw.draw(
            &self.dome,
            Shape::Pulled {
                vertices: 0..3,
                instances: 0..1,
            },
        );
        0
    }

    pub(crate) fn draw_probe_face(&self, _: &ProbeView, draw: &mut Draw<'_, Self, ProbeViews>) -> u32 {
        draw.draw(
            &self.probe_dome,
            Shape::Pulled {
                vertices: 0..3,
                instances: 0..1,
            },
        );
        0
    }

    fn write(&self, cx: &mut Prepare<'_, Self>) {
        cx.allocate_buffer(self.piece_buffer, size_of::<SkyPiece>() as u64);
        cx.write(self.piece_buffer, 0, &[self.piece]);
        cx.allocate_buffer(self.air_buffer, size_of::<SkyAir>() as u64);
        cx.write(self.air_buffer, 0, &[self.air]);
        cx.allocate_buffer(self.counts, COUNT_BYTES);
        let counts = [
            self.clouds.rows,
            self.clouds.view_pixels,
            self.tables.built,
            0,
        ];
        cx.write(self.counts, 0, &counts);
        cx.read_back::<SkyCounts>(COUNTS, self.counts, 0..COUNT_BYTES, AfterRead::Keep);
    }
}

pub(crate) type SkyReads = (
    Res<'static, Sky>,
    Res<'static, Clouds>,
    Res<'static, HazeSettings>,
    Res<'static, NightSky>,
    Res<'static, SkyLighting>,
    Res<'static, Sun>,
    Res<'static, SunPath>,
    Res<'static, Display>,
    Res<'static, StaticTables>,
    Res<'static, SurfaceUnderEye>,
    Res<'static, View>,
    Res<'static, Camera>,
    Res<'static, FrameTime>,
    Res<'static, DebugShown>,
);

impl GpuPart for SkyGpu {
    type Reads = SkyReads;

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use crate::shaders::sky::{debug, draw, piece};
        let piece_buffer = cx.buffer("sky_piece", Lifetime::Kept);
        let air_buffer = cx.buffer("sky_air", Lifetime::Kept);
        let counts = cx.buffer("sky_counts", Lifetime::Kept);
        cx.frame(piece::SKY_PIECE, Bind::Buffer(piece_buffer))?;
        cx.frame(piece::SKY_AIR, Bind::Buffer(air_buffer))?;
        let mut tables = AirTables::install(cx)?;
        let clouds = CloudPass::install(cx, tables.bindings())?;
        tables.connect(cx, piece_buffer, clouds.uniforms());
        let environment = Environment::install(cx)?;
        let haze = Haze::install(cx)?;
        let sky = |fragment, debug| RenderPipeline {
            vertex: draw::VS_SKY,
            fragment: Some(fragment),
            debug,
            vertices: &[],
            primitive: wgpu::PrimitiveState::default(),
            depth: DepthUse::Test,
            blend: None,
            writes: wgpu::ColorWrites::ALL,
        };
        let dome = cx.render_pipeline(
            MainView,
            &sky(draw::FS_SKY, Some(debug::FS_DEBUG_SKY)),
            &[(
                draw::SKY_TRANSMITTANCE_TABLE,
                Bind::Texture(tables.transmittance),
            )],
        )?;
        let probe_dome = cx.render_pipeline(ProbeViews, &sky(draw::FS_PROBE_SKY, None), &[])?;
        Ok(Self {
            tables,
            clouds,
            environment,
            haze,
            dome,
            probe_dome,
            piece_buffer,
            air_buffer,
            counts,
            piece: SkyPiece::default(),
            air: SkyAir::default(),
            frame: None,
        })
    }

    fn prepare(&mut self, reads: SystemParamItem<'_, '_, Self::Reads>, cx: &mut Prepare<'_, Self>) {
        let (sky, clouds, haze, night, light, sun, path, display, tables, under, view, camera, time, shown) =
            reads;
        self.clouds.submitted(self.frame.take());
        self.haze.submitted();
        let debug = shown.owner.is_some();
        let eye = camera.position;
        let view_projection = view.view_projection();
        let seconds = time.presented_seconds;
        let sun_direction = Vec3::from_slice(&sun.direction[..3]);
        let values = light.values();
        let exposure = display.grade[3];
        let hazes = !debug && haze_runs(&haze, &sky, &values);
        self.piece = sky_piece(&sky, &values, exposure, &self.piece);
        apply_night(&mut self.piece, &night, &path, sun_direction);
        apply_haze(&mut self.piece, &haze, view_projection, eye, hazes);
        (self.piece.cloud_shadow, self.piece.cloud_layer) = shadow_lighting(&clouds, eye);
        let (display_size, render_size) = (view.display_size(), view.render_size());
        let (exact, displayed) = (cx.exact(), cx.displayed());
        self.clouds.prepare(
            cx,
            (&*clouds, tables.cloud_noise(), (eye, under.height), seconds),
            (&*sun, &self.piece),
            exact,
        );
        self.clouds.prepare_view(
            cx,
            (&*clouds, eye, seconds),
            (view_projection, display_size, render_size),
            (&*sun, &self.piece),
        );
        [
            self.piece.domes_older,
            self.piece.domes_newer,
            self.piece.domes_older_mirror,
            self.piece.domes_newer_mirror,
            self.piece.domes_eye,
            self.piece.domes_view,
        ] = self.clouds.dome_piece();
        self.piece.domes_layers = self.clouds.dome_layers();
        let [night_light, _, moon] = night_lighting(&night, &path, sun_direction);
        let derived = self.tables.prepare(
            cx,
            (&*sky, tables.sky()),
            (
                sun_direction,
                eye,
                exposure,
                seconds,
                [moon[0], moon[1], moon[2], night_light[1] * night_light[3]],
            ),
            &mut self.air,
            || self.clouds.copy_regions(),
        );
        self.piece.state[1] = f32::from(u8::from(derived));
        let sky_light: SkyLight = [
            sun.direction,
            sun.color,
            self.piece.zenith,
            self.piece.horizon,
            self.piece.ground,
        ];
        let keys = (self.clouds.environment_key(), self.tables.environment_key());
        self.environment.prepare(cx, &sky_light, keys);
        let haze_frame = HazeFrame {
            view_projection,
            eye,
            display: display_size,
            render: render_size,
            displayed,
        };
        self.piece.haze = self.haze.prepare(cx, hazes, &haze_frame);
        if !debug {
            self.frame = self.clouds.frame();
            if let Some(frame) = self.frame {
                self.clouds.run(cx, frame);
            }
        }
        self.write(cx);
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
        if self.tables.owns(pass) {
            return self.tables.encode(pass, cx);
        }
        0
    }
}
