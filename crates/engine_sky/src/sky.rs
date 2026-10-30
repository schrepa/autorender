use crate::shaders::entry::{draw, sky};
use engine_core::table_data::sky::SkyTables;
use engine_core::table_data::sky::{MULTIPLE_SIZE, TRANSMITTANCE_SIZE};
use engine_core::{
    Applied, Clouds, Grading, NightSky, ReflectionProbeSettings, RenderSnapshot, RenderStats, Sky,
    SkyLighting, SkyLightingValues, StaticTable, Sunlight,
};
use engine_faces::environment::{EnvironmentFace, EnvironmentPlan};
use engine_faces::sun::{Sun, SunFace};
use engine_render::cached::{Bound, Cached};
use engine_render::composite::{PixelEffect, PostStage};
use engine_render::gpu_calls::Queue;
use engine_render::graph::{
    Access, FrameGraph, GraphTable, Lifetime, PassDecl, PassId, ResourceDecl, ResourceId, Stage,
    Targets,
};
use engine_render::parts::{
    Create, DrawAt, Encode, GpuPart, Hook, OpaqueSlot, PartDecl, PartEntry, Prelude, PreludeStage,
    Shading, Step, View,
};
use engine_render::pieces::{PieceAt, PieceDecl};
use engine_render::post::inverse_or_identity;
use engine_render::profiling::GpuProfiler;
use engine_render::renderer::Core;
use engine_render::screen_effects::haze_runs;
use engine_render::shaders::Entry;
use engine_render::{buffers, memory, pipelines};
use engine_time_of_day::TimeOfDay;
use glam::Vec3;

use crate::bindings::{
    COMPOSITE_HAZE, FRAME_CLOUDS, FRAME_ENVIRONMENT, FRAME_SKY_TABLES, SKY_TABLES,
    SKY_TRANSMITTANCE,
};
use crate::clouds::{CloudFrame, CloudPass};
use crate::environment::{ENVIRONMENT_FORMAT, Environment, SkyLight};
use crate::gpu_layouts::{SKY_TABLE_GROUP, SkyPiece};
use crate::haze::Haze;

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const VIEW_SIZE: (u32, u32) = (192, 108);
const ENVIRONMENT_HEIGHT_STEP: f32 = 200.0;
const DERIVED_BYTES: u64 = 7 * 16;

pub(crate) const FACES: usize = 6;

pub(crate) type AirBindings<'a> = (
    &'a wgpu::TextureView,
    &'a wgpu::TextureView,
    &'a wgpu::Sampler,
);

pub(crate) type SkyCopy<'a> = (u64, &'a wgpu::Buffer, u64, u64);

pub(crate) type CloudLight<'a> = (Sun, &'a SkyPiece);

pub(crate) const DERIVED_SUN: u64 = 0;
pub(crate) const DERIVED_ZENITH: u64 = 1;
pub(crate) const DERIVED_HORIZON: u64 = 2;
pub(crate) const DERIVED_GROUND: u64 = 3;
pub(crate) const DERIVED_AIR: u64 = 4;
pub(crate) const DERIVED_GLOW: u64 = 5;

struct Table {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

fn table(
    device: &wgpu::Device,
    label: &str,
    (width, height): (u32, u32),
    usage: wgpu::TextureUsages,
) -> Table {
    let texture = memory::texture(
        device,
        &wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage,
            view_formats: &[],
        },
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    Table { texture, view }
}

struct Passes {
    view: wgpu::ComputePipeline,
    derive: wgpu::ComputePipeline,
}

pub(crate) struct AirTables {
    layout: pipelines::GroupLayout,
    passes: Option<Passes>,
    groups: Cached<wgpu::Buffer, [wgpu::BindGroup; 2]>,
    pass_group: wgpu::BindGroup,
    transmittance: Table,
    multiple: Table,
    view: Table,
    blank: Table,
    blank_out: Table,
    sampler: wgpu::Sampler,
    derived: wgpu::Buffer,
    tables: Option<u64>,
    view_key: Option<(u64, [u32; 20], [u32; 3], u32)>,
    last_seconds: Option<f64>,
    generation: u64,
    environment_height: i32,
    active: bool,
    pub(crate) built: u32,
}

impl AirTables {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let uploaded = wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING;
        let transmittance = table(
            device,
            "engine/sky-transmittance",
            TRANSMITTANCE_SIZE,
            uploaded,
        );
        let pass_layout = pipelines::layout(device, &SKY_TRANSMITTANCE);
        let pass_group = pipelines::group(
            device,
            "engine/sky-transmittance",
            &SKY_TRANSMITTANCE,
            &pass_layout,
            &[(
                "sky_transmittance_table",
                pipelines::view(&transmittance.view),
            )],
        );
        Self {
            layout: pipelines::layout(device, &SKY_TABLES),
            passes: None,
            groups: Cached::default(),
            pass_group,
            transmittance,
            multiple: table(device, "engine/sky-multiple", MULTIPLE_SIZE, uploaded),
            view: table(
                device,
                "engine/sky-view",
                VIEW_SIZE,
                wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            ),
            blank: table(
                device,
                "engine/sky-blank",
                (1, 1),
                wgpu::TextureUsages::TEXTURE_BINDING,
            ),
            blank_out: table(
                device,
                "engine/sky-blank-out",
                (1, 1),
                wgpu::TextureUsages::STORAGE_BINDING,
            ),
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("engine/sky-table-sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            derived: buffers::storage(
                device,
                "engine/sky-derived",
                DERIVED_BYTES,
                wgpu::BufferUsages::COPY_SRC,
            ),
            tables: None,
            view_key: None,
            last_seconds: None,
            generation: 0,
            environment_height: 0,
            active: false,
            built: 0,
        }
    }

    pub(crate) fn pass_group(&self) -> &wgpu::BindGroup {
        &self.pass_group
    }

    pub(crate) fn bindings(&self) -> AirBindings<'_> {
        (&self.view.view, &self.transmittance.view, &self.sampler)
    }

    pub(crate) fn environment_key(&self) -> [f32; 4] {
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

    fn upload(&self, queue: &Queue, tables: &SkyTables) {
        for (table, texels, (width, height)) in [
            (
                &self.transmittance,
                &tables.transmittance,
                TRANSMITTANCE_SIZE,
            ),
            (&self.multiple, &tables.multiple, MULTIPLE_SIZE),
        ] {
            queue.write_texture(
                table.texture.as_image_copy(),
                bytemuck::cast_slice(texels),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(width * 8),
                    rows_per_image: Some(height),
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    pub(crate) fn encode(
        &mut self,
        device: &wgpu::Device,
        queue: &Queue,
        (settings, tables): (&Sky, Option<&StaticTable<SkyTables>>),
        (sun, eye, exposure, seconds, moon): (Vec3, Vec3, f32, f64, [f32; 4]),
        (pieces, piece, copies): (&wgpu::Buffer, &mut SkyPiece, &[SkyCopy<'_>]),
        profiler: &GpuProfiler,
    ) -> Option<wgpu::CommandEncoder> {
        self.built = 0;
        let settings = settings.sanitized();
        let uploaded = match tables {
            Some(tables) if settings.enabled && self.tables != Some(tables.id) => {
                self.upload(queue, &tables.table);
                self.tables = Some(tables.id);
                self.built += 2;
                true
            }
            _ => false,
        };
        self.active = settings.enabled && self.tables.is_some();
        let Some(uploaded_tables) = self.tables.filter(|_| self.active) else {
            self.view_key = None;
            return None;
        };
        let sun = sun.normalize_or(Vec3::Y);
        let height = engine_core::math::max(eye.y + settings.altitude, 0.5);
        let elapsed = self.last_seconds.replace(seconds).map_or(0.0, |last| {
            engine_core::math::max((seconds - last) as f32, 0.0)
        });
        piece.air_rayleigh = [
            settings.rayleigh[0],
            settings.rayleigh[1],
            settings.rayleigh[2],
            settings.rayleigh_height,
        ];
        piece.air_mie = [
            settings.mie,
            settings.mie_absorption,
            settings.mie_height,
            settings.mie_g,
        ];
        piece.air_ozone = [
            settings.ozone[0],
            settings.ozone[1],
            settings.ozone[2],
            settings.sun_intensity,
        ];
        piece.air_ground = [
            settings.ground_albedo[0],
            settings.ground_albedo[1],
            settings.ground_albedo[2],
            settings.sun_size.to_radians(),
        ];
        piece.air_radii = [
            settings.planet_radius,
            settings.planet_radius + settings.atmosphere_height,
            settings.adaptation,
            height * 0.001,
        ];
        piece.air_sun = sun.extend(exposure).to_array();
        piece.air_frame = [elapsed, 0.0, 0.0, 0.0];
        piece.air_moon = moon;
        let (ozone, ground, radii) = (piece.air_ozone, piece.air_ground, piece.air_radii);
        let air: [u32; 20] = bytemuck::cast([
            piece.air_rayleigh,
            piece.air_mie,
            [ozone[0], ozone[1], ozone[2], 0.0],
            [ground[0], ground[1], ground[2], 0.0],
            [radii[0], radii[1], 0.0, 0.0],
        ]);
        let view_key = (
            uploaded_tables,
            air,
            sun.to_array().map(f32::to_bits),
            height.to_bits(),
        );
        let build_view = uploaded || self.view_key != Some(view_key);
        let passes = self.passes.get_or_insert_with(|| passes(device));
        let bound = self.groups.get_or_insert_with(pieces.clone(), || {
            table_groups(
                device,
                &self.layout,
                [
                    &self.transmittance,
                    &self.multiple,
                    &self.view,
                    &self.blank,
                    &self.blank_out,
                ],
                &self.sampler,
                pieces,
                &self.derived,
            )
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("engine/sky-encoder"),
        });
        if let Some(span) = profiler.span("sky") {
            span.mark(&mut encoder, false);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/sky"),
                timestamp_writes: None,
            });
            let groups = |(width, height): (u32, u32)| {
                (
                    width.div_ceil(SKY_TABLE_GROUP),
                    height.div_ceil(SKY_TABLE_GROUP),
                )
            };
            if build_view {
                pass.set_pipeline(&passes.view);
                pass.set_bind_group(0, &bound[0], &[]);
                let (x, y) = groups(VIEW_SIZE);
                pass.dispatch_workgroups(x, y, 1);
                self.built += 1;
            }
            pass.set_pipeline(&passes.derive);
            pass.set_bind_group(0, &bound[1], &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        for &(index, buffer, offset, bytes) in copies {
            encoder.copy_buffer_to_buffer(&self.derived, index * 16, buffer, offset, bytes);
        }
        if let Some(span) = profiler.span("sky") {
            span.mark(&mut encoder, true);
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
        Some(encoder)
    }

    pub(crate) fn capacity_bytes(&self) -> u64 {
        let texels = |(width, height): (u32, u32)| u64::from(width) * u64::from(height);
        (texels(TRANSMITTANCE_SIZE) + texels(MULTIPLE_SIZE) + texels(VIEW_SIZE)) * 8
    }
}

fn passes(device: &wgpu::Device) -> Passes {
    let pipeline_layout =
        pipelines::layout_of(device, &crate::pipeline_layouts::SKY_TABLE_PIPELINES);
    let pipeline = |entry: Entry| pipelines::compute_entry(device, &pipeline_layout, entry);
    Passes {
        view: pipeline(sky::build_sky_view),
        derive: pipeline(sky::derive_light),
    }
}

fn table_groups(
    device: &wgpu::Device,
    layout: &pipelines::GroupLayout,
    [transmittance, multiple, view, blank, blank_out]: [&Table; 5],
    sampler: &wgpu::Sampler,
    pieces: &wgpu::Buffer,
    derived: &wgpu::Buffer,
) -> [wgpu::BindGroup; 2] {
    let group = |reads: [&Table; 3], out: &Table| {
        pipelines::group(
            device,
            "engine/sky-tables",
            &SKY_TABLES,
            layout,
            &[
                buffers::entry("pieces", pieces),
                ("transmittance_table", pipelines::view(&reads[0].view)),
                ("multiple_table", pipelines::view(&reads[1].view)),
                ("view_table", pipelines::view(&reads[2].view)),
                ("table_sampler", pipelines::sampler(sampler)),
                ("table_out", pipelines::view(&out.view)),
                buffers::entry("derived", derived),
            ],
        )
    };
    [
        group([transmittance, multiple, blank], view),
        group([transmittance, multiple, view], blank_out),
    ]
}

struct Sections {
    sky: Sky,
    clouds: Clouds,
    haze: engine_core::Haze,
    night: NightSky,
    day: TimeOfDay,
    probes: ReflectionProbeSettings,
    sun: Vec3,
    light: SkyLightingValues,
    exposure: f32,
}

impl Sections {
    fn of(snapshot: &RenderSnapshot) -> Self {
        Self {
            sky: snapshot.section::<Sky>(),
            clouds: snapshot.section::<Clouds>(),
            haze: snapshot.section::<engine_core::Haze>(),
            night: snapshot.section::<NightSky>(),
            day: snapshot.section::<TimeOfDay>(),
            probes: snapshot.section::<ReflectionProbeSettings>(),
            sun: snapshot.section::<Sunlight>().values().sun_direction,
            light: snapshot.section::<SkyLighting>().values(),
            exposure: snapshot.section::<Grading>().values().exposure,
        }
    }
}

fn night_lighting(sections: &Sections) -> [[f32; 4]; 3] {
    let (night, day, sun) = (sections.night, sections.day, sections.sun);
    let clock = if day.enabled {
        day
    } else {
        TimeOfDay::default()
    };
    let moon = (day.enabled && night.moon > 0.0).then(|| day.moon_direction(night.moon_age));
    let lit = moon.map_or(0.0, |moon| 0.5 - 0.5 * moon.dot(sun.normalize_or_zero()));
    [
        [
            night.stars,
            moon.map_or(0.0, |_| night.moon),
            night.cirrus,
            lit,
        ],
        clock.star_turn().to_array(),
        moon.unwrap_or(Vec3::Y).extend(night.moon_size).to_array(),
    ]
}

fn sky_piece(sections: &Sections, kept: &SkyPiece) -> SkyPiece {
    let (light, sky) = (sections.light, sections.sky);
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
        horizon: light.horizon.extend(sections.exposure).to_array(),
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

fn apply_night(piece: &mut SkyPiece, sections: &Sections) {
    [piece.night, piece.star_turn, piece.moon] = night_lighting(sections);
    let night = sections.night;
    piece.night_shape = [
        night.cirrus_height,
        night.cirrus_drift[0],
        night.cirrus_drift[1],
        night.star_density,
    ];
}

pub(crate) fn apply_haze(
    piece: &mut SkyPiece,
    haze: &engine_core::Haze,
    view_projection: glam::Mat4,
    eye: Vec3,
    marches: bool,
) {
    let haze = haze.sanitized();
    if !haze.enabled {
        return;
    }
    let visibility = if haze.visibility > 0.0 {
        haze.visibility
    } else {
        let far = inverse_or_identity(view_projection).project_point3(Vec3::ZERO);
        2.0 * far.distance(eye)
    };
    let extinction = 3.912 / engine_core::math::max(visibility, 1.0);
    let falloff = piece.aerial[2];
    let at_zero = extinction * engine_core::math::min(falloff * eye.y, 60.0).exp();
    piece.aerial[0] = 0.0;
    piece.aerial[1] = engine_core::math::max(piece.aerial[1], at_zero) * haze.strength;
    piece.aerial_color[3] = 1.0;
    if marches {
        piece.haze_march = [haze.reach, haze.forward, haze.spread, 1.0 / haze.size];
    }
}

const CAMERA_VIEW: &[&str] = &["refraction"];
const DOME_PASS: usize = 0;
const MARCH_PASS: usize = 1;
const COMPOSITE_PASS: usize = 2;
const HAZE_PASS: usize = 3;

pub(crate) static PART: PartEntry = PartEntry {
    decl: PartDecl {
        name: "sky",
        brick: "sky",
        passes: &[
            PassDecl {
                name: "cloud_dome",
                stage: Stage::Prepare,
            },
            PassDecl {
                name: "cloud_march",
                stage: Stage::Sky,
            },
            PassDecl {
                name: "cloud_composite",
                stage: Stage::Sky,
            },
            PassDecl {
                name: "haze",
                stage: Stage::PostScene,
            },
        ],
        resources: &[
            ResourceDecl {
                name: "cloud_dome",
                lifetime: Lifetime::Kept,
            },
            ResourceDecl {
                name: "cloud_view",
                lifetime: Lifetime::Frame,
            },
            ResourceDecl {
                name: "haze",
                lifetime: Lifetime::Kept,
            },
        ],
        hooks: &[Hook::Opaque(OpaqueSlot::Sky), Hook::View],
        contributes: &[
            FRAME_CLOUDS,
            FRAME_SKY_TABLES,
            FRAME_ENVIRONMENT,
            COMPOSITE_HAZE,
        ],
        effects: &[PixelEffect {
            stage: PostStage::Light,
            module: "sky::haze",
            reads: &["haze"],
        }],
        labels: &[],
        lazy: false,
        faces: &[engine_render::offer_face!(SkyPass => dyn EnvironmentFace)],
        needs: &[],
        piece: Some(PieceDecl {
            name: "sky",
            module: "sky::piece",
            ty: "SkyPiece",
            size: size_of::<SkyPiece>() as u64,
        }),
    },
    create,
};

fn create(cx: &Create<'_>) -> Box<dyn GpuPart> {
    let _owner = memory::owned("sky");
    Box::new(SkyPass::new(cx.core))
}

#[derive(Clone, Copy, Default)]
struct Ids {
    passes: [Option<PassId>; 4],
    cloud_dome: Option<ResourceId>,
    cloud_view: Option<ResourceId>,
    depth: Option<ResourceId>,
    shadow_maps: Option<ResourceId>,
    bounce: Option<ResourceId>,
    haze: Option<ResourceId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SkyTargets {
    Frame(Targets),
    Probe,
}

impl SkyTargets {
    fn index(self) -> usize {
        match self {
            Self::Frame(Targets::Canvas) => 0,
            Self::Frame(Targets::Post) => 1,
            Self::Frame(Targets::Temporal) => 2,
            Self::Frame(Targets::Debug) => 3,
            Self::Probe => 4,
        }
    }
}

pub(crate) struct SkyPass {
    tables: AirTables,
    clouds: CloudPass,
    environment: Environment,
    haze: Haze,
    pipelines: [Option<wgpu::RenderPipeline>; 5],
    ids: Ids,
    frame: Option<CloudFrame>,
    plan: EnvironmentPlan,
    piece: SkyPiece,
    at: PieceAt,
}

impl SkyPass {
    fn new(core: &Core) -> Self {
        let device = &core.device;
        let clouds = CloudPass::new(device, &core.queue);
        let tables = AirTables::new(device);
        let environment = Environment::new(
            device,
            core.pieces_buffer(),
            engine_core::ReflectionProbeSettings::default().resolution,
            clouds.dome(),
            tables.bindings(),
        );
        Self {
            tables,
            clouds,
            environment,
            haze: Haze::default(),
            pipelines: Default::default(),
            ids: Ids::default(),
            frame: None,
            plan: EnvironmentPlan::default(),
            piece: SkyPiece::default(),
            at: PieceAt::default(),
        }
    }

    fn prepare_clouds(
        &mut self,
        core: &Core,
        snapshot: &RenderSnapshot,
        clouds: &Clouds,
        gpu_mode: bool,
        sun: Sun,
    ) {
        let pieces = core.pieces_buffer();
        self.clouds.prepare(
            &core.device,
            &core.queue,
            (
                clouds,
                snapshot.tables.cloud_noise.as_ref(),
                (snapshot.camera_position, snapshot.surface_under_eye),
                snapshot.presented_seconds,
            ),
            ((sun, &self.piece), self.tables.bindings(), pieces),
            !gpu_mode,
        );
        self.clouds.prepare_view(
            &core.device,
            &core.queue,
            (clouds, snapshot.camera_position, snapshot.presented_seconds),
            snapshot.view_projection,
            core.display.region,
            ((sun, &self.piece), self.tables.bindings(), pieces),
            &core.depth.view,
        );
        self.environment
            .set_sources(&core.device, pieces, self.clouds.environment_dome());
        [
            self.piece.domes_older,
            self.piece.domes_newer,
            self.piece.domes_older_mirror,
            self.piece.domes_newer_mirror,
            self.piece.domes_eye,
            self.piece.domes_view,
        ] = self.clouds.dome_piece();
    }

    fn prepare_tables(
        &mut self,
        core: &Core,
        snapshot: &RenderSnapshot,
        sections: &Sections,
    ) -> Option<wgpu::CommandEncoder> {
        use std::mem::offset_of;
        let pieces = core.pieces_buffer();
        let base = self.at.offset;
        let at = |offset: usize| base + offset as u64;
        let mut copies = vec![
            (DERIVED_SUN, pieces, at(offset_of!(SkyPiece, sun_light)), 12),
            (DERIVED_ZENITH, pieces, at(offset_of!(SkyPiece, zenith)), 12),
            (
                DERIVED_HORIZON,
                pieces,
                at(offset_of!(SkyPiece, horizon)),
                16,
            ),
            (DERIVED_GROUND, pieces, at(offset_of!(SkyPiece, ground)), 12),
            (
                DERIVED_AIR,
                pieces,
                at(offset_of!(SkyPiece, aerial_color)),
                12,
            ),
            (
                DERIVED_GLOW,
                pieces,
                at(offset_of!(SkyPiece, aerial)) + 12,
                4,
            ),
        ];
        copies.extend(self.clouds.sky_copies());
        let [night, _, moon] = night_lighting(sections);
        self.tables.encode(
            &core.device,
            &core.queue,
            (&sections.sky, snapshot.tables.sky.as_ref()),
            (
                sections.sun,
                snapshot.camera_position,
                sections.exposure,
                snapshot.presented_seconds,
                [moon[0], moon[1], moon[2], night[1] * night[3]],
            ),
            (pieces, &mut self.piece, &copies),
            &core.profiler,
        )
    }

    fn prepare_environment(
        &mut self,
        core: &Core,
        snapshot: &RenderSnapshot,
        (probe_settings, sun): (ReflectionProbeSettings, Sun),
    ) -> Option<wgpu::CommandEncoder> {
        let probe_list = snapshot.probe_list();
        let most_cubes = (core.display.max_layers / FACES as u32).max(1);
        let wanted_cubes = 1 + (probe_list.len() as u32).min(most_cubes - 1);
        let cubes = if wanted_cubes == 1 {
            1
        } else if wanted_cubes > self.environment.cubes() {
            wanted_cubes.next_power_of_two().min(most_cubes)
        } else {
            self.environment.cubes()
        };
        let light: SkyLight = [
            sun.direction,
            sun.color,
            self.piece.zenith,
            self.piece.horizon,
            self.piece.ground,
        ];
        let keys = (self.clouds.environment_key(), self.tables.environment_key());
        let mut environment_encoder = self
            .environment
            .needs_encoding(probe_settings.resolution, cubes, &light, keys)
            .then(|| {
                core.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("engine/environment-encoder"),
                    })
            });
        let (lost, sky_changed) = match environment_encoder.as_mut() {
            Some(encoder) => (
                self.environment
                    .ensure(&core.device, encoder, probe_settings.resolution, cubes),
                self.environment.encode_sky(encoder, &light, keys).1,
            ),
            None => (false, false),
        };
        self.plan = EnvironmentPlan {
            cubes,
            most_cubes,
            lost,
            sky_changed,
        };
        environment_encoder
    }

    fn ensure_sky(
        &mut self,
        device: &wgpu::Device,
        canvas: wgpu::TextureFormat,
        targets: SkyTargets,
    ) {
        let index = targets.index();
        if self.pipelines[index].is_some() {
            return;
        }
        let (fragment, color_targets) = match targets {
            SkyTargets::Frame(targets) => {
                (sky_fragment(Some(targets)), targets.color_targets(canvas))
            }
            SkyTargets::Probe => (sky_fragment(None), vec![Some(ENVIRONMENT_FORMAT.into())]),
        };
        self.pipelines[index] = Some(sky_pipeline(device, fragment, &color_targets));
    }
}

impl GpuPart for SkyPass {
    fn install(&mut self, table: &GraphTable) {
        self.ids = Ids {
            passes: std::array::from_fn(|index| {
                PART.decl
                    .passes
                    .get(index)
                    .and_then(|pass| table.pass_named(pass.name))
            }),
            cloud_dome: table.resource("cloud_dome"),
            cloud_view: table.resource("cloud_view"),
            depth: table.resource("depth"),
            shadow_maps: table.resource("shadow_maps"),
            bounce: table.resource("bounce"),
            haze: table.resource("haze"),
        };
    }

    fn forget_targets(&mut self, targets: Targets) {
        self.pipelines[SkyTargets::Frame(targets).index()] = None;
    }

    fn prepare(&mut self, cx: &mut Step<'_>, prelude: &mut Prelude) {
        let frame = cx.frame;
        let core = &*cx.core;
        let snapshot = frame.snapshot;
        let sun = cx
            .parts
            .face::<dyn SunFace>()
            .map_or(Sun::NONE, |face| face.sun());
        let post = cx.parts.post();
        let post_disabled = post.is_some_and(|post| post.disabled());
        let sections = Sections::of(snapshot);
        self.piece = sky_piece(&sections, &self.piece);
        apply_night(&mut self.piece, &sections);
        apply_haze(
            &mut self.piece,
            &sections.haze,
            snapshot.view_projection,
            snapshot.camera_position,
            !frame.kind.debug && !post_disabled && haze_runs(snapshot),
        );
        (self.piece.cloud_shadow, self.piece.cloud_layer) =
            crate::clouds::shadow_lighting(&sections.clouds, snapshot.camera_position);
        self.prepare_clouds(core, snapshot, &sections.clouds, frame.kind.gpu_mode, sun);
        let derived = self.prepare_tables(core, snapshot, &sections);
        self.piece.state[1] = f32::from(u8::from(derived.is_some()));
        if let Some(encoder) = derived {
            prelude.push_encoder(PreludeStage::Tables, "pre_sky", encoder);
        }
        if let Some(encoder) = self.prepare_environment(core, snapshot, (sections.probes, sun)) {
            prelude.push_encoder(PreludeStage::Derived, "pre_environment", encoder);
        }
        let post_active = post.is_some_and(|post| post.active());
        self.haze.prepare(
            core,
            snapshot,
            (post_active, frame.kind.display, frame.kind.debug),
        );
    }

    fn finish(&mut self, cx: &mut Step<'_>) {
        let frame = cx.frame;
        let core = &*cx.core;
        self.frame = None;
        let Some(shape) = frame.shape else {
            return;
        };
        let canvas = core.canvas_format();
        self.ensure_sky(&core.device, canvas, SkyTargets::Frame(shape.targets));
        if frame.kind.gpu_mode && !frame.kind.debug && !frame.snapshot.probe_list().is_empty() {
            self.ensure_sky(&core.device, canvas, SkyTargets::Probe);
        }
        if shape.targets == Targets::Debug {
            return;
        }
        let target = if shape.targets == Targets::Canvas {
            (canvas, true)
        } else {
            (engine_render::post::SCENE_FORMAT, false)
        };
        if self.clouds.marches_view() {
            self.clouds.prepare_composite(&core.device, target);
        }
        self.frame = self.clouds.frame(target);
    }

    fn declare(&self, cx: &View<'_>, graph: &mut FrameGraph) {
        let table = cx.table;
        let ids = self.ids;
        let shape = cx.frame.shape;
        if self.haze.runs()
            && shape.is_some_and(|shape| shape.targets != Targets::Canvas)
            && let Some(haze) = ids.passes[HAZE_PASS]
        {
            let _ = graph
                .add(table, haze)
                .read(ids.depth)
                .read(ids.shadow_maps)
                .read(ids.cloud_dome)
                .read(ids.bounce)
                .read(ids.haze)
                .write(ids.haze);
        }
        let Some(frame) = self.frame else {
            return;
        };
        let [dome, march, composite, _] = ids.passes;
        if let Some(dome) = dome.filter(|_| frame.dome()) {
            let _ = graph.add(table, dome).write(ids.cloud_dome);
        }
        if frame.view().is_none() {
            return;
        }
        if let Some(march) = march {
            let _ = graph
                .add(table, march)
                .read(ids.depth)
                .write(ids.cloud_view);
        }
        if let Some(composite) = composite {
            let node = graph.add(table, composite).read(ids.cloud_view);
            let scene = cx
                .frame
                .shape
                .as_ref()
                .and_then(|shape| engine_render::parts::scene_target(table, shape));
            let _ = match scene {
                Some(scene) => node.color(scene, Access::Load),
                None => node,
            };
        }
    }

    fn encode(
        &mut self,
        pass: usize,
        cx: &mut Encode<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> u32 {
        let profiler = &cx.core.profiler;
        match pass {
            DOME_PASS => self.clouds.encode_dome(encoder, profiler),
            MARCH_PASS => self.clouds.encode_view(encoder, profiler),
            HAZE_PASS => {
                let camera = cx.core.shared_group("frame", CAMERA_VIEW);
                self.haze.encode(encoder, camera, profiler);
            }
            COMPOSITE_PASS => {
                if let Some(target) = self.frame.and_then(|frame| frame.view()) {
                    let mut render = cx.render_pass(
                        encoder,
                        "engine/cloud-composite",
                        profiler
                            .span("cloud_composite")
                            .map(engine_render::profiling::Span::render),
                    );
                    self.clouds.draw_composite(
                        &mut render,
                        cx.core.shared_group("frame", CAMERA_VIEW),
                        target,
                    );
                }
            }
            _ => {}
        }
        0
    }

    fn draw(&self, at: DrawAt<'_>, cx: &View<'_>, pass: &mut wgpu::RenderPass<'_>) -> u32 {
        let (targets, group) = match at {
            DrawAt::Opaque {
                slot: OpaqueSlot::Sky,
                ..
            } => {
                let Some(shape) = cx.frame.shape else {
                    return 0;
                };
                (
                    SkyTargets::Frame(shape.targets),
                    cx.core.shared_group("frame", CAMERA_VIEW),
                )
            }
            DrawAt::View { view, .. } => (SkyTargets::Probe, cx.core.view_group(view, CAMERA_VIEW)),
            DrawAt::Opaque { .. } | DrawAt::ShadowCasters { .. } | DrawAt::Transparent { .. } => {
                return 0;
            }
        };
        let Some(pipeline) = &self.pipelines[targets.index()] else {
            return 0;
        };
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.set_bind_group(1, self.tables.pass_group(), &[]);
        pass.draw(0..3, 0..1);
        0
    }

    fn contribute<'s>(&'s self, group: &str, out: &mut Vec<(&'static str, Bound<'s>)>) {
        if group == "composite" {
            out.extend(self.haze.binding().map(|haze| ("haze", haze)));
            return;
        }
        if group != "frame" {
            return;
        }
        let (view_table, _, table_sampler) = self.tables.bindings();
        let clouds = self.clouds.dome();
        let (environment, environment_sampler) = self.environment.binding();
        out.extend([
            (
                "cloud_shadows",
                Bound::fixed(self.clouds.shadows().as_entire_binding()),
            ),
            ("sky_view_table", Bound::fixed(pipelines::view(view_table))),
            (
                "sky_table_sampler",
                Bound::fixed(pipelines::sampler(table_sampler)),
            ),
            (
                "sky_clouds",
                Bound::counted(shown_dome(clouds.key, 1), pipelines::view(clouds.new)),
            ),
            (
                "sky_clouds_old",
                Bound::counted(shown_dome(clouds.key, 0), pipelines::view(clouds.old)),
            ),
            (
                "sky_cloud_sampler",
                Bound::fixed(pipelines::sampler(clouds.sampler)),
            ),
            ("environment_maps", environment),
            ("environment_sampler", environment_sampler),
        ]);
    }

    fn after_submit(&mut self) {
        let frame = self.frame.take();
        self.clouds.after_submit(frame.as_ref());
        self.haze.after_submit();
    }

    fn capacity_bytes(&self) -> u64 {
        self.environment.capacity_bytes()
            + self.clouds.capacity_bytes()
            + self.tables.capacity_bytes()
            + self.haze.capacity_bytes()
    }

    fn stats(&self, _cx: &View<'_>, stats: &mut RenderStats) {
        stats.clouds = self.clouds.stats();
        stats.sky.tables = self.tables.built;
        stats.haze = Applied {
            active: self.haze.runs(),
        };
    }

    fn shading(&self, snapshot: &RenderSnapshot, shading: &mut Shading) {
        let clouds = snapshot.section::<Clouds>();
        let shadows = clouds.enabled && clouds.coverage > 0.0 && clouds.shadows > 0.0;
        shading.set("CLOUD_SHADOWS", if shadows { 1.0 } else { 0.0 });
    }

    fn place_piece(&mut self, at: PieceAt) {
        self.at = at;
    }

    fn piece(&self) -> Option<&[u8]> {
        Some(bytemuck::bytes_of(&self.piece))
    }

    fn produces(&self, resource: ResourceId) -> bool {
        self.haze.runs() && self.ids.haze == Some(resource)
    }
}

impl EnvironmentFace for SkyPass {
    fn plan(&self) -> EnvironmentPlan {
        self.plan
    }

    fn size(&self) -> u32 {
        self.environment.size()
    }

    fn radiance_face(&self, face: usize) -> &wgpu::TextureView {
        self.environment.radiance_face(face)
    }

    fn finish(&self, encoder: &mut wgpu::CommandEncoder, cube: u32) {
        self.environment.finish(encoder, cube);
    }
}

fn shown_dome((generation, roles, shown): (u64, [u8; 2], bool), role: usize) -> u64 {
    (generation * 4 + u64::from(roles[role])) * 2 + u64::from(shown)
}

pub(crate) const SKY_VERTEX: Entry = draw::vs_sky;

fn sky_fragment(targets: Option<Targets>) -> Entry {
    match targets {
        Some(Targets::Canvas) => draw::fs_sky,
        Some(Targets::Post) => draw::fs_sky_post,
        Some(Targets::Temporal) => draw::fs_sky_temporal,
        Some(Targets::Debug) => draw::fs_debug_sky,
        None => draw::fs_probe_sky,
    }
}

fn sky_pipeline(
    device: &wgpu::Device,
    fragment: Entry,
    targets: &[Option<wgpu::ColorTargetState>],
) -> wgpu::RenderPipeline {
    let layout = pipelines::layout_of(device, &crate::pipeline_layouts::SKY_PIPELINES);
    pipelines::fullscreen(
        device,
        &pipelines::Fullscreen {
            label: "engine/sky",
            layout: &layout,
            vertex: SKY_VERTEX,
            fragment,
            constants: &[],
            targets,
            depth: Some(wgpu::DepthStencilState {
                format: engine_render::renderer::DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                stencil: Default::default(),
                bias: Default::default(),
            }),
        },
    )
}
