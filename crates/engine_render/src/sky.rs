use crate::shaders::entry::sky;
use engine_core::table_data::sky::SkyTables;
use engine_core::table_data::sky::{MULTIPLE_SIZE, TRANSMITTANCE_SIZE};
use engine_core::{RenderSnapshot, RenderStats, Sky, StaticTable};
use engine_time_of_day::TimeOfDay;
use glam::Vec3;

use crate::bindings::{
    Contribution, SKY_TABLES, SKY_TRANSMITTANCE, Sample, Sampling, Stages, read_of, sampler,
    texture, uniform_of,
};
use crate::cached::Bound;
use crate::clouds::{CloudFrame, CloudPass};
use crate::environment::{ENVIRONMENT_FORMAT, Environment};
use crate::gpu_data::LightingUniform;
use crate::gpu_layouts::clouds::{CLOUD_SHADOW_BYTES, DomeUniform};
use crate::gpu_layouts::constants::SKY_TABLE_GROUP;
use crate::graph::{
    Access, FrameGraph, GraphTable, Lifetime, PassDecl, PassId, ResourceDecl, ResourceId, Stage,
    Targets,
};
use crate::parts::{
    Create, DrawAt, Encode, GpuPart, Hook, OpaqueSlot, PartDecl, PartEntry, Prelude,
    PreludeStage, Step, UniformsOut, View,
};
use crate::probes::FACES;
use crate::renderer::Core;

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const VIEW_SIZE: (u32, u32) = (192, 108);
const ENVIRONMENT_HEIGHT_STEP: f32 = 200.0;
const DERIVED_BYTES: u64 = 7 * 16;

use crate::gpu_layouts::sky::SkyUniform;

pub(crate) const SKY_UNIFORM_BYTES: usize = size_of::<SkyUniform>();

pub(crate) type AirBindings<'a> = (
    &'a wgpu::Buffer,
    &'a wgpu::TextureView,
    &'a wgpu::TextureView,
    &'a wgpu::Sampler,
);

pub(crate) type SkyCopy<'a> = (u64, &'a wgpu::Buffer, u64, u64);

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
    let texture = crate::memory::texture(
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
    groups: [wgpu::BindGroup; 2],
}

pub(crate) struct AirTables {
    layout: crate::pipelines::GroupLayout,
    passes: Option<Passes>,
    pass_group: wgpu::BindGroup,
    transmittance: Table,
    multiple: Table,
    view: Table,
    blank: Table,
    blank_out: Table,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
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
        let pass_layout = crate::pipelines::layout(device, &SKY_TRANSMITTANCE);
        let pass_group = crate::pipelines::group(
            device,
            "engine/sky-transmittance",
            &SKY_TRANSMITTANCE,
            &pass_layout,
            &[(
                "sky_transmittance_table",
                crate::pipelines::view(&transmittance.view),
            )],
        );
        Self {
            layout: crate::pipelines::layout(device, &SKY_TABLES),
            passes: None,
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
            uniform: crate::buffers::uniform(
                device,
                "engine/sky-uniform",
                SKY_UNIFORM_BYTES as u64,
            ),
            derived: crate::buffers::storage(
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
        (
            &self.uniform,
            &self.view.view,
            &self.transmittance.view,
            &self.sampler,
        )
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

    fn upload(&self, queue: &crate::gpu_calls::Queue, tables: &SkyTables) {
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
        queue: &crate::gpu_calls::Queue,
        (settings, tables): (&Sky, Option<&StaticTable<SkyTables>>),
        (sun, eye, exposure, seconds, moon): (Vec3, Vec3, f32, f64, [f32; 4]),
        copies: &[SkyCopy<'_>],
        profiler: &crate::profiling::GpuProfiler,
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
        let uniform = SkyUniform {
            rayleigh: [
                settings.rayleigh[0],
                settings.rayleigh[1],
                settings.rayleigh[2],
                settings.rayleigh_height,
            ],
            mie: [
                settings.mie,
                settings.mie_absorption,
                settings.mie_height,
                settings.mie_g,
            ],
            ozone: [
                settings.ozone[0],
                settings.ozone[1],
                settings.ozone[2],
                settings.sun_intensity,
            ],
            ground: [
                settings.ground_albedo[0],
                settings.ground_albedo[1],
                settings.ground_albedo[2],
                settings.sun_size.to_radians(),
            ],
            radii: [
                settings.planet_radius,
                settings.planet_radius + settings.atmosphere_height,
                settings.adaptation,
                height * 0.001,
            ],
            sun: sun.extend(exposure).to_array(),
            frame: [elapsed, 0.0, 0.0, 0.0],
            moon,
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
        let air: [u32; 20] = bytemuck::cast([
            uniform.rayleigh,
            uniform.mie,
            [uniform.ozone[0], uniform.ozone[1], uniform.ozone[2], 0.0],
            [uniform.ground[0], uniform.ground[1], uniform.ground[2], 0.0],
            [uniform.radii[0], uniform.radii[1], 0.0, 0.0],
        ]);
        let view_key = (
            uploaded_tables,
            air,
            sun.to_array().map(f32::to_bits),
            height.to_bits(),
        );
        let build_view = uploaded || self.view_key != Some(view_key);
        let passes = self.passes.get_or_insert_with(|| {
            passes(
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
                &self.uniform,
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
                pass.set_bind_group(0, &passes.groups[0], &[]);
                let (x, y) = groups(VIEW_SIZE);
                pass.dispatch_workgroups(x, y, 1);
                self.built += 1;
            }
            pass.set_pipeline(&passes.derive);
            pass.set_bind_group(0, &passes.groups[1], &[]);
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

fn passes(
    device: &wgpu::Device,
    layout: &crate::pipelines::GroupLayout,
    [transmittance, multiple, view, blank, blank_out]: [&Table; 5],
    sampler: &wgpu::Sampler,
    uniform: &wgpu::Buffer,
    derived: &wgpu::Buffer,
) -> Passes {
    let pipeline_layout =
        crate::pipelines::layout_of(device, &crate::pipeline_layouts::SKY_TABLE_PIPELINES);
    let pipeline = |entry: crate::shaders::Entry| {
        crate::pipelines::compute_entry(device, &pipeline_layout, entry)
    };
    let group = |reads: [&Table; 3], out: &Table| {
        crate::pipelines::group(
            device,
            "engine/sky-tables",
            &SKY_TABLES,
            layout,
            &[
                crate::buffers::entry("air", uniform),
                (
                    "transmittance_table",
                    crate::pipelines::view(&reads[0].view),
                ),
                ("multiple_table", crate::pipelines::view(&reads[1].view)),
                ("view_table", crate::pipelines::view(&reads[2].view)),
                ("table_sampler", crate::pipelines::sampler(sampler)),
                ("table_out", crate::pipelines::view(&out.view)),
                crate::buffers::entry("derived", derived),
            ],
        )
    };
    Passes {
        view: pipeline(sky::build_sky_view),
        derive: pipeline(sky::derive_light),
        groups: [
            group([transmittance, multiple, blank], view),
            group([transmittance, multiple, view], blank_out),
        ],
    }
}

pub(crate) fn night_lighting(snapshot: &RenderSnapshot, sun: Vec3) -> [[f32; 4]; 3] {
    let night = snapshot.environment.night_sky.sanitized();
    let day = snapshot
        .extracted
        .get::<TimeOfDay>()
        .copied()
        .map_or_else(TimeOfDay::off, TimeOfDay::sanitized);
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

pub(crate) fn apply_night(
    lighting: &mut crate::gpu_data::LightingUniform,
    snapshot: &RenderSnapshot,
) {
    [lighting.night, lighting.star_turn, lighting.moon] =
        night_lighting(snapshot, snapshot.environment.atmosphere.sun_direction);
    let night = snapshot.environment.night_sky.sanitized();
    lighting.night_shape = [
        night.cirrus_height,
        night.cirrus_drift[0],
        night.cirrus_drift[1],
        night.star_density,
    ];
}

pub(crate) fn apply_haze(
    lighting: &mut crate::gpu_data::LightingUniform,
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
        let far = crate::post::inverse_or_identity(view_projection).project_point3(Vec3::ZERO);
        2.0 * far.distance(eye)
    };
    let extinction = 3.912 / engine_core::math::max(visibility, 1.0);
    let falloff = lighting.aerial[2];
    let at_zero = extinction * engine_core::math::min(falloff * eye.y, 60.0).exp();
    lighting.aerial[0] = 0.0;
    lighting.aerial[1] = engine_core::math::max(lighting.aerial[1], at_zero) * haze.strength;
    lighting.aerial_color[3] = 1.0;
    if marches {
        lighting.haze_march = [haze.reach, haze.forward, haze.spread, 1.0 / haze.size];
    }
}

const CAMERA_VIEW: &[&str] = &["refraction"];
const DOME_PASS: usize = 0;
const MARCH_PASS: usize = 1;
const FRAGMENT: Stages = Stages::FRAGMENT;

pub(crate) const FRAME_CLOUDS: Contribution = Contribution {
    group: "frame",
    set: "clouds",
    bindings: &[read_of("cloud_shadows", CLOUD_SHADOW_BYTES, FRAGMENT)],
};

pub(crate) const FRAME_SKY_TABLES: Contribution = Contribution {
    group: "frame",
    set: "sky_tables",
    bindings: &[
        uniform_of("sky_air", size_of::<SkyUniform>(), FRAGMENT),
        texture(
            "sky_view_table",
            Sample::Float,
            crate::bindings::View::D2,
            FRAGMENT,
        ),
        sampler("sky_table_sampler", Sampling::Filtering, FRAGMENT),
        texture(
            "sky_clouds",
            Sample::Float,
            crate::bindings::View::D2Array,
            FRAGMENT,
        ),
        texture(
            "sky_clouds_old",
            Sample::Float,
            crate::bindings::View::D2Array,
            FRAGMENT,
        ),
        sampler("sky_cloud_sampler", Sampling::Filtering, FRAGMENT),
        uniform_of("sky_domes", size_of::<DomeUniform>(), FRAGMENT),
    ],
};

pub(crate) const FRAME_ENVIRONMENT: Contribution = Contribution {
    group: "frame",
    set: "environment",
    bindings: &[
        texture(
            "environment_maps",
            Sample::Float,
            crate::bindings::View::CubeArray,
            FRAGMENT,
        ),
        sampler("environment_sampler", Sampling::Filtering, FRAGMENT),
    ],
};

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
        ],
        hooks: &[Hook::Opaque(OpaqueSlot::Sky), Hook::ProbeFace],
        contributes: &[FRAME_CLOUDS, FRAME_SKY_TABLES, FRAME_ENVIRONMENT],
        effects: &[],
        labels: &[],
        lazy: false,
    },
    create,
};

fn create(cx: &Create<'_>) -> Box<dyn GpuPart> {
    let _owner = crate::memory::owned("sky");
    Box::new(SkyPass::new(cx.core))
}

#[derive(Clone, Copy, Default)]
struct Ids {
    passes: [Option<PassId>; 3],
    cloud_dome: Option<ResourceId>,
    cloud_view: Option<ResourceId>,
    depth: Option<ResourceId>,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct EnvironmentPlan {
    pub cubes: u32,
    pub most_cubes: u32,
    pub lost: bool,
    pub sky_changed: bool,
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
    pipelines: [Option<wgpu::RenderPipeline>; 5],
    ids: Ids,
    frame: Option<CloudFrame>,
    plan: EnvironmentPlan,
}

impl SkyPass {
    fn new(core: &Core) -> Self {
        let device = &core.device;
        let clouds = CloudPass::new(device, &core.queue);
        let tables = AirTables::new(device);
        let environment = Environment::new(
            device,
            core.camera.lighting_buffer(),
            engine_core::ReflectionProbeSettings::default().resolution,
            clouds.dome(),
            tables.bindings(),
        );
        Self {
            tables,
            clouds,
            environment,
            pipelines: Default::default(),
            ids: Ids::default(),
            frame: None,
            plan: EnvironmentPlan::default(),
        }
    }

    pub(crate) fn environment(&self) -> &Environment {
        &self.environment
    }

    pub(crate) fn environment_plan(&self) -> EnvironmentPlan {
        self.plan
    }

    pub(crate) fn cloud_shadows(&self) -> &wgpu::Buffer {
        self.clouds.shadows()
    }

    fn prepare_clouds(
        &mut self,
        core: &Core,
        snapshot: &RenderSnapshot,
        gpu_mode: bool,
        lighting: &LightingUniform,
    ) {
        self.clouds.prepare(
            &core.device,
            &core.queue,
            (
                &snapshot.environment.clouds,
                snapshot.tables.cloud_noise.as_ref(),
                (snapshot.camera_position, snapshot.surface_under_eye),
                snapshot.presented_seconds,
            ),
            (lighting, self.tables.bindings()),
            !gpu_mode,
        );
        self.clouds.prepare_view(
            &core.device,
            &core.queue,
            (
                &snapshot.environment.clouds,
                snapshot.camera_position,
                snapshot.presented_seconds,
            ),
            snapshot.view_projection,
            core.display.region,
            (lighting, self.tables.bindings()),
            &core.depth.view,
        );
        self.environment
            .set_clouds(&core.device, self.clouds.environment_dome());
    }

    fn prepare_tables(
        &mut self,
        core: &Core,
        snapshot: &RenderSnapshot,
    ) -> Option<wgpu::CommandEncoder> {
        use std::mem::offset_of;
        let lighting = core.camera.lighting_buffer();
        let at = |offset: usize| offset as u64;
        let mut copies = vec![
            (
                DERIVED_SUN,
                lighting,
                at(offset_of!(LightingUniform, sun_color)),
                12,
            ),
            (
                DERIVED_ZENITH,
                lighting,
                at(offset_of!(LightingUniform, zenith)),
                12,
            ),
            (
                DERIVED_HORIZON,
                lighting,
                at(offset_of!(LightingUniform, horizon)),
                16,
            ),
            (
                DERIVED_GROUND,
                lighting,
                at(offset_of!(LightingUniform, ground)),
                12,
            ),
            (
                DERIVED_AIR,
                lighting,
                at(offset_of!(LightingUniform, aerial_color)),
                12,
            ),
            (
                DERIVED_GLOW,
                lighting,
                at(offset_of!(LightingUniform, aerial)) + 12,
                4,
            ),
        ];
        copies.extend(self.clouds.sky_copies());
        let sun = snapshot.environment.atmosphere.sun_direction;
        let [night, _, moon] = night_lighting(snapshot, sun);
        self.tables.encode(
            &core.device,
            &core.queue,
            (&snapshot.environment.sky, snapshot.tables.sky.as_ref()),
            (
                sun,
                snapshot.camera_position,
                snapshot.environment.atmosphere.exposure,
                snapshot.presented_seconds,
                [moon[0], moon[1], moon[2], night[1] * night[3]],
            ),
            &copies,
            &core.profiler,
        )
    }

    fn prepare_environment(
        &mut self,
        core: &Core,
        snapshot: &RenderSnapshot,
        lighting: &LightingUniform,
    ) -> Option<wgpu::CommandEncoder> {
        let probe_settings = snapshot.environment.reflection_probes;
        let probe_list = probe_list(snapshot);
        let most_cubes = (core.display.max_layers / FACES as u32).max(1);
        let wanted_cubes = 1 + (probe_list.len() as u32).min(most_cubes - 1);
        let cubes = if wanted_cubes == 1 {
            1
        } else if wanted_cubes > self.environment.cubes() {
            wanted_cubes.next_power_of_two().min(most_cubes)
        } else {
            self.environment.cubes()
        };
        let keys = (
            self.clouds.environment_key(),
            self.tables.environment_key(),
        );
        let mut environment_encoder = self
            .environment
            .needs_encoding(probe_settings.resolution, cubes, lighting, keys)
            .then(|| {
                core.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("engine/environment-encoder"),
                    })
            });
        let (lost, sky_changed) = match environment_encoder.as_mut() {
            Some(encoder) => (
                self.environment.ensure(
                    &core.device,
                    encoder,
                    probe_settings.resolution,
                    cubes,
                ),
                self.environment.encode_sky(encoder, lighting, keys).1,
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
            SkyTargets::Frame(targets) => (
                crate::entries::sky_fragment(Some(targets)),
                targets.color_targets(canvas),
            ),
            SkyTargets::Probe => (
                crate::entries::sky_fragment(None),
                vec![Some(ENVIRONMENT_FORMAT.into())],
            ),
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
        };
    }

    fn forget_targets(&mut self, targets: Targets) {
        self.pipelines[SkyTargets::Frame(targets).index()] = None;
    }

    fn uniforms(
        &mut self,
        cx: &mut Step<'_>,
        lighting: &mut LightingUniform,
        _out: &mut UniformsOut,
    ) {
        let frame = cx.frame;
        let snapshot = frame.snapshot;
        apply_night(lighting, snapshot);
        let post_disabled = cx
            .parts
            .iter()
            .any(|(decl, part)| decl.name == "post" && part.disabled());
        apply_haze(
            lighting,
            &snapshot.environment.haze,
            snapshot.view_projection,
            snapshot.camera_position,
            !frame.kind.debug
                && !post_disabled
                && crate::screen_effects::haze_runs(&snapshot.environment),
        );
        (lighting.cloud_shadow, lighting.cloud_layer) =
            crate::clouds::shadow_lighting(&snapshot.environment.clouds, snapshot.camera_position);
    }

    fn prepare(&mut self, cx: &mut Step<'_>, prelude: &mut Prelude) {
        let frame = cx.frame;
        let core = &*cx.core;
        let Some(lighting) = frame.lighting.as_ref() else {
            return;
        };
        let snapshot = frame.snapshot;
        self.prepare_clouds(core, snapshot, frame.kind.gpu_mode, lighting);
        if let Some(encoder) = self.prepare_tables(core, snapshot) {
            prelude.push_encoder(PreludeStage::Tables, "pre_sky", encoder);
        }
        if let Some(encoder) = self.prepare_environment(core, snapshot, lighting) {
            prelude.push_encoder(PreludeStage::Derived, "pre_environment", encoder);
        }
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
        if frame.kind.gpu_mode && !frame.kind.debug && !probe_list(frame.snapshot).is_empty() {
            self.ensure_sky(&core.device, canvas, SkyTargets::Probe);
        }
        if shape.targets == Targets::Debug {
            return;
        }
        let target = if shape.targets == Targets::Canvas {
            (canvas, true)
        } else {
            (crate::post::SCENE_FORMAT, false)
        };
        if self.clouds.marches_view() {
            self.clouds.prepare_composite(&core.device, target);
        }
        self.frame = self.clouds.frame(target);
    }

    fn declare(&self, cx: &View<'_>, graph: &mut FrameGraph) {
        let Some(frame) = self.frame else {
            return;
        };
        let table = cx.table;
        let [dome, march, composite] = self.ids.passes;
        if let Some(dome) = dome.filter(|_| frame.dome()) {
            let _ = graph.add(table, dome).write(self.ids.cloud_dome);
        }
        if frame.view().is_none() {
            return;
        }
        if let Some(march) = march {
            let _ = graph
                .add(table, march)
                .read(self.ids.depth)
                .write(self.ids.cloud_view);
        }
        if let Some(composite) = composite {
            let node = graph.add(table, composite).read(self.ids.cloud_view);
            let scene = cx
                .frame
                .shape
                .as_ref()
                .and_then(|shape| crate::parts::scene_target(table, shape));
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
            _ => {
                if let Some(target) = self.frame.and_then(|frame| frame.view()) {
                    let mut render = cx.render_pass(
                        encoder,
                        "engine/cloud-composite",
                        profiler
                            .span("cloud_composite")
                            .map(crate::profiling::Span::render),
                    );
                    self.clouds.draw_composite(
                        &mut render,
                        cx.core.shared_group("frame", CAMERA_VIEW),
                        target,
                    );
                }
            }
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
            DrawAt::ProbeFace { face, .. } => {
                (SkyTargets::Probe, cx.core.probe_group(face, CAMERA_VIEW))
            }
            DrawAt::Opaque { .. } | DrawAt::ShadowCasters { .. } => return 0,
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
        if group != "frame" {
            return;
        }
        let (air, view_table, _, table_sampler) = self.tables.bindings();
        let clouds = self.clouds.dome();
        let (environment, environment_sampler) = self.environment.binding();
        out.extend([
            (
                "cloud_shadows",
                Bound::fixed(self.clouds.shadows().as_entire_binding()),
            ),
            ("sky_air", Bound::fixed(air.as_entire_binding())),
            (
                "sky_view_table",
                Bound::fixed(crate::pipelines::view(view_table)),
            ),
            (
                "sky_table_sampler",
                Bound::fixed(crate::pipelines::sampler(table_sampler)),
            ),
            (
                "sky_clouds",
                Bound::counted(
                    shown_dome(clouds.key, 1),
                    crate::pipelines::view(clouds.new),
                ),
            ),
            (
                "sky_clouds_old",
                Bound::counted(
                    shown_dome(clouds.key, 0),
                    crate::pipelines::view(clouds.old),
                ),
            ),
            (
                "sky_cloud_sampler",
                Bound::fixed(crate::pipelines::sampler(clouds.sampler)),
            ),
            (
                "sky_domes",
                Bound::fixed(clouds.uniform.as_entire_binding()),
            ),
            ("environment_maps", environment),
            ("environment_sampler", environment_sampler),
        ]);
    }

    fn after_submit(&mut self) {
        let frame = self.frame.take();
        self.clouds.after_submit(frame.as_ref());
    }

    fn capacity_bytes(&self) -> u64 {
        self.environment.capacity_bytes()
            + self.clouds.capacity_bytes()
            + self.tables.capacity_bytes()
    }

    fn stats(&self, _cx: &View<'_>, stats: &mut RenderStats) {
        stats.clouds = self.clouds.stats();
        stats.sky.tables = self.tables.built;
    }
}

fn probe_list(snapshot: &RenderSnapshot) -> &[engine_core::RenderReflectionProbe] {
    if snapshot.environment.reflection_probes.enabled {
        &snapshot.reflection_probes.list
    } else {
        &[]
    }
}

fn shown_dome((generation, roles, shown): (u64, [u8; 2], bool), role: usize) -> u64 {
    (generation * 4 + u64::from(roles[role])) * 2 + u64::from(shown)
}

fn sky_pipeline(
    device: &wgpu::Device,
    fragment: crate::shaders::Entry,
    targets: &[Option<wgpu::ColorTargetState>],
) -> wgpu::RenderPipeline {
    let layout = crate::pipelines::layout_of(device, &crate::pipeline_layouts::SKY_PIPELINES);
    crate::pipelines::fullscreen(
        device,
        &crate::pipelines::Fullscreen {
            label: "engine/sky",
            layout: &layout,
            vertex: crate::entries::SKY_VERTEX,
            fragment,
            constants: &[],
            targets,
            depth: Some(wgpu::DepthStencilState {
                format: crate::renderer::DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                stencil: Default::default(),
                bias: Default::default(),
            }),
        },
    )
}
