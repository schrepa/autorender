use crate::render_features::{Encode, RenderFeature};
use crate::render_graph::{FrameShape, Node, RenderGraph};
use crate::shaders::entry::sky;
use engine_core::table_data::sky::SkyTables;
use engine_core::table_data::sky::{MULTIPLE_SIZE, TRANSMITTANCE_SIZE};
use engine_core::{RenderSnapshot, Sky, StaticTable};
use engine_time_of_day::TimeOfDay;
use glam::Vec3;

use crate::bindings::{SKY_TABLES, SKY_TRANSMITTANCE};
use crate::gpu_layouts::constants::SKY_TABLE_GROUP;

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

pub(crate) struct SkyPass {
    layout: crate::pipelines::GroupLayout,
    passes: Option<Passes>,
    pass_layout: crate::pipelines::GroupLayout,
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

impl SkyPass {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let _owner = crate::memory::feature(crate::render_graph::FeaturePass::CloudDome);
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
            pass_layout,
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

    pub(crate) fn pass_layout(&self) -> &crate::pipelines::GroupLayout {
        &self.pass_layout
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
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("engine/sky.wgsl"),
        source: wgpu::ShaderSource::Wgsl(crate::gpu_data::SKY_SHADER.into()),
    });
    let pipeline_layout = crate::pipelines::pipeline_layout(
        device,
        &crate::pipeline_layouts::SKY_TABLE_PIPELINES,
        &[layout],
    );
    let pipeline = |entry: crate::shaders::Entry| {
        crate::pipelines::compute_entry(device, &pipeline_layout, &module, entry)
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

impl RenderFeature for SkyPass {
    type Frame = ();
    type Stats = engine_core::SkyStats;

    fn declare(_frame: &(), _shape: &FrameShape, _graph: &mut RenderGraph) {}

    fn encode(
        &self,
        _frame: &(),
        _node: &Node,
        _encode: &Encode<'_>,
        _encoder: &mut wgpu::CommandEncoder,
    ) -> u32 {
        0
    }

    fn capacity_bytes(&self) -> u64 {
        SkyPass::capacity_bytes(self)
    }

    fn stats(
        this: Option<&Self>,
        _frame: Option<&()>,
        _snapshot: &engine_core::RenderSnapshot,
    ) -> engine_core::SkyStats {
        engine_core::SkyStats {
            tables: this.map_or(0, |sky| sky.built),
            ..engine_core::SkyStats::default()
        }
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
