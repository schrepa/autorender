//! A physically based sky (`sky_common.wgsl`, `sky.wgsl`):
//! the air's transmittance and multiple scattering tables, built when the
//! air changes; the sky-view table around the camera, built when the sun,
//! the camera's height or the air changes; and every frame the scene's
//! light the sky gives (the sun's color, the zenith's, the horizon's, the
//! ground's, the air's away from the sun and the sun's glow in it), copied
//! into the uniforms that hold the authored atmosphere's colors before any
//! pass reads them. The sky's passes and the environment maps sample the
//! sky-view and transmittance tables.

use crate::render_features::{Encode, RenderFeature};
use crate::render_graph::{FrameShape, Node, RenderGraph};
use crate::shaders::entry::sky;
use engine_core::Sky;
use glam::Vec3;

use crate::bindings::SKY_TABLES;
use crate::gpu_layouts::constants::SKY_TABLE_GROUP;

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Texels of the tables (`sky_common.wgsl`'s sizes).
const TRANSMITTANCE_SIZE: (u32, u32) = (256, 64);
const MULTIPLE_SIZE: (u32, u32) = (32, 32);
const VIEW_SIZE: (u32, u32) = (192, 108);
/// Meters of the camera's height per redraw of the sky's environment map;
/// the sky-view table follows the height exactly.
const ENVIRONMENT_HEIGHT_STEP: f32 = 200.0;
/// Bytes of the derived light: seven vec4 (`sky.wgsl`'s `derived`), the
/// last the exposure as it eased to last frame.
const DERIVED_BYTES: u64 = 7 * 16;

use crate::gpu_layouts::sky::SkyUniform;

/// Bytes of the sky's uniform.
pub(crate) const SKY_UNIFORM_BYTES: usize = size_of::<SkyUniform>();

/// The sky's uniform, its sky-view and transmittance tables and their
/// sampler, which the sky's passes and the environment maps bind.
pub(crate) type AirBindings<'a> = (
    &'a wgpu::Buffer,
    &'a wgpu::TextureView,
    &'a wgpu::TextureView,
    &'a wgpu::Sampler,
);

/// Where a derived color goes: its index in the derived light, the buffer,
/// the byte offset there and the bytes copied.
pub(crate) type SkyCopy<'a> = (u64, &'a wgpu::Buffer, u64, u64);

/// The derived light's indices: the sun's color, the zenith's, the
/// horizon's (with the adapted exposure in w), the ground's, the air's and
/// the sun's glow in it.
pub(crate) const DERIVED_SUN: u64 = 0;
pub(crate) const DERIVED_ZENITH: u64 = 1;
pub(crate) const DERIVED_HORIZON: u64 = 2;
pub(crate) const DERIVED_GROUND: u64 = 3;
pub(crate) const DERIVED_AIR: u64 = 4;
pub(crate) const DERIVED_GLOW: u64 = 5;

struct Table {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

fn table(
    device: &wgpu::Device,
    label: &str,
    (width, height): (u32, u32),
    usage: wgpu::TextureUsages,
) -> Table {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
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
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    Table {
        _texture: texture,
        view,
    }
}

/// The tables' pipelines and each pass's group, made with the first frame
/// whose sky is on.
struct Passes {
    transmittance: wgpu::ComputePipeline,
    multiple: wgpu::ComputePipeline,
    view: wgpu::ComputePipeline,
    derive: wgpu::ComputePipeline,
    groups: [wgpu::BindGroup; 4],
}

pub(crate) struct SkyPass {
    layout: crate::pipelines::GroupLayout,
    passes: Option<Passes>,
    transmittance: Table,
    multiple: Table,
    view: Table,
    // A blank texel bound where a pass writes the table, and one the light's
    // derivation writes.
    blank: Table,
    blank_out: Table,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    derived: wgpu::Buffer,
    // The air the tables hold, and the air, sun and height the sky-view
    // table holds.
    tables: Option<[u32; 20]>,
    view_key: Option<([u32; 20], [u32; 3], u32)>,
    // The presentation time of the last frame, which exposure eases from.
    last_seconds: Option<f64>,
    // Advances with each sky-view build the environment map follows.
    generation: u64,
    environment_height: i32,
    active: bool,
    /// Tables this frame built: two for the air's, one for the sky-view.
    pub(crate) built: u32,
}

impl SkyPass {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let tables = wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING;
        Self {
            layout: crate::pipelines::layout(device, &SKY_TABLES),
            passes: None,
            transmittance: table(
                device,
                "engine/sky-transmittance",
                TRANSMITTANCE_SIZE,
                tables,
            ),
            multiple: table(device, "engine/sky-multiple", MULTIPLE_SIZE, tables),
            view: table(device, "engine/sky-view", VIEW_SIZE, tables),
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

    /// The sky's uniform, its sky-view and transmittance tables and their
    /// sampler, which the sky's passes and the environment maps bind.
    pub(crate) fn bindings(&self) -> AirBindings<'_> {
        (
            &self.uniform,
            &self.view.view,
            &self.transmittance.view,
            &self.sampler,
        )
    }

    /// What the sky's environment map holds of this sky: it redraws when
    /// this changes.
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

    /// Records this frame's sky of `settings` with the sun along `sun` seen
    /// from `eye`, or nothing without one: its tables when their air
    /// changed, the sky-view table when the sun, the camera's height or the
    /// air changed, then the light it gives, copied by `copies`. Timed by
    /// `profiler` (`sky`).
    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    pub(crate) fn encode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        (settings, sun, eye, exposure, seconds): (&Sky, Vec3, Vec3, f32, f64),
        copies: &[SkyCopy<'_>],
        profiler: &crate::profiling::GpuProfiler,
    ) -> Option<wgpu::CommandEncoder> {
        self.built = 0;
        let settings = settings.sanitized();
        self.active = settings.enabled;
        if !self.active {
            self.tables = None;
            self.view_key = None;
            return None;
        }
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
            // Seconds since the last frame, over which exposure eases; 0
            // (a first or repeated time, as exact frames are) takes it at
            // once.
            frame: [elapsed, 0.0, 0.0, 0.0],
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
        let air: [u32; 20] = bytemuck::cast([
            uniform.rayleigh,
            uniform.mie,
            [uniform.ozone[0], uniform.ozone[1], uniform.ozone[2], 0.0],
            [uniform.ground[0], uniform.ground[1], uniform.ground[2], 0.0],
            [uniform.radii[0], uniform.radii[1], 0.0, 0.0],
        ]);
        let view_key = (air, sun.to_array().map(f32::to_bits), height.to_bits());
        let build_tables = self.tables != Some(air);
        let build_view = build_tables || self.view_key != Some(view_key);
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
            if build_tables {
                pass.set_pipeline(&passes.transmittance);
                pass.set_bind_group(0, &passes.groups[0], &[]);
                let (x, y) = groups(TRANSMITTANCE_SIZE);
                pass.dispatch_workgroups(x, y, 1);
                pass.set_pipeline(&passes.multiple);
                pass.set_bind_group(0, &passes.groups[1], &[]);
                let (x, y) = groups(MULTIPLE_SIZE);
                pass.dispatch_workgroups(x, y, 1);
                self.built += 2;
            }
            if build_view {
                pass.set_pipeline(&passes.view);
                pass.set_bind_group(0, &passes.groups[2], &[]);
                let (x, y) = groups(VIEW_SIZE);
                pass.dispatch_workgroups(x, y, 1);
                self.built += 1;
            }
            pass.set_pipeline(&passes.derive);
            pass.set_bind_group(0, &passes.groups[3], &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        for &(index, buffer, offset, bytes) in copies {
            encoder.copy_buffer_to_buffer(&self.derived, index * 16, buffer, offset, bytes);
        }
        if let Some(span) = profiler.span("sky") {
            span.mark(&mut encoder, true);
        }
        if build_view {
            // The environment map follows the sun's steps and the camera's
            // height more coarsely.
            let environment_height = (height / ENVIRONMENT_HEIGHT_STEP).floor() as i32;
            let sun_moved = self.view_key.is_none_or(|key| key.1 != view_key.1);
            if build_tables || sun_moved || environment_height != self.environment_height {
                self.generation += 1;
            }
            self.environment_height = environment_height;
            self.tables = Some(air);
            self.view_key = Some(view_key);
        }
        Some(encoder)
    }

    /// Bytes of the tables.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        let texels = |(width, height): (u32, u32)| u64::from(width) * u64::from(height);
        (texels(TRANSMITTANCE_SIZE) + texels(MULTIPLE_SIZE) + texels(VIEW_SIZE)) * 8
    }
}

/// The tables' pipelines and the groups of each pass: building the
/// transmittance table, the multiple scattering table, the sky-view table,
/// and deriving the light; `tables` are the transmittance, multiple
/// scattering and sky-view tables and the blank texels.
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
        transmittance: pipeline(sky::build_transmittance),
        multiple: pipeline(sky::build_multiple),
        view: pipeline(sky::build_sky_view),
        derive: pipeline(sky::derive_light),
        // A pass never reads the table it writes.
        groups: [
            group([blank, blank, blank], transmittance),
            group([transmittance, blank, blank], multiple),
            group([transmittance, multiple, blank], view),
            group([transmittance, multiple, view], blank_out),
        ],
    }
}

/// The sky works before the frame, in its prelude ([`SkyPass::encode`]):
/// it declares no nodes, so the graph encodes none of its.
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

    /// The tables this frame built.
    fn stats(
        this: Option<&Self>,
        _frame: Option<&()>,
        _snapshot: &engine_core::RenderSnapshot,
    ) -> engine_core::SkyStats {
        engine_core::SkyStats {
            tables: this.map_or(0, |sky| sky.built),
        }
    }
}
