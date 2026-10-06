//! Clouds (decision 084; `clouds.wgsl`): a volume of tiling noise made once,
//! and the dome of the sky's clouds marched from the camera, a slice of its
//! rows each ordinary frame and all of them on exact frames, when the
//! clouds change or when the camera has moved far since the last whole
//! march. The sky and the sky's environment map show the dome; frames
//! without clouds bind a clear one.

use bytemuck::{Pod, Zeroable};
use engine_core::Clouds;

use crate::bindings::{CLOUD_MARCH, CLOUD_NOISE, CLOUD_SKY};
use crate::cached::Cached;
use crate::gpu_data::LightingUniform;

/// Texels along each side of the noise volume.
const NOISE_SIZE: u32 = 128;
const DOME_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Meters the camera moves before the whole dome marches again.
const MOVED: f32 = 250.0;
/// Whole marches between redraws of the sky's environment map.
const ENVIRONMENT_EVERY: u64 = 8;
/// Cells along a side of the shadows' grid (`scene.wgsl`'s
/// `CLOUD_SHADOW_CELLS`), and the meters it spans around the camera.
const SHADOW_CELLS: u32 = 128;
const SHADOW_EXTENT: f32 = 8192.0;
/// Frames over which ordinary frames march the whole shadows' grid.
const SHADOW_SLICES: u32 = 4;
/// Bytes of the shadows' grid: a half float a cell.
pub(crate) const CLOUD_SHADOW_BYTES: usize = (SHADOW_CELLS * SHADOW_CELLS * 2) as usize;

/// WGSL `CloudUniform`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct CloudUniform {
    eye: [f32; 4],
    layer: [f32; 4],
    drift: [f32; 4],
    dome: [u32; 4],
    sun: [f32; 4],
    sun_color: [f32; 4],
    zenith: [f32; 4],
    horizon: [f32; 4],
    shadow: [f32; 4],
    shadow_rows: [u32; 4],
}

/// Bytes of the march's uniform.
pub(crate) const CLOUD_UNIFORM_BYTES: usize = size_of::<CloudUniform>();

pub(crate) struct CloudPass {
    march_layout: wgpu::BindGroupLayout,
    noise_layout: wgpu::BindGroupLayout,
    sky_layout: wgpu::BindGroupLayout,
    // The noise volume's and the march's pipelines, made with the first
    // frame that has clouds.
    pipelines: Option<[wgpu::ComputePipeline; 3]>,
    _noise: wgpu::Texture,
    noise_view: wgpu::TextureView,
    noise_made: bool,
    dome: Option<(wgpu::Texture, wgpu::TextureView)>,
    // A clear dome: no cloud, the whole sky passing.
    _clear: wgpu::Texture,
    clear_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    noise_sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    // The shadows' grid as the march writes it.
    shadow_cells: wgpu::Buffer,
    march_group: Cached<u64, wgpu::BindGroup>,
    sky_group: Cached<(u64, bool), wgpu::BindGroup>,
    // Advances whenever the dome is another texture.
    generation: u64,
    // The settings and eye of the last whole march, how many whole marches
    // there were, and how many were recorded: the sky's environment map,
    // drawn before a frame's march, follows those.
    marched: Option<(Clouds, glam::Vec3)>,
    whole: u64,
    recorded: u64,
    // Whether the dome was marched whole since it was made: until then
    // the sky shows a clear one.
    ready: bool,
    // This frame's march: the first row and the rows between those marched.
    job: Option<(u32, u32)>,
    // This frame's rows of the shadows' grid between those marched.
    shadow_stride: u32,
    frame: u64,
    active: bool,
    /// Rows of the dome this frame marched.
    pub(crate) rows: u32,
}

impl CloudPass {
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let noise = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("engine/cloud-noise"),
            size: wgpu::Extent3d {
                width: NOISE_SIZE,
                height: NOISE_SIZE,
                depth_or_array_layers: NOISE_SIZE,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let noise_view = noise.create_view(&wgpu::TextureViewDescriptor::default());
        let clear = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("engine/cloud-clear"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DOME_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        // No light scattered, the whole sky passing: half-float 0, 0, 0, 1.
        let texel: [u16; 4] = [0, 0, 0, 0x3c00];
        queue.write_texture(
            clear.as_image_copy(),
            bytemuck::cast_slice(&texel),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(8),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let clear_view = clear.create_view(&wgpu::TextureViewDescriptor::default());
        // Around the horizon the dome repeats; up and down it clamps.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("engine/cloud-dome-sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let noise_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("engine/cloud-noise-sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            march_layout: crate::pipelines::layout(device, &CLOUD_MARCH),
            noise_layout: crate::pipelines::layout(device, &CLOUD_NOISE),
            sky_layout: crate::pipelines::layout(device, &CLOUD_SKY),
            pipelines: None,
            _noise: noise,
            noise_view,
            noise_made: false,
            dome: None,
            _clear: clear,
            clear_view,
            sampler,
            noise_sampler,
            uniform: crate::buffers::uniform(
                device,
                "engine/cloud-uniform",
                CLOUD_UNIFORM_BYTES as u64,
            ),
            shadow_cells: crate::buffers::storage(
                device,
                "engine/cloud-shadow-cells",
                CLOUD_SHADOW_BYTES as u64,
                wgpu::BufferUsages::COPY_SRC,
            ),
            march_group: Cached::default(),
            sky_group: Cached::default(),
            generation: 0,
            marched: None,
            whole: 0,
            recorded: 0,
            ready: false,
            job: None,
            shadow_stride: 1,
            frame: 0,
            active: false,
            rows: 0,
        }
    }

    /// This frame's slice of the shadows' grid: all of it on exact frames,
    /// a quarter of its rows otherwise.
    fn shadow_rows(&mut self, whole: bool) -> [u32; 4] {
        let stride = if whole { 1 } else { SHADOW_SLICES };
        self.shadow_stride = stride;
        [(self.frame % u64::from(stride)) as u32, stride, 0, 0]
    }

    /// The sky pipelines' second group layout.
    pub(crate) fn sky_layout(&self) -> &wgpu::BindGroupLayout {
        &self.sky_layout
    }

    /// The dome the sky shows (a clear one without clouds), its sampler,
    /// and a key advancing whenever it is another texture.
    pub(crate) fn dome(&self) -> (&wgpu::TextureView, &wgpu::Sampler, (u64, bool)) {
        match (&self.dome, self.active && self.ready) {
            (Some((_, view)), true) => (view, &self.sampler, (self.generation, true)),
            _ => (&self.clear_view, &self.sampler, (self.generation, false)),
        }
    }

    /// What the sky's environment map holds of the clouds: it redraws when
    /// this changes, every few whole marches.
    pub(crate) fn environment_key(&self) -> [f32; 4] {
        if self.active && self.ready {
            [
                1.0,
                (self.recorded / ENVIRONMENT_EVERY) as f32,
                self.generation as f32,
                0.0,
            ]
        } else {
            [0.0; 4]
        }
    }

    /// The sky pipelines' second group, as [`Self::prepare_sky_group`] last
    /// built it.
    pub(crate) fn sky_group(&self) -> &wgpu::BindGroup {
        self.sky_group.get().expect("built each frame")
    }

    /// Builds the sky pipelines' second group unless built: the dome (or a
    /// clear one) and its sampler.
    pub(crate) fn prepare_sky_group(&mut self, device: &wgpu::Device) {
        let (view, sampler, key) = match (&self.dome, self.active && self.ready) {
            (Some((_, view)), true) => (view, &self.sampler, (self.generation, true)),
            _ => (&self.clear_view, &self.sampler, (self.generation, false)),
        };
        let layout = &self.sky_layout;
        self.sky_group.get_or_insert_with(key, || {
            crate::pipelines::group(
                device,
                "engine/cloud-sky",
                &CLOUD_SKY,
                layout,
                &[
                    ("sky_clouds", crate::pipelines::view(view)),
                    ("sky_cloud_sampler", crate::pipelines::sampler(sampler)),
                ],
            )
        });
    }

    /// Plans this frame's march of `settings`' clouds seen from `eye` at
    /// simulation time `seconds` in `lighting`'s sun and sky: the whole dome
    /// when `whole` (exact frames), when it is new or the clouds changed,
    /// or when the camera moved far; otherwise a slice of its rows.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        (settings, eye, seconds): (&Clouds, glam::Vec3, f64),
        lighting: &LightingUniform,
        whole: bool,
    ) {
        self.rows = 0;
        self.job = None;
        let settings = settings.sanitized();
        self.active = settings.enabled && settings.coverage > 0.0;
        if !self.active {
            self.marched = None;
            return;
        }
        let size = [settings.resolution, settings.resolution / 4];
        let fresh = self
            .dome
            .as_ref()
            .is_none_or(|(texture, _)| [texture.width(), texture.height()] != size);
        if fresh {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("engine/cloud-dome"),
                size: wgpu::Extent3d {
                    width: size[0],
                    height: size[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DOME_FORMAT,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.dome = Some((texture, view));
            self.generation += 1;
            self.ready = false;
        }
        let whole = whole
            || fresh
            || self
                .marched
                .is_none_or(|(marched, from)| marched != settings || from.distance(eye) > MOVED);
        self.frame += 1;
        let (first, stride) = if whole {
            self.marched = Some((settings, eye));
            self.whole += 1;
            (0, 1)
        } else {
            let slices = settings.slices.min(size[1]);
            let slice = (self.frame % u64::from(slices)) as u32;
            if slice == 0 {
                self.whole += 1;
            }
            (slice, slices)
        };
        self.job = Some((first, stride));
        self.rows = (size[1] - first).div_ceil(stride);
        // The wind's drift so far, wrapped where every repeat of the noise
        // lines up again.
        let wrap = f64::from(settings.scale) * 43.0;
        let drift = |speed: f32| ((f64::from(speed) * seconds) % wrap) as f32;
        let seed = (f64::from(settings.seed) * 7919.0 % wrap) as f32;
        let uniform = CloudUniform {
            eye: eye.extend(0.0).to_array(),
            layer: [
                settings.base,
                settings.base + settings.thickness,
                settings.coverage,
                settings.density,
            ],
            drift: [
                drift(settings.wind[0]),
                drift(settings.wind[1]),
                settings.scale,
                seed,
            ],
            dome: [size[0], size[1], first, stride],
            sun: lighting.sun,
            sun_color: lighting.sun_color,
            zenith: lighting.zenith,
            horizon: lighting.horizon,
            shadow: shadow_grid(eye).map_or([0.0; 4], |(corner, cell)| {
                [corner[0], corner[1], cell, SHADOW_CELLS as f32]
            }),
            shadow_rows: self.shadow_rows(whole),
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
    }

    /// Records the noise volume the first time, then this frame's march
    /// and the shadows' grid, copied into `shadows` (the camera group's
    /// uniform), timed by `profiler` (`clouds`).
    pub(crate) fn encode(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        shadows: &wgpu::Buffer,
        profiler: &crate::profiling::GpuProfiler,
    ) {
        let (Some((first, stride)), Some((_, dome_view))) = (self.job, &self.dome) else {
            return;
        };
        let [make, march, shade] = self
            .pipelines
            .get_or_insert_with(|| pipelines(device, &self.noise_layout, &self.march_layout));
        let (layout, uniform, noise_view, noise_sampler, cells) = (
            &self.march_layout,
            &self.uniform,
            &self.noise_view,
            &self.noise_sampler,
            &self.shadow_cells,
        );
        let group = self.march_group.get_or_insert_with(self.generation, || {
            crate::pipelines::group(
                device,
                "engine/cloud-march",
                &CLOUD_MARCH,
                layout,
                &[
                    crate::buffers::entry("clouds", uniform),
                    ("noise_volume", crate::pipelines::view(noise_view)),
                    ("noise_sampler", crate::pipelines::sampler(noise_sampler)),
                    ("dome", crate::pipelines::view(dome_view)),
                    crate::buffers::entry("shadow_cells", cells),
                ],
            )
        });
        if !self.noise_made {
            let noise_group = crate::pipelines::group(
                device,
                "engine/cloud-noise",
                &CLOUD_NOISE,
                &self.noise_layout,
                &[("noise_out", crate::pipelines::view(noise_view))],
            );
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/cloud-noise"),
                timestamp_writes: None,
            });
            pass.set_pipeline(make);
            pass.set_bind_group(0, &noise_group, &[]);
            let groups = NOISE_SIZE.div_ceil(4);
            pass.dispatch_workgroups(groups, groups, groups);
            self.noise_made = true;
        }
        let (width, height) = {
            let (texture, _) = self.dome.as_ref().expect("checked above");
            (texture.width(), texture.height())
        };
        let rows = (height - first).div_ceil(stride);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/cloud-march"),
                timestamp_writes: profiler.span("clouds").map(|span| span.compute()),
            });
            pass.set_pipeline(march);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(width.div_ceil(8), rows.div_ceil(8), 1);
        }
        {
            // A slice of the shadows' grid every frame: the clouds drift.
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/cloud-shadows"),
                timestamp_writes: profiler.span("cloud_shadows").map(|span| span.compute()),
            });
            pass.set_pipeline(shade);
            pass.set_bind_group(0, group, &[]);
            let shadow_rows = SHADOW_CELLS.div_ceil(self.shadow_stride);
            pass.dispatch_workgroups((SHADOW_CELLS / 2).div_ceil(8), shadow_rows.div_ceil(8), 1);
        }
        encoder.copy_buffer_to_buffer(cells, 0, shadows, 0, CLOUD_SHADOW_BYTES as u64);
        // A whole march makes the dome whole.
        self.ready |= stride == 1;
        self.recorded = self.whole;
    }

    /// Bytes of the noise volume and the dome.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        let noise = u64::from(NOISE_SIZE).pow(3) * 4;
        let dome = self.dome.as_ref().map_or(0, |(texture, _)| {
            u64::from(texture.width()) * u64::from(texture.height()) * 8
        });
        noise + dome
    }
}

/// The shadows' grid around `eye`: its first cell's corner (world x and z)
/// and meters a cell, cells snapped to whole cells so shadows stand still
/// as the camera moves.
pub(crate) fn shadow_grid(eye: glam::Vec3) -> Option<([f32; 2], f32)> {
    let cell = SHADOW_EXTENT / SHADOW_CELLS as f32;
    if !eye.is_finite() {
        return None;
    }
    let half = (SHADOW_CELLS / 2) as f32;
    let corner = |value: f32| ((value / cell).floor() - half) * cell;
    Some(([corner(eye.x), corner(eye.z)], cell))
}

/// The lighting uniform's clouds' shadows for `clouds` seen from `eye`:
/// the grid's corner, cells per meter and strength, and the clouds' base;
/// zero without clouds or shadows.
pub(crate) fn shadow_lighting(clouds: &Clouds, eye: glam::Vec3) -> ([f32; 4], [f32; 4]) {
    let clouds = clouds.sanitized();
    match shadow_grid(eye) {
        Some((corner, cell)) if clouds.enabled && clouds.coverage > 0.0 => (
            [corner[0], corner[1], 1.0 / cell, clouds.shadows],
            [clouds.base, 0.0, 0.0, 0.0],
        ),
        _ => ([0.0; 4], [0.0; 4]),
    }
}

/// The camera group's buffer of clouds' shadows: every cell passing all
/// the light until the clouds' pass copies its grid in.
pub(crate) fn shadow_uniform(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Buffer {
    let buffer = crate::buffers::storage(
        device,
        "engine/cloud-shadows",
        CLOUD_SHADOW_BYTES as u64,
        wgpu::BufferUsages::empty(),
    );
    // Half-float ones, two a word.
    let ones = vec![0x3c00_3c00_u32; CLOUD_SHADOW_BYTES / 4];
    queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&ones));
    buffer
}

fn pipelines(
    device: &wgpu::Device,
    noise_layout: &wgpu::BindGroupLayout,
    march_layout: &wgpu::BindGroupLayout,
) -> [wgpu::ComputePipeline; 3] {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("engine/clouds.wgsl"),
        source: wgpu::ShaderSource::Wgsl(crate::gpu_data::CLOUDS_SHADER.into()),
    });
    let layout = |group: &wgpu::BindGroupLayout| {
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("engine/clouds-layout"),
            bind_group_layouts: &[Some(group)],
            immediate_size: 0,
        })
    };
    let march = layout(march_layout);
    [
        crate::pipelines::compute_entry(device, &layout(noise_layout), &module, "make_noise"),
        crate::pipelines::compute_entry(device, &march, &module, "march_clouds"),
        crate::pipelines::compute_entry(device, &march, &module, "shadow_clouds"),
    ]
}
