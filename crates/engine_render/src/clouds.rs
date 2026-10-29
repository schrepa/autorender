use crate::shaders::entry::{clouds, meshes};
use engine_core::collections::HashMap;

use engine_core::Clouds;
use engine_core::StaticTable;
use engine_core::math::min;
use engine_core::table_data::clouds::CloudNoise;

use crate::bindings::{CLOUD_COMPOSITE, CLOUD_MARCH, CLOUD_VIEW};
use crate::cached::{Bound, Cached, GroupCache, Versioned};
use crate::gpu_data::LightingUniform;
use crate::gpu_layouts::constants::CLOUD_GROUP;
use crate::render_features::{Encode, FeaturePass, RenderFeature};
use crate::render_graph::{Access, FrameShape, Node, Pass, RenderGraph, Resource};

const DOME_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const DOME_LAYERS: u32 = MIRROR_LAYER + 1;
const ENVIRONMENT_EVERY: u64 = 16;
use crate::gpu_layouts::clouds::{
    CLOUD_LIGHT_BYTES, CLOUD_SHADOW_BYTES, LIGHT_TEXELS, MIRROR_LAYER, SHADOW_CELLS,
};
const SHADOW_EXTENT: f32 = 8192.0;
const SHADOW_SLICES: u32 = 4;
const LIGHT_EXTENT: f32 = 24_000.0;
const LIGHT_RECENTER: f32 = 2_000.0;
const LIGHT_ROWS: u32 = 48;

use crate::gpu_layouts::clouds::CloudUniform;

pub(crate) const CLOUD_UNIFORM_BYTES: usize = size_of::<CloudUniform>();

use crate::gpu_layouts::clouds::DomeUniform;

pub(crate) const DOME_UNIFORM_BYTES: usize = size_of::<DomeUniform>();

use crate::gpu_layouts::clouds::ViewUniform;

pub(crate) const CLOUD_VIEW_BYTES: usize = size_of::<ViewUniform>();

struct ViewTargets {
    clouds: (wgpu::Texture, wgpu::TextureView),
    size: [u32; 2],
}

#[derive(Default)]
struct LightMap {
    cells: Option<wgpu::Buffer>,
    held: Option<LightKey>,
    built: u32,
    job: u32,
}

#[derive(Clone, Copy, PartialEq)]
struct LightKey {
    corner: [f32; 2],
    sun: [f32; 3],
    layer: [f32; 4],
}

struct Cycle {
    origin: Origins,
    seconds: f64,
    next_row: u32,
}

#[derive(Clone, Copy, Default)]
struct Origins {
    eye: glam::Vec3,
    mirror: glam::Vec3,
}

impl Origins {
    fn of(eye: glam::Vec3, surface: Option<f32>) -> Self {
        let level = surface.map_or(eye.y, |surface| min(surface, eye.y));
        Self {
            eye,
            mirror: glam::Vec3::new(eye.x, 2.0 * level - eye.y, eye.z),
        }
    }
}

pub(crate) struct DomeBinding<'a> {
    pub old: &'a wgpu::TextureView,
    pub new: &'a wgpu::TextureView,
    pub sampler: &'a wgpu::Sampler,
    pub uniform: &'a wgpu::Buffer,
    pub key: (u64, [u8; 2], bool),
}

struct NoiseVolumes {
    id: u64,
    _textures: [wgpu::Texture; 2],
    shape: wgpu::TextureView,
    detail: wgpu::TextureView,
    bytes: u64,
}

fn levels_below(side: u32, texels: &[u8]) -> Vec<Vec<u8>> {
    let mut levels: Vec<Vec<u8>> = Vec::new();
    let mut side = side as usize;
    while side > 1 {
        let half = side / 2;
        let above = levels.last().map_or(texels, Vec::as_slice);
        let mut level = Vec::with_capacity(half * half * half);
        for z in 0..half {
            for y in 0..half {
                for x in 0..half {
                    let mut sum = 0_u32;
                    for corner in 0..8_usize {
                        let (dx, dy, dz) = (corner & 1, (corner >> 1) & 1, corner >> 2);
                        let at = ((2 * z + dz) * side + 2 * y + dy) * side + 2 * x + dx;
                        sum += u32::from(above[at]);
                    }
                    level.push(((sum + 4) / 8) as u8);
                }
            }
        }
        levels.push(level);
        side = half;
    }
    levels
}

impl NoiseVolumes {
    fn new(
        device: &wgpu::Device,
        queue: &crate::gpu_calls::Queue,
        noise: &StaticTable<CloudNoise>,
    ) -> Self {
        let volume = |label, side: u32, levels: &[&[u8]]| {
            let texture = crate::memory::texture(device, &wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: side,
                    height: side,
                    depth_or_array_layers: side,
                },
                mip_level_count: levels.len() as u32,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            for (level, &texels) in (0_u32..).zip(levels) {
                let side = (side >> level).max(1);
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture,
                        mip_level: level,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    texels,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(side),
                        rows_per_image: Some(side),
                    },
                    wgpu::Extent3d {
                        width: side,
                        height: side,
                        depth_or_array_layers: side,
                    },
                );
            }
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            (texture, view)
        };
        use engine_core::table_data::clouds::{DETAIL_SIZE, SHAPE_SIZE};
        let table = &noise.table;
        let (shape_texture, shape) = volume(
            "engine/cloud-noise-shape",
            SHAPE_SIZE,
            &[table.shape.as_slice()],
        );
        let below = levels_below(DETAIL_SIZE, &table.detail);
        let detail_levels: Vec<&[u8]> = std::iter::once(table.detail.as_slice())
            .chain(below.iter().map(Vec::as_slice))
            .collect();
        let (detail_texture, detail) =
            volume("engine/cloud-noise-detail", DETAIL_SIZE, &detail_levels);
        let bytes =
            table.shape.len() + detail_levels.iter().map(|level| level.len()).sum::<usize>();
        Self {
            id: noise.id,
            _textures: [shape_texture, detail_texture],
            shape,
            detail,
            bytes: bytes as u64,
        }
    }
}

pub(crate) struct CloudPass {
    march_layout: crate::pipelines::GroupLayout,
    pipelines: Option<[wgpu::ComputePipeline; 3]>,
    light: LightMap,
    noise: Option<NoiseVolumes>,
    shadows: wgpu::Buffer,
    domes: Option<[(wgpu::Texture, wgpu::TextureView); 3]>,
    roles: [usize; 3],
    origins: [Origins; 3],
    blend: f32,
    clear: (wgpu::Texture, wgpu::TextureView),
    sampler: wgpu::Sampler,
    noise_sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    dome_uniform: wgpu::Buffer,
    shadow_cells: wgpu::Buffer,
    march_groups: Cached<u64, [wgpu::BindGroup; 3]>,
    generation: u64,
    marched: Option<Clouds>,
    cycle: Option<Cycle>,
    completes: bool,
    completed: u64,
    recorded: u64,
    recorded_generation: Option<u64>,
    ready: bool,
    job: Option<(usize, u32, u32)>,
    shadow_stride: u32,
    frame: u64,
    active: bool,
    pub(crate) rows: u32,
    view_layout: crate::pipelines::GroupLayout,
    composite_layout: crate::pipelines::GroupLayout,
    view_clouds: wgpu::Buffer,
    view_uniform: wgpu::Buffer,
    composite_args: wgpu::Buffer,
    view_targets: Option<Versioned<ViewTargets>>,
    view_group: GroupCache<12>,
    composite_group: GroupCache<1>,
    view_pipeline: Option<wgpu::ComputePipeline>,
    composite_pipeline_layout: Option<wgpu::PipelineLayout>,
    composite_pipelines: HashMap<(wgpu::TextureFormat, bool), wgpu::RenderPipeline>,
    view_job: Option<[u32; 2]>,
    view_sky: bool,
    pub(crate) view_pixels: u32,
}

type CloudInputs<'a> = (
    &'a Clouds,
    Option<&'a StaticTable<CloudNoise>>,
    (glam::Vec3, Option<f32>),
    f64,
);

impl CloudPass {
    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    pub(crate) fn new(device: &wgpu::Device, queue: &crate::gpu_calls::Queue) -> Self {
        let clear = crate::memory::texture(device, &wgpu::TextureDescriptor {
            label: Some("engine/cloud-clear"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: DOME_LAYERS,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DOME_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let texels = [[0_u16, 0, 0, 0x3c00]; DOME_LAYERS as usize];
        queue.write_texture(
            clear.as_image_copy(),
            bytemuck::cast_slice(&texels),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(8),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: DOME_LAYERS,
            },
        );
        let clear_view = clear.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
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
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        Self {
            march_layout: crate::pipelines::layout(device, &CLOUD_MARCH),
            pipelines: None,
            light: LightMap::default(),
            noise: None,
            shadows: shadow_uniform(device, queue),
            domes: None,
            roles: [0, 1, 2],
            origins: [Origins::default(); 3],
            blend: 1.0,
            clear: (clear, clear_view),
            sampler,
            noise_sampler,
            uniform: crate::buffers::uniform(
                device,
                "engine/cloud-uniform",
                CLOUD_UNIFORM_BYTES as u64,
            ),
            dome_uniform: crate::buffers::uniform(
                device,
                "engine/cloud-domes",
                DOME_UNIFORM_BYTES as u64,
            ),
            shadow_cells: crate::buffers::storage(
                device,
                "engine/cloud-shadow-cells",
                CLOUD_SHADOW_BYTES as u64,
                wgpu::BufferUsages::COPY_SRC,
            ),
            march_groups: Cached::default(),
            generation: 0,
            marched: None,
            cycle: None,
            completes: false,
            completed: 0,
            recorded: 0,
            recorded_generation: None,
            ready: false,
            job: None,
            shadow_stride: 1,
            frame: 0,
            active: false,
            rows: 0,
            view_layout: crate::pipelines::layout(device, &CLOUD_VIEW),
            composite_layout: crate::pipelines::layout(device, &CLOUD_COMPOSITE),
            view_clouds: crate::buffers::uniform(
                device,
                "engine/cloud-view-march",
                CLOUD_UNIFORM_BYTES as u64,
            ),
            view_uniform: crate::buffers::uniform(
                device,
                "engine/cloud-view",
                CLOUD_VIEW_BYTES as u64,
            ),
            composite_args: crate::buffers::storage(
                device,
                "engine/cloud-composite-draw",
                16,
                wgpu::BufferUsages::INDIRECT,
            ),
            view_targets: None,
            view_group: GroupCache::default(),
            composite_group: GroupCache::default(),
            view_pipeline: None,
            composite_pipeline_layout: None,
            composite_pipelines: HashMap::default(),
            view_job: None,
            view_sky: false,
            view_pixels: 0,
        }
    }

    fn plan_light(
        &mut self,
        device: &wgpu::Device,
        uniform: &CloudUniform,
        whole: bool,
    ) -> [u32; 4] {
        self.light.cells.get_or_insert_with(|| {
            crate::buffers::storage(
                device,
                "engine/cloud-light",
                CLOUD_LIGHT_BYTES as u64,
                wgpu::BufferUsages::empty(),
            )
        });
        let [eye_x, _, eye_z, _] = uniform.eye;
        let [drift_x, drift_z, repeat, seed] = uniform.drift;
        let own = [eye_x + drift_x + seed, eye_z + drift_z - seed];
        let key = LightKey {
            corner: [0.0; 2],
            sun: [uniform.sun[0], uniform.sun[1], uniform.sun[2]],
            layer: [uniform.layer[0], uniform.layer[1], uniform.layer[2], repeat],
        };
        let keep = self.light.held.is_some_and(|held| {
            held.sun == key.sun
                && held.layer == key.layer
                && (0..2).all(|axis| {
                    (own[axis] - held.corner[axis] - 0.5 * LIGHT_EXTENT).abs() <= LIGHT_RECENTER
                })
        });
        if !keep {
            let cell = LIGHT_EXTENT / LIGHT_TEXELS as f32;
            let half = (LIGHT_TEXELS / 2) as f32;
            self.light.held = Some(LightKey {
                corner: own.map(|value| ((value / cell).floor() - half) * cell),
                ..key
            });
            self.light.built = 0;
        }
        let first = self.light.built;
        let rows = if whole { LIGHT_TEXELS } else { LIGHT_ROWS }.min(LIGHT_TEXELS - first);
        self.light.built += rows;
        self.light.job = rows;
        [first, rows, 0, 0]
    }

    fn light_uniform(&self) -> [f32; 4] {
        self.light.held.map_or([0.0; 4], |held| {
            [
                held.corner[0],
                held.corner[1],
                LIGHT_EXTENT / LIGHT_TEXELS as f32,
                f32::from(u8::from(self.light.built >= LIGHT_TEXELS)),
            ]
        })
    }

    fn shadow_rows(&mut self, whole: bool) -> [u32; 4] {
        let stride = if whole { 1 } else { SHADOW_SLICES };
        self.shadow_stride = stride;
        [(self.frame % u64::from(stride)) as u32, stride, 0, 0]
    }

    pub(crate) fn dome(&self) -> DomeBinding<'_> {
        match (&self.domes, self.active && self.ready) {
            (Some(domes), true) => DomeBinding {
                old: &domes[self.roles[0]].1,
                new: &domes[self.roles[1]].1,
                sampler: &self.sampler,
                uniform: &self.dome_uniform,
                key: (
                    self.generation,
                    [self.roles[0] as u8, self.roles[1] as u8],
                    true,
                ),
            },
            _ => DomeBinding {
                old: &self.clear.1,
                new: &self.clear.1,
                sampler: &self.sampler,
                uniform: &self.dome_uniform,
                key: (self.generation, [0, 0], false),
            },
        }
    }

    pub(crate) fn environment_key(&self) -> [f32; 4] {
        if self.environment_shows() {
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

    pub(crate) fn environment_dome(&self) -> DomeBinding<'_> {
        let dome = self.dome();
        if self.environment_shows() {
            return dome;
        }
        DomeBinding {
            old: &self.clear.1,
            new: &self.clear.1,
            key: (self.generation, [0, 0], false),
            ..dome
        }
    }

    fn environment_shows(&self) -> bool {
        self.active && self.ready && self.recorded_generation == Some(self.generation)
    }

    pub(crate) fn sky_copies(&self) -> [crate::sky::SkyCopy<'_>; 6] {
        let at = |offset: usize| offset as u64;
        let copies = |uniform| {
            [
                (
                    crate::sky::DERIVED_SUN,
                    uniform,
                    at(std::mem::offset_of!(CloudUniform, sun_color)),
                    12,
                ),
                (
                    crate::sky::DERIVED_ZENITH,
                    uniform,
                    at(std::mem::offset_of!(CloudUniform, zenith)),
                    12,
                ),
                (
                    crate::sky::DERIVED_HORIZON,
                    uniform,
                    at(std::mem::offset_of!(CloudUniform, horizon)),
                    12,
                ),
            ]
        };
        let [a, b, c] = copies(&self.uniform);
        let [d, e, f] = copies(&self.view_clouds);
        [a, b, c, d, e, f]
    }

    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &crate::gpu_calls::Queue,
        (settings, noise, (eye, surface), seconds): CloudInputs<'_>,
        (lighting, air): (&LightingUniform, crate::sky::AirBindings<'_>),
        whole: bool,
    ) {
        self.rows = 0;
        self.job = None;
        let settings = settings.sanitized();
        if let Some(noise) = noise.filter(|_| settings.enabled)
            && self.noise.as_ref().is_none_or(|held| held.id != noise.id)
        {
            self.noise = Some(NoiseVolumes::new(device, queue, noise));
            self.generation += 1;
            self.march_groups = Cached::default();
            self.view_group = GroupCache::default();
        }
        self.active = settings.enabled && settings.coverage > 0.0 && self.noise.is_some();
        if !self.active {
            self.marched = None;
            self.cycle = None;
            self.completes = false;
            self.write_domes(queue, eye, &settings);
            return;
        }
        let size = [settings.resolution, settings.resolution / 4];
        let fresh = self
            .domes
            .as_ref()
            .is_none_or(|domes| [domes[0].0.width(), domes[0].0.height()] != size);
        if fresh {
            self.domes = Some(std::array::from_fn(|_| dome_texture(device, size)));
            self.generation += 1;
            self.ready = false;
            self.cycle = None;
            self.completes = false;
        }
        if self.completes {
            self.completes = false;
            self.rotate();
        }
        let rows = size[1];
        let changed = self.marched.is_none_or(|marched| marched != settings);
        self.frame += 1;
        let target = self.roles[2];
        let origins = Origins::of(eye, surface);
        let (origin, at, first, count) = if whole || fresh || changed || !self.ready {
            let spare = (0..3).find(|&index| index != target).unwrap_or(0);
            self.roles = [target, target, spare];
            self.origins[target] = origins;
            self.blend = 1.0;
            self.marched = Some(settings);
            self.cycle = None;
            self.ready = true;
            self.completed += 1;
            (origins, seconds, 0, rows)
        } else {
            let cycle = self.cycle.get_or_insert(Cycle {
                origin: origins,
                seconds,
                next_row: 0,
            });
            let frames = settings.slices.max(1);
            let first = cycle.next_row;
            let count = rows.div_ceil(frames).min(rows - first);
            cycle.next_row += count;
            let (origin, at, done) = (cycle.origin, cycle.seconds, cycle.next_row >= rows);
            self.blend = cycle.next_row as f32 / rows as f32;
            if done {
                self.origins[target] = origin;
                self.cycle = None;
                self.completes = true;
            }
            (origin, at, first, count)
        };
        self.job = Some((target, first, count));
        self.rows = count;
        let mut uniform = march_uniform(&settings, (origin.eye, at), lighting);
        uniform.mirror = origin.mirror.extend(0.0).to_array();
        uniform.dome = [size[0], size[1], first, count];
        uniform.shadow = shadow_grid(eye).map_or([0.0; 4], |(first, cell)| {
            [first[0], first[1], cell, SHADOW_CELLS as f32]
        });
        uniform.shadow_rows = self.shadow_rows(whole);
        uniform.light_rows = self.plan_light(device, &uniform, whole);
        uniform.light = self.light_uniform();
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
        self.write_domes(queue, eye, &settings);
        self.prepare_dome_draws(device, air);
    }

    fn prepare_dome_draws(&mut self, device: &wgpu::Device, air: crate::sky::AirBindings<'_>) {
        let (Some(domes), Some(noise)) = (&self.domes, &self.noise) else {
            return;
        };
        let march_layout = &self.march_layout;
        self.pipelines
            .get_or_insert_with(|| pipelines(device, march_layout));
        let (layout, uniform, noise_sampler, cells) = (
            &self.march_layout,
            &self.uniform,
            &self.noise_sampler,
            &self.shadow_cells,
        );
        let Some(light) = &self.light.cells else {
            return;
        };
        self.march_groups.get_or_insert_with(self.generation, || {
            std::array::from_fn(|index| {
                crate::pipelines::group(
                    device,
                    "engine/cloud-march",
                    &CLOUD_MARCH,
                    layout,
                    &[
                        crate::buffers::entry("clouds", uniform),
                        ("noise_shape", crate::pipelines::view(&noise.shape)),
                        ("noise_sampler", crate::pipelines::sampler(noise_sampler)),
                        ("dome", crate::pipelines::view(&domes[index].1)),
                        ("noise_detail", crate::pipelines::view(&noise.detail)),
                        crate::buffers::entry("shadow_cells", cells),
                        crate::buffers::entry("cloud_air", air.0),
                        ("cloud_transmittance", crate::pipelines::view(air.2)),
                        ("cloud_table_sampler", crate::pipelines::sampler(air.3)),
                        crate::buffers::entry("light_cells", light),
                    ],
                )
            })
        });
    }

    #[expect(clippy::too_many_arguments, reason = "the view's inputs, typed")]
    pub(crate) fn prepare_view(
        &mut self,
        device: &wgpu::Device,
        queue: &crate::gpu_calls::Queue,
        (settings, eye, seconds): (&Clouds, glam::Vec3, f64),
        view_projection: glam::Mat4,
        region: crate::targets::FrameRegion,
        (lighting, air): (&LightingUniform, crate::sky::AirBindings<'_>),
        depth: &Versioned<wgpu::TextureView>,
    ) {
        let settings = settings.sanitized();
        self.view_job = None;
        self.view_pixels = 0;
        let (display, render) = (region.display, region.render);
        let marches = self.active && settings.in_view && display.width > 0 && display.height > 0;
        let view_sky = marches && eye.y > settings.base;
        if view_sky != self.view_sky {
            self.view_sky = view_sky;
            self.write_domes(queue, eye, &settings);
        }
        if !marches {
            return;
        }
        let size = [display.width, display.height];
        if self
            .view_targets
            .as_ref()
            .is_none_or(|targets| targets.size != size)
        {
            self.view_targets = Some(Versioned::new(ViewTargets {
                clouds: view_texture(device, size),
                size,
            }));
        }
        let mut uniform = march_uniform(&settings, (eye, seconds), lighting);
        uniform.light = self.light_uniform();
        queue.write_buffer(&self.view_clouds, 0, bytemuck::bytes_of(&uniform));
        let view = ViewUniform {
            inverse_view_projection: view_projection.inverse().to_cols_array_2d(),
            pixels: [render.width, render.height, u32::from(view_sky), 0],
        };
        queue.write_buffer(&self.view_uniform, 0, bytemuck::bytes_of(&view));
        queue.write_buffer(
            &self.composite_args,
            0,
            bytemuck::cast_slice(&[0_u32, 1, 0, 0]),
        );
        self.view_job = Some([render.width, render.height]);
        self.view_pixels = render.width * render.height;
        self.prepare_view_draws(device, depth, air);
    }

    pub(crate) fn marches_view(&self) -> bool {
        self.view_job.is_some()
    }

    fn prepare_view_draws(
        &mut self,
        device: &wgpu::Device,
        depth: &Versioned<wgpu::TextureView>,
        air: crate::sky::AirBindings<'_>,
    ) {
        let (Some(targets), Some(light), Some(noise)) =
            (&self.view_targets, &self.light.cells, &self.noise)
        else {
            return;
        };
        let layout = &self.view_layout;
        self.view_pipeline
            .get_or_insert_with(|| view_pipeline(device, layout));
        self.view_group.bind(
            device,
            "engine/cloud-view",
            (&CLOUD_VIEW, layout),
            [
                ("clouds", Bound::fixed(self.view_clouds.as_entire_binding())),
                (
                    "noise_shape",
                    Bound::fixed(crate::pipelines::view(&noise.shape)),
                ),
                (
                    "noise_sampler",
                    Bound::fixed(crate::pipelines::sampler(&self.noise_sampler)),
                ),
                (
                    "noise_detail",
                    Bound::fixed(crate::pipelines::view(&noise.detail)),
                ),
                (
                    "cloud_view",
                    Bound::fixed(self.view_uniform.as_entire_binding()),
                ),
                ("view_depth", depth.bound()),
                (
                    "view_out",
                    Bound::of(targets, crate::pipelines::view(&targets.clouds.1)),
                ),
                (
                    "composite_draw",
                    Bound::fixed(self.composite_args.as_entire_binding()),
                ),
                ("cloud_air", Bound::fixed(air.0.as_entire_binding())),
                (
                    "cloud_transmittance",
                    Bound::fixed(crate::pipelines::view(air.2)),
                ),
                (
                    "cloud_table_sampler",
                    Bound::fixed(crate::pipelines::sampler(air.3)),
                ),
                ("light_cells", Bound::fixed(light.as_entire_binding())),
            ],
        );
    }

    fn encode_view(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &crate::profiling::GpuProfiler,
    ) {
        let (Some(pixels), Some(pipeline), Some(group)) =
            (self.view_job, &self.view_pipeline, self.view_group.get())
        else {
            return;
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("engine/cloud-view"),
            timestamp_writes: profiler.span("cloud_view").map(|span| span.compute()),
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(
            pixels[0].div_ceil(CLOUD_GROUP),
            pixels[1].div_ceil(CLOUD_GROUP),
            1,
        );
    }

    pub(crate) fn prepare_composite(
        &mut self,
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        camera_layout: &crate::pipelines::GroupLayout,
        (format, canvas): (wgpu::TextureFormat, bool),
    ) {
        let Some(targets) = self
            .view_targets
            .as_ref()
            .filter(|_| self.view_job.is_some())
        else {
            return;
        };
        let layouts = (camera_layout, &self.composite_layout);
        let layout = self.composite_pipeline_layout.get_or_insert_with(|| {
            crate::pipelines::pipeline_layout(
                device,
                &crate::pipeline_layouts::CLOUD_COMPOSITE_PIPELINES,
                &[layouts.0, layouts.1],
            )
        });
        self.composite_pipelines
            .entry((format, canvas))
            .or_insert_with(|| composite_pipeline(device, shader, layout, format, canvas));
        self.composite_group.bind(
            device,
            "engine/cloud-composite",
            (&CLOUD_COMPOSITE, &self.composite_layout),
            [(
                "view_clouds",
                Bound::of(targets, crate::pipelines::view(&targets.clouds.1)),
            )],
        );
    }

    pub(crate) fn frame(&self, target: (wgpu::TextureFormat, bool)) -> Option<CloudFrame> {
        let dome = self.job.is_some() && self.domes.is_some();
        let view = self.view_job.map(|_| target);
        (dome || view.is_some()).then_some(CloudFrame { dome, view })
    }

    fn draw_composite(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        camera: &wgpu::BindGroup,
        (format, canvas): (wgpu::TextureFormat, bool),
    ) {
        let (Some(pipeline), Some(group)) = (
            self.composite_pipelines.get(&(format, canvas)),
            self.composite_group.get(),
        ) else {
            return;
        };
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, camera, &[]);
        pass.set_bind_group(1, group, &[]);
        pass.draw_indirect(&self.composite_args, 0);
    }

    fn rotate(&mut self) {
        let [_, new, marching] = self.roles;
        let spare = (0..3)
            .find(|&index| index != new && index != marching)
            .unwrap_or(0);
        self.roles = [new, marching, spare];
        self.blend = 0.0;
        self.completed += 1;
        self.ready = true;
    }

    fn write_domes(&self, queue: &crate::gpu_calls::Queue, eye: glam::Vec3, settings: &Clouds) {
        let shows = if self.active && self.ready { 1.0 } else { 0.0 };
        let [old, new, _] = self.roles;
        let (older, newer) = (self.origins[old], self.origins[new]);
        let uniform = DomeUniform {
            older: older.eye.extend(shows).to_array(),
            newer: newer.eye.extend(self.blend).to_array(),
            older_mirror: older.mirror.extend(0.0).to_array(),
            newer_mirror: newer.mirror.extend(0.0).to_array(),
            eye: eye
                .extend(settings.base + 0.5 * settings.thickness)
                .to_array(),
            view: [if self.view_sky { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0],
        };
        queue.write_buffer(&self.dome_uniform, 0, bytemuck::bytes_of(&uniform));
    }

    fn encode_dome(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &crate::profiling::GpuProfiler,
    ) {
        let (Some((target, _, count)), Some(domes), Some([march, shade, light]), Some(groups)) = (
            self.job,
            &self.domes,
            &self.pipelines,
            self.march_groups.get(),
        ) else {
            return;
        };
        let group = &groups[target];
        if self.light.job > 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/cloud-light"),
                timestamp_writes: profiler.span("cloud_light").map(|span| span.compute()),
            });
            pass.set_pipeline(light);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(
                LIGHT_TEXELS.div_ceil(CLOUD_GROUP),
                self.light.job.div_ceil(CLOUD_GROUP),
                1,
            );
        }
        let width = domes[target].0.width();
        if count > 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/cloud-march"),
                timestamp_writes: profiler.span("clouds").map(|span| span.compute()),
            });
            pass.set_pipeline(march);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(
                width.div_ceil(CLOUD_GROUP),
                count.div_ceil(CLOUD_GROUP),
                DOME_LAYERS,
            );
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/cloud-shadows"),
                timestamp_writes: profiler.span("cloud_shadows").map(|span| span.compute()),
            });
            pass.set_pipeline(shade);
            pass.set_bind_group(0, group, &[]);
            let shadow_rows = SHADOW_CELLS.div_ceil(self.shadow_stride);
            pass.dispatch_workgroups(
                (SHADOW_CELLS / 2).div_ceil(CLOUD_GROUP),
                shadow_rows.div_ceil(CLOUD_GROUP),
                1,
            );
        }
        encoder.copy_buffer_to_buffer(
            &self.shadow_cells,
            0,
            &self.shadows,
            0,
            CLOUD_SHADOW_BYTES as u64,
        );
    }

    pub(crate) fn shadows(&self) -> &wgpu::Buffer {
        &self.shadows
    }

    pub(crate) fn capacity_bytes(&self) -> u64 {
        let noise = self.noise.as_ref().map_or(0, |noise| noise.bytes);
        let domes = self.domes.as_ref().map_or(0, |domes| {
            domes
                .iter()
                .map(|(texture, _)| {
                    u64::from(texture.width())
                        * u64::from(texture.height())
                        * u64::from(texture.depth_or_array_layers())
                        * 8
                })
                .sum()
        });
        let view = self.view_targets.as_ref().map_or(0, |targets| {
            u64::from(targets.size[0]) * u64::from(targets.size[1]) * 8
        });
        let light = self.light.cells.as_ref().map_or(0, wgpu::Buffer::size);
        noise + domes + view + light
    }
}

fn march_uniform(
    settings: &Clouds,
    (origin, at): (glam::Vec3, f64),
    lighting: &LightingUniform,
) -> CloudUniform {
    let wrap = f64::from(settings.scale) * 43.0;
    let drift = |speed: f32| ((f64::from(speed) * at) % wrap) as f32;
    let seed = (f64::from(settings.seed) * 7919.0 % wrap) as f32;
    CloudUniform {
        eye: origin.extend(lighting.grade[2]).to_array(),
        mirror: [0.0; 4],
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
        dome: [0; 4],
        sun: lighting.sun,
        sun_color: lighting.sun_color,
        zenith: lighting.zenith,
        horizon: lighting.horizon,
        shadow: [0.0; 4],
        shadow_rows: [0; 4],
        light: [0.0; 4],
        light_rows: [0; 4],
    }
}

fn view_texture(device: &wgpu::Device, size: [u32; 2]) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = crate::memory::texture(device, &wgpu::TextureDescriptor {
        label: Some("engine/cloud-view"),
        size: wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn view_pipeline(
    device: &wgpu::Device,
    layout: &crate::pipelines::GroupLayout,
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("engine/clouds.wgsl"),
        source: wgpu::ShaderSource::Wgsl(crate::gpu_data::CLOUDS_SHADER.into()),
    });
    let layout = crate::pipelines::pipeline_layout(
        device,
        &crate::pipeline_layouts::CLOUD_VIEW_PIPELINES,
        &[layout],
    );
    crate::pipelines::compute_entry(device, &layout, &module, clouds::march_view)
}

fn composite_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    format: wgpu::TextureFormat,
    canvas: bool,
) -> wgpu::RenderPipeline {
    crate::pipelines::geometry(
        device,
        &crate::pipelines::Geometry {
            label: "engine/cloud-composite",
            layout,
            module: shader,
            vertex_constants: &[],
            vertex: crate::entries::SKY_VERTEX,
            buffers: &[],
            primitive: wgpu::PrimitiveState::default(),
            depth: None,
            fragment: Some(crate::pipelines::Fragment {
                entry: if canvas {
                    meshes::fs_view_clouds
                } else {
                    meshes::fs_view_clouds_post
                },
                constants: &[],
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::SrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::Zero,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
        },
    )
}

fn dome_texture(device: &wgpu::Device, size: [u32; 2]) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = crate::memory::texture(device, &wgpu::TextureDescriptor {
        label: Some("engine/cloud-dome"),
        size: wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: DOME_LAYERS,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DOME_FORMAT,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    (texture, view)
}

pub(crate) fn shadow_grid(eye: glam::Vec3) -> Option<([f32; 2], f32)> {
    let cell = SHADOW_EXTENT / SHADOW_CELLS as f32;
    if !eye.is_finite() {
        return None;
    }
    let half = (SHADOW_CELLS / 2) as f32;
    let first = |value: f32| (value / cell).floor() - half;
    Some(([first(eye.x), first(eye.z)], cell))
}

pub(crate) fn shadow_lighting(clouds: &Clouds, eye: glam::Vec3) -> ([f32; 4], [f32; 4]) {
    let clouds = clouds.sanitized();
    match shadow_grid(eye) {
        Some((first, cell)) if clouds.enabled && clouds.coverage > 0.0 => (
            [first[0], first[1], 1.0 / cell, clouds.shadows],
            [clouds.base, 0.0, 0.0, 0.0],
        ),
        _ => ([0.0; 4], [0.0; 4]),
    }
}

fn shadow_uniform(device: &wgpu::Device, queue: &crate::gpu_calls::Queue) -> wgpu::Buffer {
    let buffer = crate::buffers::storage(
        device,
        "engine/cloud-shadows",
        CLOUD_SHADOW_BYTES as u64,
        wgpu::BufferUsages::empty(),
    );
    let ones = vec![0x3c00_3c00_u32; CLOUD_SHADOW_BYTES / 4];
    queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&ones));
    buffer
}

fn pipelines(
    device: &wgpu::Device,
    march_layout: &crate::pipelines::GroupLayout,
) -> [wgpu::ComputePipeline; 3] {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("engine/clouds.wgsl"),
        source: wgpu::ShaderSource::Wgsl(crate::gpu_data::CLOUDS_SHADER.into()),
    });
    let march = crate::pipelines::pipeline_layout(
        device,
        &crate::pipeline_layouts::CLOUD_MARCH_PIPELINES,
        &[march_layout],
    );
    [
        crate::pipelines::compute_entry(device, &march, &module, clouds::march_clouds),
        crate::pipelines::compute_entry(device, &march, &module, clouds::shadow_clouds),
        crate::pipelines::compute_entry(device, &march, &module, clouds::light_clouds),
    ]
}

pub(crate) struct CloudFrame {
    dome: bool,
    view: Option<(wgpu::TextureFormat, bool)>,
}

impl RenderFeature for CloudPass {
    type Frame = CloudFrame;
    type Stats = engine_core::CloudStats;

    fn declare(frame: &CloudFrame, shape: &FrameShape, graph: &mut RenderGraph) {
        if frame.dome {
            graph
                .add(Pass::Feature(FeaturePass::CloudDome))
                .write(Resource::CloudDome);
        }
        if frame.view.is_some() {
            graph
                .add(Pass::Feature(FeaturePass::CloudMarch))
                .read(Resource::Depth)
                .write(Resource::CloudView);
            graph
                .add(Pass::Feature(FeaturePass::CloudComposite))
                .read(Resource::CloudView)
                .color(shape.scene(), Access::Load);
        }
    }

    fn encode(
        &self,
        frame: &CloudFrame,
        node: &Node,
        encode: &Encode<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> u32 {
        let pass = node.pass();
        if pass == Pass::Feature(FeaturePass::CloudDome) {
            self.encode_dome(encoder, encode.profiler);
        } else if pass == Pass::Feature(FeaturePass::CloudMarch) {
            self.encode_view(encoder, encode.profiler);
        } else if let Some(target) = frame.view {
            let mut pass = encode.render_pass(
                encoder,
                node,
                "engine/cloud-composite",
                encode
                    .profiler
                    .span("cloud_composite")
                    .map(crate::profiling::Span::render),
            );
            self.draw_composite(&mut pass, encode.camera, target);
        }
        0
    }

    fn after_submit(&mut self, frame: Option<&CloudFrame>) {
        if frame.is_some_and(|frame| frame.dome) {
            self.recorded = self.completed;
            self.recorded_generation = Some(self.generation);
        }
    }

    fn capacity_bytes(&self) -> u64 {
        CloudPass::capacity_bytes(self)
    }

    fn stats(
        this: Option<&Self>,
        _frame: Option<&CloudFrame>,
        _snapshot: &engine_core::RenderSnapshot,
    ) -> engine_core::CloudStats {
        engine_core::CloudStats {
            rows: this.map_or(0, |clouds| clouds.rows),
            view_pixels: this.map_or(0, |clouds| clouds.view_pixels),
        }
    }
}
