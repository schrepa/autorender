use engine_core::Error;
use engine_core::math::min;
use engine_places::sun::Sun;
use engine_render::{
    Attach, Bind, Buffer, ComputePipeline, Copied, CopyRegion, Count, DepthUse, Draw, DrawInput,
    Encode, Install, Layer, Lifetime, Own, Pass, Pipeline, Prepare, RenderPipeline, Sampler,
    SamplerDesc, Shape, Texture, ViewTarget, wgpu,
};

use crate::gpu_layouts::{
    CLOUD_GROUP, CLOUD_LIGHT_BYTES, CLOUD_SHADOW_BYTES, CLOUD_SHADOW_CELLS, CloudUniform,
    CloudView, DOME_LAYERS, LIGHT_TEXELS, SkyPiece,
};
use crate::settings::Clouds;
use crate::sky::SkyGpu;
use crate::tables::StaticTable;
use crate::tables::clouds::{CloudNoise, DETAIL_SIZE, SHAPE_SIZE};

const DOME_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const DOME_ROLES: u32 = 3;
const ENVIRONMENT_EVERY: u64 = 16;
const SHADOW_EXTENT: f32 = 8192.0;
const SHADOW_SLICES: u32 = 4;
const LIGHT_EXTENT: f32 = 24_000.0;
const LIGHT_RECENTER: f32 = 2_000.0;
const LIGHT_ROWS: u32 = 48;
const COMPOSITE_ARGS: [u32; 4] = [0, 1, 0, 0];

pub(crate) type CloudLight<'a> = (&'a Sun, &'a SkyPiece);

type CloudInputs<'a> = (
    &'a Clouds,
    Option<&'a StaticTable<CloudNoise>>,
    (glam::Vec3, Option<f32>),
    f64,
);

#[derive(Default)]
struct LightMap {
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

fn upload_volume(
    cx: &mut Prepare<'_, SkyGpu>,
    texture: Texture<SkyGpu>,
    side: u32,
    levels: &[&[u8]],
) {
    cx.allocate(
        texture,
        wgpu::Extent3d {
            width: side,
            height: side,
            depth_or_array_layers: side,
        },
        levels.len() as u32,
    );
    for (level, &texels) in (0_u32..).zip(levels) {
        let side = (side >> level).max(1);
        cx.write_texture_level(
            texture,
            level,
            wgpu::Origin3d::ZERO,
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
}

fn upload_noise(
    cx: &mut Prepare<'_, SkyGpu>,
    (shape, detail): (Texture<SkyGpu>, Texture<SkyGpu>),
    noise: &StaticTable<CloudNoise>,
) {
    let table = &noise.table;
    upload_volume(cx, shape, SHAPE_SIZE, &[table.shape.as_slice()]);
    let below = levels_below(DETAIL_SIZE, &table.detail);
    let detail_levels: Vec<&[u8]> = std::iter::once(table.detail.as_slice())
        .chain(below.iter().map(Vec::as_slice))
        .collect();
    upload_volume(cx, detail, DETAIL_SIZE, &detail_levels);
}

pub(crate) fn cloud_regions() -> [CopyRegion; 3] {
    let at = |offset: usize| offset as u64;
    [
        CopyRegion::Bytes {
            from: crate::sky::DERIVED_SUN * 16,
            to: at(std::mem::offset_of!(CloudUniform, sun_color)),
            size: 12,
        },
        CopyRegion::Bytes {
            from: crate::sky::DERIVED_ZENITH * 16,
            to: at(std::mem::offset_of!(CloudUniform, zenith)),
            size: 12,
        },
        CopyRegion::Bytes {
            from: crate::sky::DERIVED_HORIZON * 16,
            to: at(std::mem::offset_of!(CloudUniform, horizon)),
            size: 12,
        },
    ]
}

pub(crate) struct CloudPlan {
    light: LightMap,
    noise: Option<u64>,
    dome_size: Option<[u32; 2]>,
    roles: [usize; 3],
    origins: [Origins; 3],
    blend: f32,
    dome_piece: [[f32; 4]; 6],
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
    shadowing: bool,
    pub(crate) rows: u32,
    view_size: [u32; 2],
    view_job: Option<[u32; 2]>,
    view_sky: bool,
    pub(crate) view_pixels: u32,
    uniform: CloudUniform,
    view_clouds: CloudUniform,
    view: CloudView,
}

impl Default for CloudPlan {
    fn default() -> Self {
        Self {
            light: LightMap::default(),
            noise: None,
            dome_size: None,
            roles: [0, 1, 2],
            origins: [Origins::default(); 3],
            blend: 1.0,
            dome_piece: [[0.0; 4]; 6],
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
            shadowing: false,
            rows: 0,
            view_size: [0, 0],
            view_job: None,
            view_sky: false,
            view_pixels: 0,
            uniform: CloudUniform::default(),
            view_clouds: CloudUniform::default(),
            view: CloudView::default(),
        }
    }
}

impl CloudPlan {
    fn plan_light(&mut self, uniform: &CloudUniform) -> [u32; 4] {
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
        let rows = LIGHT_ROWS.min(LIGHT_TEXELS - first);
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

    fn shadow_rows(&mut self) -> [u32; 4] {
        self.shadow_stride = SHADOW_SLICES;
        [(self.frame % u64::from(SHADOW_SLICES)) as u32, SHADOW_SLICES, 0, 0]
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

    fn environment_shows(&self) -> bool {
        self.active && self.ready && self.recorded_generation == Some(self.generation)
    }

    pub(crate) fn dome_layers(&self) -> [u32; 4] {
        [
            self.roles[0] as u32 * DOME_LAYERS,
            self.roles[1] as u32 * DOME_LAYERS,
            u32::from(self.environment_shows()),
            0,
        ]
    }

    pub(crate) fn submitted(&mut self, frame: Option<CloudFrame>) {
        if frame.is_some_and(|frame| frame.dome) {
            self.recorded = self.completed;
            self.recorded_generation = Some(self.generation);
        }
    }

    pub(crate) fn prepare(
        &mut self,
        (settings, noise, (eye, surface), seconds): CloudInputs<'_>,
        light: CloudLight<'_>,
    ) {
        self.rows = 0;
        self.job = None;
        let wanted = settings.enabled && settings.coverage > 0.0;
        if let Some(noise) = noise.filter(|_| wanted)
            && self.noise.is_none_or(|held| held != noise.id)
        {
            self.noise = Some(noise.id);
            self.generation += 1;
        }
        self.active = wanted && self.noise.is_some();
        self.shadowing = settings.shadows > 0.0;
        if !self.active {
            self.marched = None;
            self.cycle = None;
            self.completes = false;
            self.release();
            self.write_domes(eye, settings);
            return;
        }
        let size = [settings.resolution, settings.resolution / 4];
        let fresh = self.dome_size.is_none_or(|held| held != size);
        if fresh {
            self.dome_size = Some(size);
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
        let changed = self.marched.is_none_or(|marched| marched != *settings);
        self.frame += 1;
        let target = self.roles[2];
        let origins = Origins::of(eye, surface);
        let (origin, at, first, count) = if fresh || changed || !self.ready {
            let spare = (0..3).find(|&index| index != target).unwrap_or(0);
            self.roles = [target, target, spare];
            self.origins[target] = origins;
            self.blend = 1.0;
            self.marched = Some(*settings);
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
        let mut uniform = march_uniform(settings, (origin.eye, at), light);
        uniform.mirror = origin.mirror.extend(0.0).to_array();
        uniform.dome = [size[0], size[1], first, count];
        uniform.shadow = shadow_grid(eye).map_or([0.0; 4], |(first, cell)| {
            [first[0], first[1], cell, CLOUD_SHADOW_CELLS as f32]
        });
        uniform.shadow_rows = self.shadow_rows();
        uniform.light_rows = self.plan_light(&uniform);
        uniform.light = self.light_uniform();
        uniform.layers = [target as u32 * DOME_LAYERS, 0, 0, 0];
        self.uniform = uniform;
        self.write_domes(eye, settings);
    }

    fn release(&mut self) {
        self.dome_size = None;
        self.noise = None;
        self.light = LightMap::default();
    }

    pub(crate) fn prepare_view(
        &mut self,
        (settings, eye, seconds): (&Clouds, glam::Vec3, f64),
        (view_projection, display, render): (glam::Mat4, glam::UVec2, glam::UVec2),
        light: CloudLight<'_>,
    ) {
        self.view_job = None;
        self.view_pixels = 0;
        let marches = self.active && settings.in_view && display.x > 0 && display.y > 0;
        let view_sky = marches && eye.y > settings.base;
        if view_sky != self.view_sky {
            self.view_sky = view_sky;
            self.write_domes(eye, settings);
        }
        if !marches {
            return;
        }
        self.view_size = [display.x, display.y];
        let mut uniform = march_uniform(settings, (eye, seconds), light);
        uniform.light = self.light_uniform();
        self.view_clouds = uniform;
        self.view = CloudView {
            inverse_view_projection: view_projection.inverse().to_cols_array_2d(),
            pixels: [render.x, render.y, u32::from(view_sky), 0],
        };
        self.view_job = Some([render.x, render.y]);
        self.view_pixels = render.x * render.y;
    }

    pub(crate) fn frame(&self) -> Option<CloudFrame> {
        let dome = self.job.is_some() && self.dome_size.is_some();
        let view = self.view_job.is_some();
        (dome || view).then_some(CloudFrame { dome, view })
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

    fn write_domes(&mut self, eye: glam::Vec3, settings: &Clouds) {
        let shows = if self.active && self.ready { 1.0 } else { 0.0 };
        let [old, new, _] = self.roles;
        let (older, newer) = (self.origins[old], self.origins[new]);
        self.dome_piece = [
            older.eye.extend(shows).to_array(),
            newer.eye.extend(self.blend).to_array(),
            older.mirror.extend(0.0).to_array(),
            newer.mirror.extend(0.0).to_array(),
            eye.extend(settings.base + 0.5 * settings.thickness)
                .to_array(),
            [if self.view_sky { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0],
        ];
    }

    pub(crate) fn dome_piece(&self) -> [[f32; 4]; 6] {
        self.dome_piece
    }
}

#[derive(Clone, Copy, Default)]
struct CloudRun {
    light_rows: u32,
    width: u32,
    count: u32,
    shadow_stride: u32,
    view: [u32; 2],
}

pub(crate) struct CloudPass {
    noise_shape: Texture<SkyGpu>,
    noise_detail: Texture<SkyGpu>,
    domes: Texture<SkyGpu>,
    view_texture: Texture<SkyGpu>,
    shadows: Buffer<SkyGpu>,
    shadow_cells: Buffer<SkyGpu>,
    light_cells: Buffer<SkyGpu>,
    uniform: Buffer<SkyGpu>,
    view_clouds: Buffer<SkyGpu>,
    view_uniform: Buffer<SkyGpu>,
    composite_args: Buffer<SkyGpu>,
    light_pass: Pass<SkyGpu>,
    march_pass: Pass<SkyGpu>,
    shadow_pass: Pass<SkyGpu>,
    shadow_copy: Pass<SkyGpu>,
    view_pass: Pass<SkyGpu>,
    composite_pass: Pass<SkyGpu>,
    marching: ComputePipeline<SkyGpu>,
    shading: ComputePipeline<SkyGpu>,
    lighting: ComputePipeline<SkyGpu>,
    viewing: ComputePipeline<SkyGpu>,
    compositing: Pipeline<SkyGpu, Own>,
    composite_draws: DrawInput<SkyGpu, Own>,
    noise: Option<u64>,
    shadows_filled: bool,
    run: CloudRun,
}

fn composite_pipeline(
    cx: &mut Install<'_, SkyGpu>,
    pass: Pass<SkyGpu>,
    view_texture: Texture<SkyGpu>,
) -> Result<Pipeline<SkyGpu, Own>, Error> {
    use crate::shaders::sky::draw;
    cx.render_pipeline(
        pass,
        &RenderPipeline {
            vertex: draw::VS_SKY,
            fragment: Some(draw::FS_VIEW_CLOUDS_POST),
            debug: None,
            vertices: &[],
            primitive: wgpu::PrimitiveState::default(),
            depth: DepthUse::None,
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
            writes: wgpu::ColorWrites::ALL,
        },
        &[(draw::VIEW_CLOUDS, Bind::Texture(view_texture))],
    )
}

fn samplers(cx: &mut Install<'_, SkyGpu>) -> [Sampler<SkyGpu>; 2] {
    let dome = cx.sampler(SamplerDesc {
        filter: wgpu::FilterMode::Linear,
        mipmap: wgpu::MipmapFilterMode::Nearest,
        address: [
            wgpu::AddressMode::Repeat,
            wgpu::AddressMode::ClampToEdge,
            wgpu::AddressMode::ClampToEdge,
        ],
        compare: None,
        anisotropy: 1,
    });
    let noise = cx.sampler(SamplerDesc {
        filter: wgpu::FilterMode::Linear,
        mipmap: wgpu::MipmapFilterMode::Linear,
        address: [wgpu::AddressMode::Repeat; 3],
        compare: None,
        anisotropy: 1,
    });
    [dome, noise]
}

impl CloudPass {
    pub(crate) fn install(
        cx: &mut Install<'_, SkyGpu>,
        (transmittance, table_sampler): (Texture<SkyGpu>, Sampler<SkyGpu>),
        environment: Pass<SkyGpu>,
    ) -> Result<Self, Error> {
        use crate::shaders::sky::{clouds, faces::cloud_shadows, light};
        let noise_shape =
            cx.texture("cloud_noise_shape", wgpu::TextureFormat::R8Unorm, Lifetime::Kept);
        let noise_detail =
            cx.texture("cloud_noise_detail", wgpu::TextureFormat::R8Unorm, Lifetime::Kept);
        let domes = cx.texture("cloud_domes", DOME_FORMAT, Lifetime::Kept);
        let view_texture =
            cx.texture("cloud_view", wgpu::TextureFormat::Rgba16Float, Lifetime::Kept);
        let shadows = cx.buffer("cloud_shadows", Lifetime::Kept);
        let shadow_cells = cx.buffer("cloud_shadow_cells", Lifetime::Kept);
        let light_cells = cx.buffer("cloud_light", Lifetime::Kept);
        let uniform = cx.buffer("cloud_uniform", Lifetime::Kept);
        let view_clouds = cx.buffer("cloud_view_march", Lifetime::Kept);
        let view_uniform = cx.buffer("cloud_view_uniform", Lifetime::Kept);
        let composite_args = cx.buffer("cloud_composite_draw", Lifetime::Kept);
        let [dome_sampler, noise_sampler] = samplers(cx);
        cx.frame(light::SKY_DOMES, Bind::Texture(domes))?;
        cx.frame(light::SKY_CLOUD_SAMPLER, Bind::Sampler(dome_sampler))?;
        cx.frame(cloud_shadows::CLOUD_SHADOWS, Bind::Buffer(shadows))?;
        let light_pass = cx
            .pass("cloud_light")
            .after(environment)
            .label("engine/cloud-light")
            .span("cloud_light")
            .id();
        let march_pass = cx
            .pass("clouds")
            .after(environment)
            .after(light_pass)
            .label("engine/cloud-march")
            .span("clouds")
            .id();
        let shadow_pass = cx
            .pass("cloud_shadows")
            .after(march_pass)
            .label("engine/cloud-shadows")
            .span("cloud_shadows")
            .id();
        let shadow_copy = cx.copy(
            "cloud_shadow_copy",
            Copied::Buffer(shadow_cells),
            Copied::Buffer(shadows),
        );
        let view_pass = cx
            .pass("cloud_march")
            .at(Layer::Over)
            .label("engine/cloud-view")
            .span("cloud_view")
            .id();
        let composite_pass = cx
            .pass("cloud_composite")
            .at(Layer::Over)
            .after(view_pass)
            .label("engine/cloud-composite")
            .span("cloud_composite")
            .id();
        let noise = || {
            [
                (clouds::CLOUDS, Bind::Buffer(uniform)),
                (clouds::NOISE_SHAPE, Bind::Texture(noise_shape)),
                (clouds::NOISE_SAMPLER, Bind::Sampler(noise_sampler)),
            ]
        };
        let lit = || {
            [
                (clouds::CLOUD_TRANSMITTANCE, Bind::Texture(transmittance)),
                (clouds::CLOUD_TABLE_SAMPLER, Bind::Sampler(table_sampler)),
                (clouds::LIGHT_CELLS, Bind::Buffer(light_cells)),
            ]
        };
        let mut lighted = Vec::from(noise());
        lighted.extend([
            (clouds::NOISE_DETAIL, Bind::Texture(noise_detail)),
            (clouds::LIGHT_CELLS, Bind::Buffer(light_cells)),
        ]);
        let lighting = cx.compute_pipeline(light_pass, clouds::LIGHT_CLOUDS, &lighted)?;
        let mut marched = Vec::from(noise());
        marched.extend([
            (clouds::NOISE_DETAIL, Bind::Texture(noise_detail)),
            (clouds::DOME, Bind::Texture(domes)),
        ]);
        marched.extend(lit());
        let marching = cx.compute_pipeline(march_pass, clouds::MARCH_CLOUDS, &marched)?;
        let mut shaded = Vec::from(noise());
        shaded.extend([
            (clouds::NOISE_DETAIL, Bind::Texture(noise_detail)),
            (clouds::SHADOW_CELLS, Bind::Buffer(shadow_cells)),
        ]);
        let shading = cx.compute_pipeline(shadow_pass, clouds::SHADOW_CLOUDS, &shaded)?;
        let mut viewed = vec![
            (clouds::CLOUDS, Bind::Buffer(view_clouds)),
            (clouds::NOISE_SHAPE, Bind::Texture(noise_shape)),
            (clouds::NOISE_SAMPLER, Bind::Sampler(noise_sampler)),
            (clouds::NOISE_DETAIL, Bind::Texture(noise_detail)),
            (clouds::CLOUD_VIEW, Bind::Buffer(view_uniform)),
            (clouds::VIEW_DEPTH, Bind::Target(ViewTarget::Depth)),
            (clouds::VIEW_OUT, Bind::Texture(view_texture)),
            (clouds::COMPOSITE_DRAW, Bind::Buffer(composite_args)),
        ];
        viewed.extend(lit());
        let viewing = cx.compute_pipeline(view_pass, clouds::MARCH_VIEW, &viewed)?;
        let compositing = composite_pipeline(cx, composite_pass, view_texture)?;
        let composite_draws = cx.draw_input(composite_pass, composite_args);
        Ok(Self {
            noise_shape,
            noise_detail,
            domes,
            view_texture,
            shadows,
            shadow_cells,
            light_cells,
            uniform,
            view_clouds,
            view_uniform,
            composite_args,
            light_pass,
            march_pass,
            shadow_pass,
            shadow_copy,
            view_pass,
            composite_pass,
            marching,
            shading,
            lighting,
            viewing,
            compositing,
            composite_draws,
            noise: None,
            shadows_filled: false,
            run: CloudRun::default(),
        })
    }

    pub(crate) fn uniforms(&self) -> [Buffer<SkyGpu>; 2] {
        [self.uniform, self.view_clouds]
    }

    fn allocate(&mut self, cx: &mut Prepare<'_, SkyGpu>) {
        cx.allocate_buffer(self.uniform, size_of::<CloudUniform>() as u64);
        cx.allocate_buffer(self.view_clouds, size_of::<CloudUniform>() as u64);
        cx.allocate_buffer(self.view_uniform, size_of::<CloudView>() as u64);
        cx.allocate_buffer(self.composite_args, size_of_val(&COMPOSITE_ARGS) as u64);
        cx.allocate_buffer(self.shadows, CLOUD_SHADOW_BYTES as u64);
        cx.allocate_buffer(self.shadow_cells, CLOUD_SHADOW_BYTES as u64);
        if self.shadows_filled {
            return;
        }
        let ones = vec![0x3c00_3c00_u32; CLOUD_SHADOW_BYTES / 4];
        cx.write(self.shadows, 0, &ones);
        self.shadows_filled = true;
    }

    pub(crate) fn prepare(
        &mut self,
        cx: &mut Prepare<'_, SkyGpu>,
        (plan, noise): (&CloudPlan, Option<&StaticTable<CloudNoise>>),
        frame: Option<CloudFrame>,
    ) {
        self.allocate(cx);
        if plan.noise.is_none() {
            self.noise = None;
        }
        if !plan.active {
            return;
        }
        if let Some(noise) = noise.filter(|noise| Some(noise.id) == plan.noise)
            && self.noise != plan.noise
        {
            upload_noise(cx, (self.noise_shape, self.noise_detail), noise);
            self.noise = plan.noise;
        }
        if let Some(size) = plan.dome_size {
            cx.allocate(
                self.domes,
                wgpu::Extent3d {
                    width: size[0],
                    height: size[1],
                    depth_or_array_layers: DOME_ROLES * DOME_LAYERS,
                },
                1,
            );
        }
        cx.allocate_buffer(self.light_cells, CLOUD_LIGHT_BYTES as u64);
        cx.write(self.uniform, 0, &[plan.uniform]);
        if let Some(pixels) = plan.view_job {
            cx.allocate(
                self.view_texture,
                wgpu::Extent3d {
                    width: plan.view_size[0],
                    height: plan.view_size[1],
                    depth_or_array_layers: 1,
                },
                1,
            );
            cx.write(self.view_clouds, 0, &[plan.view_clouds]);
            cx.write(self.view_uniform, 0, &[plan.view]);
            cx.write(self.composite_args, 0, &COMPOSITE_ARGS);
            self.run.view = pixels;
        }
        let Some(frame) = frame else {
            return;
        };
        if let (true, Some((_, _, count)), Some(size)) = (frame.dome, plan.job, plan.dome_size) {
            self.run.light_rows = plan.light.job;
            self.run.width = size[0];
            self.run.count = count;
            self.run.shadow_stride = plan.shadow_stride;
            if plan.light.job > 0 {
                cx.run(self.light_pass);
            }
            if count > 0 {
                cx.run(self.march_pass);
            }
            if plan.shadowing {
                cx.run(self.shadow_pass);
                cx.copy(
                    self.shadow_copy,
                    &[CopyRegion::Bytes {
                        from: 0,
                        to: 0,
                        size: CLOUD_SHADOW_BYTES as u64,
                    }],
                );
            }
        }
        if frame.view {
            cx.run(self.view_pass);
            cx.run(self.composite_pass);
        }
    }

    pub(crate) fn owns(&self, pass: Pass<SkyGpu>) -> bool {
        pass == self.light_pass
            || pass == self.march_pass
            || pass == self.shadow_pass
            || pass == self.view_pass
            || pass == self.composite_pass
    }

    pub(crate) fn encode(&self, pass: Pass<SkyGpu>, cx: &mut Encode<'_, SkyGpu>) -> u32 {
        let run = self.run;
        if pass == self.light_pass {
            cx.dispatch(
                &self.lighting,
                Count::Groups([
                    LIGHT_TEXELS.div_ceil(CLOUD_GROUP),
                    run.light_rows.div_ceil(CLOUD_GROUP),
                    1,
                ]),
            );
        } else if pass == self.march_pass {
            cx.dispatch(
                &self.marching,
                Count::Groups([
                    run.width.div_ceil(CLOUD_GROUP),
                    run.count.div_ceil(CLOUD_GROUP),
                    DOME_LAYERS,
                ]),
            );
        } else if pass == self.shadow_pass {
            let shadow_rows = CLOUD_SHADOW_CELLS.div_ceil(run.shadow_stride);
            cx.dispatch(
                &self.shading,
                Count::Groups([
                    (CLOUD_SHADOW_CELLS / 2).div_ceil(CLOUD_GROUP),
                    shadow_rows.div_ceil(CLOUD_GROUP),
                    1,
                ]),
            );
        } else if pass == self.view_pass {
            cx.dispatch(
                &self.viewing,
                Count::Groups([
                    run.view[0].div_ceil(CLOUD_GROUP),
                    run.view[1].div_ceil(CLOUD_GROUP),
                    1,
                ]),
            );
        } else if pass == self.composite_pass {
            let mut draw = cx.render(Attach::Declared, None);
            self.draw_composite(&mut draw);
        }
        0
    }

    fn draw_composite(&self, draw: &mut Draw<'_, SkyGpu, Own>) {
        draw.draw(
            &self.compositing,
            Shape::Indirect {
                vertices: &[],
                indices: None,
                args: self.composite_draws,
                offset: 0,
            },
        );
    }
}

fn march_uniform(
    settings: &Clouds,
    (origin, at): (glam::Vec3, f64),
    (sun, piece): CloudLight<'_>,
) -> CloudUniform {
    let wrap = f64::from(settings.scale) * 43.0;
    let drift = |speed: f32| ((f64::from(speed) * at) % wrap) as f32;
    let seed = (f64::from(settings.seed) * 7919.0 % wrap) as f32;
    CloudUniform {
        eye: origin.extend(piece.state[0]).to_array(),
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
        sun: sun.direction,
        sun_color: sun.color,
        zenith: piece.zenith,
        horizon: piece.horizon,
        shadow: [0.0; 4],
        shadow_rows: [0; 4],
        light: [0.0; 4],
        light_rows: [0; 4],
        layers: [0; 4],
    }
}

pub(crate) fn shadow_grid(eye: glam::Vec3) -> Option<([f32; 2], f32)> {
    let cell = SHADOW_EXTENT / CLOUD_SHADOW_CELLS as f32;
    if !eye.is_finite() {
        return None;
    }
    let half = (CLOUD_SHADOW_CELLS / 2) as f32;
    let first = |value: f32| (value / cell).floor() - half;
    Some(([first(eye.x), first(eye.z)], cell))
}

pub(crate) fn shadow_lighting(clouds: &Clouds, eye: glam::Vec3) -> ([f32; 4], [f32; 4]) {
    match shadow_grid(eye) {
        Some((first, cell)) if clouds.enabled && clouds.coverage > 0.0 => (
            [first[0], first[1], 1.0 / cell, clouds.shadows],
            [clouds.base, 0.0, 0.0, 0.0],
        ),
        _ => ([0.0; 4], [0.0; 4]),
    }
}

#[derive(Clone, Copy)]
pub(crate) struct CloudFrame {
    dome: bool,
    view: bool,
}
