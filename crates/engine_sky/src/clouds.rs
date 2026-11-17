use bevy_ecs::prelude::Resource;
use engine_core::math::min;
use engine_core::{Error, Filter, FrameBudget, Readable, Readback, Settling, Wrap};
use engine_places_rendering::sun::Sun;
use engine_render::{
    Allocation, Attach, AttachPoint, Bind, Buffer, ComputePipeline, CopyRegion, Count, DepthUse,
    Draw, DrawInput, Encode, Install, Lifetime, Own, Pass, Pipeline, Prepare, RenderPipeline,
    Sampler, Shape, TargetName, Texture, wgpu,
};

use crate::gpu_layouts::{
    CLOUD_GROUP, CLOUD_LIGHT_BYTES, CloudUniform, DOME_FIRST, DOME_LAYERS, LIGHT_STEPS,
    LIGHT_TEXELS, SkyPiece,
};
use crate::settings::Clouds;
use crate::sky::SkyGpu;
use crate::tables::StaticTable;
use crate::tables::clouds::{CloudNoise, DETAIL_SIZE, SHAPE_SIZE};

const DOME_ROLES: u32 = 3;
const ENVIRONMENT_EVERY: u64 = 16;
const LIGHT_EXTENT: f32 = 24_000.0;
const LIGHT_RECENTER: f32 = 2_000.0;
const LIGHT_ROW_BYTES: u64 = CLOUD_LIGHT_BYTES as u64 / LIGHT_TEXELS as u64;
const COMPOSITE_ARGS: [u32; 4] = [0; 4];
const BUDGET: &str = "clouds";

pub(crate) type CloudLight<'a> = (&'a Sun, &'a SkyPiece);

type CloudInputs<'a> = (
    (&'a Clouds, bool),
    Option<&'a StaticTable<CloudNoise>>,
    (glam::Vec3, Option<f32>),
    f64,
);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CloudStep {
    completed: u64,
    generation: u64,
}

impl CloudStep {
    fn bytes(self, ran: bool) -> Vec<u8> {
        let mut bytes = vec![u8::from(ran)];
        bytes.extend_from_slice(&self.completed.to_le_bytes());
        bytes.extend_from_slice(&self.generation.to_le_bytes());
        bytes
    }
}

#[derive(Resource, Default)]
pub(crate) struct CloudSteps {
    done: Vec<(bool, CloudStep)>,
}

impl Readable for CloudSteps {
    fn arrived(&mut self, _: u32, read: Readback<&[u8]>) {
        let Some((
            &ran,
            [
                c0,
                c1,
                c2,
                c3,
                c4,
                c5,
                c6,
                c7,
                g0,
                g1,
                g2,
                g3,
                g4,
                g5,
                g6,
                g7,
            ],
        )) = read.value.split_first()
        else {
            return;
        };
        let step = CloudStep {
            completed: u64::from_le_bytes([*c0, *c1, *c2, *c3, *c4, *c5, *c6, *c7]),
            generation: u64::from_le_bytes([*g0, *g1, *g2, *g3, *g4, *g5, *g6, *g7]),
        };
        self.done.push((ran != 0, step));
    }
}

#[derive(Default)]
struct LightMap {
    held: Option<LightKey>,
    built: u32,
    job: u32,
}

impl LightMap {
    fn complete(&self) -> bool {
        self.built >= LIGHT_TEXELS
    }
}

#[derive(Clone, Copy, PartialEq)]
struct LightKey {
    corner: [f32; 2],
    sun: [f32; 3],
    shape: Clouds,
}

struct Cycle {
    origin: Origins,
    seconds: f64,
    next_row: u32,
    lit: bool,
}

struct March {
    origin: Origins,
    at: f64,
    first: u32,
    count: u32,
    whole: bool,
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
    let volume = wgpu::Extent3d {
        width: side,
        height: side,
        depth_or_array_layers: side,
    };
    let (Allocation::Fresh | Allocation::Kept) = cx.allocate(texture, volume, levels.len() as u32);
    write_volume(cx, texture, side, levels);
}

fn write_volume(
    cx: &mut Prepare<'_, SkyGpu>,
    texture: Texture<SkyGpu>,
    side: u32,
    levels: &[&[u8]],
) {
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
    for (texture, side, texels) in [
        (shape, SHAPE_SIZE, &table.shape),
        (detail, DETAIL_SIZE, &table.detail),
    ] {
        let below = levels_below(side, texels);
        let levels: Vec<&[u8]> = std::iter::once(texels.as_slice())
            .chain(below.iter().map(Vec::as_slice))
            .collect();
        upload_volume(cx, texture, side, &levels);
    }
}

pub(crate) fn cloud_regions() -> [CopyRegion; 2] {
    let at = |offset: usize| offset as u64;
    [
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
    lit: [bool; 3],
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
    pub(crate) step: CloudStep,
    job: Option<(usize, u32, u32)>,
    active: bool,
    pub(crate) rows: u32,
    view_size: [u32; 2],
    view_job: Option<[u32; 2]>,
    view_sky: bool,
    pub(crate) view_pixels: u32,
    uniform: CloudUniform,
    view_clouds: CloudUniform,
    settling: Settling,
}

impl Default for CloudPlan {
    fn default() -> Self {
        Self {
            light: LightMap::default(),
            noise: None,
            dome_size: None,
            roles: [0, 1, 2],
            origins: [Origins::default(); 3],
            lit: [false; 3],
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
            step: CloudStep {
                completed: 0,
                generation: 0,
            },
            job: None,
            active: false,
            rows: 0,
            view_size: [0, 0],
            view_job: None,
            view_sky: false,
            view_pixels: 0,
            uniform: CloudUniform::default(),
            view_clouds: CloudUniform::default(),
            settling: Settling::on_view(),
        }
    }
}

impl CloudPlan {
    fn plan_light(
        &mut self,
        (uniform, settings): (&CloudUniform, &Clouds),
        budget: &mut FrameBudget,
    ) -> [u32; 4] {
        let [eye_x, _, eye_z, _] = uniform.eye;
        let [drift_x, drift_z, _, seed] = uniform.drift;
        let own = [eye_x + drift_x + seed, eye_z + drift_z - seed];
        let key = LightKey {
            corner: [0.0; 2],
            sun: [uniform.sun[0], uniform.sun[1], uniform.sun[2]],
            shape: *settings,
        };
        let keep = self.light.held.is_some_and(|held| {
            held.sun == key.sun
                && held.shape == key.shape
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
        let mut rows = 0;
        while first + rows < LIGHT_TEXELS && budget.spend(BUDGET, LIGHT_ROW_BYTES) {
            rows += 1;
        }
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
                f32::from(u8::from(self.light.complete())),
            ]
        })
    }

    pub(crate) fn environment_key(&self) -> [f32; 4] {
        if self.environment_shows() {
            [
                1.0,
                (self.recorded / ENVIRONMENT_EVERY) as f32,
                self.generation as f32,
                f32::from(u8::from(self.shows_lit())),
            ]
        } else {
            [0.0; 4]
        }
    }

    fn environment_shows(&self) -> bool {
        self.active && self.ready && self.recorded_generation == Some(self.generation)
    }

    pub(crate) fn layers_extent(&self, view: (u32, u32)) -> wgpu::Extent3d {
        let ((width, height), layers) = match self.dome_size {
            Some([width, height]) => (
                (width.max(view.0), height.max(view.1)),
                DOME_FIRST + DOME_ROLES * DOME_LAYERS,
            ),
            None => (view, DOME_FIRST),
        };
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: layers,
        }
    }

    pub(crate) fn dome_size(&self) -> [f32; 4] {
        let [width, height] = self.dome_size.unwrap_or([1, 1]);
        [width as f32, height as f32, 0.0, 0.0]
    }

    pub(crate) fn dome_layers(&self) -> [u32; 4] {
        [
            DOME_FIRST + self.roles[0] as u32 * DOME_LAYERS,
            DOME_FIRST + self.roles[1] as u32 * DOME_LAYERS,
            u32::from(self.environment_shows()),
            0,
        ]
    }

    pub(crate) fn acknowledge(&mut self, steps: &mut CloudSteps) {
        for (ran, step) in steps.done.drain(..) {
            if ran {
                self.recorded = step.completed;
                self.recorded_generation = Some(step.generation);
            } else {
                self.ready = false;
                self.cycle = None;
                self.completes = false;
                self.light.built = 0;
            }
        }
    }

    pub(crate) fn prepare(
        &mut self,
        ((settings, wanted), noise, (eye, surface), seconds): CloudInputs<'_>,
        light: CloudLight<'_>,
        (budget, preparing): (&mut FrameBudget, bool),
    ) {
        self.rows = 0;
        self.job = None;
        if let Some(noise) = noise.filter(|_| wanted)
            && self.noise.is_none_or(|held| held != noise.id)
        {
            self.noise = Some(noise.id);
            self.generation += 1;
        }
        self.active = wanted && self.noise.is_some();
        if !self.active {
            self.marched = None;
            self.cycle = None;
            self.completes = false;
            self.release();
            self.write_domes(eye, settings);
            self.settling.left(0);
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
        let target = self.roles[2];
        let march = self.next_march(settings, Origins::of(eye, surface), (seconds, preparing));
        let mut uniform = march_uniform(settings, (march.origin.eye, march.at), light);
        uniform.mirror = march.origin.mirror.extend(0.0).to_array();
        uniform.light_rows = self.plan_light((&uniform, settings), budget);
        uniform.light = self.light_uniform();
        let lit = self.light.complete();
        let held = preparing && (!lit || self.light.job > 0);
        let count = if held {
            0
        } else {
            self.commit(&march, (settings, size[1], lit))
        };
        self.job = Some((target, march.first, count));
        self.rows = count;
        uniform.dome = [size[0], size[1], march.first, count];
        uniform.layers = [
            DOME_FIRST + target as u32 * DOME_LAYERS,
            (self.completed % 1024) as u32,
            0,
            0,
        ];
        self.uniform = uniform;
        if count > 0 {
            self.step = CloudStep {
                completed: self.completed,
                generation: self.generation,
            };
        }
        self.write_domes(eye, settings);
        self.settling.left(self.left());
    }

    fn shows_lit(&self) -> bool {
        self.lit[self.roles[0]] && self.lit[self.roles[1]]
    }

    fn left(&self) -> u32 {
        let light = LIGHT_TEXELS.saturating_sub(self.light.built);
        let dome = u32::from(!self.ready || !self.shows_lit());
        let told = u32::from(self.ready && self.recorded_generation != Some(self.generation));
        light + dome + told
    }

    fn next_march(
        &mut self,
        settings: &Clouds,
        origins: Origins,
        (seconds, preparing): (f64, bool),
    ) -> March {
        let rows = self.dome_size.map_or(0, |[_, rows]| rows);
        let changed = self.marched.is_none_or(|marched| marched != *settings);
        if changed || !self.ready || (preparing && !self.shows_lit()) {
            return March {
                origin: origins,
                at: seconds,
                first: 0,
                count: rows,
                whole: true,
            };
        }
        let cycle = self.cycle.get_or_insert(Cycle {
            origin: origins,
            seconds,
            next_row: 0,
            lit: true,
        });
        let first = cycle.next_row;
        March {
            origin: cycle.origin,
            at: cycle.seconds,
            first,
            count: rows.div_ceil(settings.slices.max(1)).min(rows - first),
            whole: false,
        }
    }

    fn commit(&mut self, march: &March, (settings, rows, lit): (&Clouds, u32, bool)) -> u32 {
        let target = self.roles[2];
        if march.whole {
            let spare = (0..3).find(|&index| index != target).unwrap_or(0);
            self.roles = [target, target, spare];
            self.origins[target] = march.origin;
            self.lit[target] = lit;
            self.blend = 1.0;
            self.marched = Some(*settings);
            self.cycle = None;
            self.ready = true;
            self.completed += 1;
            return march.count;
        }
        let Some(cycle) = self.cycle.as_mut() else {
            return 0;
        };
        cycle.next_row += march.count;
        cycle.lit &= lit;
        self.blend = cycle.next_row as f32 / rows as f32;
        if cycle.next_row >= rows {
            self.origins[target] = cycle.origin;
            self.lit[target] = cycle.lit;
            self.cycle = None;
            self.completes = true;
        }
        march.count
    }

    fn release(&mut self) {
        self.dome_size = None;
        self.noise = None;
        self.light = LightMap::default();
    }

    pub(crate) fn prepare_view(
        &mut self,
        (settings, eye, seconds): (&Clouds, glam::Vec3, f64),
        (display, render): (glam::UVec2, glam::UVec2),
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
        uniform.eye[3] = f32::from(u8::from(view_sky));
        self.view_clouds = uniform;
        self.view_job = Some([render.x, render.y]);
        self.view_pixels = render.x * render.y;
    }

    pub(crate) fn marched_layer(&self) -> Option<u32> {
        self.job
            .filter(|&(_, _, count)| count > 0 && self.dome_size.is_some())
            .map(|(target, _, _)| DOME_FIRST + target as u32 * DOME_LAYERS)
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
    view: [u32; 2],
}

pub(crate) struct CloudPass {
    noise_shape: Texture<SkyGpu>,
    noise_detail: Texture<SkyGpu>,
    view_texture: Texture<SkyGpu>,
    light_cells: Buffer<SkyGpu>,
    uniform: Buffer<SkyGpu>,
    view_clouds: Buffer<SkyGpu>,
    composite_args: Buffer<SkyGpu>,
    view_tiles: Buffer<SkyGpu>,
    light_pass: Pass<SkyGpu>,
    march_pass: Pass<SkyGpu>,
    view_pass: Pass<SkyGpu>,
    composite_pass: Pass<SkyGpu>,
    marching: ComputePipeline<SkyGpu>,
    lighting: ComputePipeline<SkyGpu>,
    viewing: ComputePipeline<SkyGpu>,
    compositing: Pipeline<SkyGpu, Own>,
    composite_draws: DrawInput<SkyGpu, Own>,
    noise: Option<u64>,
    stepped: Option<(CloudStep, bool, bool)>,
    dispatched: [bool; 2],
    run: CloudRun,
    tile_bytes: u64,
}

fn composite_pipeline(
    cx: &mut Install<'_, SkyGpu>,
    pass: Pass<SkyGpu>,
    (view_texture, view_tiles): (Texture<SkyGpu>, Buffer<SkyGpu>),
) -> Result<Pipeline<SkyGpu, Own>, Error> {
    use crate::shaders::sky::draw;
    cx.render_pipeline(
        pass,
        &RenderPipeline {
            vertex: draw::VS_CLOUD_TILE,
            fragment: Some(draw::FS_VIEW_CLOUDS_POST),
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
        &[
            (draw::VIEW_CLOUDS, Bind::Texture(view_texture)),
            (draw::CLOUD_TILES, Bind::Buffer(view_tiles)),
        ],
    )
}

fn samplers(cx: &mut Install<'_, SkyGpu>) -> [Sampler<SkyGpu>; 2] {
    let dome = cx.sampler(engine_core::Sampler {
        wrap_u: Wrap::Repeat,
        mipmap_filter: Some(Filter::Nearest),
        ..engine_core::Sampler::LINEAR_CLAMP
    });
    let noise = cx.sampler(engine_core::Sampler {
        wrap_u: Wrap::Repeat,
        wrap_v: Wrap::Repeat,
        wrap_w: Wrap::Repeat,
        ..engine_core::Sampler::LINEAR_CLAMP
    });
    [dome, noise]
}

fn cloud_passes(
    cx: &mut Install<'_, SkyGpu>,
    (environment, dome): (Pass<SkyGpu>, Pass<SkyGpu>),
) -> [Pass<SkyGpu>; 4] {
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
    let view_pass = cx
        .pass("cloud_march")
        .at(AttachPoint::LateOpaque)
        .label("engine/cloud-view")
        .span("cloud_view")
        .id();
    let composite_pass = cx
        .pass("cloud_composite")
        .at(AttachPoint::LateOpaque)
        .after(view_pass)
        .after(dome)
        .label("engine/cloud-composite")
        .span("cloud_composite")
        .id();
    [light_pass, march_pass, view_pass, composite_pass]
}

impl CloudPass {
    pub(crate) fn install(
        cx: &mut Install<'_, SkyGpu>,
        (transmittance, table_sampler): (Texture<SkyGpu>, Sampler<SkyGpu>),
        (environment, dome): (Pass<SkyGpu>, Pass<SkyGpu>),
        domes: Texture<SkyGpu>,
    ) -> Result<Self, Error> {
        use crate::shaders::sky::{clouds, light};
        let noise_shape = cx.texture(
            "cloud_noise_shape",
            wgpu::TextureFormat::R8Unorm,
            Lifetime::Kept,
        );
        let noise_detail = cx.texture(
            "cloud_noise_detail",
            wgpu::TextureFormat::R8Unorm,
            Lifetime::Kept,
        );
        let view_texture = cx.texture(
            "cloud_view",
            wgpu::TextureFormat::Rgba16Float,
            Lifetime::Kept,
        );
        let light_cells = cx.buffer("cloud_light", Lifetime::Kept);
        let uniform = cx.buffer("cloud_uniform", Lifetime::Kept);
        let view_clouds = cx.buffer("cloud_view_march", Lifetime::Kept);
        let composite_args = cx.buffer("cloud_composite_draw", Lifetime::Kept);
        let view_tiles = cx.buffer("cloud_view_tiles", Lifetime::Kept);
        let [dome_sampler, noise_sampler] = samplers(cx);
        cx.frame(light::SKY_CLOUD_SAMPLER, Bind::Sampler(dome_sampler))?;
        let [light_pass, march_pass, view_pass, composite_pass] =
            cloud_passes(cx, (environment, dome));
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
        lighted.push((clouds::LIGHT_CELLS, Bind::Buffer(light_cells)));
        let lighting = cx.compute_pipeline(light_pass, clouds::LIGHT_CLOUDS, &lighted)?;
        let mut marched = Vec::from(noise());
        marched.extend([
            (clouds::NOISE_DETAIL, Bind::Texture(noise_detail)),
            (clouds::DOME, Bind::Texture(domes)),
        ]);
        marched.extend(lit());
        let marching = cx.compute_pipeline(march_pass, clouds::MARCH_CLOUDS, &marched)?;
        let mut viewed = vec![
            (clouds::CLOUDS, Bind::Buffer(view_clouds)),
            (clouds::NOISE_SHAPE, Bind::Texture(noise_shape)),
            (clouds::NOISE_SAMPLER, Bind::Sampler(noise_sampler)),
            (clouds::NOISE_DETAIL, Bind::Texture(noise_detail)),
            (clouds::VIEW_DEPTH, Bind::Target(TargetName::DEPTH)),
            (clouds::VIEW_OUT, Bind::Texture(view_texture)),
            (clouds::COMPOSITE_DRAW, Bind::Buffer(composite_args)),
            (clouds::VIEW_TILES, Bind::Buffer(view_tiles)),
        ];
        viewed.extend(lit());
        let viewing = cx.compute_pipeline(view_pass, clouds::MARCH_VIEW, &viewed)?;
        let compositing = composite_pipeline(cx, composite_pass, (view_texture, view_tiles))?;
        let composite_draws = cx.args_input(composite_pass, composite_args);
        Ok(Self {
            noise_shape,
            noise_detail,
            view_texture,
            light_cells,
            uniform,
            view_clouds,
            composite_args,
            view_tiles,
            light_pass,
            march_pass,
            view_pass,
            composite_pass,
            marching,
            lighting,
            viewing,
            compositing,
            composite_draws,
            noise: None,
            stepped: None,
            dispatched: [false; 2],
            run: CloudRun::default(),
            tile_bytes: 0,
        })
    }

    pub(crate) fn uniforms(&self) -> [Buffer<SkyGpu>; 2] {
        [self.uniform, self.view_clouds]
    }

    fn allocate(&mut self, cx: &mut Prepare<'_, SkyGpu>) {
        let sizes = [
            (self.uniform, size_of::<CloudUniform>() as u64),
            (self.view_clouds, size_of::<CloudUniform>() as u64),
            (self.composite_args, size_of_val(&COMPOSITE_ARGS) as u64),
        ];
        for (buffer, bytes) in sizes {
            let (Allocation::Fresh | Allocation::Kept) = cx.allocate_buffer(buffer, bytes);
        }
    }

    pub(crate) fn prepare(
        &mut self,
        cx: &mut Prepare<'_, SkyGpu>,
        (plan, noise): (&CloudPlan, Option<&StaticTable<CloudNoise>>),
        (frame, fresh): (Option<CloudFrame>, bool),
    ) {
        if let Some((step, light, march)) = self.stepped.take() {
            let ran = (!light || (cx.ran(self.light_pass) && self.dispatched[0]))
                && (!march || (cx.ran(self.march_pass) && self.dispatched[1]));
            cx.deliver::<CloudSteps>(0, step.bytes(ran));
        }
        self.allocate(cx);
        if !plan.active {
            return;
        }
        if let Some(noise) = noise.filter(|noise| Some(noise.id) == plan.noise)
            && self.noise != plan.noise
        {
            upload_noise(cx, (self.noise_shape, self.noise_detail), noise);
            self.noise = plan.noise;
        }
        let lost = cx.allocate_buffer(self.light_cells, CLOUD_LIGHT_BYTES as u64)
            == Allocation::Fresh
            || (fresh && plan.dome_size.is_some());
        if lost {
            cx.deliver::<CloudSteps>(0, plan.step.bytes(false));
        }
        cx.write(self.uniform, 0, &[plan.uniform]);
        if let Some(pixels) = plan.view_job {
            let size = wgpu::Extent3d {
                width: plan.view_size[0],
                height: plan.view_size[1],
                depth_or_array_layers: 1,
            };
            cx.write(self.view_clouds, 0, &[plan.view_clouds]);
            cx.write(self.composite_args, 0, &COMPOSITE_ARGS);
            let tiles = pixels.map(|side| side.div_ceil(CLOUD_GROUP));
            let bytes = u64::from(tiles[0] * tiles[1]).max(1) * 4;
            self.tile_bytes = self.tile_bytes.max(bytes);
            let (Allocation::Fresh | Allocation::Kept) =
                cx.allocate_buffer(self.view_tiles, self.tile_bytes);
            self.run.view = pixels;
            let (Allocation::Fresh | Allocation::Kept) = cx.allocate(self.view_texture, size, 1);
        }
        let Some(frame) = frame else {
            return;
        };
        if let (true, Some((_, _, count)), Some(size)) = (frame.dome, plan.job, plan.dome_size) {
            self.run.light_rows = plan.light.job;
            self.run.width = size[0];
            self.run.count = count;
            self.stepped = Some((plan.step, plan.light.job > 0, count > 0));
            self.dispatched = [false; 2];
            if plan.light.job > 0 {
                cx.run(self.light_pass);
            }
            if count > 0 {
                cx.run(self.march_pass);
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
            || pass == self.view_pass
            || pass == self.composite_pass
    }

    pub(crate) fn encode(&mut self, pass: Pass<SkyGpu>, cx: &mut Encode<'_, SkyGpu>) -> u32 {
        let run = self.run;
        if pass == self.light_pass {
            self.dispatched[0] = cx.dispatch(
                &self.lighting,
                Count::Groups([
                    LIGHT_TEXELS.div_ceil(CLOUD_GROUP),
                    run.light_rows.div_ceil(CLOUD_GROUP),
                    1,
                ]),
            ) != 0;
        } else if pass == self.march_pass {
            self.dispatched[1] = cx.dispatch(
                &self.marching,
                Count::Groups([
                    run.width.div_ceil(CLOUD_GROUP),
                    run.count.div_ceil(CLOUD_GROUP),
                    DOME_LAYERS,
                ]),
            ) != 0;
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
            let mut draw = cx.render(Attach::Declared);
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
                material: None,
            },
        );
    }
}

fn march_uniform(
    settings: &Clouds,
    (origin, at): (glam::Vec3, f64),
    (sun, piece): CloudLight<'_>,
) -> CloudUniform {
    let field = settings.scale * settings.weather as f32;
    let wrap = f64::from(field);
    let drift = |speed: f32| ((f64::from(speed) * at) % wrap) as f32;
    let seed = (f64::from(settings.seed) * 7919.0 % wrap) as f32;
    let growth = settings.sun_step_growth;
    let light_reach = settings.sun_step * (growth.powi(LIGHT_STEPS as i32) - 1.0) / (growth - 1.0);
    let fibre = settings.scale * settings.high_fibre;
    let (turn_sin, turn_cos) = settings.high_turn.to_radians().sin_cos();
    CloudUniform {
        eye: origin.extend(0.0).to_array(),
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
        zenith: piece.zenith,
        horizon: piece.horizon,
        light: [0.0; 4],
        light_rows: [0; 4],
        layers: [0; 4],
        counts: [
            settings.fewest_steps,
            settings.most_steps,
            settings.step_limit,
            0,
        ],
        march: [
            settings.step,
            settings.nearest_step,
            settings.step_share,
            settings.empty_stride,
        ],
        sampling: [
            settings.step_blur,
            settings.sun_step,
            growth,
            light_reach,
        ],
        weather: [
            settings.groups as f32 / field,
            settings.banks as f32 / field,
            settings.warp,
            settings.high_sheets as f32 / field,
        ],
        bank: [
            settings.bank_cover[0],
            settings.bank_cover[1],
            settings.bank_noise[0],
            settings.bank_noise[1],
        ],
        gather: [
            settings.group_cover[0],
            settings.group_cover[1],
            settings.group_noise[0],
            settings.group_noise[1],
        ],
        profile: [
            settings.cover_noise[0],
            settings.cover_noise[1],
            1.0 / settings.base_rise,
            1.0 / (settings.scale * settings.detail),
        ],
        top: [
            settings.top[0],
            settings.top[1],
            1.0 / settings.wisp_height,
            settings.erosion,
        ],
        phase: [
            settings.forward,
            settings.backward,
            settings.backward_share,
            0.0,
        ],
        lighting: [
            settings.powder,
            settings.powder_depth,
            settings.sky_gray,
            settings.far_shade * 0.001,
        ],
        sky_light: [
            settings.sky_light[0],
            settings.sky_light[1],
            1.0 / settings.fade,
            0.0,
        ],
        high: [
            settings.high,
            settings.high_base,
            settings.high_opacity,
            settings.high_soft,
        ],
        high_fibre: [
            1.0 / (fibre * settings.high_stretch),
            1.0 / fibre,
            settings.high_wisps[0],
            settings.high_wisps[1],
        ],
        high_turn: [
            turn_cos,
            turn_sin,
            settings.high_noise[0],
            settings.high_noise[1],
        ],
    }
}

#[derive(Clone, Copy)]
pub(crate) struct CloudFrame {
    dome: bool,
    view: bool,
}
