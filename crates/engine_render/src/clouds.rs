//! Clouds (`clouds.wgsl`): domes of the sky's clouds, each marched from
//! one fixed point through the noise the scene's static tables hold
//! (`engine_core::table_data::clouds` makes it), uploaded once it arrives;
//! until then no cloud shows.
//! Ordinary frames march a back dome a band of rows at a time, over
//! the settings' slices of frames, from where the camera was when it began;
//! once whole it blends in over the newer shown dome while the next one
//! marches, so no displayed frame shows a part-marched dome or marches one
//! whole. Lookups correct each dome's parallax through the cloud
//! layer's middle (`scene.wgsl`'s `cloud_dome_direction`). Exact frames, new
//! domes and changed clouds march a dome whole from the camera, shown alone.
//! The sky and the sky's environment map show the domes; frames without
//! clouds bind a clear one.
//!
//! The view marches the clouds again, a ray through each
//! block of its pixels after the opaque scene and water, up to the farthest
//! of the block's pixels: the clouds' light, the share of the scene passing
//! and how far they lie, which a composite lays over the scene where they
//! lie nearer than what each pixel shows. Blocks seeing the sky march too
//! while the camera is in or above the layer, and the view's sky shows no
//! dome then; under it, the dome shows the sky's clouds as before.

#![expect(
    clippy::expect_used,
    reason = "the sky's group is built each frame before it is read"
)]

use crate::shaders::entry::{clouds, meshes};
use engine_core::collections::HashMap;

use engine_core::Clouds;
use engine_core::StaticTable;
use engine_core::table_data::clouds::CloudNoise;

use crate::bindings::{CLOUD_COMPOSITE, CLOUD_MARCH, CLOUD_SKY, CLOUD_VIEW};
use crate::cached::{Bound, Cached, GroupCache, Versioned};
use crate::gpu_data::LightingUniform;
use crate::gpu_layouts::constants::CLOUD_GROUP;
use crate::render_features::{Encode, FeaturePass, RenderFeature};
use crate::render_graph::{Access, FrameShape, Node, Pass, RenderGraph, Resource};

const DOME_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Domes completed between redraws of the sky's environment map.
const ENVIRONMENT_EVERY: u64 = 16;
use crate::gpu_layouts::clouds::{
    CLOUD_LIGHT_BYTES, CLOUD_SHADOW_BYTES, LIGHT_TEXELS, SHADOW_CELLS,
};
/// The meters the shadows' grid spans around the camera.
const SHADOW_EXTENT: f32 = 8192.0;
/// Frames over which ordinary frames march the whole shadows' grid.
const SHADOW_SLICES: u32 = 4;
/// The meters the light map spans around the eye's place in the clouds' own
/// frame, how far that place may move from its middle before a new map
/// begins, and the map's rows ordinary frames build.
const LIGHT_EXTENT: f32 = 24_000.0;
const LIGHT_RECENTER: f32 = 2_000.0;
const LIGHT_ROWS: u32 = 48;

use crate::gpu_layouts::clouds::CloudUniform;

/// Bytes of the march's uniform.
pub(crate) const CLOUD_UNIFORM_BYTES: usize = size_of::<CloudUniform>();

use crate::gpu_layouts::clouds::DomeUniform;

/// Bytes of the domes' uniform.
pub(crate) const DOME_UNIFORM_BYTES: usize = size_of::<DomeUniform>();

use crate::gpu_layouts::clouds::ViewUniform;

/// Bytes of the view march's uniform.
pub(crate) const CLOUD_VIEW_BYTES: usize = size_of::<ViewUniform>();

/// The view's marched blocks: their clouds' light and
/// passing share, how far those lie, and the blocks along x and y.
struct ViewTargets {
    clouds: (wgpu::Texture, wgpu::TextureView),
    front: (wgpu::Texture, wgpu::TextureView),
    size: [u32; 2],
}

/// The light map (`clouds.wgsl`'s): its cells, made with the first frame
/// that has clouds; what it holds the light for; its rows built so far;
/// and this frame's rows to build.
#[derive(Default)]
struct LightMap {
    cells: Option<wgpu::Buffer>,
    held: Option<LightKey>,
    built: u32,
    job: u32,
}

/// What a light map holds the light for: its first column's corner in the
/// clouds' own frame (meters), the sun's direction, and the layer's base,
/// top, coverage and noise repeat.
#[derive(Clone, Copy, PartialEq)]
struct LightKey {
    corner: [f32; 2],
    sun: [f32; 3],
    layer: [f32; 4],
}

/// A back dome's march: where and when it began, and the next row it
/// marches.
struct Cycle {
    origin: glam::Vec3,
    seconds: f64,
    next_row: u32,
}

/// What the sky binds of the domes: the older and newer shown domes (clear
/// ones without clouds), their sampler and uniform, and a key advancing
/// whenever other textures show.
pub(crate) struct DomeBinding<'a> {
    pub old: &'a wgpu::TextureView,
    pub new: &'a wgpu::TextureView,
    pub sampler: &'a wgpu::Sampler,
    pub uniform: &'a wgpu::Buffer,
    pub key: (u64, [u8; 2], bool),
}

/// The noise's volumes on the GPU, the shapes' and the detail's with its
/// mipmaps, which the march reads as wide as its rays' cones
/// (`clouds.wgsl`'s `detail_lod`), with the table they were uploaded from.
struct NoiseVolumes {
    id: u64,
    _textures: [wgpu::Texture; 2],
    shape: wgpu::TextureView,
    detail: wgpu::TextureView,
    bytes: u64,
}

/// The levels below `texels`, a cube `side` texels a side, a byte a texel
/// (rows along x, then y, then layers along z), down to one texel: each the
/// mean of every 2 × 2 × 2 block of the level above, rounded to the nearest
/// byte, as mipmaps average what a footprint spans.
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
    /// `noise`'s volumes, uploaded.
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, noise: &StaticTable<CloudNoise>) -> Self {
        let volume = |label, side: u32, levels: &[&[u8]]| {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
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
        let (shape_texture, shape) =
            volume("engine/cloud-noise-shape", SHAPE_SIZE, &[table.shape.as_slice()]);
        let below = levels_below(DETAIL_SIZE, &table.detail);
        let detail_levels: Vec<&[u8]> = std::iter::once(table.detail.as_slice())
            .chain(below.iter().map(Vec::as_slice))
            .collect();
        let (detail_texture, detail) =
            volume("engine/cloud-noise-detail", DETAIL_SIZE, &detail_levels);
        let bytes = table.shape.len() + detail_levels.iter().map(|level| level.len()).sum::<usize>();
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
    sky_layout: crate::pipelines::GroupLayout,
    // The march's pipelines, made with the first frame that has clouds.
    pipelines: Option<[wgpu::ComputePipeline; 3]>,
    light: LightMap,
    // The noise's volumes, once its table arrived.
    noise: Option<NoiseVolumes>,
    // The shadows' grid, as the camera group binds it.
    shadows: wgpu::Buffer,
    // Three domes of the settings' size: two shown, blended, and one
    // marching.
    domes: Option<[(wgpu::Texture, wgpu::TextureView); 3]>,
    // Which dome is the older shown one, the newer and the marching one,
    // each dome's origin, and the newer's share of the blend.
    roles: [usize; 3],
    origins: [glam::Vec3; 3],
    blend: f32,
    // A clear dome: no cloud, the whole sky passing; and its view.
    clear: (wgpu::Texture, wgpu::TextureView),
    sampler: wgpu::Sampler,
    noise_sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    dome_uniform: wgpu::Buffer,
    // The shadows' grid as the march writes it.
    shadow_cells: wgpu::Buffer,
    // The march's groups, one writing each dome.
    march_groups: Cached<u64, [wgpu::BindGroup; 3]>,
    sky_group: Cached<(u64, [u8; 2], bool), wgpu::BindGroup>,
    // Advances whenever the domes are other textures.
    generation: u64,
    // The settings the domes march, the back dome's march, and whether
    // this frame's completes it.
    marched: Option<Clouds>,
    cycle: Option<Cycle>,
    completes: bool,
    // Domes completed, and how many were recorded: the sky's environment
    // map, drawn before a frame's march, follows those, and shows the domes
    // of the generation whose march a frame recorded.
    completed: u64,
    recorded: u64,
    recorded_generation: Option<u64>,
    // Whether a whole dome shows: until then the sky shows a clear one.
    ready: bool,
    // This frame's march: the dome written, its first row and its rows.
    job: Option<(usize, u32, u32)>,
    // This frame's rows of the shadows' grid between those marched.
    shadow_stride: u32,
    frame: u64,
    active: bool,
    /// Rows of the dome this frame marched.
    pub(crate) rows: u32,
    // Clouds in the view: the march's and the composite's
    // layouts, the march's uniform from the camera and the view's, the
    // marched blocks, the groups binding them (by the depth buffer and the
    // blocks), the march's pipeline and the composite's by target, and the
    // blocks' sampler; this frame's blocks to march, and whether they march
    // the sky.
    view_layout: crate::pipelines::GroupLayout,
    composite_layout: crate::pipelines::GroupLayout,
    view_clouds: wgpu::Buffer,
    view_uniform: wgpu::Buffer,
    // The composite's indirect draw, which the march gives a triangle once
    // any block meets a cloud.
    composite_args: wgpu::Buffer,
    view_targets: Option<Versioned<ViewTargets>>,
    view_group: GroupCache<13>,
    composite_group: GroupCache<4>,
    view_pipeline: Option<wgpu::ComputePipeline>,
    composite_pipeline_layout: Option<wgpu::PipelineLayout>,
    composite_pipelines: HashMap<(wgpu::TextureFormat, bool), wgpu::RenderPipeline>,
    view_sampler: wgpu::Sampler,
    view_job: Option<[u32; 2]>,
    view_sky: bool,
    /// Blocks of the view whose clouds this frame marched.
    pub(crate) view_blocks: u32,
}

impl CloudPass {
    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
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
        // The detail's levels blend as a ray's cone widens, so its wisps
        // average away with no step between levels.
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
        let view_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("engine/cloud-view-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            march_layout: crate::pipelines::layout(device, &CLOUD_MARCH),
            sky_layout: crate::pipelines::layout(device, &CLOUD_SKY),
            pipelines: None,
            light: LightMap::default(),
            noise: None,
            shadows: shadow_uniform(device, queue),
            domes: None,
            roles: [0, 1, 2],
            origins: [glam::Vec3::ZERO; 3],
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
            sky_group: Cached::default(),
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
            view_sampler,
            view_job: None,
            view_sky: false,
            view_blocks: 0,
        }
    }

    /// Plans this frame's rows of the light map for the march `uniform`: a
    /// new map around the eye's place in the clouds' own frame when the sun
    /// or the layer changed or that place moved too far from its middle;
    /// all its rows still due on exact frames (`whole`), otherwise a slice.
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

    /// The light map as the marches read it: its first column's corner,
    /// meters a column, and 1 once it holds every row.
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

    /// This frame's slice of the shadows' grid: all of it on exact frames,
    /// a quarter of its rows otherwise.
    fn shadow_rows(&mut self, whole: bool) -> [u32; 4] {
        let stride = if whole { 1 } else { SHADOW_SLICES };
        self.shadow_stride = stride;
        [(self.frame % u64::from(stride)) as u32, stride, 0, 0]
    }

    /// The sky pipelines' second group layout.
    pub(crate) fn sky_layout(&self) -> &crate::pipelines::GroupLayout {
        &self.sky_layout
    }

    /// The domes the sky shows (clear ones without clouds), their sampler
    /// and uniform, and a key advancing whenever other textures show.
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

    /// What the sky's environment map holds of the clouds: it redraws when
    /// this changes, every few completed domes.
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

    /// The domes the sky's environment map shows: [`Self::dome`]'s once a
    /// frame recorded a march of the domes shown, clear ones until then.
    /// The map draws before a frame's march, so a dome a frame first
    /// marches shows in it from the next frame on.
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

    /// Whether the environment map shows the domes: they show, and a frame
    /// recorded a march of their generation.
    fn environment_shows(&self) -> bool {
        self.active && self.ready && self.recorded_generation == Some(self.generation)
    }

    /// The sky pipelines' second group, as [`Self::prepare_sky_group`] last
    /// built it.
    pub(crate) fn sky_group(&self) -> &wgpu::BindGroup {
        self.sky_group.get().expect("built each frame")
    }

    /// Where the physically based sky's light goes in the
    /// marches' uniforms, the dome's and the view's: the
    /// sun's, the zenith's and the horizon's colors.
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

    /// Builds the sky pipelines' second group unless built: the domes (or
    /// clear ones), their sampler and uniform, and the physically based
    /// sky's `air` (its uniform, sky-view and transmittance tables and
    /// sampler).
    pub(crate) fn prepare_sky_group(
        &mut self,
        device: &wgpu::Device,
        air: crate::sky::AirBindings<'_>,
    ) {
        let key = self.dome().key;
        if !self.sky_group.stale(&key) {
            return;
        }
        let dome = self.dome();
        let (old, new, sampler, uniform) = (dome.old, dome.new, dome.sampler, dome.uniform);
        let group = {
            crate::pipelines::group(
                device,
                "engine/cloud-sky",
                &CLOUD_SKY,
                &self.sky_layout,
                &[
                    ("sky_clouds", crate::pipelines::view(new)),
                    ("sky_cloud_sampler", crate::pipelines::sampler(sampler)),
                    crate::buffers::entry("sky_air", air.0),
                    ("sky_view_table", crate::pipelines::view(air.1)),
                    ("sky_transmittance_table", crate::pipelines::view(air.2)),
                    ("sky_table_sampler", crate::pipelines::sampler(air.3)),
                    ("sky_clouds_old", crate::pipelines::view(old)),
                    crate::buffers::entry("sky_domes", uniform),
                ],
            )
        };
        self.sky_group.set(key, group);
    }

    /// Plans this frame's march of `settings`' clouds, shaped by `noise`
    /// (the latest the scene's tables hold, uploaded unless it was), for an
    /// eye at `eye` at simulation time `seconds` in `lighting`'s sun and sky
    /// (the physically based one's `air` lighting each cloud where it
    /// stands): a whole dome from the eye, shown alone, when `whole` (exact
    /// frames), when the domes are new or the clouds changed; otherwise a
    /// band of the back dome's rows from where it began, which shows once
    /// whole. No cloud shows before the noise first arrived.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        (settings, noise, eye, seconds): (
            &Clouds,
            Option<&StaticTable<CloudNoise>>,
            glam::Vec3,
            f64,
        ),
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
            // Groups binding the volumes it replaces bind them anew.
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
        // Last frame's band completed the back dome: it blends in over the
        // newer shown one, which becomes the older.
        if self.completes {
            self.completes = false;
            self.rotate();
        }
        let rows = size[1];
        let changed = self.marched.is_none_or(|marched| marched != settings);
        self.frame += 1;
        let target = self.roles[2];
        let (origin, at, first, count) = if whole || fresh || changed || !self.ready {
            // A whole dome from the eye, shown alone this frame.
            let spare = (0..3).find(|&index| index != target).unwrap_or(0);
            self.roles = [target, target, spare];
            self.origins[target] = eye;
            self.blend = 1.0;
            self.marched = Some(settings);
            self.cycle = None;
            self.ready = true;
            self.completed += 1;
            (eye, seconds, 0, rows)
        } else {
            let cycle = self.cycle.get_or_insert(Cycle {
                origin: eye,
                seconds,
                next_row: 0,
            });
            let frames = settings.slices.max(1);
            let first = cycle.next_row;
            let count = rows.div_ceil(frames).min(rows - first);
            cycle.next_row += count;
            let (origin, at, done) = (cycle.origin, cycle.seconds, cycle.next_row >= rows);
            // The newer shown dome blends in as the back one marches.
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
        let mut uniform = march_uniform(&settings, (origin, at), lighting);
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

    /// The pipelines and groups this frame's march of the dome binds.
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

    /// Plans this frame's clouds in the view of `settings`
    /// from an eye at `eye` at simulation time `seconds`, through
    /// `view_projection` (unjittered) onto the frame's `region` in
    /// `lighting`'s sun and sky: blocks made for its display, marched over
    /// its render size; call after [`Self::prepare`]. Nothing marches
    /// without clouds or with the view's march off.
    #[expect(clippy::too_many_arguments, reason = "the view's inputs, typed")]
    pub(crate) fn prepare_view(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        (settings, eye, seconds): (&Clouds, glam::Vec3, f64),
        view_projection: glam::Mat4,
        region: crate::targets::FrameRegion,
        (lighting, air): (&LightingUniform, crate::sky::AirBindings<'_>),
        depth: &Versioned<wgpu::TextureView>,
    ) {
        let settings = settings.sanitized();
        self.view_job = None;
        self.view_blocks = 0;
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
        let side = settings.view_block;
        // Blocks over the display's pixels, those over the frame's render
        // size marched.
        let room = [display.width.div_ceil(side), display.height.div_ceil(side)];
        let blocks = [render.width.div_ceil(side), render.height.div_ceil(side)];
        if self
            .view_targets
            .as_ref()
            .is_none_or(|targets| targets.size != room)
        {
            self.view_targets = Some(Versioned::new(ViewTargets {
                clouds: view_texture(device, room, wgpu::TextureFormat::Rgba16Float),
                front: view_texture(device, room, wgpu::TextureFormat::R32Float),
                size: room,
            }));
        }
        let mut uniform = march_uniform(&settings, (eye, seconds), lighting);
        uniform.light = self.light_uniform();
        queue.write_buffer(&self.view_clouds, 0, bytemuck::bytes_of(&uniform));
        let view = ViewUniform {
            inverse_view_projection: view_projection.inverse().to_cols_array_2d(),
            blocks: [blocks[0], blocks[1], side, u32::from(view_sky)],
            pixels: [render.width, render.height, 0, 0],
        };
        queue.write_buffer(&self.view_uniform, 0, bytemuck::bytes_of(&view));
        // No vertices until the march meets a cloud; one instance.
        queue.write_buffer(
            &self.composite_args,
            0,
            bytemuck::cast_slice(&[0_u32, 1, 0, 0]),
        );
        self.view_job = Some(blocks);
        self.view_blocks = blocks[0] * blocks[1];
        self.prepare_view_draws(device, depth, air);
    }

    /// Whether this frame marches clouds in the view.
    pub(crate) fn marches_view(&self) -> bool {
        self.view_job.is_some()
    }

    /// The view march's pipeline and group, up to what the depth buffer
    /// `depth` shows.
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
                    "front_out",
                    Bound::of(targets, crate::pipelines::view(&targets.front.1)),
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

    /// Records this frame's march of the view's clouds, timed by `profiler`
    /// (`cloud_view`); after the dome's march made the noise.
    fn encode_view(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &crate::profiling::GpuProfiler,
    ) {
        let (Some(blocks), Some(pipeline), Some(group)) =
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
            blocks[0].div_ceil(CLOUD_GROUP),
            blocks[1].div_ceil(CLOUD_GROUP),
            1,
        );
    }

    /// Builds the composite's pipeline into `format` (`canvas`: the
    /// canvas, encoded; otherwise the screen effects' scene target) from
    /// `shader` with the camera group's `camera_layout`, and its group over
    /// the depth buffer `depth`.
    pub(crate) fn prepare_composite(
        &mut self,
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        camera_layout: &crate::pipelines::GroupLayout,
        (format, canvas): (wgpu::TextureFormat, bool),
        depth: &Versioned<wgpu::TextureView>,
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
            [
                (
                    "view_clouds",
                    Bound::of(targets, crate::pipelines::view(&targets.clouds.1)),
                ),
                (
                    "view_cloud_front",
                    Bound::of(targets, crate::pipelines::view(&targets.front.1)),
                ),
                ("view_cloud_depth", depth.bound()),
                (
                    "view_cloud_sampler",
                    Bound::fixed(crate::pipelines::sampler(&self.view_sampler)),
                ),
            ],
        );
    }

    /// What this frame records: the dome's march, and the view's march and
    /// composite into `target` (the format of the scene's color, and whether
    /// that is the canvas); `None` when neither marches.
    pub(crate) fn frame(&self, target: (wgpu::TextureFormat, bool)) -> Option<CloudFrame> {
        let dome = self.job.is_some() && self.domes.is_some();
        let view = self.view_job.map(|_| target);
        (dome || view.is_some()).then_some(CloudFrame { dome, view })
    }

    /// Lays the view's clouds over `pass`'s target of `format` (`canvas`:
    /// the canvas), the camera's `camera` group bound; after
    /// [`Self::prepare_composite`].
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
        // The march's count: no triangle while no block met a cloud.
        pass.draw_indirect(&self.composite_args, 0);
    }

    /// The back dome, complete, shows as the newer one; the newer shown one
    /// becomes the older; the third marches next.
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

    /// Writes the domes' uniform: the shown domes' origins and blend, the
    /// eye with the cloud layer's middle, and whether the view's march
    /// shows the sky's clouds.
    fn write_domes(&self, queue: &wgpu::Queue, eye: glam::Vec3, settings: &Clouds) {
        let shows = if self.active && self.ready { 1.0 } else { 0.0 };
        let [old, new, _] = self.roles;
        let uniform = DomeUniform {
            older: self.origins[old].extend(shows).to_array(),
            newer: self.origins[new].extend(self.blend).to_array(),
            eye: eye
                .extend(settings.base + 0.5 * settings.thickness)
                .to_array(),
            view: [if self.view_sky { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0],
        };
        queue.write_buffer(&self.dome_uniform, 0, bytemuck::bytes_of(&uniform));
    }

    /// Records this frame's march and the shadows' grid, copied into the
    /// uniform the camera group binds, timed by `profiler` (`clouds`).
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
            // The light map's slice before the march that reads it.
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
            pass.dispatch_workgroups(width.div_ceil(CLOUD_GROUP), count.div_ceil(CLOUD_GROUP), 1);
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

    /// The shadows' grid, as the camera group binds it.
    pub(crate) fn shadows(&self) -> &wgpu::Buffer {
        &self.shadows
    }

    /// Bytes of the noise's volumes, the domes, the view's blocks and the
    /// light map.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        let noise = self.noise.as_ref().map_or(0, |noise| noise.bytes);
        let domes = self.domes.as_ref().map_or(0, |domes| {
            domes
                .iter()
                .map(|(texture, _)| u64::from(texture.width()) * u64::from(texture.height()) * 8)
                .sum()
        });
        let view = self.view_targets.as_ref().map_or(0, |targets| {
            u64::from(targets.size[0]) * u64::from(targets.size[1]) * 12
        });
        let light = self.light.cells.as_ref().map_or(0, wgpu::Buffer::size);
        noise + domes + view + light
    }
}

/// The march's uniform for `settings`' clouds marched from `origin` at
/// simulation time `at` in `lighting`'s sun and sky; no dome rows or
/// shadows' grid.
fn march_uniform(
    settings: &Clouds,
    (origin, at): (glam::Vec3, f64),
    lighting: &LightingUniform,
) -> CloudUniform {
    // The wind's drift by then, wrapped where every repeat of the noise
    // lines up again.
    let wrap = f64::from(settings.scale) * 43.0;
    let drift = |speed: f32| ((f64::from(speed) * at) % wrap) as f32;
    let seed = (f64::from(settings.seed) * 7919.0 % wrap) as f32;
    CloudUniform {
        // The physically based sky lights each cloud where it stands.
        eye: origin.extend(lighting.grade[2]).to_array(),
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

/// A texture of the view's blocks, `size` of them, in
/// `format`, which the view's march writes and its composite reads.
fn view_texture(
    device: &wgpu::Device,
    size: [u32; 2],
    format: wgpu::TextureFormat,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("engine/cloud-view-blocks"),
        size: wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// The view's march.
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

/// The composite of the view's clouds into `format`, the
/// canvas's when `canvas`: their light over the scene, which keeps the
/// share they pass, and the target's alpha as it was.
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

/// A dome `size` texels wide and tall that the march writes and the sky
/// reads.
fn dome_texture(device: &wgpu::Device, size: [u32; 2]) -> (wgpu::Texture, wgpu::TextureView) {
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
    (texture, view)
}

/// The shadows' grid around `eye`: its first cell (world
/// cell indices along x and z, as floats) and meters a cell. Cells snap to
/// whole cells and keep their slots as the grid moves (cell index modulo
/// the cells along a side), so moving the grid leaves every cell it still
/// covers where it was.
pub(crate) fn shadow_grid(eye: glam::Vec3) -> Option<([f32; 2], f32)> {
    let cell = SHADOW_EXTENT / SHADOW_CELLS as f32;
    if !eye.is_finite() {
        return None;
    }
    let half = (SHADOW_CELLS / 2) as f32;
    let first = |value: f32| (value / cell).floor() - half;
    Some(([first(eye.x), first(eye.z)], cell))
}

/// The lighting uniform's clouds' shadows for `clouds` seen from `eye`:
/// the grid's first cell, cells per meter and strength, and the clouds'
/// base; zero without clouds or shadows.
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

/// The camera group's buffer of clouds' shadows: every cell passing all
/// the light until the clouds' pass copies its grid in.
fn shadow_uniform(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Buffer {
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

/// What a frame records of the clouds: the dome's march, and the view's
/// march and composite into the scene's color of the given format (and
/// whether that is the canvas).
pub(crate) struct CloudFrame {
    dome: bool,
    view: Option<(wgpu::TextureFormat, bool)>,
}

impl RenderFeature for CloudPass {
    type Frame = CloudFrame;
    type Stats = engine_core::CloudStats;

    /// The dome's slice, which the sky and its environment map show; then
    /// the view's march up to what the opaque scene and water show, and its
    /// composite over the scene before transparent surfaces draw.
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
                .read(Resource::Depth)
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

    /// A march recorded completes its dome for the sky's environment map.
    fn after_submit(&mut self, frame: Option<&CloudFrame>) {
        if frame.is_some_and(|frame| frame.dome) {
            self.recorded = self.completed;
            self.recorded_generation = Some(self.generation);
        }
    }

    fn capacity_bytes(&self) -> u64 {
        CloudPass::capacity_bytes(self)
    }

    /// The dome's rows marched and the view's blocks.
    fn stats(
        this: Option<&Self>,
        _frame: Option<&CloudFrame>,
        _snapshot: &engine_core::RenderSnapshot,
    ) -> engine_core::CloudStats {
        engine_core::CloudStats {
            rows: this.map_or(0, |clouds| clouds.rows),
            view_blocks: this.map_or(0, |clouds| clouds.view_blocks),
        }
    }
}
