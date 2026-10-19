//! Water: terrains' water as their tiles hold it, and the sea. Each frame
//! the camera's terrain patches holding water draw on terrain's own grid,
//! with its morph and projection, standing at the surfaces the tiles' water
//! blocks hold (`crate::terrain::TerrainPass::water_draws`): the depth each
//! pixel shades by is the tiles' water less their ground on the triangles
//! the ground is drawn on, and the shore falls where the ground rises
//! through the water. Water draws after the opaque scene and its decals and
//! blends over the ground by how little light comes back through it,
//! writing the depth and surface targets, so later passes and screen-space
//! reflections see its surface. Before it draws, the opaque depth copies
//! into a texture of its own: what stands in the water behind a pixel
//! shortens the light's path to it. In frames drawing the effects' targets
//! the opaque color copies too, and water covers it whole with what shows
//! through it, bent by its waves and dimmed channel by channel over the
//! light's path; other frames blend water over the ground.
//!
//! Water's waves (`crate::water_waves`) evolve before it draws, the same
//! tiles of them for terrains' water and the sea.
//!
//! Where terrains' rivers leave their beds, their jets draw last, as
//! sheets (`crate::water_falls`) over the water they land in, writing their
//! depth where they cover the scene.
//!
//! Water is a feature of the renderer (`crate::render_features`).
//!
//! The sea draws first in the same pass, from a grid around
//! the eye: rings of [`OCEAN_SEGMENTS`] cells, a cell apart near the eye
//! and then each wider by a cell's share of its circumference, so cells
//! stay square, out to the sea's reach. Its sectors cull on their own;
//! vertices move by the waves longer than a few of their cells, and pixels
//! shade every wave longer than a few pixels. It draws through the
//! terrain's projection, jittered alike, so its shores meet the ground.

use engine_core::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use bytemuck::Zeroable;
use engine_core::glam::{Mat4, Vec3};
use engine_core::math::max;
use engine_core::terrain_data::Fall;
use engine_core::{
    Environment, MAX_OCEAN_WAVES, Ocean, OceanStats, RenderSnapshot, RenderTerrain, Water,
    WaterStats, Wind,
};

use crate::bindings::{WATER, WATER_BEHIND};
use crate::gpu_layouts::terrain::{TerrainUniform, ViewUniform};
use crate::profiling::Span;
use crate::render_features::{Encode, FeaturePass, RenderFeature};
use crate::render_graph::{Access, FrameShape, Node, Pass, RenderGraph, Resource, Targets};
use crate::targets::{FrameSize, ScreenTarget};
use crate::terrain::{TerrainWaterDraws, WATER_GRID_INDICES, water_patch_layout};
use crate::water_falls::{fall_layout, sheets};
use crate::water_waves::{WaterWaves, WaveShape};

/// Cells around the sea's grid, the sectors it splits into for culling,
/// and the rings a cell apart at its middle: about the cells around over
/// 2 pi, where a cell's share of the circumference reaches a cell.
const OCEAN_SEGMENTS: u32 = 192;
const OCEAN_SECTORS: u32 = 16;
const OCEAN_FIRST_RINGS: u32 = 31;
/// The sea grid's finest cell (meters), and how much wider it grows per
/// meter the eye stands over the sea.
const OCEAN_CELL: f32 = 0.3;
const OCEAN_CELL_PER_HEIGHT: f32 = 0.004;

use crate::gpu_layouts::water::{FallVertex, WaterUniform};

/// Bytes of the water group's uniform.
pub(crate) const WATER_UNIFORM_BYTES: usize = size_of::<WaterUniform>();

/// One-texel stand-ins for what the water group binds that the sea reads
/// none of: the terrain's height array and uniforms and tiles' water
/// blocks; and for the scene's color, which frames blending water read
/// none of.
struct Placeholders {
    surfaces: wgpu::Texture,
    info: wgpu::Texture,
    views: wgpu::Buffer,
    terrains: wgpu::Buffer,
    scene: wgpu::TextureView,
}

/// The sea on the GPU: its uniform and group, and this frame's rings and
/// the runs of sectors the camera sees.
struct OceanDraw {
    uniform: wgpu::Buffer,
    group: wgpu::BindGroup,
    rings: u32,
    runs: Vec<Range<u32>>,
}

/// Tiles' water this frame: its instances, the grid's indices and each
/// terrain's uniform offset and range of instances.
struct TileDraws {
    instances: wgpu::Buffer,
    indices: wgpu::Buffer,
    terrains: Vec<(u32, Range<u32>)>,
}

/// Terrains' falls' sheets: their vertices and triangles' indices, and the
/// falls they were built from (each terrain's origin and falls).
struct FallDraws {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    count: u32,
    built: Vec<(Vec3, Arc<[Fall]>)>,
}

/// The copies of the opaque scene water reads, the depth and, in frames
/// drawing the effects' targets, the color; and their group.
struct BehindCopies {
    depth: wgpu::Texture,
    scene: Option<wgpu::Texture>,
    group: wgpu::BindGroup,
}

/// What water draws this frame: tiles' patches holding water and their
/// blocks resident and dropped, and the sea's sectors and their triangles.
pub(crate) struct WaterFrame {
    patches: u32,
    blocks: [u32; 2],
    sectors: u32,
    triangles: u32,
}

pub(crate) struct WaterPass {
    pub(crate) layout: crate::pipelines::GroupLayout,
    behind_layout: crate::pipelines::GroupLayout,
    behind: ScreenTarget<BehindCopies>,
    placeholders: Placeholders,
    // Created with the first frame drawing water.
    waves: Option<WaterWaves>,
    // Tiles' water's uniform, and the group binding it with the terrain
    // pass's arrays and uniforms, as of their generations.
    uniform: wgpu::Buffer,
    group: Option<((u64, u64, u64), wgpu::BindGroup)>,
    tiles: Option<TileDraws>,
    pipeline_layout: Option<wgpu::PipelineLayout>,
    pipelines: HashMap<Targets, wgpu::RenderPipeline>,
    ocean_pipelines: HashMap<Targets, wgpu::RenderPipeline>,
    ocean: Option<OceanDraw>,
    // Terrains' falls' sheets, as of the falls they were built from, and
    // what builds them.
    falls: Option<FallDraws>,
    fall_pipelines: HashMap<Targets, wgpu::RenderPipeline>,
    fall_vertices: Vec<FallVertex>,
    fall_indices: Vec<u32>,
}

impl WaterPass {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        Self {
            layout: crate::pipelines::layout(device, &WATER),
            behind_layout: crate::pipelines::layout(device, &WATER_BEHIND),
            behind: ScreenTarget::default(),
            placeholders: placeholders(device),
            waves: None,
            uniform: crate::buffers::uniform(device, "engine/water", WATER_UNIFORM_BYTES as u64),
            group: None,
            tiles: None,
            pipeline_layout: None,
            pipelines: HashMap::default(),
            ocean_pipelines: HashMap::default(),
            ocean: None,
            falls: None,
            fall_pipelines: HashMap::default(),
            fall_vertices: Vec::new(),
            fall_indices: Vec::new(),
        }
    }

    /// Builds the sheets of `terrains`' falls unless they were built from
    /// the same falls; drops them where none fall.
    fn prepare_falls(&mut self, device: &wgpu::Device, terrains: &[RenderTerrain]) {
        let falling = || terrains.iter().filter(|terrain| !terrain.falls.is_empty());
        let same = self.falls.as_ref().is_some_and(|built| {
            built.built.len() == falling().count()
                && built
                    .built
                    .iter()
                    .zip(falling())
                    .all(|(a, b)| a.0 == b.origin && Arc::ptr_eq(&a.1, &b.falls))
        });
        if same || (self.falls.is_none() && falling().next().is_none()) {
            return;
        }
        sheets(
            falling().map(|terrain| (terrain.origin, &*terrain.falls)),
            &mut self.fall_vertices,
            &mut self.fall_indices,
        );
        if self.fall_indices.is_empty() {
            self.falls = None;
            return;
        }
        let buffer = |label, contents: &[u8], usage| {
            wgpu::util::DeviceExt::create_buffer_init(
                device,
                &wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents,
                    usage,
                },
            )
        };
        self.falls = Some(FallDraws {
            vertices: buffer(
                "engine/water-falls",
                bytemuck::cast_slice(&self.fall_vertices),
                wgpu::BufferUsages::VERTEX,
            ),
            indices: buffer(
                "engine/water-fall-indices",
                bytemuck::cast_slice(&self.fall_indices),
                wgpu::BufferUsages::INDEX,
            ),
            count: self.fall_indices.len() as u32,
            built: falling()
                .map(|terrain| (terrain.origin, terrain.falls.clone()))
                .collect(),
        });
    }

    /// The sea's sectors drawn this frame and their triangles.
    fn ocean_drawn(&self) -> (u32, u32) {
        self.ocean.as_ref().map_or((0, 0), |ocean| {
            let sectors: u32 = ocean.runs.iter().map(|run| run.len() as u32).sum();
            let per_sector = OCEAN_SEGMENTS / OCEAN_SECTORS;
            (sectors, sectors * ocean.rings * per_sector * 2)
        })
    }

    /// Copies the opaque `depth` and, when water refracts it, the
    /// `scene`'s color into the copies water reads, before it draws.
    fn copy_behind(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        depth: &wgpu::Texture,
        scene: Option<&wgpu::Texture>,
    ) {
        let Some(copies) = self.behind.get() else {
            return;
        };
        let pairs = [
            Some((depth, &copies.depth)),
            scene.zip(copies.scene.as_ref()),
        ];
        for (from, to) in pairs.into_iter().flatten() {
            if from.size() == to.size() {
                encoder.copy_texture_to_texture(
                    from.as_image_copy(),
                    to.as_image_copy(),
                    from.size(),
                );
            }
        }
    }

    /// Prepares this frame's water: tiles' patches holding water as the
    /// terrain pass lays them out (`tiles`) and the falls of `terrains`,
    /// shaded as `settings` say, and
    /// the sea `ocean`, culled through `cull` and drawn through `view`
    /// (both eye-relative) over the opaque `depth` and, when `refracts`,
    /// the color it copies; seen from `eye` in the wind `wind`, its waves
    /// at the simulation time `seconds` the frame presents; `None` when
    /// none of it draws.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        (tiles, terrains): (Option<TerrainWaterDraws<'_>>, &[RenderTerrain]),
        (settings, ocean): (Water, Ocean),
        (eye, wind, seconds): (Vec3, Wind, f64),
        (cull, view, depth, refracts): (Mat4, Mat4, &wgpu::Texture, bool),
    ) -> Option<WaterFrame> {
        self.tiles = None;
        let blocks = tiles.as_ref().map_or([0; 2], |tiles| tiles.blocks);
        let tiles = tiles.filter(|tiles| settings.enabled && !tiles.terrains.is_empty());
        self.prepare_falls(device, if settings.enabled { terrains } else { &[] });
        // Falls draw with tiles' water's group, once any tiles' water has.
        let falling = self.falls.is_some() && (tiles.is_some() || self.group.is_some());
        if tiles.is_none() && !ocean.enabled && !falling {
            self.ocean = None;
            return None;
        }
        // Terrains' water and the sea ripple with the same waves.
        let waves = self.waves.get_or_insert_with(|| WaterWaves::new(device));
        let shape = WaveShape {
            wind: wind.velocity,
            size: settings.ripple_size,
        };
        waves.prepare(device, queue, shape, seconds);
        let repeats = waves.repeats();
        if let Some(tiles) = tiles {
            if let Some(waves) = &self.waves
                && self
                    .group
                    .as_ref()
                    .is_none_or(|(bound, _)| *bound != tiles.generation)
            {
                let group = self.tile_group(device, &tiles, waves);
                self.group = Some((tiles.generation, group));
            }
            self.tiles = Some(TileDraws {
                instances: tiles.instances.clone(),
                indices: tiles.indices.clone(),
                terrains: tiles.terrains.to_vec(),
            });
            // Terrains' water gives way to the sea where it lies under it.
            let sea = if ocean.enabled {
                [ocean.level - eye.y, 1.0, 0.0, 0.0]
            } else {
                [0.0; 4]
            };
            let [red, green, blue] = settings.color;
            let uniform = WaterUniform {
                color: [red, green, blue, settings.clarity],
                ripples: ripples(settings, wind),
                foam: [settings.foam, settings.foam_width, 0.0, 0.0],
                absorption: absorption(settings.color, settings.clarity),
                waves: repeats,
                ocean: sea,
                ..WaterUniform::zeroed()
            };
            queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
        }
        let sea = self.prepare_ocean(
            device,
            queue,
            (cull, view),
            (settings, ocean),
            (eye, wind, seconds),
            repeats,
        );
        let patches = self.tiles.as_ref().map_or(0, |tiles| {
            tiles
                .terrains
                .iter()
                .map(|(_, range)| range.len() as u32)
                .sum()
        });
        if patches == 0 && !sea && !falling {
            return None;
        }
        // The copies, at the depth buffer's size: the frame's.
        let size = FrameSize {
            width: depth.width(),
            height: depth.height(),
        };
        let refracted = self
            .behind
            .get()
            .is_some_and(|copies| copies.scene.is_some());
        if refracted != refracts {
            self.behind.release();
        }
        if self.behind.resizes(size) {
            let copies = self.behind_copies(device, size, depth.format(), refracts);
            self.behind.set(size, copies);
        }
        let (sectors, triangles) = self.ocean_drawn();
        Some(WaterFrame {
            patches,
            blocks,
            sectors,
            triangles,
        })
    }

    /// The copies of the opaque scene at `size`, the depth's of `format`
    /// and, when water `refracts`, the color's; and their group.
    fn behind_copies(
        &self,
        device: &wgpu::Device,
        size: FrameSize,
        format: wgpu::TextureFormat,
        refracts: bool,
    ) -> BehindCopies {
        let copy = |label, format| {
            crate::targets::texture_2d(
                device,
                label,
                size.pixels(),
                1,
                1,
                format,
                wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            )
        };
        let depth = copy("engine/water-depth", format);
        let scene = refracts.then(|| copy("engine/water-scene", crate::post::SCENE_FORMAT));
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
        let scene_view = scene
            .as_ref()
            .map(|scene| scene.create_view(&wgpu::TextureViewDescriptor::default()));
        let group = crate::pipelines::group(
            device,
            "engine/water-behind",
            &WATER_BEHIND,
            &self.behind_layout,
            &[
                ("water_depth", crate::pipelines::view(&depth_view)),
                (
                    "water_scene",
                    crate::pipelines::view(scene_view.as_ref().unwrap_or(&self.placeholders.scene)),
                ),
            ],
        );
        BehindCopies {
            depth,
            scene,
            group,
        }
    }

    /// The group tiles' water draws with: the terrain pass's height array,
    /// views' and terrains' uniforms and water blocks, water's uniform and
    /// `waves`.
    fn tile_group(
        &self,
        device: &wgpu::Device,
        tiles: &TerrainWaterDraws<'_>,
        waves: &WaterWaves,
    ) -> wgpu::BindGroup {
        let (moments, sampler) = waves.sampled();
        crate::pipelines::group(
            device,
            "engine/water-group",
            &WATER,
            &self.layout,
            &[
                ("terrain_heights", crate::pipelines::view(tiles.heights)),
                ("terrain_view", sized(tiles.views, size_of::<ViewUniform>())),
                (
                    "terrain",
                    sized(tiles.uniforms, size_of::<TerrainUniform>()),
                ),
                ("water_surface", crate::pipelines::view(tiles.surfaces)),
                ("water_info", crate::pipelines::view(tiles.info)),
                ("water", self.uniform.as_entire_binding()),
                ("water_waves", crate::pipelines::view(moments)),
                ("water_wave_sampler", crate::pipelines::sampler(sampler)),
            ],
        )
    }

    /// Prepares the sea culled through `cull` and drawn through `view`
    /// (both eye-relative) from `eye`, its ripples as `water`'s in `wind`, the
    /// waves' tiles repeating `repeats` times a meter, its own waves at
    /// simulation time `seconds`; returns whether any of it draws.
    fn prepare_ocean(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        (cull, view): (Mat4, Mat4),
        (water, ocean): (Water, Ocean),
        (eye, wind, seconds): (Vec3, Wind, f64),
        repeats: [f32; 4],
    ) -> bool {
        let (true, Some(waves)) = (ocean.enabled, &self.waves) else {
            self.ocean = None;
            return false;
        };
        let draw = self
            .ocean
            .get_or_insert_with(|| ocean_draw(device, &self.layout, &self.placeholders, waves));
        let waves = ocean.spectrum();
        let amplitude = waves.amplitude();
        // Cells widen as the eye rises over the sea.
        let above = max(eye.y - ocean.level, 0.0);
        let cell = OCEAN_CELL + above * OCEAN_CELL_PER_HEIGHT;
        let growth = 1.0 + std::f32::consts::TAU / OCEAN_SEGMENTS as f32;
        let first = OCEAN_FIRST_RINGS as f32 * cell;
        let widening = max((ocean.reach / first).ln() / growth.ln(), 0.0).ceil() as u32;
        draw.rings = OCEAN_FIRST_RINGS + widening;
        // Each sector a wedge from the eye out to the reach, as high as the
        // waves rise and fall, culled eye-relative.
        let planes = crate::terrain::planes(cull, false);
        let (low, high) = (
            ocean.level - eye.y - amplitude,
            ocean.level - eye.y + amplitude,
        );
        draw.runs.clear();
        for sector in 0..OCEAN_SECTORS {
            let angle =
                |step: f32| (sector as f32 + step) / OCEAN_SECTORS as f32 * std::f32::consts::TAU;
            let (mut min, mut max) = (Vec3::new(0.0, low, 0.0), Vec3::new(0.0, high, 0.0));
            for step in [0.0, 0.5, 1.0] {
                // A little past the reach, so the arc between stays inside.
                let (sin, cos) = engine_core::math::sin_cos(angle(step));
                let point = Vec3::new(cos, 0.0, sin) * (ocean.reach * 1.02);
                min = min.min(point.with_y(low));
                max = max.max(point.with_y(high));
            }
            if !crate::terrain::visible(&planes, min, max) {
                continue;
            }
            match draw.runs.last_mut() {
                Some(run) if run.end == sector => run.end = sector + 1,
                _ => draw.runs.push(sector..sector + 1),
            }
        }
        let phases = waves.phases(f64::from(eye.x), f64::from(eye.z), seconds);
        let mut ocean_waves = [[0.0; 4]; MAX_OCEAN_WAVES];
        let mut ocean_reach = [[0.0; 4]; MAX_OCEAN_WAVES / 4];
        for (index, wave) in waves.waves().iter().enumerate() {
            ocean_waves[index] = [
                wave.vector[0],
                wave.vector[1],
                wave.amplitude,
                phases[index],
            ];
            ocean_reach[index / 4][index % 4] = wave.reach;
        }
        let [red, green, blue] = ocean.color;
        // Whitecaps cover a share of the sea growing with the wind's speed
        // to the power 3.41 (Monahan's fit; about 2 % at 12 m/s), twice it
        // at full foam: crests foam where the waves compress the surface
        // more than that share of it, which the spectrum's shape fixes.
        let speed = ocean.wind[0].hypot(ocean.wind[1]);
        let share = 3.84e-6 * speed.powf(3.41) * 2.0 * ocean.foam;
        let crests = 0.67 + 0.069 * (max(share, 1.0e-4) / 0.01).ln();
        let heading = [ocean.wind[0] / speed, ocean.wind[1] / speed];
        let uniform = WaterUniform {
            relative_view_projection: view.to_cols_array_2d(),
            origin: [heading[0], heading[1], speed, 0.0],
            color: [red, green, blue, ocean.clarity],
            ripples: ripples(water, wind),
            // Shores foam as wide as the waves rise.
            foam: [ocean.foam, 0.4 + amplitude, crests, amplitude],
            absorption: absorption(ocean.color, ocean.clarity),
            waves: repeats,
            ocean: [
                ocean.level - eye.y,
                cell,
                OCEAN_SEGMENTS as f32,
                OCEAN_FIRST_RINGS as f32,
            ],
            ocean_grid: [
                growth,
                ocean.reach,
                (OCEAN_SEGMENTS / OCEAN_SECTORS) as f32,
                waves.count as f32,
            ],
            ocean_waves,
            ocean_reach,
        };
        queue.write_buffer(&draw.uniform, 0, bytemuck::bytes_of(&uniform));
        !draw.runs.is_empty()
    }

    /// Creates the pipelines into `targets` this frame's water and sea
    /// draw with unless they exist, and tiles' water's when `coming`,
    /// before a bake gives any; `camera` is the main passes'
    /// first group.
    pub(crate) fn ensure_pipeline(
        &mut self,
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        camera: &crate::pipelines::GroupLayout,
        (targets, coming): (Targets, bool),
        formats: (wgpu::TextureFormat, wgpu::TextureFormat),
    ) {
        if (coming || self.tiles.is_some()) && !self.pipelines.contains_key(&targets) {
            let pipeline = self.pipeline(device, shader, camera, targets, formats, Surface::Tiles);
            self.pipelines.insert(targets, pipeline);
        }
        if self.ocean.is_some() && !self.ocean_pipelines.contains_key(&targets) {
            let pipeline = self.pipeline(device, shader, camera, targets, formats, Surface::Sea);
            self.ocean_pipelines.insert(targets, pipeline);
        }
        if self.falls.is_some() && !self.fall_pipelines.contains_key(&targets) {
            let pipeline = self.pipeline(device, shader, camera, targets, formats, Surface::Falls);
            self.fall_pipelines.insert(targets, pipeline);
        }
    }

    /// The pipeline of `surface` into `targets`.
    fn pipeline(
        &mut self,
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        camera: &crate::pipelines::GroupLayout,
        targets: Targets,
        (canvas, depth): (wgpu::TextureFormat, wgpu::TextureFormat),
        surface: Surface,
    ) -> wgpu::RenderPipeline {
        // Water and the sea share one layout.
        let layout = self.pipeline_layout.get_or_insert_with(|| {
            crate::pipelines::pipeline_layout(
                device,
                &crate::pipeline_layouts::WATER_PIPELINES,
                &[camera, &self.layout, &self.behind_layout],
            )
        });
        let (vertex, fragment) = match surface {
            Surface::Tiles => crate::entries::water(targets),
            Surface::Sea => crate::entries::ocean(targets),
            Surface::Falls => crate::entries::falls(targets),
        };
        // The scene color blends premultiplied over the ground, keeping the
        // ground's ambient share in its alpha (water refracting the scene
        // covers it whole but where its shores fade); the other targets
        // take the water's.
        let blended = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let mut color_targets = targets.color_targets(canvas);
        if let Some(Some(scene)) = color_targets.first_mut() {
            scene.blend = Some(blended);
        }
        let patches = [Some(water_patch_layout())];
        let sheets = [Some(fall_layout())];
        crate::pipelines::geometry(
            device,
            &crate::pipelines::Geometry {
                label: match surface {
                    Surface::Tiles => "engine/water",
                    Surface::Sea => "engine/ocean",
                    Surface::Falls => "engine/water-falls",
                },
                layout,
                module: shader,
                vertex_constants: &[],
                vertex,
                // The sea's vertices come from their indices alone.
                buffers: match surface {
                    Surface::Tiles => &patches,
                    Surface::Sea => &[],
                    Surface::Falls => &sheets,
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    // Water is seen from above and below alike, falls from
                    // either side.
                    cull_mode: None,
                    ..Default::default()
                },
                depth: Some(wgpu::DepthStencilState {
                    format: depth,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Greater),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                fragment: Some(crate::pipelines::Fragment {
                    entry: fragment,
                    constants: &[],
                    targets: &color_targets,
                }),
            },
        )
    }

    /// Records this frame's water into `pass` with the main passes' first
    /// group `camera`; returns its draws.
    fn draw(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        camera: &wgpu::BindGroup,
        targets: Targets,
    ) -> u32 {
        let Some(behind) = self.behind.get() else {
            return 0;
        };
        let mut draws = 0;
        // The sea first: terrains' water below its surface stays hidden.
        if let (Some(ocean), Some(pipeline)) = (&self.ocean, self.ocean_pipelines.get(&targets)) {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, camera, &[]);
            pass.set_bind_group(1, &ocean.group, &[0, 0]);
            pass.set_bind_group(2, &behind.group, &[]);
            let vertices = ocean.rings * (OCEAN_SEGMENTS / OCEAN_SECTORS) * 6;
            for run in &ocean.runs {
                pass.draw(0..vertices, run.clone());
                draws += 1;
            }
        }
        let Some((_, group)) = &self.group else {
            return draws;
        };
        if let (Some(tiles), Some(pipeline)) = (&self.tiles, self.pipelines.get(&targets)) {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, camera, &[]);
            pass.set_bind_group(2, &behind.group, &[]);
            pass.set_vertex_buffer(0, tiles.instances.slice(..));
            pass.set_index_buffer(tiles.indices.slice(..), wgpu::IndexFormat::Uint16);
            // Each terrain's patches through its uniform, seen by the
            // camera's view, the first.
            for (uniform, range) in &tiles.terrains {
                pass.set_bind_group(1, group, &[0, *uniform]);
                pass.draw_indexed(0..WATER_GRID_INDICES, 0, range.clone());
                draws += 1;
            }
        }
        // Falls last, over the water they land in: of its group they read
        // only water's uniform and waves.
        if let (Some(falls), Some(pipeline)) = (&self.falls, self.fall_pipelines.get(&targets)) {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, camera, &[]);
            pass.set_bind_group(1, group, &[0, 0]);
            pass.set_bind_group(2, &behind.group, &[]);
            pass.set_vertex_buffer(0, falls.vertices.slice(..));
            pass.set_index_buffer(falls.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..falls.count, 0, 0..1);
            draws += 1;
        }
        draws
    }
}

impl RenderFeature for WaterPass {
    type Frame = WaterFrame;
    type Stats = (WaterStats, OceanStats);

    /// Water blends over the opaque scene, with its shadows and probes, and
    /// writes its depth, which transparent surfaces and screen effects then
    /// see.
    fn declare(_frame: &WaterFrame, shape: &FrameShape, graph: &mut RenderGraph) {
        let mut water = graph
            .add(Pass::Feature(FeaturePass::Water))
            .read_if(shape.shadows, Resource::ShadowMaps)
            .read(Resource::Probes);
        for &target in shape.main_targets() {
            water = water.color(target, Access::Load);
        }
        water.depth(Resource::Depth, Access::Load);
    }

    fn encode(
        &self,
        _frame: &WaterFrame,
        node: &Node,
        encode: &Encode<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> u32 {
        // The waves, and what lies behind the water, before it draws.
        if let Some(waves) = &self.waves {
            waves.encode(encoder, encode.profiler);
        }
        self.copy_behind(encoder, encode.depth, encode.scene);
        let mut pass = encode.render_pass(
            encoder,
            node,
            "engine/water",
            encode.profiler.span("water").map(Span::render),
        );
        self.draw(&mut pass, encode.camera, encode.targets)
    }

    /// Bytes of the copies and the waves: tiles' water blocks count with
    /// the terrain's arrays.
    fn capacity_bytes(&self) -> u64 {
        let copies = self.behind.size().map_or(0, |size| {
            let texels = u64::from(size.width) * u64::from(size.height);
            let scene = self
                .behind
                .get()
                .is_some_and(|copies| copies.scene.is_some());
            texels * if scene { 8 } else { 4 }
        });
        let falls = self
            .falls
            .as_ref()
            .map_or(0, |falls| falls.vertices.size() + falls.indices.size());
        copies + falls + self.waves.as_ref().map_or(0, WaterWaves::capacity_bytes)
    }

    /// Tiles' patches holding water drawn, and the sea's sectors drawn and
    /// their triangles.
    fn stats(
        _this: Option<&Self>,
        frame: Option<&WaterFrame>,
        _snapshot: &RenderSnapshot,
    ) -> (WaterStats, OceanStats) {
        (
            WaterStats {
                patches: frame.map_or(0, |frame| frame.patches),
                blocks: frame.map_or(0, |frame| frame.blocks[0]),
                dropped_blocks: frame.map_or(0, |frame| frame.blocks[1]),
            },
            OceanStats {
                sectors: frame.map_or(0, |frame| frame.sectors),
                triangles: frame.map_or(0, |frame| frame.triangles),
            },
        )
    }
}

/// What a water pipeline draws: tiles' water, the sea, or falls' sheets.
#[derive(Clone, Copy)]
enum Surface {
    Tiles,
    Sea,
    Falls,
}

/// The first `bytes` of `buffer`, bound at a dynamic offset.
fn sized(buffer: &wgpu::Buffer, bytes: usize) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer,
        offset: 0,
        size: wgpu::BufferSize::new(bytes as u64),
    })
}

/// The light of the sea around an eye at `eye` under its surface,
/// its waves at simulation time `seconds`, as the lighting uniform's
/// `underwater`: what its water scatters of the sun and sky, dimmed by the
/// water above the eye (rgb), and the meters through which half the light
/// passes (w); zero above the sea.
pub(crate) fn underwater(environment: &Environment, eye: Vec3, seconds: f64) -> [f32; 4] {
    let ocean = environment.ocean;
    if !ocean.enabled {
        return [0.0; 4];
    }
    let waves = ocean.spectrum();
    if eye.y > ocean.level + waves.amplitude() {
        return [0.0; 4];
    }
    let surface = waves.surface(f64::from(eye.x), f64::from(eye.z), seconds);
    let depth = surface.height - eye.y;
    if depth <= 0.0 {
        return [0.0; 4];
    }
    let atmosphere = &environment.atmosphere;
    let light = atmosphere.sun_color * max(atmosphere.sun_direction.normalize_or(Vec3::Y).y, 0.0)
        + atmosphere.zenith * atmosphere.ambient;
    // Water scatters back many times what its color says it returns of a
    // single bounce.
    let color = Vec3::from_array(ocean.color) * light * 3.0 * (-depth / ocean.clarity).exp2();
    [color.x, color.y, color.z, ocean.clarity]
}

/// The water uniform's ripples: their strength in `water`, and how strongly
/// `wind` gusts and its velocity.
fn ripples(water: Water, wind: Wind) -> [f32; 4] {
    [
        water.ripples,
        wind.gusts,
        wind.velocity[0],
        wind.velocity[1],
    ]
}

/// Per meter, how fast water whose light halves over `clarity` meters and
/// which scatters back `color` takes each channel (rgb): a channel as fast
/// as the color's luminance exceeds the channel's share of it (at most
/// eight times), weighted so the light's luminance halves over the
/// clarity. The channels water's color holds least go first, so the
/// ground behind it tints toward that color, and water as gray as its
/// color takes all alike.
fn absorption(color: [f32; 3], clarity: f32) -> [f32; 4] {
    const LUMINANCE: [f32; 3] = [0.2126, 0.7152, 0.0722];
    let luminance = max(
        color[0] * LUMINANCE[0] + color[1] * LUMINANCE[1] + color[2] * LUMINANCE[2],
        1.0e-4,
    );
    let weights = color.map(|channel| luminance / max(channel, luminance * 0.125));
    let mean = weights[0] * LUMINANCE[0] + weights[1] * LUMINANCE[1] + weights[2] * LUMINANCE[2];
    let rate = std::f32::consts::LN_2 / max(clarity, 0.01) / mean;
    [weights[0] * rate, weights[1] * rate, weights[2] * rate, 0.0]
}

/// The water group's stand-ins for what the sea reads none of, and the
/// color's for frames blending water.
fn placeholders(device: &wgpu::Device) -> Placeholders {
    let array = |label, format| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    };
    let scene = array(
        "engine/water-placeholder-scene",
        wgpu::TextureFormat::Rgba8Unorm,
    );
    Placeholders {
        scene: scene.create_view(&wgpu::TextureViewDescriptor::default()),
        surfaces: array("engine/water-placeholder", wgpu::TextureFormat::R32Float),
        info: array(
            "engine/water-placeholder-info",
            wgpu::TextureFormat::Rgba8Snorm,
        ),
        views: crate::buffers::uniform(
            device,
            "engine/water-placeholder-view",
            size_of::<ViewUniform>() as u64,
        ),
        terrains: crate::buffers::uniform(
            device,
            "engine/water-placeholder-terrain",
            size_of::<TerrainUniform>() as u64,
        ),
    }
}

/// The sea's GPU state: its uniform, and its group binding `waves` and
/// `placeholders` for what only tiles' water reads.
fn ocean_draw(
    device: &wgpu::Device,
    layout: &crate::pipelines::GroupLayout,
    placeholders: &Placeholders,
    waves: &WaterWaves,
) -> OceanDraw {
    let (moments, sampler) = waves.sampled();
    let array = |texture: &wgpu::Texture| {
        texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        })
    };
    let (surfaces, info) = (array(&placeholders.surfaces), array(&placeholders.info));
    let uniform = crate::buffers::uniform(device, "engine/ocean", WATER_UNIFORM_BYTES as u64);
    let group = crate::pipelines::group(
        device,
        "engine/ocean-group",
        &WATER,
        layout,
        &[
            ("terrain_heights", crate::pipelines::view(&surfaces)),
            (
                "terrain_view",
                sized(&placeholders.views, size_of::<ViewUniform>()),
            ),
            (
                "terrain",
                sized(&placeholders.terrains, size_of::<TerrainUniform>()),
            ),
            ("water_surface", crate::pipelines::view(&surfaces)),
            ("water_info", crate::pipelines::view(&info)),
            ("water", uniform.as_entire_binding()),
            ("water_waves", crate::pipelines::view(moments)),
            ("water_wave_sampler", crate::pipelines::sampler(sampler)),
        ],
    );
    OceanDraw {
        uniform,
        group,
        rings: 0,
        runs: Vec::new(),
    }
}
