//! Water (decision 067): terrains' lakes and rivers where their hydrology
//! bakes found them (decision 066). Each terrain's water grid uploads once
//! as two textures, its surface heights and its depth and flow, and splits
//! into chunks of [`CHUNK_QUADS`] quads a side, those holding water kept.
//! Each frame the chunks the camera sees draw, coarser with distance,
//! after the opaque scene and its decals: water blends over the ground by
//! how little light comes back through it and writes the depth and surface
//! targets, so later passes and screen-space reflections see its surface.
//! Before it draws, the opaque depth copies into a texture of its own, from
//! which its pixels find the ground behind them (decision 074): the
//! distance light travels through the water, and where it runs shallow
//! enough to foam.
//!
//! The sea (decision 076) draws first in the same pass, from a grid around
//! the eye: rings of [`OCEAN_SEGMENTS`] cells, a cell apart near the eye
//! and then each wider by a cell's share of its circumference, so cells
//! stay square, out to the sea's reach. Its sectors cull on their own;
//! vertices move by the waves longer than a few of their cells, and pixels
//! shade every wave longer than a few pixels.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use engine_core::glam::{Mat4, Vec3};
use engine_core::math::max;
use engine_core::{Environment, MAX_OCEAN_WAVES, Ocean, RenderTerrains, Water, WaterGrid};

use crate::bindings::{WATER, WATER_DEPTH};
use crate::render_graph::Targets;

/// Quads along each side of a chunk at full detail; coarser chunks draw a
/// vertex every second, fourth or eighth sample.
pub(crate) const CHUNK_QUADS: u32 = 32;
/// Samples between a chunk's vertices by distance: full detail within the
/// first reach, half within the second, and so on (meters).
const STRIDES: [(u32, f32); 4] = [(1, 700.0), (2, 1600.0), (4, 3200.0), (8, f32::INFINITY)];

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

/// WGSL `WaterUniform` of `meshes/water.wgsl`; the sea's parts are zero
/// for terrains' water, and the terrain parts for the sea's.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
struct WaterUniform {
    relative_view_projection: [[f32; 4]; 4],
    origin: [f32; 4],
    grid: [u32; 4],
    color: [f32; 4],
    ripples: [f32; 4],
    foam: [f32; 4],
    ocean: [f32; 4],
    ocean_grid: [f32; 4],
    ocean_waves: [[f32; 4]; MAX_OCEAN_WAVES],
    ocean_reach: [[f32; 4]; MAX_OCEAN_WAVES / 4],
}

/// Bytes of the water group's uniform.
pub(crate) const WATER_UNIFORM_BYTES: usize = size_of::<WaterUniform>();

/// The sea on the GPU (decision 076): placeholders for the water group's
/// textures, its uniform and group, and this frame's rings and the runs of
/// sectors the camera sees.
struct OceanDraw {
    _textures: [wgpu::Texture; 2],
    uniform: wgpu::Buffer,
    group: wgpu::BindGroup,
    rings: u32,
    runs: Vec<Range<u32>>,
}

/// WGSL `WaterChunk`'s instance part: its first sample and the samples
/// between its vertices.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
struct ChunkRecord {
    first: [u32; 2],
    stride: u32,
}

/// A chunk holding water: its first sample, and its terrain-local bounds.
struct Chunk {
    first: [u32; 2],
    min: Vec3,
    max: Vec3,
}

/// One terrain's water on the GPU, keyed by its grid.
struct TerrainWater {
    grid: Arc<WaterGrid>,
    _textures: [wgpu::Texture; 2],
    uniform: wgpu::Buffer,
    group: wgpu::BindGroup,
    chunks: Vec<Chunk>,
    // This frame: its instance range for each stride, and whether it draws.
    draws: Vec<(u32, Range<u32>)>,
    drawn: bool,
}

/// The opaque depth's copy water reads (decision 074), its group and its
/// size.
struct DepthCopy {
    texture: wgpu::Texture,
    group: wgpu::BindGroup,
    size: (u32, u32),
}

pub(crate) struct WaterPass {
    pub(crate) layout: wgpu::BindGroupLayout,
    depth_layout: wgpu::BindGroupLayout,
    depth: Option<DepthCopy>,
    terrains: HashMap<u64, TerrainWater>,
    instances: wgpu::Buffer,
    pipeline_layout: Option<wgpu::PipelineLayout>,
    pipelines: HashMap<Targets, wgpu::RenderPipeline>,
    ocean_pipelines: HashMap<Targets, wgpu::RenderPipeline>,
    records: Vec<ChunkRecord>,
    chunks_drawn: u32,
    ocean: Option<OceanDraw>,
}

impl WaterPass {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        Self {
            layout: crate::pipelines::layout(device, &WATER),
            depth_layout: crate::pipelines::layout(device, &WATER_DEPTH),
            depth: None,
            terrains: HashMap::new(),
            instances: chunk_buffer(device, 256),
            pipeline_layout: None,
            pipelines: HashMap::new(),
            ocean_pipelines: HashMap::new(),
            records: Vec::new(),
            chunks_drawn: 0,
            ocean: None,
        }
    }

    /// Chunks drawn this frame.
    pub(crate) fn chunks_drawn(&self) -> u32 {
        self.chunks_drawn
    }

    /// The sea's sectors drawn this frame and their triangles.
    pub(crate) fn ocean_drawn(&self) -> (u32, u32) {
        self.ocean.as_ref().map_or((0, 0), |ocean| {
            let sectors: u32 = ocean.runs.iter().map(|run| run.len() as u32).sum();
            let per_sector = OCEAN_SEGMENTS / OCEAN_SECTORS;
            (sectors, sectors * ocean.rings * per_sector * 2)
        })
    }

    /// Bytes of the water's textures and its depth copy.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        let depth = self.depth.as_ref().map_or(0, |copy| {
            u64::from(copy.size.0) * u64::from(copy.size.1) * 4
        });
        self.terrains
            .values()
            .map(|water| (water.grid.width * water.grid.height * 8) as u64)
            .sum::<u64>()
            + depth
    }

    /// Copies the opaque `depth` into the copy water reads, before it
    /// draws.
    pub(crate) fn copy_depth(&self, encoder: &mut wgpu::CommandEncoder, depth: &wgpu::Texture) {
        if let Some(copy) = &self.depth
            && depth.size() == copy.texture.size()
        {
            encoder.copy_texture_to_texture(
                depth.as_image_copy(),
                copy.texture.as_image_copy(),
                depth.size(),
            );
        }
    }

    /// Prepares this frame's water of `terrains` as `settings` shade it and
    /// the sea `ocean` (decision 076), seen from `eye` with the wind's
    /// `wind` velocity, its waves at the simulation time `seconds` the
    /// frame presents (decision 077); returns whether any of it draws.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        terrains: &RenderTerrains,
        (settings, ocean): (Water, Ocean),
        (eye, wind, seconds): (Vec3, [f32; 2], f64),
        depth: &wgpu::Texture,
    ) -> bool {
        self.chunks_drawn = 0;
        self.records.clear();
        self.terrains
            .retain(|id, _| terrains.terrains.iter().any(|terrain| terrain.id == *id));
        for water in self.terrains.values_mut() {
            water.drawn = false;
        }
        if settings.enabled {
            self.prepare_terrains(device, queue, terrains, settings, (eye, wind));
        }
        let sea = self.prepare_ocean(
            device,
            queue,
            terrains.relative_view_projection,
            (settings, ocean),
            (eye, wind, seconds),
        );
        if self.records.is_empty() && !sea {
            return false;
        }
        // The depth copy, at the frame's size.
        let depth_size = (depth.width(), depth.height());
        if self
            .depth
            .as_ref()
            .is_none_or(|copy| copy.size != depth_size)
        {
            let texture = crate::targets::texture_2d(
                device,
                "engine/water-depth",
                depth_size,
                1,
                1,
                depth.format(),
                wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            );
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let group = crate::pipelines::group(
                device,
                "engine/water-depth",
                &WATER_DEPTH,
                &self.depth_layout,
                &[("water_depth", crate::pipelines::view(&view))],
            );
            self.depth = Some(DepthCopy {
                texture,
                group,
                size: depth_size,
            });
        }
        if !self.records.is_empty() {
            let bytes = std::mem::size_of_val(self.records.as_slice()) as u64;
            if bytes > self.instances.size() {
                self.instances = chunk_buffer(device, bytes.next_power_of_two());
            }
            queue.write_buffer(&self.instances, 0, bytemuck::cast_slice(&self.records));
        }
        true
    }

    /// Prepares the sea (decision 076) seen through `view` (eye-relative)
    /// from `eye`, its ripples as `water`'s, its waves at simulation time
    /// `seconds`; returns whether any of it draws.
    fn prepare_ocean(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: Mat4,
        (water, ocean): (Water, Ocean),
        (eye, wind, seconds): (Vec3, [f32; 2], f64),
    ) -> bool {
        if !ocean.enabled {
            self.ocean = None;
            return false;
        }
        let draw = self
            .ocean
            .get_or_insert_with(|| ocean_draw(device, queue, &self.layout));
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
        let planes = crate::terrain::planes(view, false);
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
            ripples: [water.ripples, water.ripple_size, wind[0], wind[1]],
            // Shores foam as wide as the waves rise.
            foam: [ocean.foam, 0.4 + amplitude, crests, amplitude],
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
            ..WaterUniform::zeroed()
        };
        queue.write_buffer(&draw.uniform, 0, bytemuck::bytes_of(&uniform));
        !draw.runs.is_empty()
    }

    /// Prepares terrains' water chunks the camera sees.
    fn prepare_terrains(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        terrains: &RenderTerrains,
        settings: Water,
        (eye, wind): (Vec3, [f32; 2]),
    ) {
        // Chunks cull eye-relative, as terrain's patches do.
        let planes = crate::terrain::planes(terrains.relative_view_projection, false);
        for terrain in &terrains.terrains {
            let Some(grid) = &terrain.water else {
                self.terrains.remove(&terrain.id);
                continue;
            };
            let stale = self
                .terrains
                .get(&terrain.id)
                .is_none_or(|water| !Arc::ptr_eq(&water.grid, grid));
            if stale {
                let water = upload(device, queue, &self.layout, grid);
                self.terrains.insert(terrain.id, water);
            }
            let water = self.terrains.get_mut(&terrain.id).expect("inserted above");
            water.draws.clear();
            let relative_origin = terrain.origin - eye;
            // The chunks the camera sees, coarser with distance, grouped by
            // stride.
            let mut seen: Vec<(u32, [u32; 2])> = Vec::new();
            for chunk in &water.chunks {
                let (min, max) = (relative_origin + chunk.min, relative_origin + chunk.max);
                if !crate::terrain::visible(&planes, min, max) {
                    continue;
                }
                let distance = ((min + max) * 0.5).length();
                let stride = STRIDES
                    .iter()
                    .find(|(_, reach)| distance < *reach)
                    .map_or(8, |(stride, _)| *stride);
                seen.push((stride, chunk.first));
            }
            seen.sort_unstable_by_key(|(stride, _)| *stride);
            for (stride, first) in seen {
                let start = self.records.len() as u32;
                self.records.push(ChunkRecord { first, stride });
                match water.draws.last_mut() {
                    Some((last, range)) if *last == stride => range.end = start + 1,
                    _ => water.draws.push((stride, start..start + 1)),
                }
            }
            water.drawn = !water.draws.is_empty();
            self.chunks_drawn += water
                .draws
                .iter()
                .map(|(_, range)| range.len() as u32)
                .sum::<u32>();
            let [red, green, blue] = settings.color;
            let uniform = WaterUniform {
                relative_view_projection: terrains.relative_view_projection.to_cols_array_2d(),
                origin: relative_origin.extend(water.grid.spacing).to_array(),
                grid: [
                    water.grid.width as u32,
                    water.grid.height as u32,
                    CHUNK_QUADS,
                    0,
                ],
                color: [red, green, blue, settings.clarity],
                ripples: [settings.ripples, settings.ripple_size, wind[0], wind[1]],
                foam: [settings.foam, settings.foam_width, 0.0, 0.0],
                ..WaterUniform::zeroed()
            };
            queue.write_buffer(&water.uniform, 0, bytemuck::bytes_of(&uniform));
        }
    }

    /// Creates the pipelines into `targets` this frame's water and sea
    /// draw with unless they exist; `camera` is the main passes' first
    /// group.
    pub(crate) fn ensure_pipeline(
        &mut self,
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        camera: &wgpu::BindGroupLayout,
        targets: Targets,
        formats: (wgpu::TextureFormat, wgpu::TextureFormat),
    ) {
        if !self.records.is_empty() && !self.pipelines.contains_key(&targets) {
            let pipeline = self.pipeline(device, shader, camera, targets, formats, false);
            self.pipelines.insert(targets, pipeline);
        }
        if self.ocean.is_some() && !self.ocean_pipelines.contains_key(&targets) {
            let pipeline = self.pipeline(device, shader, camera, targets, formats, true);
            self.ocean_pipelines.insert(targets, pipeline);
        }
    }

    /// The pipeline of terrains' water, or of the sea when `ocean`, into
    /// `targets`.
    fn pipeline(
        &mut self,
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        camera: &wgpu::BindGroupLayout,
        targets: Targets,
        (canvas, depth): (wgpu::TextureFormat, wgpu::TextureFormat),
        ocean: bool,
    ) -> wgpu::RenderPipeline {
        // Water and the sea share one layout.
        let layout = self.pipeline_layout.get_or_insert_with(|| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("engine/water-layout"),
                bind_group_layouts: &[Some(camera), Some(&self.layout), Some(&self.depth_layout)],
                immediate_size: 0,
            })
        });
        let (vertex, fragment) = if ocean {
            crate::entries::ocean(targets)
        } else {
            crate::entries::water(targets)
        };
        // The scene color blends premultiplied over the ground, keeping the
        // ground's ambient share in its alpha; the other targets take the
        // water's.
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
        let chunks = [Some(chunk_layout())];
        crate::pipelines::geometry(
            device,
            &crate::pipelines::Geometry {
                label: if ocean {
                    "engine/ocean"
                } else {
                    "engine/water"
                },
                layout,
                module: shader,
                vertex,
                // The sea's vertices come from their indices alone.
                buffers: if ocean { &[] } else { &chunks },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    // Water is seen from above and below alike.
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
    pub(crate) fn draw(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        camera: &wgpu::BindGroup,
        targets: Targets,
    ) -> u32 {
        let Some(depth) = &self.depth else {
            return 0;
        };
        let mut draws = 0;
        // The sea first: terrains' water below its surface stays hidden.
        if let (Some(ocean), Some(pipeline)) = (&self.ocean, self.ocean_pipelines.get(&targets)) {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, camera, &[]);
            pass.set_bind_group(1, &ocean.group, &[]);
            pass.set_bind_group(2, &depth.group, &[]);
            let vertices = ocean.rings * (OCEAN_SEGMENTS / OCEAN_SECTORS) * 6;
            for run in &ocean.runs {
                pass.draw(0..vertices, run.clone());
                draws += 1;
            }
        }
        let Some(pipeline) = self.pipelines.get(&targets) else {
            return draws;
        };
        if self.records.is_empty() {
            return draws;
        }
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, camera, &[]);
        pass.set_bind_group(2, &depth.group, &[]);
        pass.set_vertex_buffer(0, self.instances.slice(..));
        for water in self.terrains.values().filter(|water| water.drawn) {
            pass.set_bind_group(1, &water.group, &[]);
            for (stride, range) in &water.draws {
                let quads = CHUNK_QUADS / stride;
                pass.draw(0..quads * quads * 6, range.clone());
                draws += 1;
            }
        }
        draws
    }
}

/// `WaterChunk`'s instance locations: its first sample and stride.
fn chunk_layout() -> wgpu::VertexBufferLayout<'static> {
    const ATTRIBUTES: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Uint32x2, 1 => Uint32];
    wgpu::VertexBufferLayout {
        array_stride: size_of::<ChunkRecord>() as u64,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &ATTRIBUTES,
    }
}

/// `grid` on the GPU: its surface heights and its depth and flow as
/// textures, its uniform and group, and its chunks holding water.
fn upload(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    grid: &Arc<WaterGrid>,
) -> TerrainWater {
    let (width, height) = (grid.width as u32, grid.height as u32);
    let texture = |label, format| {
        crate::targets::texture_2d(
            device,
            label,
            (width, height),
            1,
            1,
            format,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        )
    };
    let surface = texture("engine/water-surface", wgpu::TextureFormat::R32Float);
    let info = texture("engine/water-info", wgpu::TextureFormat::Rgba8Unorm);
    let unit = |value: f32| (engine_core::math::clamp(value, 0.0, 1.0) * 255.0 + 0.5) as u8;
    let packed: Vec<u8> = grid
        .depth
        .iter()
        .zip(&grid.flow)
        .flat_map(|(&depth, &[x, z])| {
            [
                unit(depth / 32.0),
                unit(x / 16.0 + 0.5),
                unit(z / 16.0 + 0.5),
                0,
            ]
        })
        .collect();
    let write = |texture: &wgpu::Texture, bytes: &[u8], texel: u32| {
        queue.write_texture(
            texture.as_image_copy(),
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * texel),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    };
    write(&surface, bytemuck::cast_slice(&grid.surface), 4);
    write(&info, &packed, 4);
    let uniform = crate::buffers::uniform(device, "engine/water", size_of::<WaterUniform>() as u64);
    let group = crate::pipelines::group(
        device,
        "engine/water-group",
        &WATER,
        layout,
        &[
            (
                "water_surface",
                crate::pipelines::view(&surface.create_view(&Default::default())),
            ),
            (
                "water_info",
                crate::pipelines::view(&info.create_view(&Default::default())),
            ),
            ("water", uniform.as_entire_binding()),
        ],
    );
    // Chunks holding water, with their surfaces' bounds.
    let quads = CHUNK_QUADS as usize;
    let mut chunks = Vec::new();
    for z in (0..grid.height.saturating_sub(1)).step_by(quads) {
        for x in (0..grid.width.saturating_sub(1)).step_by(quads) {
            let Some([low, high]) = grid.block(x, z, quads) else {
                continue;
            };
            let spacing = grid.spacing;
            chunks.push(Chunk {
                first: [x as u32, z as u32],
                min: Vec3::new(x as f32 * spacing, low, z as f32 * spacing),
                max: Vec3::new(
                    (x + quads) as f32 * spacing,
                    high,
                    (z + quads) as f32 * spacing,
                ),
            });
        }
    }
    TerrainWater {
        grid: Arc::clone(grid),
        _textures: [surface, info],
        uniform,
        group,
        chunks,
        draws: Vec::new(),
        drawn: false,
    }
}

/// The light of the sea around an eye at `eye` under its surface (decision
/// 077), its waves at simulation time `seconds`, as the lighting uniform's
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

/// The sea's GPU state: its uniform, and one-texel placeholders for the
/// water group's textures, which only terrains' water reads.
fn ocean_draw(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
) -> OceanDraw {
    let texture = |label, format| {
        crate::targets::texture_2d(
            device,
            label,
            (1, 1),
            1,
            1,
            format,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        )
    };
    let surface = texture("engine/ocean-surface", wgpu::TextureFormat::R32Float);
    let info = texture("engine/ocean-info", wgpu::TextureFormat::Rgba8Unorm);
    for (texture, bytes) in [(&surface, [0_u8; 4]), (&info, [0_u8; 4])] {
        queue.write_texture(
            texture.as_image_copy(),
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
    }
    let uniform = crate::buffers::uniform(device, "engine/ocean", WATER_UNIFORM_BYTES as u64);
    let group = crate::pipelines::group(
        device,
        "engine/ocean-group",
        &WATER,
        layout,
        &[
            (
                "water_surface",
                crate::pipelines::view(&surface.create_view(&Default::default())),
            ),
            (
                "water_info",
                crate::pipelines::view(&info.create_view(&Default::default())),
            ),
            ("water", uniform.as_entire_binding()),
        ],
    );
    OceanDraw {
        _textures: [surface, info],
        uniform,
        group,
        rings: 0,
        runs: Vec::new(),
    }
}

/// A buffer of `bytes` of chunk records.
fn chunk_buffer(device: &wgpu::Device, bytes: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("engine/water-chunks"),
        size: bytes,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
