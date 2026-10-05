//! Water (decision 067): terrains' lakes and rivers where their hydrology
//! bakes found them (decision 066). Each terrain's water grid uploads once
//! as two textures, its surface heights and its depth and flow, and splits
//! into chunks of [`CHUNK_QUADS`] quads a side, those holding water kept.
//! Each frame the chunks the camera sees draw, coarser with distance,
//! after the opaque scene and its decals: water blends over the ground by
//! how little light comes back through its depth and writes the depth and
//! surface targets, so later passes and screen-space reflections see its
//! surface.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use engine_core::glam::Vec3;
use engine_core::{RenderTerrains, Water, WaterGrid};

use crate::bindings::WATER;
use crate::render_graph::Targets;

/// Quads along each side of a chunk at full detail; coarser chunks draw a
/// vertex every second, fourth or eighth sample.
pub(crate) const CHUNK_QUADS: u32 = 32;
/// Samples between a chunk's vertices by distance: full detail within the
/// first reach, half within the second, and so on (meters).
const STRIDES: [(u32, f32); 4] = [(1, 700.0), (2, 1600.0), (4, 3200.0), (8, f32::INFINITY)];

/// WGSL `WaterUniform` of `meshes/water.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
struct WaterUniform {
    relative_view_projection: [[f32; 4]; 4],
    origin: [f32; 4],
    grid: [u32; 4],
    color: [f32; 4],
    ripples: [f32; 4],
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

pub(crate) struct WaterPass {
    pub(crate) layout: wgpu::BindGroupLayout,
    terrains: HashMap<u64, TerrainWater>,
    instances: wgpu::Buffer,
    pipelines: HashMap<Targets, wgpu::RenderPipeline>,
    records: Vec<ChunkRecord>,
    chunks_drawn: u32,
}

impl WaterPass {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        Self {
            layout: crate::pipelines::layout(device, &WATER),
            terrains: HashMap::new(),
            instances: chunk_buffer(device, 256),
            pipelines: HashMap::new(),
            records: Vec::new(),
            chunks_drawn: 0,
        }
    }

    /// Chunks drawn this frame.
    pub(crate) fn chunks_drawn(&self) -> u32 {
        self.chunks_drawn
    }

    /// Bytes of the water's textures.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.terrains
            .values()
            .map(|water| (water.grid.width * water.grid.height * 8) as u64)
            .sum()
    }

    /// Prepares this frame's water of `terrains` as `settings` shade it,
    /// seen from `eye` with the wind's `wind` velocity; returns whether any
    /// chunk draws.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        terrains: &RenderTerrains,
        settings: Water,
        (eye, wind): (Vec3, [f32; 2]),
    ) -> bool {
        self.chunks_drawn = 0;
        self.records.clear();
        self.terrains
            .retain(|id, _| terrains.terrains.iter().any(|terrain| terrain.id == *id));
        if !settings.enabled {
            return false;
        }
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
            };
            queue.write_buffer(&water.uniform, 0, bytemuck::bytes_of(&uniform));
        }
        if self.records.is_empty() {
            return false;
        }
        let bytes = std::mem::size_of_val(self.records.as_slice()) as u64;
        if bytes > self.instances.size() {
            self.instances = chunk_buffer(device, bytes.next_power_of_two());
        }
        queue.write_buffer(&self.instances, 0, bytemuck::cast_slice(&self.records));
        true
    }

    /// Creates the pipeline into `targets` unless it exists; `camera` is
    /// the main passes' first group.
    pub(crate) fn ensure_pipeline(
        &mut self,
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        camera: &wgpu::BindGroupLayout,
        targets: Targets,
        (canvas, depth): (wgpu::TextureFormat, wgpu::TextureFormat),
    ) {
        if self.pipelines.contains_key(&targets) {
            return;
        }
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("engine/water-layout"),
            bind_group_layouts: &[Some(camera), Some(&self.layout)],
            immediate_size: 0,
        });
        let (vertex, fragment) = crate::entries::water(targets);
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
        let pipeline = crate::pipelines::geometry(
            device,
            &crate::pipelines::Geometry {
                label: "engine/water",
                layout: &layout,
                module: shader,
                vertex,
                buffers: &[Some(chunk_layout())],
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
        );
        self.pipelines.insert(targets, pipeline);
    }

    /// Records this frame's water into `pass` with the main passes' first
    /// group `camera`; returns its draws.
    pub(crate) fn draw(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        camera: &wgpu::BindGroup,
        targets: Targets,
    ) -> u32 {
        let Some(pipeline) = self.pipelines.get(&targets) else {
            return 0;
        };
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, camera, &[]);
        pass.set_vertex_buffer(0, self.instances.slice(..));
        let mut draws = 0;
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

/// A buffer of `bytes` of chunk records.
fn chunk_buffer(device: &wgpu::Device, bytes: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("engine/water-chunks"),
        size: bytes,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
