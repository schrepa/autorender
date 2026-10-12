//! Terrain sky visibility: each terrain's sky map, a grid
//! over the whole terrain holding at every texel the share of the sky its
//! ground sees about its normal, from the horizons of the terrain around it
//! in sixteen directions (`terrain_sky.wgsl`). Terrain scales its ambient
//! light by it, taking the hidden sky's share back as its own light.
//!
//! A map builds from the terrain's finest pinned level once every tile of
//! that level is resident, a band of rows per frame in a submission of its
//! own (shadows or not), and builds again when those tiles or the map's
//! size change; until its first build completes, a terrain's ground sees
//! the whole sky.

use crate::shaders::entry::terrain_sky;
use engine_core::collections::HashMap;

use engine_core::{RenderTerrain, RenderTerrains, TerrainSettings, TileKey};

use crate::bindings::TERRAIN_SKY;
use crate::gpu_calls::Queue;
use crate::gpu_layouts::constants::MISSING;
use crate::gpu_layouts::constants::SKY_MAP_GROUP as GROUP;
use crate::terrain_horizons::{source_level, tile_table};

/// A sky map texel: the visibility in red.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// Texels a frame builds of each terrain's map.
const TEXELS_PER_FRAME: u32 = 65_536;
/// How far horizons are looked for, in meters.
const HORIZON_REACH: f32 = 3_000.0;
/// Uniform offset alignment between terrains' builds.
const SLOT: u64 = 256;

use crate::gpu_layouts::terrain_sky::BuildUniform;

/// What a terrain's layer is built from: its build's shape and counts
/// (without the table's place), and its tile table.
#[derive(Clone, Debug, PartialEq)]
struct Source {
    uniform: BuildUniform,
    table: Vec<u32>,
}

struct Maps {
    view: wgpu::TextureView,
    // Each layer as a build writes it.
    layers: Vec<wgpu::TextureView>,
    size: u32,
}

pub(crate) struct SkyMaps {
    layout: crate::pipelines::GroupLayout,
    pipeline: Option<wgpu::ComputePipeline>,
    maps: Option<Maps>,
    // A one-texel map of open sky bound while no terrain has one.
    empty: wgpu::TextureView,
    uniforms: wgpu::Buffer,
    tables: crate::buffers::Growable,
    // What each layer holds once built, and the build under way with the
    // next row it builds.
    built: Vec<Option<Source>>,
    building: Vec<Option<(Source, u32)>>,
    // Builds each layer finished, which pages of the runtime virtual
    // texture baked from it follow.
    finished: Vec<u64>,
    // Advances when the bound view is replaced.
    generation: u64,
    /// Rows built this frame, over every terrain.
    pub(crate) rows: u32,
}

impl SkyMaps {
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let empty = crate::targets::texture_2d(
            device,
            "engine/terrain-sky-empty",
            (1, 1),
            1,
            1,
            FORMAT,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        queue.write_texture(
            empty.as_image_copy(),
            &[255; 4],
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
        Self {
            layout: crate::pipelines::layout(device, &TERRAIN_SKY),
            pipeline: None,
            maps: None,
            empty: array_view(&empty),
            uniforms: crate::buffers::uniform(device, "engine/terrain-sky-builds", SLOT),
            tables: crate::buffers::Growable::new(
                device,
                "engine/terrain-sky-tables",
                16,
                wgpu::BufferUsages::empty(),
            ),
            built: Vec::new(),
            building: Vec::new(),
            finished: Vec::new(),
            generation: 0,
            rows: 0,
        }
    }

    /// Advances when [`Self::view`] is another view.
    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    /// The maps, one layer per terrain, or a texel of open sky.
    pub(crate) fn view(&self) -> &wgpu::TextureView {
        self.maps.as_ref().map_or(&self.empty, |maps| &maps.view)
    }

    /// The layer plus one of terrain `index`'s map once built, 0 before.
    pub(crate) fn layer(&self, index: usize) -> u32 {
        match self.built.get(index) {
            Some(Some(_)) => index as u32 + 1,
            _ => 0,
        }
    }

    /// Builds terrain `index`'s map finished while none is under way, so
    /// what bakes read of it holds still; `None` while one is; 0 without
    /// maps, which leave open sky everywhere.
    pub(crate) fn settled(&self, index: usize) -> Option<u64> {
        if self.maps.is_none() {
            return Some(0);
        }
        match (self.built.get(index), self.building.get(index)) {
            (Some(Some(_)), Some(None)) => self.finished.get(index).copied(),
            _ => None,
        }
    }

    /// Terrains whose maps are built.
    pub(crate) fn count(&self) -> u32 {
        self.built.iter().flatten().count() as u32
    }

    /// Bytes of the maps.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.maps.as_ref().map_or(0, |maps| {
            u64::from(maps.size) * u64::from(maps.size) * maps.layers.len() as u64 * 4
        })
    }

    /// Plans this frame's builds: every terrain's map whose source changed,
    /// once its source level is wholly resident, with terrain `settings`
    /// asking for sky visibility; `slots` names each resident tile's layer
    /// of the height array. Without it the maps are released.
    /// Creates the pipeline sky maps build with unless it exists.
    pub(crate) fn warm(&mut self, device: &wgpu::Device) {
        self.pipeline
            .get_or_insert_with(|| pipeline(device, &self.layout));
    }

    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        terrains: &RenderTerrains,
        settings: &TerrainSettings,
        slots: &HashMap<(u64, TileKey), u32>,
    ) {
        self.rows = 0;
        if !(settings.enabled && settings.sky_visibility) || terrains.terrains.is_empty() {
            if self.maps.take().is_some() {
                self.generation += 1;
            }
            self.built.clear();
            self.building.clear();
            self.finished.clear();
            return;
        }
        let size = settings.sky_resolution;
        let layers = terrains.terrains.len();
        if self
            .maps
            .as_ref()
            .is_none_or(|maps| maps.size != size || maps.layers.len() != layers)
        {
            self.maps = Some(maps(device, size, layers as u32));
            self.built = vec![None; layers];
            self.building = vec![None; layers];
            self.finished = vec![self.generation + 1; layers];
            self.generation += 1;
        }
        let mut sources = Vec::with_capacity(layers);
        for (index, terrain) in terrains.terrains.iter().enumerate() {
            let source = Source {
                uniform: build_uniform(terrain, settings, size),
                table: tile_table(terrain, settings, slots),
            };
            // Builds start once their level is wholly resident, and again
            // when its tiles change.
            let whole = source
                .table
                .iter()
                .all(|&entry| entry != MISSING && entry >> 16 == 0);
            let current = self.building[index]
                .as_ref()
                .map(|(building, _)| building)
                .or(self.built[index].as_ref());
            if whole && current != Some(&source) {
                self.building[index] = Some((source.clone(), 0));
            }
            sources.push(source);
        }
        if self.building.iter().all(Option::is_none) {
            return;
        }
        self.write_builds(device, queue, &sources, size);
        self.pipeline
            .get_or_insert_with(|| pipeline(device, &self.layout));
    }

    /// Writes the uniforms of this frame's builds, each with its band of
    /// rows, and every terrain's tile table, placing each table's entries.
    fn write_builds(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        sources: &[Source],
        size: u32,
    ) {
        let band = rows_per_frame(size);
        let mut tables: Vec<u32> = Vec::new();
        let mut uniforms = vec![0_u8; sources.len() * SLOT as usize];
        for (index, building) in self.building.iter().enumerate() {
            let Some((source, row)) = building else {
                continue;
            };
            let mut uniform = source.uniform;
            uniform.counts[3] = tables.len() as u32;
            uniform.rows = [*row, band.min(size - row), 0, 0];
            tables.extend_from_slice(&source.table);
            let at = index * SLOT as usize;
            uniforms[at..at + size_of::<BuildUniform>()]
                .copy_from_slice(bytemuck::bytes_of(&uniform));
        }
        if uniforms.len() as u64 > self.uniforms.size() {
            self.uniforms = crate::buffers::uniform(
                device,
                "engine/terrain-sky-builds",
                (uniforms.len() as u64).next_power_of_two(),
            );
        }
        queue.write_buffer(&self.uniforms, 0, &uniforms);
        let bytes = (tables.len() * 4) as u64;
        self.tables.grow(device, bytes);
        if !tables.is_empty() {
            queue.write_buffer(&self.tables, 0, bytemuck::cast_slice(&tables));
        }
    }

    /// Builds this frame's bands from the height array `heights` and
    /// submits them; a map whose last band this is counts as built.
    pub(crate) fn build(
        &mut self,
        device: &wgpu::Device,
        queue: &Queue,
        heights: Option<&wgpu::TextureView>,
    ) {
        let (Some(maps), Some(pipeline), Some(heights)) = (&self.maps, &self.pipeline, heights)
        else {
            return;
        };
        if self.building.iter().all(Option::is_none) {
            return;
        }
        let band = rows_per_frame(maps.size);
        let groups: Vec<(usize, wgpu::BindGroup)> = self
            .building
            .iter()
            .enumerate()
            .filter(|(_, building)| building.is_some())
            .map(|(layer, _)| {
                let group = crate::pipelines::group(
                    device,
                    "engine/terrain-sky-group",
                    &TERRAIN_SKY,
                    &self.layout,
                    &[
                        ("heights", crate::pipelines::view(heights)),
                        crate::buffers::entry("tile_layers", &self.tables),
                        (
                            "build",
                            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &self.uniforms,
                                offset: layer as u64 * SLOT,
                                size: wgpu::BufferSize::new(size_of::<BuildUniform>() as u64),
                            }),
                        ),
                        ("sky", crate::pipelines::view(&maps.layers[layer])),
                    ],
                );
                (layer, group)
            })
            .collect();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("engine/terrain-sky"),
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("engine/terrain-sky"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        for (_, group) in &groups {
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(maps.size.div_ceil(GROUP), band.div_ceil(GROUP), 1);
        }
        drop(pass);
        queue.submit_now([encoder.finish()]);
        for (layer, _) in groups {
            let Some((source, row)) = self.building[layer].take() else {
                continue;
            };
            let next = row + band.min(maps.size - row);
            self.rows += next - row;
            if next >= maps.size {
                self.built[layer] = Some(source);
                self.finished[layer] += 1;
            } else {
                self.building[layer] = Some((source, next));
            }
        }
    }
}

/// Rows of a map `size` texels wide that a frame builds.
fn rows_per_frame(size: u32) -> u32 {
    (TEXELS_PER_FRAME / size).max(GROUP)
}

/// `terrain`'s build over a grid of `size` texels a side covering it.
fn build_uniform(terrain: &RenderTerrain, settings: &TerrainSettings, size: u32) -> BuildUniform {
    let level = source_level(terrain, settings);
    let pyramid = &terrain.pyramid;
    let extent = pyramid.extent();
    let [tiles_x, tiles_z] = pyramid.tiles(level);
    BuildUniform {
        shape: [extent.x, extent.y, pyramid.spacing(level), HORIZON_REACH],
        counts: [size, tiles_x, tiles_z, 0],
        rows: [0; 4],
    }
}

/// The maps: `layers` layers of `size` texels a side, viewed whole and by
/// layer.
fn maps(device: &wgpu::Device, size: u32, layers: u32) -> Maps {
    let texture = crate::targets::texture_2d(
        device,
        "engine/terrain-sky",
        (size, size),
        layers,
        1,
        FORMAT,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
    );
    Maps {
        view: array_view(&texture),
        layers: (0..layers)
            .map(|layer| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("engine/terrain-sky-layer"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect(),
        size,
    }
}

fn array_view(texture: &wgpu::Texture) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("engine/terrain-sky"),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    })
}

fn pipeline(
    device: &wgpu::Device,
    layout: &crate::pipelines::GroupLayout,
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("engine/terrain-sky.wgsl"),
        source: wgpu::ShaderSource::Wgsl(crate::gpu_data::TERRAIN_SKY_SHADER.into()),
    });
    let pipeline_layout = crate::pipelines::pipeline_layout(
        device,
        &crate::pipeline_layouts::TERRAIN_SKY_PIPELINES,
        &[layout],
    );
    crate::pipelines::compute_entry(device, &pipeline_layout, &module, terrain_sky::build_sky)
}
