//! Terrain's long sun shadows: each terrain's horizon map, a
//! grid aligned with the sun over the terrain holding at every texel how far
//! above its ground the sun clears the terrain toward it, how far along the
//! sun the slope setting that lies, and the ground's height
//! (`terrain_horizon.wgsl`). A layer is rebuilt when the sun, the terrain or
//! the tiles of its finest pinned level it reads change; every lit surface
//! compares its height above that ground with it, terrain's own surfaces
//! lying on it, softening with that distance (`scene.wgsl`). Two texels per
//! terrain place its map for those lookups.

use std::collections::HashMap;

use engine_core::glam::{Vec2, Vec3};
use engine_core::{RenderTerrain, RenderTerrains, TerrainSettings, TileKey};

use crate::bindings::TERRAIN_HORIZON;
use crate::gpu_layouts::constants::HORIZON_GROUP as GROUP;
use crate::gpu_layouts::constants::MISSING;

/// A horizon map texel: the horizon above the ground, its distance and the
/// ground's height, filtered.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// The records' texels: two per terrain.
const RECORD_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;
/// Uniform offset alignment between terrains' builds.
const SLOT: u64 = 256;
/// The steepest rise toward the sun builds sweep with, near the zenith.
const MOST_RISE: f32 = 1.0e3;
/// Texels of the horizon map within which terrain still draws into the
/// near cascades, which shadow fine detail its texels miss.
const CASCADE_TEXELS: f32 = 4.0;

use crate::gpu_layouts::terrain_horizons::BuildUniform;

/// What a terrain's layer was built from: its build's uniform (whose last
/// word, the table's place, is ignored) and its tile table.
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

pub(crate) struct Horizons {
    layout: wgpu::BindGroupLayout,
    pipeline: Option<wgpu::ComputePipeline>,
    maps: Option<Maps>,
    // A one-texel map bound while no terrain casts, and the maps' sampler.
    empty: wgpu::TextureView,
    sampler: wgpu::Sampler,
    records: wgpu::Texture,
    records_view: wgpu::TextureView,
    records_written: Vec<[f32; 4]>,
    // Advances when the bound views are replaced.
    generation: u64,
    uniforms: wgpu::Buffer,
    tables: wgpu::Buffer,
    // What each layer holds, and the layers this frame builds with what
    // they will then hold: a layer counts as built once its build is
    // recorded.
    built: Vec<Option<Source>>,
    pending: Vec<(usize, Source)>,
    // Each terrain's reach of its near cascade casters, by terrain ID.
    reaches: HashMap<u64, f32>,
    /// Layers built this frame.
    pub(crate) builds: u32,
}

impl Horizons {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let empty = device
            .create_texture(&texture_descriptor(
                "engine/terrain-horizons-empty",
                FORMAT,
                1,
                1,
                wgpu::TextureUsages::TEXTURE_BINDING,
            ))
            .create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            });
        let (records, records_view) = records_texture(device, 1);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("engine/terrain-horizon-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            layout: crate::pipelines::layout(device, &TERRAIN_HORIZON),
            pipeline: None,
            maps: None,
            empty,
            sampler,
            records,
            records_view,
            records_written: Vec::new(),
            generation: 0,
            uniforms: crate::buffers::uniform(device, "engine/terrain-horizon-builds", SLOT),
            tables: crate::buffers::storage(
                device,
                "engine/terrain-horizon-tables",
                16,
                wgpu::BufferUsages::empty(),
            ),
            built: Vec::new(),
            pending: Vec::new(),
            reaches: HashMap::new(),
            builds: 0,
        }
    }

    /// Advances when [`Self::entries`] binds other views.
    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    /// The main passes' bindings (`bindings::CAMERA`): the maps, their
    /// records and their sampler.
    pub(crate) fn entries(&self) -> [(&'static str, wgpu::BindingResource<'_>); 3] {
        let maps = self.maps.as_ref().map_or(&self.empty, |maps| &maps.view);
        [
            ("terrain_horizons", crate::pipelines::view(maps)),
            (
                "terrain_horizon_records",
                crate::pipelines::view(&self.records_view),
            ),
            (
                "terrain_horizon_sampler",
                crate::pipelines::sampler(&self.sampler),
            ),
        ]
    }

    /// Terrains the records texture holds (two texels each), which the
    /// main passes' lookups loop over.
    pub(crate) fn record_count(&self) -> u32 {
        self.records.width() / 2
    }

    /// Terrains casting long shadows.
    pub(crate) fn count(&self) -> u32 {
        self.maps
            .as_ref()
            .map_or(0, |_| self.built.iter().flatten().count() as u32)
    }

    /// Bytes of the maps.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.maps.as_ref().map_or(0, |maps| {
            u64::from(maps.size) * u64::from(maps.size) * maps.layers.len() as u64 * 8
        })
    }

    /// How far toward the sun of a near cascade's receivers terrain `id`
    /// still casts into it, when its horizon map shadows the rest.
    pub(crate) fn cascade_reach(&self, id: u64) -> Option<f32> {
        self.reaches.get(&id).copied()
    }

    /// Plans this frame's builds: every terrain's layer whose sun, terrain
    /// or source tiles changed, with shadows on and terrain `settings`
    /// casting; `sun` points toward the sun and `slots` names each resident
    /// tile's layer of the height array. Without casters the maps are
    /// released.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        terrains: &RenderTerrains,
        (settings, shadows): (&TerrainSettings, bool),
        sun: Vec3,
        slots: &HashMap<(u64, TileKey), u32>,
    ) {
        self.builds = 0;
        self.pending.clear();
        self.reaches.clear();
        if !(shadows && settings.enabled && settings.sun_shadows) || terrains.terrains.is_empty() {
            self.release(device);
            return;
        }
        let size = settings.shadow_resolution;
        let layers = terrains.terrains.len();
        if self
            .maps
            .as_ref()
            .is_none_or(|maps| maps.size != size || maps.layers.len() != layers)
        {
            self.maps = Some(maps(device, size, layers as u32));
            self.built = vec![None; layers];
            self.generation += 1;
        }
        let mut sources = Vec::with_capacity(layers);
        let mut records = Vec::with_capacity(2 * layers);
        for terrain in &terrains.terrains {
            let (uniform, half) = build_uniform(terrain, settings, sun, size);
            let table = tile_table(terrain, settings, slots);
            records.push([
                uniform.place[0],
                uniform.place[1],
                uniform.place[2],
                uniform.place[3],
            ]);
            records.push([
                half.x,
                half.y,
                settings.shadow_softness,
                bias(terrain, settings),
            ]);
            self.reaches
                .insert(terrain.id, CASCADE_TEXELS * 2.0 * half.x / size as f32);
            sources.push(Source { uniform, table });
        }
        for (layer, source) in sources.iter().enumerate() {
            if self.built[layer].as_ref() != Some(source) {
                self.pending.push((layer, source.clone()));
            }
        }
        if !self.pending.is_empty() {
            self.write_builds(device, queue, &sources);
            self.pipeline
                .get_or_insert_with(|| pipeline(device, &self.layout));
        }
        if records != self.records_written {
            if records.len() != self.records_written.len() {
                (self.records, self.records_view) = records_texture(device, records.len() as u32);
                self.generation += 1;
            }
            queue.write_texture(
                self.records.as_image_copy(),
                bytemuck::cast_slice(&records),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(records.len() as u32 * 16),
                    rows_per_image: Some(1),
                },
                wgpu::Extent3d {
                    width: records.len() as u32,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
            self.records_written = records;
        }
    }

    /// Writes the uniforms and tile tables of `sources`, placing each
    /// table's entries.
    fn write_builds(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, sources: &[Source]) {
        let mut tables: Vec<u32> = Vec::new();
        let mut uniforms = vec![0_u8; sources.len() * SLOT as usize];
        for (index, source) in sources.iter().enumerate() {
            let mut uniform = source.uniform;
            uniform.counts[3] = tables.len() as u32;
            tables.extend_from_slice(&source.table);
            let at = index * SLOT as usize;
            uniforms[at..at + size_of::<BuildUniform>()]
                .copy_from_slice(bytemuck::bytes_of(&uniform));
        }
        if uniforms.len() as u64 > self.uniforms.size() {
            self.uniforms = crate::buffers::uniform(
                device,
                "engine/terrain-horizon-builds",
                (uniforms.len() as u64).next_power_of_two(),
            );
        }
        queue.write_buffer(&self.uniforms, 0, &uniforms);
        let bytes = (tables.len() * 4) as u64;
        if bytes > self.tables.size() {
            self.tables = crate::buffers::storage(
                device,
                "engine/terrain-horizon-tables",
                bytes.next_power_of_two(),
                wgpu::BufferUsages::empty(),
            );
        }
        if !tables.is_empty() {
            queue.write_buffer(&self.tables, 0, bytemuck::cast_slice(&tables));
        }
    }

    /// Records this frame's builds from the height array `heights`, timed
    /// by `profiler` (`terrain_horizons`).
    pub(crate) fn encode(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        heights: Option<&wgpu::TextureView>,
        profiler: &crate::profiling::GpuProfiler,
    ) {
        let (Some(maps), Some(pipeline), Some(heights)) = (&self.maps, &self.pipeline, heights)
        else {
            return;
        };
        if self.pending.is_empty() {
            return;
        }
        let groups: Vec<wgpu::BindGroup> = self
            .pending
            .iter()
            .map(|&(layer, _)| {
                crate::pipelines::group(
                    device,
                    "engine/terrain-horizon-group",
                    &TERRAIN_HORIZON,
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
                        ("horizon", crate::pipelines::view(&maps.layers[layer])),
                    ],
                )
            })
            .collect();
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("engine/terrain-horizons"),
            timestamp_writes: profiler.span("terrain_horizons").map(|span| span.compute()),
        });
        pass.set_pipeline(pipeline);
        for group in &groups {
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(maps.size.div_ceil(GROUP), 1, 1);
        }
        self.builds = groups.len() as u32;
        for (layer, source) in self.pending.drain(..) {
            self.built[layer] = Some(source);
        }
    }

    /// Drops the maps, binding the stand-ins.
    fn release(&mut self, device: &wgpu::Device) {
        if self.maps.take().is_some() {
            self.generation += 1;
        }
        self.built.clear();
        if !self.records_written.is_empty() {
            (self.records, self.records_view) = records_texture(device, 1);
            self.records_written.clear();
            self.generation += 1;
        }
    }
}

/// `terrain`'s build over a grid of `size` texels a side aligned with `sun`
/// (toward the sun), and the grid's half extents along and across the sun.
fn build_uniform(
    terrain: &RenderTerrain,
    settings: &TerrainSettings,
    sun: Vec3,
    size: u32,
) -> (BuildUniform, Vec2) {
    let level = source_level(terrain, settings);
    let pyramid = &terrain.pyramid;
    let flat = Vec2::new(sun.x, sun.z);
    let length = flat.length();
    let toward = if length > 1.0e-6 {
        flat / length
    } else {
        Vec2::X
    };
    let rise = engine_core::math::min(sun.y / engine_core::math::max(length, 1.0e-6), MOST_RISE);
    let extent = pyramid.extent();
    // The terrain's rectangle seen along and across the sun, a texel wider
    // on each side so its edges are inside.
    let half = Vec2::new(
        toward.x.abs() * extent.x + toward.y.abs() * extent.y,
        toward.y.abs() * extent.x + toward.x.abs() * extent.y,
    ) * (0.5 * (size as f32 + 2.0) / size as f32);
    let center = Vec2::new(terrain.origin.x, terrain.origin.z) + extent * 0.5;
    let [tiles_x, tiles_z] = pyramid.tiles(level);
    (
        BuildUniform {
            place: [center.x, center.y, toward.x, toward.y],
            shape: [half.x, half.y, rise, terrain.origin.y],
            origin: [
                terrain.origin.x,
                terrain.origin.z,
                pyramid.spacing(level),
                0.0,
            ],
            counts: [size, tiles_x, tiles_z, 0],
        },
        half,
    )
}

/// The level horizon maps read: the finest of the levels kept resident
/// over the whole terrain.
pub(crate) fn source_level(terrain: &RenderTerrain, settings: &TerrainSettings) -> u32 {
    let top = terrain.pyramid.levels() - 1;
    top.saturating_sub(settings.pinned_levels.saturating_sub(1))
}

/// Meters receivers rise before they compare with the horizon: a tenth of
/// the source level's spacing, which slopes the sun grazes stay lit within.
fn bias(terrain: &RenderTerrain, settings: &TerrainSettings) -> f32 {
    0.1 * terrain.pyramid.spacing(source_level(terrain, settings))
}

/// Each source tile's entry (`terrain_horizon.wgsl`): the layer of the tile
/// or of its nearest resident ancestor, and how many levels up that is.
pub(crate) fn tile_table(
    terrain: &RenderTerrain,
    settings: &TerrainSettings,
    slots: &HashMap<(u64, TileKey), u32>,
) -> Vec<u32> {
    let level = source_level(terrain, settings);
    let pyramid = &terrain.pyramid;
    let [tiles_x, tiles_z] = pyramid.tiles(level);
    let mut table = Vec::with_capacity((tiles_x * tiles_z) as usize);
    for z in 0..tiles_z {
        for x in 0..tiles_x {
            let mut key = Some(TileKey::new(level as u8, x as u16, z as u16));
            let mut entry = MISSING;
            for coarser in 0_u32.. {
                let Some(current) = key else {
                    break;
                };
                if let Some(&layer) = slots.get(&(terrain.id, current)) {
                    entry = layer | (coarser << 16);
                    break;
                }
                key = pyramid.parent(current);
            }
            table.push(entry);
        }
    }
    table
}

fn texture_descriptor(
    label: &'static str,
    format: wgpu::TextureFormat,
    size: u32,
    layers: u32,
    usage: wgpu::TextureUsages,
) -> wgpu::TextureDescriptor<'static> {
    wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: layers,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    }
}

/// Horizon maps of `layers` terrains, `size` texels a side.
fn maps(device: &wgpu::Device, size: u32, layers: u32) -> Maps {
    let texture = device.create_texture(&texture_descriptor(
        "engine/terrain-horizons",
        FORMAT,
        size,
        layers,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
    ));
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    let layers = (0..layers)
        .map(|layer| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: layer,
                array_layer_count: Some(1),
                ..Default::default()
            })
        })
        .collect();
    Maps { view, layers, size }
}

/// The records texture of `texels` texels (one stands in without terrain).
fn records_texture(device: &wgpu::Device, texels: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("engine/terrain-horizon-records"),
        size: wgpu::Extent3d {
            width: texels.max(1),
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: RECORD_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn pipeline(device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("engine/terrain-horizon.wgsl"),
        source: wgpu::ShaderSource::Wgsl(crate::gpu_data::TERRAIN_HORIZON_SHADER.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("engine/terrain-horizon-layout"),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    crate::pipelines::compute_entry(device, &pipeline_layout, &module, "build_horizon")
}
