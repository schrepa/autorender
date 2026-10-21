//! Terrain's long sun shadows: each terrain's horizon map, cooked with its
//! lighting ([`engine_core::TerrainLighting`]): a grid over the terrain
//! holding at every texel, in sixteen directions around, how high above its
//! ground and how far away the crest of the highest ground ahead lies, and
//! the ground's height. Every lit surface finds how far the sun's ray toward
//! it clears those crests, between the directions either side of the sun,
//! and compares its own height above the ground with it, softening with the
//! crest's distance (`scene.wgsl`): the sun moves and nothing builds. A
//! terrain's layers upload as its lighting arrives and again when it
//! changes; two texels per terrain place its map for those lookups.

use std::sync::Arc;

use engine_core::collections::HashMap;
use engine_core::terrain_data::lighting::HORIZON_LAYERS;
use engine_core::{RenderTerrain, RenderTerrains, TerrainLighting, TerrainSettings};

use crate::cached::{Bound, Versioned};
use crate::gpu_layouts::constants::HORIZON_RECORD_TEXELS;

/// A horizon map texel: two directions' crest heights and distances, or the
/// ground's height, filtered.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// The records' texels.
const RECORD_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;
/// Texels of the horizon map within which terrain still draws into the
/// near cascades, which shadow fine detail its texels miss.
const CASCADE_TEXELS: f32 = 4.0;

struct Maps {
    texture: wgpu::Texture,
    view: Versioned<wgpu::TextureView>,
    size: u32,
    terrains: usize,
}

pub(crate) struct Horizons {
    maps: Option<Maps>,
    // A one-texel map bound while no terrain casts, and the maps' sampler.
    empty: wgpu::TextureView,
    sampler: wgpu::Sampler,
    records: wgpu::Texture,
    records_view: Versioned<wgpu::TextureView>,
    records_written: Vec<[f32; 4]>,
    // The lighting each terrain's layers hold.
    held: Vec<Option<Arc<TerrainLighting>>>,
    // Each terrain's reach of its near cascade casters, by terrain ID.
    reaches: HashMap<u64, f32>,
    /// Terrains whose maps uploaded this frame.
    pub(crate) uploaded: u32,
}

impl Horizons {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let empty = crate::targets::texture_2d(
            device,
            "engine/terrain-horizons-empty",
            (1, 1),
            1,
            1,
            FORMAT,
            wgpu::TextureUsages::TEXTURE_BINDING,
        )
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
            maps: None,
            empty,
            sampler,
            records,
            records_view,
            records_written: Vec::new(),
            held: Vec::new(),
            reaches: HashMap::default(),
            uploaded: 0,
        }
    }

    /// The main passes' bindings (`bindings::CAMERA`): the maps (the empty
    /// stand-in without them), their records and their sampler.
    pub(crate) fn entries(&self) -> [(&'static str, Bound<'_>); 3] {
        let maps = self.maps.as_ref().map_or_else(
            || Bound::fixed(crate::pipelines::view(&self.empty)),
            |maps| maps.view.bound(),
        );
        [
            ("terrain_horizons", maps),
            ("terrain_horizon_records", self.records_view.bound()),
            (
                "terrain_horizon_sampler",
                Bound::fixed(crate::pipelines::sampler(&self.sampler)),
            ),
        ]
    }

    /// Terrains the records texture holds, which the main passes' lookups
    /// loop over.
    pub(crate) fn record_count(&self) -> u32 {
        self.records.width() / HORIZON_RECORD_TEXELS
    }

    /// Terrains casting long shadows.
    pub(crate) fn count(&self) -> u32 {
        self.held.iter().flatten().count() as u32
    }

    /// Bytes of the maps.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.maps.as_ref().map_or(0, |maps| {
            u64::from(maps.size)
                * u64::from(maps.size)
                * (HORIZON_LAYERS * maps.terrains) as u64
                * 8
        })
    }

    /// How far toward the sun of a near cascade's receivers terrain `id`
    /// still casts into it, when its horizon map shadows the rest.
    pub(crate) fn cascade_reach(&self, id: u64) -> Option<f32> {
        self.reaches.get(&id).copied()
    }

    /// Uploads the maps of every terrain whose lighting arrived or changed
    /// since and writes their records, with shadows on and terrain
    /// `settings` casting at their resolution. Without casters the maps
    /// are released.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        terrains: &RenderTerrains,
        (settings, shadows): (&TerrainSettings, bool),
    ) {
        self.uploaded = 0;
        self.reaches.clear();
        if !(shadows && settings.enabled && settings.sun_shadows) || terrains.terrains.is_empty() {
            self.release(device);
            return;
        }
        let size = settings.shadow_resolution;
        let count = terrains.terrains.len();
        if self
            .maps
            .as_ref()
            .is_none_or(|maps| maps.size != size || maps.terrains != count)
        {
            self.maps = Some(maps(device, size, count));
            self.held = vec![None; count];
        }
        let Some(maps) = &self.maps else {
            return;
        };
        let mut records = Vec::with_capacity(HORIZON_RECORD_TEXELS as usize * count);
        for (index, terrain) in terrains.terrains.iter().enumerate() {
            let lighting = terrain
                .lighting
                .as_ref()
                .filter(|lighting| lighting.horizon_size == size);
            let held = &mut self.held[index];
            match (lighting, &*held) {
                (Some(lighting), Some(current)) if Arc::ptr_eq(lighting, current) => {}
                (Some(lighting), _) => {
                    write_layers(
                        queue,
                        &maps.texture,
                        index,
                        size,
                        &lighting.horizon_layers(),
                    );
                    *held = Some(lighting.clone());
                    self.uploaded += 1;
                }
                (None, _) => *held = None,
            }
            match held {
                Some(lighting) => {
                    records.extend(record(terrain, lighting, index, settings));
                    let texel = max_extent(terrain) / size as f32;
                    self.reaches.insert(terrain.id, CASCADE_TEXELS * texel);
                }
                // Nothing reads a terrain without its map: its place
                // covers no ground.
                None => records.extend([[0.0; 4]; HORIZON_RECORD_TEXELS as usize]),
            }
        }
        if records != self.records_written {
            if records.len() != self.records_written.len() {
                (self.records, self.records_view) = records_texture(device, records.len() as u32);
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

    /// Drops the maps, binding the stand-ins.
    fn release(&mut self, device: &wgpu::Device) {
        self.maps = None;
        self.held.clear();
        if !self.records_written.is_empty() {
            (self.records, self.records_view) = records_texture(device, 1);
            self.records_written.clear();
        }
    }
}

/// The larger side of `terrain`'s extent, in meters.
fn max_extent(terrain: &RenderTerrain) -> f32 {
    let extent = terrain.pyramid.extent();
    engine_core::math::max(extent.x, extent.y)
}

/// Terrain `index`'s record (`scene.wgsl`): its map's first corner (world x
/// and z) and one over its extent; its first layer, how fast its shadows
/// soften per meter to their crests, the meters receivers rise before they
/// compare (a tenth of a texel, within which slopes the sun grazes stay
/// lit) and the world height its ground's heights count from.
fn record(
    terrain: &RenderTerrain,
    lighting: &TerrainLighting,
    index: usize,
    settings: &TerrainSettings,
) -> [[f32; 4]; 2] {
    let extent = terrain.pyramid.extent();
    let texel = max_extent(terrain) / lighting.horizon_size as f32;
    [
        [
            terrain.origin.x,
            terrain.origin.z,
            1.0 / extent.x,
            1.0 / extent.y,
        ],
        [
            (HORIZON_LAYERS * index) as f32,
            settings.shadow_softness,
            0.1 * texel,
            terrain.origin.y + lighting.lowest,
        ],
    ]
}

/// Writes terrain `index`'s layers of `texture`, `size` texels a side, from
/// `texels`.
fn write_layers(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    index: usize,
    size: u32,
    texels: &[u8],
) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: 0,
                y: 0,
                z: (HORIZON_LAYERS * index) as u32,
            },
            aspect: wgpu::TextureAspect::All,
        },
        texels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(size * 8),
            rows_per_image: Some(size),
        },
        wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: HORIZON_LAYERS as u32,
        },
    );
}

/// Horizon maps of `terrains` terrains, `size` texels a side.
fn maps(device: &wgpu::Device, size: u32, terrains: usize) -> Maps {
    let texture = crate::targets::texture_2d(
        device,
        "engine/terrain-horizons",
        (size, size),
        (HORIZON_LAYERS * terrains) as u32,
        1,
        FORMAT,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    Maps {
        texture,
        view: Versioned::new(view),
        size,
        terrains,
    }
}

/// The records texture of `texels` texels (one stands in without terrain).
fn records_texture(
    device: &wgpu::Device,
    texels: u32,
) -> (wgpu::Texture, Versioned<wgpu::TextureView>) {
    let texture = crate::targets::texture_2d(
        device,
        "engine/terrain-horizon-records",
        (texels.max(1), 1),
        1,
        1,
        RECORD_FORMAT,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, Versioned::new(view))
}
