//! Terrain's long sun shadows: each terrain's horizon map, cooked with its
//! lighting ([`engine_core::TerrainLighting`]): a grid over the terrain
//! holding at every texel, in sixteen directions around, how high above its
//! ground and how far away the crest of the highest ground ahead lies, and
//! the ground's height. Every lit surface finds how far the sun's ray toward
//! it clears those crests, between the directions either side of the sun,
//! and compares its own height above the ground with it, softening with the
//! crest's distance (`scene.wgsl`): the sun moves and nothing builds. A
//! terrain's layers upload as its lighting arrives and again when it
//! changes; a record of four vectors per terrain, in a uniform the lookups
//! loop over, places its map. Terrains past what the device's array layers
//! or the records hold cast no long shadows (`room`).
//!
//! With bounce light on, the maps hold after every terrain's horizon layers
//! the bounce layers of each terrain it relights, by slot
//! (`crate::terrain_bounce`: surroundings hold none), which its probes
//! around the camera write every frame; the records place them too.

use std::sync::Arc;

use engine_core::terrain_data::lighting::HORIZON_LAYERS;
use engine_core::{RenderTerrain, RenderTerrains, TerrainLighting, TerrainSettings};

use crate::cached::{Bound, Versioned};
use crate::gpu_layouts::constants::{HORIZON_RECORD_VECTORS, HORIZON_RECORDS};
use crate::terrain_bounce::BOUNCE_LAYERS;
use crate::terrain_bounce::BounceWindow;

/// A horizon map texel: two directions' crest heights and distances, or the
/// ground's height, filtered; or bounce light's coefficients.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// A terrain's record without a map: its place covers no ground and it
/// holds no bounce light.
const NO_RECORD: [[f32; 4]; HORIZON_RECORD_VECTORS as usize] =
    [[0.0; 4], [0.0; 4], [-1.0, 0.0, 0.0, 0.0], [0.0; 4]];

struct Maps {
    texture: wgpu::Texture,
    // Every layer, as the main passes read them; the horizon layers alone,
    // as bounce light reads them; and the bounce layers, as it writes them:
    // disjoint layers, so one dispatch reads the first and writes the other.
    view: Versioned<wgpu::TextureView>,
    horizon_view: Versioned<wgpu::TextureView>,
    bounce_view: Option<Versioned<wgpu::TextureView>>,
    size: u32,
    terrains: usize,
    slots: u32,
}

pub(crate) struct Horizons {
    maps: Option<Maps>,
    // A one-texel map bound while no terrain casts, and the maps' sampler.
    empty: wgpu::TextureView,
    sampler: wgpu::Sampler,
    // Every casting terrain's record, in a uniform holding the most there
    // may be, and what it holds.
    records: wgpu::Buffer,
    records_written: Vec<[f32; 4]>,
    // The lighting each terrain's layers hold.
    held: Vec<Option<Arc<TerrainLighting>>>,
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
            records: crate::buffers::uniform(
                device,
                "engine/terrain-horizon-records",
                crate::bindings::HORIZON_RECORDS_BYTES as u64,
            ),
            records_written: Vec::new(),
            held: Vec::new(),
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
            (
                "terrain_horizon_records",
                Bound::fixed(self.records.as_entire_binding()),
            ),
            (
                "terrain_horizon_sampler",
                Bound::fixed(crate::pipelines::sampler(&self.sampler)),
            ),
        ]
    }

    /// Bounce light's bindings (`bindings::BOUNCE`) while the maps hold its
    /// layers: the horizon layers alone with the records and sampler, as
    /// the sun's lookups read them, and the bounce layers to write.
    pub(crate) fn bounce_entries(&self) -> Option<([(&'static str, Bound<'_>); 3], Bound<'_>)> {
        let maps = self.maps.as_ref()?;
        let written = maps.bounce_view.as_ref()?.bound();
        Some((
            [
                ("terrain_horizons", maps.horizon_view.bound()),
                (
                    "terrain_horizon_records",
                    Bound::fixed(self.records.as_entire_binding()),
                ),
                (
                    "terrain_horizon_sampler",
                    Bound::fixed(crate::pipelines::sampler(&self.sampler)),
                ),
            ],
            written,
        ))
    }

    /// Whether the maps hold bounce layers.
    pub(crate) fn bounces(&self) -> bool {
        self.maps
            .as_ref()
            .is_some_and(|maps| maps.bounce_view.is_some())
    }

    /// Whether terrain `index`'s map is here.
    pub(crate) fn held(&self, index: usize) -> bool {
        self.held.get(index).is_some_and(Option::is_some)
    }

    /// Terrains the records hold, which the main passes' lookups loop
    /// over.
    pub(crate) fn record_count(&self) -> u32 {
        (self.records_written.len() / HORIZON_RECORD_VECTORS as usize) as u32
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
                * u64::from(maps.texture.depth_or_array_layers())
                * 8
        })
    }

    /// Uploads the maps of every terrain whose lighting arrived or changed
    /// since and writes their records, with shadows on and terrain
    /// `settings` casting at their resolution, each placing the window of
    /// `windows` (by terrain index) its bounce light relights there, the
    /// maps holding the bounce layers of `slots` terrains. Without casters
    /// the maps are released.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        terrains: &RenderTerrains,
        (settings, shadows): (&TerrainSettings, bool),
        (windows, slots): (&[Option<BounceWindow>], u32),
    ) {
        self.uploaded = 0;
        if !(shadows && settings.enabled && settings.sun_shadows) || terrains.terrains.is_empty() {
            self.release();
            return;
        }
        let size = settings.shadow_resolution;
        // Terrains past the device's array layers and the records cast none.
        let (count, _) = room(device, terrains.terrains.len());
        if self
            .maps
            .as_ref()
            .is_none_or(|maps| maps.size != size || maps.terrains != count || maps.slots != slots)
        {
            self.maps = Some(maps(device, size, count, slots));
            self.held = vec![None; count];
        }
        let Some(maps) = &self.maps else {
            return;
        };
        let mut records = Vec::with_capacity(HORIZON_RECORD_VECTORS as usize * count);
        for (index, terrain) in terrains.terrains.iter().take(count).enumerate() {
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
                    // Bounce light where its layers are and it relights.
                    let window = windows
                        .get(index)
                        .copied()
                        .flatten()
                        .filter(|_| maps.bounce_view.is_some());
                    let place = Place {
                        index,
                        count,
                        size,
                        window,
                    };
                    records.extend(record(terrain, lighting, place, settings));
                }
                // Nothing reads a terrain without its map: its place
                // covers no ground.
                None => records.extend(NO_RECORD),
            }
        }
        if records != self.records_written {
            queue.write_buffer(&self.records, 0, bytemuck::cast_slice(&records));
            self.records_written = records;
        }
    }

    /// Drops the maps, binding the stand-in; the lookups read no record.
    fn release(&mut self) {
        self.maps = None;
        self.held.clear();
        self.records_written.clear();
    }
}

/// What the horizon maps hold on `device` for `terrains` terrains: the
/// terrains casting long shadows, the first as many as the device's array
/// layers and the records hold; and the slots of bounce layers the array's
/// layers hold after theirs, which bounce light gives the terrains it
/// relights among those (`crate::terrain_bounce`).
pub(crate) fn room(device: &wgpu::Device, terrains: usize) -> (usize, u32) {
    let layers = device.limits().max_texture_array_layers as usize;
    let casting = terrains
        .min(layers / HORIZON_LAYERS)
        .min(HORIZON_RECORDS / HORIZON_RECORD_VECTORS as usize);
    let slots = (layers - HORIZON_LAYERS * casting) / BOUNCE_LAYERS;
    (casting, slots as u32)
}

/// The larger side of `terrain`'s extent, in meters.
fn max_extent(terrain: &RenderTerrain) -> f32 {
    let extent = terrain.pyramid.extent();
    engine_core::math::max(extent.x, extent.y)
}

/// Where a terrain's record places its layers: its index among `count`
/// terrains whose maps are `size` texels a side, and the window its bounce
/// light relights in its slot's layers, if it does.
struct Place {
    index: usize,
    count: usize,
    size: u32,
    window: Option<BounceWindow>,
}

/// A terrain's record (`scene.wgsl`): its map's first corner (world x and
/// z) and one over its extent; its first layer, how fast its shadows soften
/// per meter to their crests, the meters receivers rise before they compare
/// (a tenth of a texel, within which slopes the sun grazes stay lit) and
/// the world height its ground's heights count from; its bounce layers'
/// first (-1 without bounce light), its probe columns along a side and the
/// first column (x and z) of its window; and the window's columns along a
/// side and one over the maps' side.
fn record(
    terrain: &RenderTerrain,
    lighting: &TerrainLighting,
    place: Place,
    settings: &TerrainSettings,
) -> [[f32; 4]; HORIZON_RECORD_VECTORS as usize] {
    let extent = terrain.pyramid.extent();
    let texel = max_extent(terrain) / lighting.horizon_size as f32;
    let (bounce, window) = match place.window {
        Some(window) => (
            [
                (HORIZON_LAYERS * place.count + BOUNCE_LAYERS * window.slot as usize) as f32,
                window.probes as f32,
                window.first[0] as f32,
                window.first[1] as f32,
            ],
            [window.columns as f32, 1.0 / place.size as f32, 0.0, 0.0],
        ),
        None => (NO_RECORD[2], NO_RECORD[3]),
    };
    [
        [
            terrain.origin.x,
            terrain.origin.z,
            1.0 / extent.x,
            1.0 / extent.y,
        ],
        [
            (HORIZON_LAYERS * place.index) as f32,
            settings.shadow_softness,
            0.1 * texel,
            terrain.origin.y + lighting.lowest,
        ],
        bounce,
        window,
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

/// Horizon maps of `terrains` terrains, `size` texels a side, with the
/// bounce layers of `slots` of them after every terrain's horizon layers.
fn maps(device: &wgpu::Device, size: u32, terrains: usize, slots: u32) -> Maps {
    let horizon_layers = (HORIZON_LAYERS * terrains) as u32;
    let bounce_layers = BOUNCE_LAYERS as u32 * slots;
    let bounce = bounce_layers > 0;
    let mut usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
    if bounce {
        usage |= wgpu::TextureUsages::STORAGE_BINDING;
    }
    let texture = crate::targets::texture_2d(
        device,
        "engine/terrain-horizons",
        (size, size),
        horizon_layers + bounce_layers,
        1,
        FORMAT,
        usage,
    );
    let layers = |first: u32, count: u32| wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        base_array_layer: first,
        array_layer_count: Some(count),
        ..Default::default()
    };
    Maps {
        view: Versioned::new(texture.create_view(&layers(0, horizon_layers + bounce_layers))),
        horizon_view: Versioned::new(texture.create_view(&layers(0, horizon_layers))),
        bounce_view: bounce
            .then(|| Versioned::new(texture.create_view(&layers(horizon_layers, bounce_layers)))),
        texture,
        size,
        terrains,
        slots,
    }
}
