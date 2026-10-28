use std::sync::Arc;

use engine_terrain::lighting::HORIZON_LAYERS;
use engine_terrain::lighting::TerrainLighting;
use engine_terrain::{RenderTerrain, RenderTerrains, TerrainSettings};

use crate::cached::{Bound, Versioned};
use crate::gpu_layouts::constants::{HORIZON_RECORD_VECTORS, HORIZON_RECORDS};
use crate::terrain_bounce::{BOUNCE_LAYERS, BounceLevels, MOST_LEVELS};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const WINDOW_ROW: u32 = engine_terrain::bounce::MAX_PROBES;

const _: () = assert!(
    MOST_LEVELS <= 4 * (HORIZON_RECORD_VECTORS as usize - 4),
    "a record holds every level's window, four a vector"
);

const NO_RECORD: [[f32; 4]; HORIZON_RECORD_VECTORS as usize] = {
    let mut record = [[0.0; 4]; HORIZON_RECORD_VECTORS as usize];
    record[2][0] = -1.0;
    record
};

struct Maps {
    texture: wgpu::Texture,
    view: Versioned<wgpu::TextureView>,
    horizon_view: Versioned<wgpu::TextureView>,
    bounce_view: Option<Versioned<wgpu::TextureView>>,
    size: u32,
    terrains: usize,
    slots: u32,
}

pub(crate) struct Horizons {
    maps: Option<Maps>,
    empty: wgpu::TextureView,
    sampler: wgpu::Sampler,
    records: wgpu::Buffer,
    records_written: Vec<[f32; 4]>,
    held: Vec<Option<Arc<TerrainLighting>>>,
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

    pub(crate) fn bounces(&self) -> bool {
        self.maps
            .as_ref()
            .is_some_and(|maps| maps.bounce_view.is_some())
    }

    pub(crate) fn held(&self, index: usize) -> bool {
        self.held.get(index).is_some_and(Option::is_some)
    }

    pub(crate) fn record_count(&self) -> u32 {
        (self.records_written.len() / HORIZON_RECORD_VECTORS as usize) as u32
    }

    pub(crate) fn count(&self) -> u32 {
        self.held.iter().flatten().count() as u32
    }

    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.maps.as_ref().map_or(0, |maps| {
            u64::from(maps.size)
                * u64::from(maps.size)
                * u64::from(maps.texture.depth_or_array_layers())
                * 8
        })
    }

    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        terrains: &RenderTerrains,
        (settings, shadows): (&TerrainSettings, bool),
        (windows, slots): (&[Option<BounceLevels>], u32),
    ) {
        self.uploaded = 0;
        if !(shadows && settings.enabled && settings.sun_shadows) || terrains.terrains.is_empty() {
            self.release();
            return;
        }
        let size = settings.shadow_resolution;
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
                None => records.extend(NO_RECORD),
            }
        }
        if records != self.records_written {
            queue.write_buffer(&self.records, 0, bytemuck::cast_slice(&records));
            self.records_written = records;
        }
    }

    fn release(&mut self) {
        self.maps = None;
        self.held.clear();
        self.records_written.clear();
    }
}

pub(crate) fn room(device: &wgpu::Device, terrains: usize) -> (usize, u32) {
    let layers = device.limits().max_texture_array_layers as usize;
    let casting = terrains
        .min(layers / HORIZON_LAYERS)
        .min(HORIZON_RECORDS / HORIZON_RECORD_VECTORS as usize);
    let slots = (layers - HORIZON_LAYERS * casting) / BOUNCE_LAYERS;
    (casting, slots as u32)
}

fn max_extent(terrain: &RenderTerrain) -> f32 {
    let extent = terrain.pyramid.extent();
    engine_core::math::max(extent.x, extent.y)
}

struct Place {
    index: usize,
    count: usize,
    size: u32,
    window: Option<BounceLevels>,
}

fn record(
    terrain: &RenderTerrain,
    lighting: &TerrainLighting,
    place: Place,
    settings: &TerrainSettings,
) -> [[f32; 4]; HORIZON_RECORD_VECTORS as usize] {
    let extent = terrain.pyramid.extent();
    let texel = max_extent(terrain) / lighting.horizon_size as f32;
    let mut record = NO_RECORD;
    record[0] = [
        terrain.origin.x,
        terrain.origin.z,
        1.0 / extent.x,
        1.0 / extent.y,
    ];
    record[1] = [
        (HORIZON_LAYERS * place.index) as f32,
        settings.shadow_softness,
        0.1 * texel,
        terrain.origin.y + lighting.lowest,
    ];
    if let Some(window) = place.window {
        record[2] = [
            (HORIZON_LAYERS * place.count + BOUNCE_LAYERS * window.slot as usize) as f32,
            window.probes as f32,
            window.levels as f32,
            window.columns as f32,
        ];
        record[3] = [
            1.0 / place.size as f32,
            window.reach,
            window.row as f32,
            window.finest as f32,
        ];
        for (level, &[x, z]) in window.first.iter().enumerate().take(MOST_LEVELS) {
            record[4 + level / 4][level % 4] = (x + z * WINDOW_ROW) as f32;
        }
    }
    record
}

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
