use std::sync::Arc;

use engine_core::Error;
use engine_places::ground::{TerrainEntry, TerrainTiles};
use engine_render::{Install, Lifetime, Prepare, Texture, wgpu};

use crate::gpu::{HORIZON_RECORD_VECTORS, HORIZON_RECORDS};
use crate::lighting::{HORIZON_LAYERS, TerrainLighting};
use crate::settings::TerrainShading;
use crate::state::Bakes;
use crate::terrain::TerrainGpu;
use crate::terrain_bounce::{BOUNCE_LAYERS, BounceLevels, MOST_LEVELS};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const WINDOW_ROW: u32 = crate::bounce::MAX_PROBES;

const _: () = assert!(
    MOST_LEVELS <= 4 * (HORIZON_RECORD_VECTORS as usize - 4),
    "record_holds_level_windows"
);

const NO_RECORD: [[f32; 4]; HORIZON_RECORD_VECTORS as usize] = {
    let mut record = [[0.0; 4]; HORIZON_RECORD_VECTORS as usize];
    record[2][0] = -1.0;
    record
};

pub(crate) struct Horizons {
    pub(crate) texture: Texture<TerrainGpu>,
    pub(crate) probes: Texture<TerrainGpu>,
    shape: Option<(u32, usize, u32)>,
    held: Vec<Option<Arc<TerrainLighting>>>,
    pub(crate) uploaded: u32,
}

impl Horizons {
    pub(crate) fn install(cx: &mut Install<'_, TerrainGpu>) -> Result<Self, Error> {
        Ok(Self {
            texture: cx.texture("horizons", FORMAT, Lifetime::Kept),
            probes: cx.texture("bounce_probes", FORMAT, Lifetime::Kept),
            shape: None,
            held: Vec::new(),
            uploaded: 0,
        })
    }

    pub(crate) fn bounces(&self) -> bool {
        self.shape.is_some_and(|(_, _, slots)| slots > 0)
    }

    pub(crate) fn held(&self, index: usize) -> bool {
        self.held.get(index).is_some_and(Option::is_some)
    }

    pub(crate) fn count(&self) -> u32 {
        self.held.iter().flatten().count() as u32
    }

    pub(crate) fn prepare(
        &mut self,
        cx: &mut Prepare<'_, TerrainGpu>,
        (tiles, bakes, settings, shadows, layers): (&TerrainTiles, &Bakes, &TerrainShading, bool, u32),
        slots: u32,
    ) {
        if !(shadows && settings.sun_shadows) || tiles.terrains.is_empty() {
            self.rest();
            return;
        }
        self.uploaded = 0;
        let size = settings.shadow_resolution;
        let (count, _) = room(layers, tiles.terrains.len());
        if self.shape != Some((size, count, slots)) {
            self.shape = Some((size, count, slots));
            self.held = vec![None; count];
        }
        let horizons = (HORIZON_LAYERS * count) as u32;
        cx.allocate(
            self.texture,
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: horizons.max(1),
            },
            1,
        );
        cx.allocate(
            self.probes,
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: (BOUNCE_LAYERS as u32 * slots).max(1),
            },
            1,
        );
        for (index, terrain) in tiles.terrains.iter().take(count).enumerate() {
            let lighting = bakes
                .of(terrain.terrain)
                .and_then(|held| held.lighting.made.as_ref())
                .filter(|lighting| lighting.horizon_size == size);
            let held = &mut self.held[index];
            match (lighting, &*held) {
                (Some(lighting), Some(current)) if Arc::ptr_eq(lighting, current) => {}
                (Some(lighting), _) => {
                    cx.write_texture(
                        self.texture,
                        wgpu::Origin3d {
                            x: 0,
                            y: 0,
                            z: (HORIZON_LAYERS * index) as u32,
                        },
                        &lighting.horizon_layers(),
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
                    *held = Some(Arc::clone(lighting));
                    self.uploaded += 1;
                }
                (None, _) => *held = None,
            }
        }
    }

    pub(crate) fn rest(&mut self) {
        self.uploaded = 0;
        self.shape = None;
        self.held.clear();
    }
}

pub(crate) fn records(
    terrains: &[TerrainEntry],
    bakes: &Bakes,
    settings: &TerrainShading,
    windows: Vec<Option<BounceLevels>>,
    layers: u32,
) -> Vec<[f32; 4]> {
    let size = settings.shadow_resolution;
    let (count, _) = room(layers, terrains.len());
    let bounces = windows.iter().any(Option::is_some);
    let mut records = Vec::with_capacity(HORIZON_RECORD_VECTORS as usize * count);
    for (index, terrain) in terrains.iter().take(count).enumerate() {
        let lighting = bakes
            .of(terrain.terrain)
            .and_then(|held| held.lighting.made.as_ref())
            .filter(|lighting| lighting.horizon_size == size);
        match lighting {
            Some(lighting) => {
                let window = windows.get(index).copied().flatten().filter(|_| bounces);
                let place = Place {
                    index,
                    size,
                    window,
                };
                records.extend(record(terrain, lighting, place, settings));
            }
            None => records.extend(NO_RECORD),
        }
    }
    records
}

pub(crate) fn room(layers: u32, terrains: usize) -> (usize, u32) {
    let layers = layers as usize;
    let casting = terrains
        .min(layers / HORIZON_LAYERS)
        .min(HORIZON_RECORDS as usize / HORIZON_RECORD_VECTORS as usize);
    let slots = (layers - HORIZON_LAYERS * casting) / BOUNCE_LAYERS;
    (casting, slots as u32)
}

fn max_extent(terrain: &TerrainEntry) -> f32 {
    let extent = terrain.pyramid.extent();
    engine_math::max(extent.x, extent.y)
}

struct Place {
    index: usize,
    size: u32,
    window: Option<BounceLevels>,
}

fn record(
    terrain: &TerrainEntry,
    lighting: &TerrainLighting,
    place: Place,
    settings: &TerrainShading,
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
            (BOUNCE_LAYERS * window.slot as usize) as f32,
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
