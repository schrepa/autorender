use std::sync::Arc;

use engine_core::Error;
use engine_open_world_kit::ground::{TerrainEntry, TerrainTiles};
use engine_render::{Allocation, Install, Lifetime, Prepare, Texture, wgpu};

use crate::gpu::{HORIZON_RECORD_VECTORS, HORIZON_RECORDS, SUN_LEVEL_LAYERS, SUN_LEVELS};
use crate::jobs::lighting::{HORIZON_LAYERS, TerrainLighting};
use crate::settings::TerrainShading;
use crate::state::Bakes;
use crate::terrain::TerrainGpu;
use crate::terrain_bounce::{BOUNCE_LAYERS, BounceLevels, MOST_LEVELS};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const WINDOW_ROW: u32 = crate::jobs::bounce::MAX_PROBES;
pub(crate) const SUN_LAYERS: usize = (SUN_LEVELS * SUN_LEVEL_LAYERS) as usize;

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
    shape: Option<(u32, usize, u32)>,
    held: Vec<Option<Arc<TerrainLighting>>>,
    pub(crate) uploaded: u32,
    pub(crate) fresh: bool,
}

impl Horizons {
    pub(crate) fn install(cx: &mut Install<'_, TerrainGpu>) -> Result<Self, Error> {
        Ok(Self {
            texture: cx.texture("horizons", FORMAT, Lifetime::Kept),
            shape: None,
            held: Vec::new(),
            uploaded: 0,
            fresh: false,
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
        (tiles, bakes, settings, layers): (&TerrainTiles, &Bakes, &TerrainShading, u32),
        slots: u32,
    ) {
        if !settings.sun_shadows || tiles.terrains.is_empty() {
            self.rest();
            return;
        }
        self.uploaded = 0;
        let size = settings.shadow_resolution;
        let (count, _) = room(layers, tiles.terrains.len());
        self.shape = Some((size, count, slots));
        let total = sun_base(count, slots) + SUN_LAYERS as u32;
        let allocation = cx.allocate(
            self.texture,
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: total,
            },
            1,
        );
        self.fresh = allocation == Allocation::Fresh;
        if allocation == Allocation::Fresh || self.held.len() != count {
            self.held = vec![None; count];
        }
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
        self.fresh = false;
    }
}

pub(crate) fn sun_base(count: usize, slots: u32) -> u32 {
    (HORIZON_LAYERS * count) as u32 + BOUNCE_LAYERS as u32 * slots
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
                    count,
                    size,
                    window,
                };
                records.extend(record(terrain, lighting, place));
            }
            None => records.extend(NO_RECORD),
        }
    }
    records
}

pub(crate) fn room(layers: u32, terrains: usize) -> (usize, u32) {
    let layers = (layers as usize).saturating_sub(SUN_LAYERS);
    let casting = terrains
        .min(layers / HORIZON_LAYERS)
        .min(HORIZON_RECORDS as usize / HORIZON_RECORD_VECTORS as usize);
    let slots = (layers - HORIZON_LAYERS * casting) / BOUNCE_LAYERS;
    (casting, slots as u32)
}

struct Place {
    index: usize,
    count: usize,
    size: u32,
    window: Option<BounceLevels>,
}

fn record(
    terrain: &TerrainEntry,
    lighting: &TerrainLighting,
    place: Place,
) -> [[f32; 4]; HORIZON_RECORD_VECTORS as usize] {
    let extent = terrain.pyramid.extent();
    let mut record = NO_RECORD;
    record[0] = [
        terrain.origin.x,
        terrain.origin.z,
        1.0 / extent.x,
        1.0 / extent.y,
    ];
    record[1] = [
        (HORIZON_LAYERS * place.index) as f32,
        0.0,
        0.0,
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
