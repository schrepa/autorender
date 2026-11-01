use std::sync::Arc;

use engine_core::ErrorCode;

use crate::bounce::{MAX_PROBES, TerrainBounce};
use crate::codec::{self, Error, record};
use crate::lighting::{Ground, TerrainLighting};
use crate::macro_colors::{MACRO_BLOCK, MAX_MACRO_TEXELS};
use crate::shading::ShadingMaking;
use crate::{Generator, Layers, Pyramid, Stamp, TILE_QUADS, Tile, TileKey, TileShading};

const SHADING: u8 = 4;

pub fn share_shading(record: &[u8]) -> Vec<u8> {
    let mut shared = Vec::with_capacity(record.len() + 1);
    shared.push(SHADING);
    shared.extend_from_slice(record);
    shared
}

pub fn shading_key(record: &[u8; 32]) -> String {
    format!("shading-{}", hex(record))
}

#[derive(Clone, Debug, PartialEq)]
pub struct LightingJob {
    pub recipe: String,
    pub level: u32,
    pub tiles: Vec<(TileKey, Vec<Stamp>)>,
    pub sky: u32,
    pub horizons: u32,
}

record!(LightingJob {
    recipe,
    level,
    tiles,
    sky,
    horizons,
});

impl LightingJob {
    pub const KIND: JobKind = JobKind::Lighting;

    pub fn input(&self) -> Vec<u8> {
        codec::encode(self)
    }

    fn check(&self) -> Result<(), Error> {
        for (path, texels) in [("$.sky", self.sky), ("$.horizons", self.horizons)] {
            if !(1..=MAX_LIGHTING_TEXELS).contains(&texels) {
                return Err(Error::new(ErrorCode::OutOfRange, path)
                    .range(1.0, f64::from(MAX_LIGHTING_TEXELS))
                    .got(texels));
            }
        }
        Ok(())
    }
}

pub const MAX_LIGHTING_TEXELS: u32 = 4096;

pub fn lighting_output(bytes: &[u8]) -> Result<TerrainLighting, Error> {
    codec::decode(bytes)
}

#[derive(Clone, Debug, PartialEq)]
pub enum ShadingReads {
    Tiles(Vec<(TileKey, Vec<Stamp>)>),
    Shadings(Vec<(TileKey, [u8; 32])>),
}

codec::tagged!(ShadingReads {
    0 => Tiles(tiles),
    1 => Shadings(shadings),
});

#[derive(Clone, Debug, PartialEq)]
pub struct ShadingJob {
    pub recipe: String,
    pub key: TileKey,
    pub reads: ShadingReads,
}

record!(ShadingJob { recipe, key, reads });

impl ShadingJob {
    pub const KIND: JobKind = JobKind::Shading;

    pub fn identity(&self) -> Vec<u8> {
        codec::encode(self)
    }
}

pub fn shading_output(bytes: &[u8]) -> Result<TileShading, Error> {
    codec::decode(bytes)
}

#[derive(Clone, Debug, PartialEq)]
pub struct BounceJob {
    pub recipe: String,
    pub level: u32,
    pub tiles: Vec<(TileKey, Vec<Stamp>)>,
    pub probes: u32,
}

record!(BounceJob {
    recipe,
    level,
    tiles,
    probes,
});

impl BounceJob {
    pub const KIND: JobKind = JobKind::Bounce;

    pub fn input(&self) -> Vec<u8> {
        codec::encode(self)
    }

    fn check(&self) -> Result<(), Error> {
        if (2..=MAX_PROBES).contains(&self.probes) && self.probes.is_multiple_of(2) {
            Ok(())
        } else {
            Err(Error::new(ErrorCode::OutOfRange, "$.probes")
                .range(2.0, f64::from(MAX_PROBES))
                .got(self.probes))
        }
    }
}

pub fn bounce_output(bytes: &[u8]) -> Result<TerrainBounce, Error> {
    codec::decode(bytes)
}

#[derive(Clone, Debug, PartialEq)]
pub struct MacroJob {
    pub recipe: String,
    pub level: u32,
    pub size: u32,
}

record!(MacroJob {
    recipe,
    level,
    size,
});

impl MacroJob {
    pub const KIND: JobKind = JobKind::Macro;

    pub fn identity(&self) -> Vec<u8> {
        codec::encode(self)
    }

    fn check(&self) -> Result<(), Error> {
        if (MACRO_BLOCK..=MAX_MACRO_TEXELS).contains(&self.size)
            && self.size.is_multiple_of(MACRO_BLOCK)
        {
            Ok(())
        } else {
            Err(Error::new(ErrorCode::OutOfRange, "$.size")
                .range(f64::from(MACRO_BLOCK), f64::from(MAX_MACRO_TEXELS))
                .got(self.size))
        }
    }
}

struct GroundMaking {
    layers: Arc<Layers>,
    pyramid: Pyramid,
    level: u32,
    tiles: Vec<(TileKey, Vec<Stamp>)>,
    made: Vec<(TileKey, Tile)>,
    generator: Option<Generator>,
    next: usize,
    samples: [u32; 2],
    heights: Vec<f32>,
}

impl GroundMaking {
    fn new(recipe: &Recipe, level: u32, tiles: Vec<(TileKey, Vec<Stamp>)>) -> Result<Self, Error> {
        let pyramid = recipe.pyramid;
        if level >= pyramid.levels() {
            return Err(level_outside(level, &pyramid));
        }
        let [columns, rows] = pyramid.tiles(level);
        let stands = |key: TileKey, held: TileKey| {
            held.level >= key.level && pyramid.contains(held) && {
                let up = u32::from(held.level - key.level);
                key.x >> up == held.x && key.z >> up == held.z
            }
        };
        if tiles.len() != columns as usize * rows as usize
            || !pyramid
                .level_tiles(level)
                .zip(&tiles)
                .all(|(key, &(held, _))| stands(key, held))
        {
            return Err(Error::new(ErrorCode::InvalidValue, "$.tiles"));
        }
        let quads = TILE_QUADS as usize;
        let samples = [columns as usize * quads + 1, rows as usize * quads + 1];
        Ok(Self {
            layers: recipe.layers.clone(),
            pyramid,
            level,
            tiles,
            made: Vec::new(),
            generator: None,
            next: 0,
            samples: samples.map(|count| count as u32),
            heights: vec![0.0; samples[0] * samples[1]],
        })
    }

    fn advance(&mut self, budget: &mut u64) -> Result<bool, Error> {
        let columns = self.pyramid.tiles(self.level)[0] as usize;
        while let Some(&(held, _)) = self.tiles.get(self.next) {
            if let Some((_, tile)) = self.made.iter().find(|(made, _)| *made == held) {
                let key = TileKey::new(
                    self.level as u8,
                    (self.next % columns) as u16,
                    (self.next / columns) as u16,
                );
                fill(
                    (self.heights.as_mut_slice(), self.samples[0] as usize),
                    (key, self.level),
                    tile,
                );
                self.next += 1;
                continue;
            }
            if *budget == 0 {
                return Ok(false);
            }
            if self.generator.is_none() {
                let layers = stamped(&self.layers, &self.tiles[self.next].1)?;
                self.generator = Some(Generator::new(layers, &self.pyramid, held));
            }
            let Some(generator) = self.generator.as_mut() else {
                continue;
            };
            let Some(made) = generator.advance(budget) else {
                if generator.failed() {
                    return Err(invalid_heights());
                }
                return Ok(false);
            };
            self.generator = None;
            let tile: Tile = codec::decode(&codec::encode(&made))?;
            self.made.push((held, tile));
        }
        Ok(true)
    }

    fn ground(&self) -> Ground<'_> {
        Ground {
            samples: self.samples,
            spacing: self.pyramid.spacing(self.level),
            heights: &self.heights,
        }
    }

    fn extent(&self) -> [f32; 2] {
        self.pyramid.extent().to_array()
    }
}

fn fill((heights, columns): (&mut [f32], usize), (key, level): (TileKey, u32), tile: &Tile) {
    let quads = TILE_QUADS as usize;
    let first = [usize::from(key.x) * quads, usize::from(key.z) * quads];
    let held = tile.key();
    let coarser = u32::from(held.level) - level;
    let scale = 1.0 / (1u32 << coarser) as f32;
    let offset = [
        (u32::from(key.x) * TILE_QUADS) as f32 * scale - (u32::from(held.x) * TILE_QUADS) as f32,
        (u32::from(key.z) * TILE_QUADS) as f32 * scale - (u32::from(held.z) * TILE_QUADS) as f32,
    ];
    for z in 0..=quads {
        for x in 0..=quads {
            let height = if coarser == 0 {
                tile.height(x as i32, z as i32)
            } else {
                between(
                    tile,
                    offset[0] + x as f32 * scale,
                    offset[1] + z as f32 * scale,
                )
            };
            heights[(first[1] + z) * columns + first[0] + x] = height;
        }
    }
}

fn between(tile: &Tile, x: f32, z: f32) -> f32 {
    let (x0, z0) = (x.floor(), z.floor());
    let (tx, tz) = (x - x0, z - z0);
    let (x0, z0) = (x0 as i32, z0 as i32);
    let a = tile.height(x0, z0);
    let b = tile.height(x0 + 1, z0);
    let c = tile.height(x0, z0 + 1);
    let d = tile.height(x0 + 1, z0 + 1);
    let near = a + (b - a) * tx;
    let far = c + (d - c) * tx;
    near + (far - near) * tz
}

impl TerrainWorker {
    fn shading(&self, job: ShadingJob) -> Result<ShadingMaking, Error> {
        let recipe = self.recipe(&job.recipe)?;
        let (pyramid, key) = (recipe.pyramid, job.key);
        let reads = TileShading::reads(&pyramid, key);
        if !pyramid.contains(key) || reads.is_empty() {
            return Err(outside(key));
        }
        match job.reads {
            ShadingReads::Tiles(tiles)
                if key.level == 1 && names(tiles.iter().map(|&(at, _)| at), &reads) =>
            {
                let surface = (recipe.layers.clone(), recipe.rules.clone());
                Ok(ShadingMaking::from_ground(surface, pyramid, key, tiles))
            }
            ShadingReads::Shadings(shadings)
                if key.level > 1 && names(shadings.iter().map(|&(at, _)| at), &reads) =>
            {
                let finer = shadings
                    .iter()
                    .map(|&(at, record)| self.shared_shading(&record, at))
                    .collect::<Result<Vec<_>, Error>>()?;
                Ok(ShadingMaking::from_finer(pyramid, key, finer))
            }
            ShadingReads::Tiles(_) | ShadingReads::Shadings(_) => {
                Err(Error::new(ErrorCode::InvalidValue, "$.reads"))
            }
        }
    }

    fn shared_shading(&self, record: &[u8; 32], key: TileKey) -> Result<TileShading, Error> {
        let name = shading_key(record);
        let shading: TileShading = match self.records.get(&name).map(|bytes| bytes.split_first()) {
            Some(Some((&SHADING, rest))) => codec::decode(rest)?,
            _ => return Err(unshared("$.reads", &name)),
        };
        if shading.key() != key {
            return Err(mismatched("$.reads", &name));
        }
        Ok(shading)
    }
}

fn names(named: impl Iterator<Item = TileKey>, reads: &[TileKey]) -> bool {
    named.eq(reads.iter().copied())
}
