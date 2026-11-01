use std::sync::Arc;

use engine_codec::{self as codec, Error, record};
use engine_core::{ErrorCode, Job, JobCx, JobKind, RecordKey};
use engine_heightfield::layer::LayerKind;
use engine_heightfield::tile::{Generator, Recipe, Tile};
use engine_heightfield::{Pyramid, Stamp, TILE_QUADS, TileKey};
use engine_places::terrain::TerrainLayers;

use crate::bounce::{MAX_PROBES, TerrainBounce, bounce};
use crate::lighting::{Ground, TerrainLighting, light};
use crate::macro_colors::{MACRO_BLOCK, MAX_MACRO_TEXELS, macro_colors};
use crate::shading::{ShadingMaking, TileShading};

pub static LIGHTING: JobKind =
    JobKind::new("terrain_shading::lighting", &crate::SOURCE, start_lighting);
pub static BOUNCE: JobKind = JobKind::new("terrain_shading::bounce", &crate::SOURCE, start_bounce);
pub static MACRO_COLORS: JobKind =
    JobKind::new("terrain_shading::macro_colors", &crate::SOURCE, start_macro);
pub static SHADING: JobKind =
    JobKind::new("terrain_shading::shading", &crate::SOURCE, start_shading);

pub const MAX_LIGHTING_TEXELS: u32 = 4096;

#[derive(Clone, Debug, PartialEq)]
pub struct LightingJob {
    pub recipe: [u8; 32],
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
    pub recipe: [u8; 32],
    pub key: TileKey,
    pub reads: ShadingReads,
}

record!(ShadingJob { recipe, key, reads });

#[derive(Clone, Debug, PartialEq)]
pub struct BounceJob {
    pub recipe: [u8; 32],
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

#[derive(Clone, Debug, PartialEq)]
pub struct MacroJob {
    pub recipe: [u8; 32],
    pub level: u32,
    pub size: u32,
}

record!(MacroJob {
    recipe,
    level,
    size,
});

impl MacroJob {
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

pub fn lighting_output(bytes: &[u8]) -> Result<TerrainLighting, Error> {
    codec::decode(bytes)
}

pub fn bounce_output(bytes: &[u8]) -> Result<TerrainBounce, Error> {
    codec::decode(bytes)
}

pub fn shading_output(bytes: &[u8]) -> Result<TileShading, Error> {
    codec::decode(bytes)
}

fn kinds(cx: &JobCx<'_>) -> Vec<&'static LayerKind> {
    cx.members::<TerrainLayers>().copied().collect()
}

fn recipe(
    cx: &JobCx<'_>,
    shared: [u8; 32],
    kinds: &[&'static LayerKind],
) -> Result<Arc<Recipe>, Error> {
    cx.held(shared, || {
        let key = RecordKey::from_hash(shared);
        let bytes = cx
            .share(key)
            .ok_or_else(|| unshared("$.recipe", &key.hex()))?;
        Recipe::read(bytes, &|stack| cx.share(RecordKey::from_hash(stack)), kinds)
    })
}

struct LightingMaking {
    making: GroundMaking,
    job: LightingJob,
}

impl Job for LightingMaking {
    fn step(&mut self, budget: u64, _: &mut JobCx<'_>) -> Result<Option<Vec<u8>>, Error> {
        let mut budget = budget;
        if !self.making.advance(&mut budget)? {
            return Ok(None);
        }
        Ok(Some(codec::encode(&light(
            &self.making.ground(),
            self.making.extent(),
            self.job.sky,
            self.job.horizons,
        ))))
    }
}

fn start_lighting(input: &[u8], cx: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    let mut job: LightingJob = codec::decode(input)?;
    job.check()?;
    let kinds = kinds(cx);
    let recipe = recipe(cx, job.recipe, &kinds)?;
    let tiles = std::mem::take(&mut job.tiles);
    let making = GroundMaking::new((&recipe, kinds), job.level, tiles)?;
    Ok(Box::new(LightingMaking { making, job }))
}

struct BounceMaking {
    making: GroundMaking,
    job: BounceJob,
    recipe: Arc<Recipe>,
}

impl Job for BounceMaking {
    fn step(&mut self, budget: u64, _: &mut JobCx<'_>) -> Result<Option<Vec<u8>>, Error> {
        let mut budget = budget;
        if !self.making.advance(&mut budget)? {
            return Ok(None);
        }
        Ok(Some(codec::encode(&bounce(
            &self.making.ground(),
            self.making.extent(),
            (self.recipe.layers.as_ref(), self.recipe.rules.as_ref()),
            self.job.probes,
        ))))
    }
}

fn start_bounce(input: &[u8], cx: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    let mut job: BounceJob = codec::decode(input)?;
    job.check()?;
    let kinds = kinds(cx);
    let recipe = recipe(cx, job.recipe, &kinds)?;
    let tiles = std::mem::take(&mut job.tiles);
    let making = GroundMaking::new((&recipe, kinds), job.level, tiles)?;
    Ok(Box::new(BounceMaking {
        making,
        job,
        recipe,
    }))
}

struct MacroMaking {
    recipe: Arc<Recipe>,
    job: MacroJob,
}

impl Job for MacroMaking {
    fn step(&mut self, _: u64, _: &mut JobCx<'_>) -> Result<Option<Vec<u8>>, Error> {
        let causes = self.recipe.rules.authored().macro_colors;
        macro_colors(
            &self.recipe.layers,
            &self.recipe.pyramid,
            self.job.level,
            self.job.size,
            &causes,
        )
        .map(Some)
        .ok_or_else(invalid_heights)
    }
}

fn start_macro(input: &[u8], cx: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    let job: MacroJob = codec::decode(input)?;
    let recipe = recipe(cx, job.recipe, &kinds(cx))?;
    job.check()?;
    if job.level >= recipe.pyramid.levels() {
        return Err(level_outside(job.level, &recipe.pyramid));
    }
    Ok(Box::new(MacroMaking { recipe, job }))
}

impl Job for ShadingMaking {
    fn step(&mut self, budget: u64, _: &mut JobCx<'_>) -> Result<Option<Vec<u8>>, Error> {
        let mut budget = budget;
        Ok(self
            .advance(&mut budget)?
            .map(|shading| codec::encode(&shading)))
    }
}

fn start_shading(input: &[u8], cx: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    let job: ShadingJob = codec::decode(input)?;
    let kinds = kinds(cx);
    let recipe = recipe(cx, job.recipe, &kinds)?;
    let (pyramid, key) = (recipe.pyramid, job.key);
    let reads = TileShading::reads(&pyramid, key);
    if !pyramid.contains(key) || reads.is_empty() {
        return Err(outside(key));
    }
    match job.reads {
        ShadingReads::Tiles(tiles)
            if key.level == 1 && names(tiles.iter().map(|&(at, _)| at), &reads) =>
        {
            let surface = (recipe.layers.clone(), recipe.rules.clone(), kinds);
            Ok(Box::new(ShadingMaking::from_ground(
                surface, pyramid, key, tiles,
            )))
        }
        ShadingReads::Shadings(shadings)
            if key.level > 1 && names(shadings.iter().map(|&(at, _)| at), &reads) =>
        {
            let finer = shadings
                .iter()
                .map(|&(at, record)| shared_shading(cx, record, at))
                .collect::<Result<Vec<_>, Error>>()?;
            Ok(Box::new(ShadingMaking::from_finer(pyramid, key, finer)))
        }
        ShadingReads::Tiles(_) | ShadingReads::Shadings(_) => {
            Err(Error::new(ErrorCode::InvalidValue, "$.reads"))
        }
    }
}

fn shared_shading(cx: &JobCx<'_>, record: [u8; 32], key: TileKey) -> Result<TileShading, Error> {
    let shared = RecordKey::from_hash(record);
    let bytes = cx
        .share(shared)
        .ok_or_else(|| unshared("$.reads", &shared.hex()))?;
    let shading: TileShading = codec::decode(bytes)?;
    if shading.key() != key {
        return Err(mismatched("$.reads", &shared.hex()));
    }
    Ok(shading)
}

fn names(named: impl Iterator<Item = TileKey>, reads: &[TileKey]) -> bool {
    named.eq(reads.iter().copied())
}

pub(crate) struct GroundMaking {
    recipe: Arc<Recipe>,
    kinds: Vec<&'static LayerKind>,
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
    fn new(
        (recipe, kinds): (&Arc<Recipe>, Vec<&'static LayerKind>),
        level: u32,
        tiles: Vec<(TileKey, Vec<Stamp>)>,
    ) -> Result<Self, Error> {
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
            recipe: Arc::clone(recipe),
            kinds,
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
        while let Some((held, _)) = self.tiles.get(self.next) {
            let held = *held;
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
                let layers = self.recipe.stamped(&self.tiles[self.next].1, &self.kinds)?;
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

pub(crate) fn invalid_heights() -> Error {
    Error::new(ErrorCode::InvalidTerrain, "$.heights").expected(engine_codec::Expected::Finite)
}

fn unshared(path: &str, name: &str) -> Error {
    Error::new(ErrorCode::UnknownReference, path).got(name)
}

fn mismatched(path: &str, name: &str) -> Error {
    Error::new(ErrorCode::InvalidValue, path).got(name)
}

fn outside(key: TileKey) -> Error {
    Error::new(ErrorCode::OutOfRange, "$.key").got([u16::from(key.level), key.x, key.z])
}

fn level_outside(level: u32, pyramid: &Pyramid) -> Error {
    Error::new(ErrorCode::OutOfRange, "$.level")
        .at_most(u64::from(pyramid.levels().saturating_sub(1)))
        .got(level)
}
