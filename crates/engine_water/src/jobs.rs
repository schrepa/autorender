use std::sync::Arc;

use engine_codec::{Expected, Reader, record};
use engine_core::{Error, ErrorCode, Job, JobCx, JobKind, RecordKey, codec};
use engine_heightfield::layer::LayerKind;
use engine_heightfield::tile::{Recipe, Solid, TileJob};
use engine_heightfield::{Layers, Pyramid, Stamp, TileKey};
use engine_places::terrain::TerrainLayers;

use crate::SOURCE;
use crate::drawn::{start_drawn, start_rewater};
use crate::steady_water::{PartMaking, SettledReads, Steadying};
use crate::tile_water::BaseWater;
use crate::{SteadyCell, SteadySummary, SteadyWater, WaterPart, settle_regions, watered_tiles};

pub static TILE_WATER: JobKind = JobKind::new("water::tile", &SOURCE, start_tile_water);
pub static PARTS: JobKind = JobKind::new("water::part", &SOURCE, start_part);
pub static STEADIED: JobKind = JobKind::new("water::steadied", &SOURCE, start_steadied);
pub static SETTLED: JobKind =
    JobKind::new("water::settled", &SOURCE, start_settled).kept_alike(settled_alike);
pub static DRAWN: JobKind = JobKind::new("water::drawn", &SOURCE, start_drawn);
pub static REWATER: JobKind = JobKind::new("water::rewater", &SOURCE, start_rewater);
pub static REACH: JobKind = JobKind::new("water::reach", &SOURCE, start_reach);

const SETTLED_SHARE: u8 = 2;
const SUMMARY_SHARE: u8 = 3;

pub fn settled_alike(handed: &[u8], made: &[u8]) -> bool {
    let (Ok(handed), Ok(made)) = (
        engine_codec::decode::<SteadyWater>(handed),
        engine_codec::decode::<SteadyWater>(made),
    ) else {
        return false;
    };
    if handed.key != made.key || handed.first != made.first || handed.cells != made.cells {
        return false;
    }
    let [columns, rows] = handed.cells;
    let first = handed.first;
    let (mut cells, mut wetness, mut wet, mut apart) = (0_u32, 0_u32, 0_u32, 0_u32);
    for j in 0..i64::from(rows) {
        for i in 0..i64::from(columns) {
            let at = |water: &SteadyWater| match water.at(first[0] + i, first[1] + j) {
                Some(SteadyCell::Held { level, .. }) => Some(level),
                Some(SteadyCell::Flowing { surface, .. }) => Some(surface),
                Some(SteadyCell::Dry) | None => None,
            };
            cells += 1;
            match (at(&handed), at(&made)) {
                (Some(a), Some(b)) => {
                    wet += 1;
                    apart += u32::from((a - b).abs() > 0.05);
                }
                (None, None) => {}
                (Some(_), None) | (None, Some(_)) => wetness += 1,
            }
        }
    }
    wetness * 50 <= cells && apart * 100 <= wet.max(1)
}

pub fn settled_share(record: RecordKey) -> RecordKey {
    RecordKey::shared_tagged(SETTLED_SHARE, &record.hash())
}

pub fn summary_share(record: RecordKey) -> RecordKey {
    RecordKey::shared_tagged(SUMMARY_SHARE, &record.hash())
}

#[derive(Clone, Debug, PartialEq)]
pub struct WaterJob {
    pub recipe: [u8; 32],
    pub key: TileKey,
    pub stamps: Vec<Stamp>,
    pub solids: Vec<Option<Solid>>,
    pub cell: f32,
}

record!(WaterJob {
    recipe,
    key,
    stamps,
    solids,
    cell,
});

impl WaterJob {
    pub fn input(&self) -> Vec<u8> {
        codec::encode(self)
    }
}

pub fn water_output(bytes: &[u8]) -> Result<WaterPart, Error> {
    let part: WaterPart = codec::decode(bytes)?;
    if !part.valid() {
        return Err(engine_codec::invalid("$.water_part"));
    }
    Ok(part)
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReachJob {
    pub recipe: [u8; 32],
    pub cell: f32,
}

record!(ReachJob { recipe, cell });

#[derive(Clone, Debug, PartialEq)]
pub struct WaterRegion {
    pub tiles: [u32; 4],
    pub keys: Vec<TileKey>,
}

record!(WaterRegion { tiles, keys });

#[derive(Clone, Debug, PartialEq)]
pub struct WaterReach {
    pub watered: Vec<TileKey>,
    pub regions: Vec<WaterRegion>,
    pub of: Vec<u32>,
}

record!(WaterReach {
    watered,
    regions,
    of,
});

#[derive(Clone, Debug, PartialEq)]
pub struct SteadiedJob {
    pub tile: TileJob,
    pub cell: f32,
    pub settled: Vec<(TileKey, [u8; 32])>,
}

record!(SteadiedJob {
    tile,
    cell,
    settled,
});

impl SteadiedJob {
    pub fn input(&self) -> Vec<u8> {
        codec::encode(self)
    }
}

pub(crate) fn invalid_heights() -> Error {
    Error::new(ErrorCode::InvalidTerrain, "$.heights").expected(Expected::Finite)
}

pub(crate) fn kinds(cx: &JobCx<'_>) -> Vec<&'static LayerKind> {
    cx.members::<TerrainLayers>().copied().collect()
}

pub(crate) fn recipe(
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

fn start_tile_water(input: &[u8], cx: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    let mut reader = Reader::new(input);
    let job: TileJob = reader.get()?;
    reader.finish()?;
    let kinds = kinds(cx);
    let recipe = recipe(cx, job.recipe, &kinds)?;
    if !recipe.pyramid.contains(job.key) {
        return Err(outside(job.key));
    }
    Ok(Box::new(BaseWater::new(
        recipe.stamped(&job.stamps, &kinds)?,
        recipe.pyramid,
        job.key,
    )))
}

impl Job for BaseWater {
    fn step(&mut self, budget: u64, _: &mut JobCx<'_>) -> Result<Option<Vec<u8>>, Error> {
        let mut budget = budget;
        Ok(self
            .advance(&mut budget)?
            .map(|water| codec::encode(&water)))
    }
}

fn start_part(input: &[u8], cx: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    let mut reader = Reader::new(input);
    let WaterJob {
        recipe: shared,
        key,
        stamps,
        solids,
        cell,
    } = reader.get()?;
    reader.finish()?;
    let kinds = kinds(cx);
    let recipe = recipe(cx, shared, &kinds)?;
    if key.level != 0 || !recipe.pyramid.contains(key) {
        return Err(outside(key));
    }
    if !(cell.is_finite() && cell > 0.0) {
        return Err(Error::new(ErrorCode::OutOfRange, "$.cell").got(cell));
    }
    if !solids.iter().flatten().all(Solid::valid) {
        return Err(Error::new(ErrorCode::InvalidValue, "$.solids"));
    }
    Ok(Box::new(PartMaking::new(
        (recipe.stamped(&stamps, &kinds)?, recipe.rules.clone()),
        recipe.pyramid,
        key,
        (cell, solids),
    )))
}

impl Job for PartMaking {
    fn step(&mut self, budget: u64, _: &mut JobCx<'_>) -> Result<Option<Vec<u8>>, Error> {
        let mut budget = budget;
        Ok(self.advance(&mut budget)?.map(|part| codec::encode(&part)))
    }
}

fn start_steadied(input: &[u8], cx: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    let mut reader = Reader::new(input);
    let job: SteadiedJob = reader.get()?;
    reader.finish()?;
    let kinds = kinds(cx);
    let recipe = recipe(cx, job.tile.recipe, &kinds)?;
    let (key, cell) = (job.tile.key, job.cell);
    if !recipe.pyramid.contains(key) {
        return Err(outside(key));
    }
    if !(cell.is_finite() && cell > 0.0) {
        return Err(Error::new(ErrorCode::OutOfRange, "$.cell").got(cell));
    }
    let reads = if key.level == 0 {
        SettledReads::Records(
            job.settled
                .iter()
                .map(|&(at, record)| Ok((at, settled(cx, record, at, cell)?)))
                .collect::<Result<_, Error>>()?,
        )
    } else {
        SettledReads::Summaries(
            job.settled
                .iter()
                .map(|&(at, record)| Ok((at, summary(cx, record, at, cell)?)))
                .collect::<Result<_, Error>>()?,
        )
    };
    Ok(Box::new(Steadying::new(
        recipe.stamped(&job.tile.stamps, &kinds)?,
        recipe.pyramid,
        key,
        (cell, reads),
    )))
}

impl Job for Steadying {
    fn step(&mut self, budget: u64, _: &mut JobCx<'_>) -> Result<Option<Vec<u8>>, Error> {
        let mut budget = budget;
        Ok(self
            .advance(&mut budget)?
            .map(|water| codec::encode(&water)))
    }
}

pub(crate) struct Finished(pub(crate) Option<Vec<u8>>);

impl Job for Finished {
    fn step(&mut self, _: u64, _: &mut JobCx<'_>) -> Result<Option<Vec<u8>>, Error> {
        Ok(self.0.take())
    }
}

fn water_reach(layers: &Layers, pyramid: &Pyramid, cell: f32) -> WaterReach {
    let watered = watered_tiles(layers, pyramid);
    let mut of = vec![0; watered.len()];
    let mut regions = Vec::new();
    for tiles in settle_regions(layers, pyramid, cell) {
        let [x, z, columns, rows] = tiles;
        let mut keys = Vec::new();
        for (at, key) in watered.iter().enumerate() {
            let inside = (x..x + columns).contains(&u32::from(key.x))
                && (z..z + rows).contains(&u32::from(key.z));
            if inside {
                of[at] = regions.len() as u32;
                keys.push(*key);
            }
        }
        if !keys.is_empty() {
            regions.push(WaterRegion { tiles, keys });
        }
    }
    WaterReach {
        watered,
        regions,
        of,
    }
}

fn start_reach(input: &[u8], cx: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    let mut reader = Reader::new(input);
    let ReachJob {
        recipe: shared,
        cell,
    } = reader.get()?;
    reader.finish()?;
    if !(cell.is_finite() && cell > 0.0) {
        return Err(Error::new(ErrorCode::OutOfRange, "$.cell").got(cell));
    }
    let kinds = kinds(cx);
    let recipe = recipe(cx, shared, &kinds)?;
    let reach = water_reach(&recipe.layers, &recipe.pyramid, cell);
    Ok(Box::new(Finished(Some(codec::encode(&reach)))))
}

fn start_settled(_: &[u8], _: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    Err(Error::new(ErrorCode::JobUnsupported, "$.jobs.water::settled").got("water::settled"))
}

fn settled(
    cx: &JobCx<'_>,
    record: [u8; 32],
    key: TileKey,
    cell: f32,
) -> Result<SteadyWater, Error> {
    let shared = RecordKey::from_hash(record);
    let Some(bytes) = cx.share(shared) else {
        return Err(unshared("$.settled", &shared.hex()));
    };
    let water: SteadyWater = codec::decode(bytes)?;
    if water.key != key || water.cell != cell {
        return Err(mismatched("$.settled", &shared.hex()));
    }
    Ok(water)
}

fn summary(
    cx: &JobCx<'_>,
    record: [u8; 32],
    key: TileKey,
    cell: f32,
) -> Result<SteadySummary, Error> {
    let shared = RecordKey::from_hash(record);
    let Some(bytes) = cx.share(shared) else {
        return Err(unshared("$.settled", &shared.hex()));
    };
    let (head, _) = SteadyWater::head(bytes)?;
    if head.key != key || head.cell != cell {
        return Err(mismatched("$.settled", &shared.hex()));
    }
    Ok(head.summary)
}

pub(crate) fn outside(key: TileKey) -> Error {
    Error::new(ErrorCode::OutOfRange, "$.key").got([u16::from(key.level), key.x, key.z])
}

fn unshared(path: &str, name: &str) -> Error {
    Error::new(ErrorCode::UnknownReference, path).got(name)
}

fn mismatched(path: &str, name: &str) -> Error {
    Error::new(ErrorCode::InvalidValue, path).got(name)
}
