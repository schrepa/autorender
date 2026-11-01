use engine_codec::{Expected, Reader, record};
use engine_core::{Error, ErrorCode, Job, JobCx, JobKind, RecordKey, codec};
use engine_heightfield::tile::{Recipe, Solid, TileJob};
use engine_heightfield::{Stamp, TileKey};

use crate::SOURCE;
use crate::settled::settled_alike;
use crate::steady_water::{PartMaking, SettledReads, Steadying};
use crate::tile_water::BaseWater;
use crate::{SteadySummary, SteadyWater, TileWater, WaterPart};

pub static TILE_WATER: JobKind = JobKind::new("water::tile", &SOURCE, start_tile_water);
pub static PARTS: JobKind = JobKind::new("water::part", &SOURCE, start_part);
pub static STEADIED: JobKind = JobKind::new("water::steadied", &SOURCE, start_steadied);
pub static SETTLED: JobKind =
    JobKind::new("water::settled", &SOURCE, start_settled).kept_alike(settled_alike);

const SETTLED_SHARE: u8 = 2;
const SUMMARY_SHARE: u8 = 3;

pub fn share_settled(record: &[u8]) -> Vec<u8> {
    let mut shared = Vec::with_capacity(record.len() + 1);
    shared.push(SETTLED_SHARE);
    shared.extend_from_slice(record);
    shared
}

pub fn share_summary(head: &[u8]) -> Vec<u8> {
    let mut shared = Vec::with_capacity(head.len() + 1);
    shared.push(SUMMARY_SHARE);
    shared.extend_from_slice(head);
    shared
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

pub fn tile_water_output(bytes: &[u8]) -> Result<Option<TileWater>, Error> {
    codec::decode(bytes)
}

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

pub fn steadied_output(bytes: &[u8]) -> Result<Option<TileWater>, Error> {
    codec::decode(bytes)
}

pub(crate) fn invalid_heights() -> Error {
    Error::new(ErrorCode::InvalidTerrain, "$.heights").expected(Expected::Finite)
}

fn recipe(cx: &JobCx<'_>, shared: [u8; 32]) -> Result<Recipe, Error> {
    let key = RecordKey::from_hash(shared);
    let bytes = cx
        .share(key)
        .ok_or_else(|| unshared("$.recipe", &key.hex()))?;
    codec::decode(bytes)
}

fn start_tile_water(input: &[u8], cx: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    let mut reader = Reader::new(input);
    let job: TileJob = reader.get()?;
    reader.finish()?;
    let recipe = recipe(cx, job.recipe)?;
    if !recipe.pyramid.contains(job.key) {
        return Err(outside(job.key));
    }
    Ok(Box::new(BaseWater::new(
        recipe.stamped(&job.stamps)?,
        recipe.pyramid,
        job.key,
    )))
}

impl Job for BaseWater {
    fn step(&mut self, budget: u64, _: &mut JobCx<'_>) -> Result<Option<Vec<u8>>, Error> {
        let mut budget = budget;
        Ok(self.advance(&mut budget)?.map(|water| codec::encode(&water)))
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
    let recipe = recipe(cx, shared)?;
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
        (recipe.stamped(&stamps)?, recipe.rules.clone()),
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
    let recipe = recipe(cx, job.tile.recipe)?;
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
        recipe.stamped(&job.tile.stamps)?,
        recipe.pyramid,
        key,
        (cell, reads),
    )))
}

impl Job for Steadying {
    fn step(&mut self, budget: u64, _: &mut JobCx<'_>) -> Result<Option<Vec<u8>>, Error> {
        let mut budget = budget;
        Ok(self.advance(&mut budget)?.map(|water| codec::encode(&water)))
    }
}

fn start_settled(_: &[u8], _: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    Err(Error::new(ErrorCode::JobUnsupported, "$.jobs.water::settled").got("water::settled"))
}

fn settled(cx: &JobCx<'_>, record: [u8; 32], key: TileKey, cell: f32) -> Result<SteadyWater, Error> {
    let shared = RecordKey::from_hash(record);
    let water: SteadyWater = match cx.share(shared).map(<[u8]>::split_first) {
        Some(Some((&SETTLED_SHARE, rest))) => codec::decode(rest)?,
        _ => return Err(unshared("$.settled", &shared.hex())),
    };
    if water.key != key || water.cell != cell {
        return Err(mismatched("$.settled", &shared.hex()));
    }
    Ok(water)
}

fn summary(cx: &JobCx<'_>, record: [u8; 32], key: TileKey, cell: f32) -> Result<SteadySummary, Error> {
    let shared = RecordKey::from_hash(record);
    let (head, _) = match cx.share(shared).map(<[u8]>::split_first) {
        Some(Some((&SUMMARY_SHARE, rest))) => SteadyWater::head(rest)?,
        _ => return Err(unshared("$.settled", &shared.hex())),
    };
    if head.key != key || head.cell != cell {
        return Err(mismatched("$.settled", &shared.hex()));
    }
    Ok(head.summary)
}

fn outside(key: TileKey) -> Error {
    Error::new(ErrorCode::OutOfRange, "$.key").got([u16::from(key.level), key.x, key.z])
}

fn unshared(path: &str, name: &str) -> Error {
    Error::new(ErrorCode::UnknownReference, path).got(name)
}

fn mismatched(path: &str, name: &str) -> Error {
    Error::new(ErrorCode::InvalidValue, path).got(name)
}
