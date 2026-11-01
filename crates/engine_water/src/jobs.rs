use engine_core::ErrorCode;

use crate::codec::{self, Error, record};
use crate::steady_water::{SettledReads, Steadying};
use crate::{Solid, Stamp, SteadySummary, SteadyWater, TileKey, TileWater, WaterPart};

const SETTLED: u8 = 2;
const SUMMARY: u8 = 3;

pub fn share_settled(record: &[u8]) -> Vec<u8> {
    let mut shared = Vec::with_capacity(record.len() + 1);
    shared.push(SETTLED);
    shared.extend_from_slice(record);
    shared
}

pub fn share_summary(head: &[u8]) -> Vec<u8> {
    let mut shared = Vec::with_capacity(head.len() + 1);
    shared.push(SUMMARY);
    shared.extend_from_slice(head);
    shared
}

pub fn settled_key(record: &[u8; 32]) -> String {
    format!("settled-{}", hex(record))
}

pub fn summary_key(record: &[u8; 32]) -> String {
    format!("summary-{}", hex(record))
}

#[derive(Clone, Debug, PartialEq)]
pub struct WaterJob {
    pub recipe: String,
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
    pub const KIND: JobKind = JobKind::Water;

    pub fn input(&self) -> Vec<u8> {
        codec::encode(self)
    }
}

pub fn water_output(bytes: &[u8]) -> Result<WaterPart, Error> {
    let part: WaterPart = codec::decode(bytes)?;
    if !part.valid() {
        return Err(codec::invalid("$.water_part"));
    }
    Ok(part)
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
    pub const KIND: JobKind = JobKind::Steadied;

    pub fn input(&self) -> Vec<u8> {
        codec::encode(self)
    }
}

pub fn steadied_output(bytes: &[u8]) -> Result<Option<TileWater>, Error> {
    codec::decode(bytes)
}

impl TerrainWorker {
    fn steadying(&self, job: &SteadiedJob) -> Result<Steadying, Error> {
        let recipe = self.recipe(&job.tile.recipe)?;
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
                    .map(|&(at, record)| Ok((at, self.settled(&record, at, cell)?)))
                    .collect::<Result<_, Error>>()?,
            )
        } else {
            SettledReads::Summaries(
                job.settled
                    .iter()
                    .map(|&(at, record)| Ok((at, self.summary(&record, at, cell)?)))
                    .collect::<Result<_, Error>>()?,
            )
        };
        Ok(Steadying::new(
            recipe.stamped(&job.tile.stamps)?,
            recipe.pyramid,
            key,
            (cell, reads),
        ))
    }

    fn settled(&self, record: &[u8; 32], key: TileKey, cell: f32) -> Result<SteadyWater, Error> {
        let name = settled_key(record);
        let water: SteadyWater = match self.records.get(&name).map(|bytes| bytes.split_first()) {
            Some(Some((&SETTLED, rest))) => codec::decode(rest)?,
            _ => return Err(unshared("$.settled", &name)),
        };
        if water.key != key || water.cell != cell {
            return Err(mismatched("$.settled", &name));
        }
        Ok(water)
    }

    fn summary(&self, record: &[u8; 32], key: TileKey, cell: f32) -> Result<SteadySummary, Error> {
        let name = summary_key(record);
        let (head, _) = match self.records.get(&name).map(|bytes| bytes.split_first()) {
            Some(Some((&SUMMARY, rest))) => SteadyWater::head(rest)?,
            _ => return Err(unshared("$.settled", &name)),
        };
        if head.key != key || head.cell != cell {
            return Err(mismatched("$.settled", &name));
        }
        Ok(head.summary)
    }
}
