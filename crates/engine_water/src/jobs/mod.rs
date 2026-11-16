use std::sync::Arc;

use engine_codec::{Expected, Reader};
use engine_core::{Error, ErrorCode, Job, JobCx, JobKind, RecordKey, codec};
use engine_heightfield::layer::LayerKind;
use engine_heightfield::tile::{Recipe, TileJob};
use engine_heightfield::{INVALID_TERRAIN, TileKey};
use engine_open_world_kit::terrain::TerrainLayers;

use drawn::{start_drawn, start_rewater};
use tile_water::BaseWater;

pub(crate) mod drawn;
pub(crate) mod gpu;
pub mod shallow_water;
pub(crate) mod tile_water;

include!(concat!(env!("OUT_DIR"), "/jobs.rs"));

pub static TILE_WATER: JobKind = JobKind::new("water::tile", &SOURCE, start_tile_water);
pub static DRAWN: JobKind = JobKind::new("water::drawn", &SOURCE, start_drawn);
pub static REWATER: JobKind = JobKind::new("water::rewater", &SOURCE, start_rewater);

pub(crate) fn invalid_heights() -> Error {
    Error::new(INVALID_TERRAIN, "$.heights").expected(Expected::Finite)
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

pub(crate) struct Finished(pub(crate) Option<Vec<u8>>);

impl Job for Finished {
    fn step(&mut self, _: u64, _: &mut JobCx<'_>) -> Result<Option<Vec<u8>>, Error> {
        Ok(self.0.take())
    }
}

pub(crate) fn outside(key: TileKey) -> Error {
    Error::new(ErrorCode::OutOfRange, "$.key").got([u16::from(key.level), key.x, key.z])
}

fn unshared(path: &str, name: &str) -> Error {
    Error::new(ErrorCode::UnknownReference, path).got(name)
}
