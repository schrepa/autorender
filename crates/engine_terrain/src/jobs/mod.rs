use std::sync::Arc;

use engine_codec::{self as codec, Reader, Writer};
use engine_core::{Error, ErrorCode, Expected, Job, JobCx, JobKind, RecordKey};
use engine_heightfield::grid::{Pyramid, TileKey};
use engine_heightfield::layer::LayerKind;
use engine_heightfield::tile::{Generator, Recipe, TerrainRecipe, Tile, TileJob, shaded_elsewhere};
use engine_heightfield::{INVALID_TERRAIN, Layers};
use engine_open_world_kit::terrain::TerrainLayers;

pub mod bake;
pub mod expr_layer;

include!(concat!(env!("OUT_DIR"), "/jobs.rs"));

pub static TILE_BUILD: JobKind = JobKind::new("terrain::tile", &SOURCE, start_tile);
pub static BAKE: JobKind = JobKind::new("terrain::bake", &SOURCE, start_bake);

pub fn share_stack(layers: &Layers) -> Arc<[u8]> {
    let mut writer = Writer::new();
    layers.write(&mut writer);
    Arc::from(writer.into_bytes())
}

pub fn share_recipe(recipe: &TerrainRecipe) -> Arc<[u8]> {
    let mut writer = Writer::new();
    writer.put(recipe);
    Arc::from(writer.into_bytes())
}

pub fn tile_output(bytes: &[u8]) -> Result<Tile, Error> {
    codec::decode(bytes)
}

#[derive(Clone, Debug, PartialEq)]
pub struct BakeJob {
    pub layers: Layers,
    pub pyramid: Pyramid,
}

impl BakeJob {
    pub fn input(&self) -> Vec<u8> {
        let mut writer = Writer::new();
        self.layers.write(&mut writer);
        writer.put(&self.pyramid);
        writer.into_bytes()
    }

    fn read(input: &[u8], kinds: &[&'static LayerKind]) -> Result<Self, Error> {
        let mut reader = Reader::new(input);
        let layers = Layers::read(&mut reader, kinds)?;
        let pyramid = reader.get()?;
        reader.finish()?;
        Ok(Self { layers, pyramid })
    }
}

pub fn bake_output(bytes: &[u8], kinds: &[&'static LayerKind]) -> Result<Layers, Error> {
    let mut reader = Reader::new(bytes);
    let layers = Layers::read(&mut reader, kinds)?;
    reader.finish()?;
    Ok(layers)
}

fn baked_bytes(layers: &Layers) -> Vec<u8> {
    let mut writer = Writer::new();
    layers.write(&mut writer);
    writer.into_bytes()
}

fn kinds(cx: &JobCx<'_>) -> Vec<&'static LayerKind> {
    cx.members::<TerrainLayers>().copied().collect()
}

fn shared<'c>(cx: &'c JobCx<'_>, key: [u8; 32], path: &str) -> Result<&'c [u8], Error> {
    cx.share(RecordKey::from_hash(key)).ok_or_else(|| {
        Error::new(ErrorCode::UnknownReference, path).got(RecordKey::from_hash(key).hex())
    })
}

fn recipe(
    cx: &JobCx<'_>,
    key: [u8; 32],
    kinds: &[&'static LayerKind],
) -> Result<Arc<Recipe>, Error> {
    cx.held(key, || {
        Recipe::read(
            shared(cx, key, "$.recipe")?,
            &|stack| cx.share(RecordKey::from_hash(stack)),
            kinds,
        )
    })
}

fn start_tile(input: &[u8], cx: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    let mut reader = Reader::new(input);
    let TileJob {
        recipe: key,
        key: tile,
        stamps,
    } = reader.get()?;
    reader.finish()?;
    let kinds = kinds(cx);
    let recipe = recipe(cx, key, &kinds)?;
    if !recipe.pyramid.contains(tile) {
        return Err(outside(tile));
    }
    let materials = recipe.materials && !shaded_elsewhere(tile.level, recipe.around);
    Ok(Box::new(TileBuild(Box::new(
        Generator::new(recipe.stamped(&stamps, &kinds)?, &recipe.pyramid, tile)
            .with_surface(recipe.rules.clone())
            .with_materials(materials),
    ))))
}

struct TileBuild(Box<Generator>);

impl Job for TileBuild {
    fn step(&mut self, budget: u64, _: &mut JobCx<'_>) -> Result<Option<Vec<u8>>, Error> {
        let mut budget = budget.max(1);
        match self.0.advance(&mut budget) {
            Some(tile) => Ok(Some(codec::encode(&tile))),
            None if self.0.failed() => Err(invalid_heights()),
            None => Ok(None),
        }
    }
}

fn start_bake(input: &[u8], cx: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    let BakeJob { layers, pyramid } = BakeJob::read(input, &kinds(cx))?;
    Ok(Box::new(Baking(Box::new((layers, pyramid)))))
}

struct Baking(Box<(Layers, Pyramid)>);

impl Job for Baking {
    fn step(&mut self, _: u64, _: &mut JobCx<'_>) -> Result<Option<Vec<u8>>, Error> {
        let (layers, pyramid) = &*self.0;
        Ok(Some(baked_bytes(&bake::bake_all(layers, pyramid))))
    }
}

pub(crate) fn invalid_heights() -> Error {
    Error::new(INVALID_TERRAIN, "$.heights").expected(Expected::Finite)
}

pub(crate) fn outside(key: TileKey) -> Error {
    Error::new(ErrorCode::OutOfRange, "$.key").got([u16::from(key.level), key.x, key.z])
}
