use std::cell::RefCell;
use std::sync::Arc;

use engine_core::{Error, ErrorCode, Expected, Job, JobCx, JobKind, RecordKey};
use engine_places::ground::{GroundBridge, GroundGrid};
use glam::Vec2;

use crate::codec::{self, Reader, Writer, record};
use crate::layers::with_stamps;
use crate::runtime::reaches;
use crate::{
    BiomeRules, Biomes, Generator, Layers, Placing, Pyramid, SOURCE, Stamp, TILE_QUADS,
    TILE_SAMPLES, TILE_STRIDE, Tile, TileKey,
};

pub static TILE_BUILD: JobKind = JobKind::new("terrain::tile", &SOURCE, start_tile);
pub static BAKE: JobKind = JobKind::new("terrain::bake", &SOURCE, start_bake);
pub static GROUND_GRID: JobKind = JobKind::new("terrain::ground_grid", &SOURCE, start_grid);

codec::tagged!(Placing {
    0 => All,
    1 => Near,
    2 => Far,
});

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StackView {
    pub offset: Vec2,
    pub low: Vec2,
    pub high: Vec2,
    pub blend: f32,
}

record!(StackView {
    offset,
    low,
    high,
    blend,
});

#[derive(Clone, Debug, PartialEq)]
pub struct TerrainRecipe {
    pub stack: [u8; 32],
    pub view: Option<StackView>,
    pub pyramid: Pyramid,
    pub surface: Biomes,
    pub materials: bool,
}

record!(TerrainRecipe {
    stack,
    view,
    pyramid,
    surface,
    materials,
});

pub fn share_stack(layers: &Layers) -> Arc<[u8]> {
    let mut writer = Writer::new();
    writer.put(layers);
    Arc::from(writer.into_bytes())
}

pub fn share_recipe(recipe: &TerrainRecipe) -> Arc<[u8]> {
    let mut writer = Writer::new();
    writer.put(recipe);
    Arc::from(writer.into_bytes())
}

#[derive(Clone, Debug, PartialEq)]
pub struct TileJob {
    pub recipe: [u8; 32],
    pub key: TileKey,
    pub stamps: Vec<Stamp>,
}

record!(TileJob {
    recipe,
    key,
    stamps,
});

impl TileJob {
    pub fn identity(&self) -> Vec<u8> {
        codec::encode(self)
    }
}

pub fn tile_output(bytes: &[u8]) -> Result<Tile, Error> {
    codec::decode(bytes)
}

#[derive(Clone, Debug, PartialEq)]
pub struct BakeJob {
    pub layers: Layers,
    pub pyramid: Pyramid,
}

record!(BakeJob { layers, pyramid });

impl BakeJob {
    pub fn input(&self) -> Vec<u8> {
        codec::encode(self)
    }
}

pub fn bake_output(bytes: &[u8]) -> Result<Layers, Error> {
    codec::decode(bytes)
}

#[derive(Clone, Debug, PartialEq)]
pub struct GroundIdentity {
    pub stack: [u8; 32],
    pub pyramid: Pyramid,
    pub edits: Vec<Stamp>,
}

record!(GroundIdentity {
    stack,
    pyramid,
    edits,
});

#[derive(Clone, Debug, PartialEq)]
pub struct GridJob {
    pub identity: GroundIdentity,
    pub key: TileKey,
}

record!(GridJob { identity, key });

struct Recipe {
    layers: Arc<Layers>,
    pyramid: Pyramid,
    rules: Arc<BiomeRules>,
    materials: bool,
    around: bool,
}

impl Recipe {
    fn stamped(&self, stamps: &[Stamp]) -> Result<Arc<Layers>, Error> {
        stamped(&self.layers, stamps)
    }
}

const HELD_RECIPES: usize = 8;

thread_local! {
    static RECIPES: RefCell<Vec<([u8; 32], Arc<Recipe>)>> = const { RefCell::new(Vec::new()) };
    static STACKS: RefCell<Vec<([u8; 32], Arc<Layers>)>> = const { RefCell::new(Vec::new()) };
}

fn held<T>(
    cache: &'static std::thread::LocalKey<RefCell<Vec<([u8; 32], Arc<T>)>>>,
    key: [u8; 32],
    make: impl FnOnce() -> Result<T, Error>,
) -> Result<Arc<T>, Error> {
    if let Some(found) = cache.with_borrow(|held| {
        held.iter()
            .find(|(at, _)| *at == key)
            .map(|(_, value)| Arc::clone(value))
    }) {
        return Ok(found);
    }
    let made = Arc::new(make()?);
    cache.with_borrow_mut(|held| {
        if held.len() >= HELD_RECIPES {
            held.remove(0);
        }
        held.push((key, Arc::clone(&made)));
    });
    Ok(made)
}

fn shared<'c>(cx: &'c JobCx<'_>, key: [u8; 32], path: &str) -> Result<&'c [u8], Error> {
    cx.share(RecordKey::from_hash(key)).ok_or_else(|| {
        Error::new(ErrorCode::UnknownReference, path).got(RecordKey::from_hash(key).hex())
    })
}

fn stack(cx: &JobCx<'_>, key: [u8; 32]) -> Result<Arc<Layers>, Error> {
    held(&STACKS, key, || codec::decode(shared(cx, key, "$.stack")?))
}

fn recipe(cx: &JobCx<'_>, key: [u8; 32]) -> Result<Arc<Recipe>, Error> {
    held(&RECIPES, key, || {
        let recipe: TerrainRecipe = codec::decode(shared(cx, key, "$.recipe")?)?;
        let stack = stack(cx, recipe.stack)?;
        let layers = match recipe.view {
            Some(view) => {
                Arc::new(stack.surroundings(view.offset, (view.low, view.high), view.blend))
            }
            None => stack,
        };
        let rules = recipe.surface.compile()?;
        Ok(Recipe {
            layers,
            pyramid: recipe.pyramid,
            rules: Arc::new(rules),
            materials: recipe.materials,
            around: recipe.view.is_some(),
        })
    })
}

pub fn stamped(layers: &Arc<Layers>, stamps: &[Stamp]) -> Result<Arc<Layers>, Error> {
    if stamps.is_empty() {
        return Ok(layers.clone());
    }
    with_stamps(layers, stamps).map(Arc::new)
}

pub(crate) fn shaded_elsewhere(level: u8, around: bool) -> bool {
    level > 0 && !around
}

fn start_tile(input: &[u8], cx: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    let mut reader = Reader::new(input);
    let TileJob {
        recipe: key,
        key: tile,
        stamps,
    } = reader.get()?;
    reader.finish()?;
    let recipe = recipe(cx, key)?;
    if !recipe.pyramid.contains(tile) {
        return Err(outside(tile));
    }
    let materials = recipe.materials && !shaded_elsewhere(tile.level, recipe.around);
    Ok(Box::new(TileBuild(Box::new(
        Generator::new(recipe.stamped(&stamps)?, &recipe.pyramid, tile)
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

fn start_bake(input: &[u8], _: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    let mut reader = Reader::new(input);
    let BakeJob { layers, pyramid } = reader.get()?;
    reader.finish()?;
    Ok(Box::new(Baking(Box::new((layers, pyramid)))))
}

struct Baking(Box<(Layers, Pyramid)>);

impl Job for Baking {
    fn step(&mut self, _: u64, _: &mut JobCx<'_>) -> Result<Option<Vec<u8>>, Error> {
        let (layers, pyramid) = &*self.0;
        Ok(Some(codec::encode(&crate::bake::bake_all(layers, pyramid))))
    }
}

fn start_grid(input: &[u8], cx: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    let mut reader = Reader::new(input);
    let GridJob { identity, key } = reader.get()?;
    reader.finish()?;
    if !identity.pyramid.contains(key) {
        return Err(outside(key));
    }
    let layers = stack(cx, identity.stack)?;
    let reaching: Vec<Stamp> = identity
        .edits
        .iter()
        .copied()
        .filter(|stamp| reaches(stamp, &identity.pyramid, key))
        .collect();
    let layers = stamped(&layers, &reaching)?;
    Ok(Box::new(Gridding(Some(Box::new((layers, identity.pyramid, key))))))
}

struct Gridding(Option<Box<(Arc<Layers>, Pyramid, TileKey)>>);

impl Job for Gridding {
    fn step(&mut self, _: u64, _: &mut JobCx<'_>) -> Result<Option<Vec<u8>>, Error> {
        let Some(held) = self.0.take() else {
            return Err(Error::new(ErrorCode::InvalidValue, "$.step"));
        };
        let (layers, pyramid, key) = *held;
        let mut whole = u64::MAX;
        let tile = Generator::new(layers.clone(), &pyramid, key)
            .advance(&mut whole)
            .ok_or_else(invalid_heights)?;
        let tile: Tile = codec::decode(&codec::encode(&tile))?;
        let bridges = ground_bridges(&layers);
        Ok(Some(codec::encode(&ground_grid(
            &tile, &layers, &pyramid, bridges,
        ))))
    }
}

pub(crate) fn ground_bridges(layers: &Layers) -> Arc<[GroundBridge]> {
    layers
        .bridges()
        .into_iter()
        .map(|bridge| GroundBridge {
            deck: bridge.deck,
            width: bridge.width,
            tunnel: bridge.tunnel,
        })
        .collect()
}

fn ground_grid(
    tile: &Tile,
    layers: &Layers,
    pyramid: &Pyramid,
    bridges: Arc<[GroundBridge]>,
) -> GroundGrid {
    let side = TILE_SAMPLES as usize;
    let key = tile.key();
    let spacing = pyramid.spacing(u32::from(key.level));
    let stride = TILE_STRIDE as usize;
    let stored = |x: usize, z: usize| (z + 1) * stride + x + 1;
    let mut heights = vec![0.0; side * side].into_boxed_slice();
    for z in 0..side {
        for x in 0..side {
            heights[z * side + x] = tile.heights()[stored(x, z)];
        }
    }
    let first = pyramid.first_sample(key);
    GroundGrid {
        key,
        spacing,
        heights,
        holes: holes(layers, first, spacing),
        bridges,
    }
}

fn position(sample: [i64; 2], spacing: f32) -> Vec2 {
    Vec2::new(sample[0] as f32, sample[1] as f32) * spacing
}

fn holes(layers: &Layers, first: [i64; 2], spacing: f32) -> Option<Box<[u64]>> {
    let side = TILE_SAMPLES as usize;
    let quads = TILE_QUADS as i64;
    let last = [first[0] + quads, first[1] + quads];
    if !layers.holes_reach(position(first, spacing), position(last, spacing)) {
        return None;
    }
    let mut bits = vec![0_u64; (side * side).div_ceil(64)];
    let mut any = false;
    for z in 0..side {
        for x in 0..side {
            let point = position([first[0] + x as i64, first[1] + z as i64], spacing);
            if layers.hole_at(point.x, point.y) {
                let index = z * side + x;
                bits[index / 64] |= 1 << (index % 64);
                any = true;
            }
        }
    }
    any.then(|| bits.into_boxed_slice())
}

pub(crate) fn invalid_heights() -> Error {
    Error::new(ErrorCode::InvalidValue, "$.heights").expected(Expected::Finite)
}

pub(crate) fn outside(key: TileKey) -> Error {
    Error::new(ErrorCode::OutOfRange, "$.key").got([u16::from(key.level), key.x, key.z])
}
