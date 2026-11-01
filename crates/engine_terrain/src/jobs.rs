use engine_math::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use engine_core::{ErrorCode, Expected};
use glam::Vec2;

use crate::bounce::{MAX_PROBES, TerrainBounce, bounce};
use crate::codec::{self, Error, Reader, Writer, record};
use crate::lighting::{Ground, TerrainLighting, light};
use crate::macro_colors::{MACRO_BLOCK, MAX_MACRO_TEXELS, macro_colors};
use crate::shading::ShadingMaking;
use crate::steady_water::{PartMaking, SettledReads, Steadying};
use crate::{
    BiomeRules, Biomes, Generator, Layers, MacroCauses, Placing, Pyramid, Solid, Stamp,
    SteadySummary, SteadyWater, TILE_QUADS, Tile, TileKey, TileShading, TileWater, WaterPart,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobKind {
    Tile,
    Bake,
    Lighting,
    Water,
    Bounce,
    Macro,
    Steadied,
    Shading,
}

impl JobKind {
    pub const ALL: [Self; 8] = [
        Self::Tile,
        Self::Bake,
        Self::Lighting,
        Self::Water,
        Self::Bounce,
        Self::Macro,
        Self::Steadied,
        Self::Shading,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Tile => "terrain_tile",
            Self::Bake => "terrain_bake",
            Self::Lighting => "terrain_lighting",
            Self::Water => "terrain_water",
            Self::Bounce => "terrain_bounce",
            Self::Macro => "terrain_macro",
            Self::Steadied => "terrain_steadied",
            Self::Shading => "terrain_shading",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.name() == name)
    }
}

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
    pub stack: String,
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

const STACK: u8 = 0;
const RECIPE: u8 = 1;

pub fn share_stack(layers: &Layers) -> Vec<u8> {
    let mut writer = Writer::new();
    writer.put(&STACK).put(layers);
    writer.into_bytes()
}

pub fn share_recipe(recipe: &TerrainRecipe) -> Vec<u8> {
    let mut writer = Writer::new();
    writer.put(&RECIPE).put(recipe);
    writer.into_bytes()
}

fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[derive(Clone, Debug, PartialEq)]
pub struct TileJob {
    pub recipe: String,
    pub key: TileKey,
    pub stamps: Vec<Stamp>,
}

record!(TileJob {
    recipe,
    key,
    stamps,
});

impl TileJob {
    pub const KIND: JobKind = JobKind::Tile;

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
    pub const KIND: JobKind = JobKind::Bake;

    pub fn input(&self) -> Vec<u8> {
        codec::encode(self)
    }
}

pub fn bake_output(bytes: &[u8]) -> Result<Layers, Error> {
    codec::decode(bytes)
}

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

pub fn stamped(layers: &Arc<Layers>, stamps: &[Stamp]) -> Result<Arc<Layers>, Error> {
    if stamps.is_empty() {
        return Ok(layers.clone());
    }
    layers.with_stamps(stamps).map(Arc::new)
}

enum Running {
    Tile(Box<Generator>),
    Bake(Box<(Layers, Pyramid)>),
    Lighting(Box<(GroundMaking, LightingJob)>),
    Water(Box<PartMaking>),
    Bounce(Box<(GroundMaking, BounceJob, Arc<Layers>, Arc<BiomeRules>)>),
    Macro(Box<(Arc<Layers>, Pyramid, MacroCauses, MacroJob)>),
    Steadied(Box<Steadying>),
    Shading(Box<ShadingMaking>),
}

pub(crate) struct WalkedTile {
    pub(crate) identity: Box<[u8]>,
    pub(crate) key: TileKey,
    pub(crate) tile: Tile,
}

#[derive(Default)]
pub struct TerrainWorker {
    stacks: HashMap<String, Arc<Layers>>,
    recipes: HashMap<String, Recipe>,
    records: HashMap<String, Arc<[u8]>>,
    running: HashMap<u64, Running>,
    last_walk: Mutex<Option<WalkedTile>>,
}

impl TerrainWorker {
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn last_walk(&self) -> MutexGuard<'_, Option<WalkedTile>> {
        self.last_walk
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    pub fn share(&mut self, key: &str, bytes: impl Into<Arc<[u8]>>) -> Result<(), String> {
        self.keep(key, bytes.into())
            .map_err(|error| error.to_string())
    }

    fn keep(&mut self, key: &str, bytes: Arc<[u8]>) -> Result<(), Error> {
        let Some((&tag, rest)) = bytes.split_first() else {
            return Err(codec::invalid("$.tag").expected(Expected::Present));
        };
        match tag {
            STACK => {
                let layers: Layers = codec::decode(rest)?;
                self.stacks.insert(key.to_owned(), Arc::new(layers));
            }
            RECIPE => {
                let recipe: TerrainRecipe = codec::decode(rest)?;
                let stack = self.stacks.get(&recipe.stack).ok_or_else(|| {
                    Error::new(ErrorCode::UnknownReference, "$.stack").got(&recipe.stack)
                })?;
                let layers = match recipe.view {
                    Some(view) => {
                        Arc::new(stack.surroundings(view.offset, (view.low, view.high), view.blend))
                    }
                    None => stack.clone(),
                };
                let rules = recipe.surface.compile()?;
                self.recipes.insert(
                    key.to_owned(),
                    Recipe {
                        layers,
                        pyramid: recipe.pyramid,
                        rules: Arc::new(rules),
                        materials: recipe.materials,
                        around: recipe.view.is_some(),
                    },
                );
            }
            SETTLED | SUMMARY | SHADING => {
                self.records.insert(key.to_owned(), Arc::clone(&bytes));
            }
            _ => {
                let tags = [STACK, RECIPE, SETTLED, SUMMARY, SHADING].map(|tag| tag.to_string());
                return Err(codec::invalid("$.tag").got(tag).variants(tags));
            }
        }
        Ok(())
    }

    pub fn unshare(&mut self, key: &str) {
        self.stacks.remove(key);
        self.recipes.remove(key);
        self.records.remove(key);
    }

    pub fn stack(&self, key: &str) -> Option<&Arc<Layers>> {
        self.stacks.get(key)
    }

    pub fn run(&mut self, kind: &str, input: &[u8]) -> Result<Vec<u8>, String> {
        const WHOLE: u64 = u64::MAX;
        self.start(WHOLE, kind, input)?;
        loop {
            if let Some(output) = self.step(WHOLE, u64::MAX)? {
                return Ok(output);
            }
        }
    }

    pub fn start(&mut self, id: u64, kind: &str, input: &[u8]) -> Result<(), String> {
        let running = self.begin(kind, input).map_err(|error| error.to_string())?;
        self.running.insert(id, running);
        Ok(())
    }

    fn begin(&self, kind: &str, input: &[u8]) -> Result<Running, Error> {
        let kind = JobKind::from_name(kind).ok_or_else(|| {
            Error::new(ErrorCode::UnknownVariant, "$.kind")
                .got(kind)
                .variants(JobKind::ALL.map(JobKind::name))
        })?;
        let mut reader = Reader::new(input);
        Ok(match kind {
            JobKind::Tile => {
                let TileJob {
                    recipe,
                    key,
                    stamps,
                } = reader.get()?;
                reader.finish()?;
                let recipe = self.recipe(&recipe)?;
                if !recipe.pyramid.contains(key) {
                    return Err(outside(key));
                }
                let materials = recipe.materials && !TileShading::shades(key.level, recipe.around);
                Running::Tile(Box::new(
                    Generator::new(recipe.stamped(&stamps)?, &recipe.pyramid, key)
                        .with_surface(recipe.rules.clone())
                        .with_materials(materials),
                ))
            }
            JobKind::Bake => {
                let BakeJob { layers, pyramid } = reader.get()?;
                reader.finish()?;
                Running::Bake(Box::new((layers, pyramid)))
            }
            JobKind::Lighting => {
                let mut job: LightingJob = reader.get()?;
                reader.finish()?;
                job.check()?;
                let recipe = self.recipe(&job.recipe)?;
                let making = GroundMaking::new(recipe, job.level, std::mem::take(&mut job.tiles))?;
                Running::Lighting(Box::new((making, job)))
            }
            JobKind::Water => {
                let WaterJob {
                    recipe,
                    key,
                    stamps,
                    solids,
                    cell,
                } = reader.get()?;
                reader.finish()?;
                let recipe = self.recipe(&recipe)?;
                if key.level != 0 || !recipe.pyramid.contains(key) {
                    return Err(outside(key));
                }
                if !(cell.is_finite() && cell > 0.0) {
                    return Err(Error::new(ErrorCode::OutOfRange, "$.cell").got(cell));
                }
                if !solids.iter().flatten().all(Solid::valid) {
                    return Err(Error::new(ErrorCode::InvalidValue, "$.solids"));
                }
                Running::Water(Box::new(PartMaking::new(
                    (recipe.stamped(&stamps)?, recipe.rules.clone()),
                    recipe.pyramid,
                    key,
                    (cell, solids),
                )))
            }
            JobKind::Bounce => {
                let mut job: BounceJob = reader.get()?;
                reader.finish()?;
                job.check()?;
                let recipe = self.recipe(&job.recipe)?;
                let making = GroundMaking::new(recipe, job.level, std::mem::take(&mut job.tiles))?;
                let (layers, rules) = (recipe.layers.clone(), recipe.rules.clone());
                Running::Bounce(Box::new((making, job, layers, rules)))
            }
            JobKind::Macro => {
                let job: MacroJob = reader.get()?;
                reader.finish()?;
                let recipe = self.recipe(&job.recipe)?;
                job.check()?;
                if job.level >= recipe.pyramid.levels() {
                    return Err(level_outside(job.level, &recipe.pyramid));
                }
                let causes = recipe.rules.authored().macro_colors;
                Running::Macro(Box::new((
                    recipe.layers.clone(),
                    recipe.pyramid,
                    causes,
                    job,
                )))
            }
            JobKind::Steadied => {
                let job: SteadiedJob = reader.get()?;
                reader.finish()?;
                Running::Steadied(Box::new(self.steadying(&job)?))
            }
            JobKind::Shading => {
                let job: ShadingJob = reader.get()?;
                reader.finish()?;
                Running::Shading(Box::new(self.shading(job)?))
            }
        })
    }

    pub fn step(&mut self, id: u64, budget: u64) -> Result<Option<Vec<u8>>, String> {
        let Some(running) = self.running.get_mut(&id) else {
            return Err(Error::new(ErrorCode::UnknownReference, "$.id")
                .got(id)
                .to_string());
        };
        let output = match advance(running, budget.max(1)) {
            Ok(None) => return Ok(None),
            Ok(Some(output)) => Ok(Some(output)),
            Err(error) => Err(error.to_string()),
        };
        self.running.remove(&id);
        output
    }

    pub fn cancel(&mut self, id: u64) {
        self.running.remove(&id);
    }

    fn recipe(&self, key: &str) -> Result<&Recipe, Error> {
        self.recipes
            .get(key)
            .ok_or_else(|| Error::new(ErrorCode::UnknownReference, "$.recipe").got(key))
    }
}

fn advance(running: &mut Running, mut budget: u64) -> Result<Option<Vec<u8>>, Error> {
    let budget = &mut budget;
    Ok(Some(match running {
        Running::Tile(generator) => match generator.advance(budget) {
            Some(tile) => codec::encode(&tile),
            None if generator.failed() => return Err(invalid_heights()),
            None => return Ok(None),
        },
        Running::Bake(bake) => {
            let (layers, pyramid) = &**bake;
            codec::encode(&layers.bake_all(pyramid))
        }
        Running::Lighting(held) => {
            let (making, job) = &mut **held;
            if !making.advance(budget)? {
                return Ok(None);
            }
            codec::encode(&light(
                &making.ground(),
                making.extent(),
                job.sky,
                job.horizons,
            ))
        }
        Running::Water(part) => match part.advance(budget)? {
            Some(part) => codec::encode(&part),
            None => return Ok(None),
        },
        Running::Bounce(held) => {
            let (making, job, layers, rules) = &mut **held;
            if !making.advance(budget)? {
                return Ok(None);
            }
            codec::encode(&bounce(
                &making.ground(),
                making.extent(),
                (layers.as_ref(), rules.as_ref()),
                job.probes,
            ))
        }
        Running::Macro(colors) => {
            let (layers, pyramid, causes, job) = &**colors;
            macro_colors(layers, pyramid, job.level, job.size, causes)
                .ok_or_else(invalid_heights)?
        }
        Running::Steadied(steadying) => match steadying.advance(budget)? {
            Some(water) => codec::encode(&water),
            None => return Ok(None),
        },
        Running::Shading(making) => match making.advance(budget)? {
            Some(shading) => codec::encode(&shading),
            None => return Ok(None),
        },
    }))
}

pub(crate) fn invalid_heights() -> Error {
    Error::new(ErrorCode::InvalidTerrain, "$.heights").expected(Expected::Finite)
}

pub(crate) fn outside(key: TileKey) -> Error {
    Error::new(ErrorCode::OutOfRange, "$.key").got([u16::from(key.level), key.x, key.z])
}

fn level_outside(level: u32, pyramid: &Pyramid) -> Error {
    Error::new(ErrorCode::OutOfRange, "$.level")
        .at_most(u64::from(pyramid.levels().saturating_sub(1)))
        .got(level)
}

fn unshared(path: &str, name: &str) -> Error {
    Error::new(ErrorCode::UnknownReference, path).got(name)
}

fn mismatched(path: &str, name: &str) -> Error {
    Error::new(ErrorCode::InvalidValue, path).got(name)
}
