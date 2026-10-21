//! Terrain jobs: terrain work engines hand workers, as bytes
//! in and bytes out: a tile generated whole, a walk searched, a stack
//! baked. Workers hold once what many jobs read, shared by key: layer
//! stacks with their bakes, and the recipes generating a terrain's tiles
//! from one. Jobs run the same code wherever they run (a worker, a host
//! without workers, the cook), so their results are the same bits, and a
//! tile's and a bake's output is a record hosts keep and the cook ships:
//! every field of a job is something its output depends on.
//!
//! A tile's job carries the edits reaching its ground alone, its recipe's
//! stack none: an edit leaves every other tile's job, and its record, as
//! it was.

use engine_math::collections::HashMap;
use std::sync::Arc;

use glam::Vec2;

use crate::codec::{self, CodecError, Reader, Writer, record};
use crate::{
    BiomeRules, Biomes, Generator, Layers, NavOptions, NavPath, Obstacle, Placing, Pyramid, Stamp,
    Tile, TileKey, WalkSearch,
};

/// The kinds of terrain jobs, which hosts carry by name: a tile of a
/// shared recipe, a walk over one, and a stack's bakes. Workers match every
/// kind, so a kind added without its work fails to build.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobKind {
    Tile,
    Walk,
    Bake,
}

impl JobKind {
    /// Every kind.
    pub const ALL: [Self; 3] = [Self::Tile, Self::Walk, Self::Bake];

    /// Its name, as hosts carry it and key the records they keep by.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Tile => "terrain_tile",
            Self::Walk => "terrain_walk",
            Self::Bake => "terrain_bake",
        }
    }

    /// The kind named `name`.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.name() == name)
    }
}

codec::tagged!(Placing, "invalid placing" {
    0 => All,
    1 => Near,
    2 => Far,
});

/// How a terrain's surroundings see a stack: as
/// [`Layers::surroundings`] takes them.
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

/// How a terrain's tiles generate: from the stack shared as `stack`, seen
/// through `view` for surroundings, over `pyramid`, with `surface`'s
/// control values and placed objects.
#[derive(Clone, Debug, PartialEq)]
pub struct TerrainRecipe {
    pub stack: String,
    pub view: Option<StackView>,
    pub pyramid: Pyramid,
    pub surface: Biomes,
    /// Whether tiles hold control values; without, the GPU gives them.
    pub materials: bool,
}

record!(TerrainRecipe {
    stack,
    view,
    pyramid,
    surface,
    materials,
});

/// What shared bytes hold: a stack, or a recipe naming one.
const STACK: u8 = 0;
const RECIPE: u8 = 1;

/// `layers` (with their bakes) as workers share them.
pub fn share_stack(layers: &Layers) -> Vec<u8> {
    let mut writer = Writer::new();
    writer.put(&STACK).put(layers);
    writer.into_bytes()
}

/// `recipe` as workers share it: after the stack it names.
pub fn share_recipe(recipe: &TerrainRecipe) -> Vec<u8> {
    let mut writer = Writer::new();
    writer.put(&RECIPE).put(recipe);
    writer.into_bytes()
}

/// A [`JobKind::Tile`]: tile `key` of the recipe shared as `recipe`, whole,
/// with every object its level places, under `stamps`, the terrain's edits
/// reaching its ground, in order. Like every terrain job, it is its own
/// identity and its own input.
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

    /// What its output is kept by, and its input: the job.
    pub fn identity(&self) -> Vec<u8> {
        codec::encode(self)
    }
}

/// A [`JobKind::Tile`]'s output: the tile.
pub fn tile_output(bytes: &[u8]) -> Result<Tile, CodecError> {
    codec::decode(bytes)
}

/// A [`JobKind::Walk`]: the shortest walk over the recipe shared as `recipe`
/// under `stamps`, the terrain's edits in order, from terrain-local `from`
/// to `to` among `obstacles`, as `options` walk.
#[derive(Clone, Debug, PartialEq)]
pub struct WalkJob {
    pub recipe: String,
    pub stamps: Vec<Stamp>,
    pub from: Vec2,
    pub to: Vec2,
    pub obstacles: Vec<Obstacle>,
    pub options: NavOptions,
}

record!(WalkJob {
    recipe,
    stamps,
    from,
    to,
    obstacles,
    options,
});

impl WalkJob {
    pub const KIND: JobKind = JobKind::Walk;

    /// What its output is kept by, and its input: the job.
    pub fn identity(&self) -> Vec<u8> {
        codec::encode(self)
    }
}

/// A [`JobKind::Walk`]'s output: the walk, `None` where none reaches.
pub fn walk_output(bytes: &[u8]) -> Result<Option<NavPath>, CodecError> {
    codec::decode(bytes)
}

/// A [`JobKind::Bake`]: every bake of `layers` over `pyramid`. Its input is the
/// job; as its layers can hold whole heightmaps, hosts keep its output by
/// that input's content hash, which engines compute.
#[derive(Clone, Debug, PartialEq)]
pub struct BakeJob {
    pub layers: Layers,
    pub pyramid: Pyramid,
}

record!(BakeJob { layers, pyramid });

impl BakeJob {
    pub const KIND: JobKind = JobKind::Bake;

    /// Its input: the job.
    pub fn input(&self) -> Vec<u8> {
        codec::encode(self)
    }
}

/// A [`JobKind::Bake`]'s output: the layers with every bake done.
pub fn bake_output(bytes: &[u8]) -> Result<Layers, CodecError> {
    codec::decode(bytes)
}

/// A shared recipe as a worker holds it: its layers seen as its terrain
/// sees them, its pyramid and its compiled surface.
struct Recipe {
    layers: Arc<Layers>,
    pyramid: Pyramid,
    rules: Arc<BiomeRules>,
    materials: bool,
}

impl Recipe {
    /// Its layers under `stamps`, edits in order.
    fn stamped(&self, stamps: &[Stamp]) -> Result<Arc<Layers>, String> {
        if stamps.is_empty() {
            return Ok(self.layers.clone());
        }
        self.layers
            .with_stamps(stamps)
            .map(Arc::new)
            .map_err(|error| error.to_string())
    }
}

/// A job a worker started, to step until done.
enum Running {
    Tile(Box<Generator>),
    Walk(Box<WalkSearch>),
    Bake(Box<(Layers, Pyramid)>),
}

/// What a worker holds of terrains, and runs their jobs over.
#[derive(Default)]
pub struct TerrainWorker {
    stacks: HashMap<String, Arc<Layers>>,
    recipes: HashMap<String, Recipe>,
    running: HashMap<u64, Running>,
}

impl TerrainWorker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Holds the shared `bytes` as `key`: a stack, or a recipe whose stack
    /// it already holds.
    pub fn share(&mut self, key: &str, bytes: &[u8]) -> Result<(), String> {
        let (&tag, rest) = bytes.split_first().ok_or("empty share")?;
        match tag {
            STACK => {
                let layers: Layers = codec::decode(rest).map_err(|error| error.to_string())?;
                self.stacks.insert(key.to_owned(), Arc::new(layers));
            }
            RECIPE => {
                let recipe: TerrainRecipe =
                    codec::decode(rest).map_err(|error| error.to_string())?;
                let stack = self
                    .stacks
                    .get(&recipe.stack)
                    .ok_or_else(|| format!("recipe {key} names unknown stack {}", recipe.stack))?;
                let layers = match recipe.view {
                    Some(view) => {
                        Arc::new(stack.surroundings(view.offset, (view.low, view.high), view.blend))
                    }
                    None => stack.clone(),
                };
                let rules = recipe
                    .surface
                    .compile()
                    .map_err(|error| error.to_string())?;
                self.recipes.insert(
                    key.to_owned(),
                    Recipe {
                        layers,
                        pyramid: recipe.pyramid,
                        rules: Arc::new(rules),
                        materials: recipe.materials,
                    },
                );
            }
            _ => return Err("unknown share".into()),
        }
        Ok(())
    }

    /// Drops what was shared as `key`.
    pub fn unshare(&mut self, key: &str) {
        self.stacks.remove(key);
        self.recipes.remove(key);
    }

    /// Runs job `kind` on `input` whole; its output.
    pub fn run(&mut self, kind: &str, input: &[u8]) -> Result<Vec<u8>, String> {
        const WHOLE: u64 = u64::MAX;
        self.start(WHOLE, kind, input)?;
        loop {
            if let Some(output) = self.step(WHOLE, u64::MAX)? {
                return Ok(output);
            }
        }
    }

    /// Starts job `id` of `kind` on `input`, to
    /// [`Self::step`] until done; another of the same id stops.
    pub fn start(&mut self, id: u64, kind: &str, input: &[u8]) -> Result<(), String> {
        let invalid = |error: CodecError| error.to_string();
        let kind = JobKind::from_name(kind).ok_or_else(|| format!("unknown job kind {kind}"))?;
        let mut reader = Reader::new(input);
        let running = match kind {
            JobKind::Tile => {
                let TileJob {
                    recipe,
                    key,
                    stamps,
                } = reader.get().map_err(invalid)?;
                reader.finish().map_err(invalid)?;
                let recipe = self.recipe(&recipe)?;
                if !recipe.pyramid.contains(key) {
                    return Err("tile outside its pyramid".into());
                }
                Running::Tile(Box::new(
                    Generator::new(recipe.stamped(&stamps)?, &recipe.pyramid, key)
                        .with_surface(recipe.rules.clone())
                        .with_placing(Placing::All)
                        .with_materials(recipe.materials),
                ))
            }
            JobKind::Walk => {
                let WalkJob {
                    recipe,
                    stamps,
                    from,
                    to,
                    obstacles,
                    options,
                } = reader.get().map_err(invalid)?;
                reader.finish().map_err(invalid)?;
                options.validate().map_err(|invalid| invalid.to_string())?;
                let recipe = self.recipe(&recipe)?;
                Running::Walk(Box::new(WalkSearch::new(
                    recipe.stamped(&stamps)?,
                    &recipe.pyramid,
                    (from, to),
                    obstacles,
                    &options,
                )))
            }
            JobKind::Bake => {
                let BakeJob { layers, pyramid } = reader.get().map_err(invalid)?;
                reader.finish().map_err(invalid)?;
                Running::Bake(Box::new((layers, pyramid)))
            }
        };
        self.running.insert(id, running);
        Ok(())
    }

    /// Works on job `id` for about `budget` work units (about one per
    /// sample; bakes run whole): its output once done, `None` while work is
    /// left. A job that failed or finished is gone.
    pub fn step(&mut self, id: u64, budget: u64) -> Result<Option<Vec<u8>>, String> {
        let Some(running) = self.running.get_mut(&id) else {
            return Err("no such job".into());
        };
        let mut budget = budget.max(1);
        let output = match running {
            Running::Tile(generator) => match generator.advance(&mut budget) {
                Some(tile) => Ok(codec::encode(&tile)),
                None if generator.failed() => Err("the tile's heights are invalid".into()),
                None => return Ok(None),
            },
            Running::Walk(search) => match search.step(&mut budget) {
                Some(walk) => Ok(codec::encode(&walk)),
                None => return Ok(None),
            },
            Running::Bake(bake) => {
                let (layers, pyramid) = &**bake;
                Ok(codec::encode(&layers.bake_all(pyramid)))
            }
        };
        self.running.remove(&id);
        output.map(Some)
    }

    /// Stops job `id`, if it runs.
    pub fn cancel(&mut self, id: u64) {
        self.running.remove(&id);
    }

    fn recipe(&self, key: &str) -> Result<&Recipe, String> {
        self.recipes
            .get(key)
            .ok_or_else(|| format!("unknown recipe {key}"))
    }
}
