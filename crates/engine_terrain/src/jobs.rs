//! Terrain jobs (decision 110): terrain work engines hand workers, as bytes
//! in and bytes out: a tile generated, a walk searched, a stack baked.
//! Workers hold once what many jobs read, shared by key: layer stacks with
//! their bakes, and the recipes generating a terrain's tiles from one. Jobs
//! run the same code in workers as engines without them run in place, so
//! their results are the same bits.

use std::collections::HashMap;
use std::sync::Arc;

use glam::Vec2;

use crate::codec::{self, CodecError, Reader, Writer, record};
use crate::{
    BiomeRules, Biomes, Generator, Layers, NavOptions, NavPath, Obstacle, Pyramid, Tile, TileKey,
    WalkSearch,
};

/// The kinds of terrain jobs: a tile of a shared recipe, a walk over one,
/// and a stack's bakes.
pub const TILE_JOB: &str = "terrain_tile";
pub const WALK_JOB: &str = "terrain_walk";
pub const BAKE_JOB: &str = "terrain_bake";

/// How a terrain's surroundings (decision 094) see a stack: as
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
}

record!(TerrainRecipe {
    stack,
    view,
    pyramid,
    surface,
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

/// The input of a [`TILE_JOB`]: tile `key` of the recipe shared as
/// `recipe`.
pub fn tile_input(recipe: &str, key: TileKey) -> Vec<u8> {
    let mut writer = Writer::new();
    writer.put(&recipe.to_owned()).put(&key);
    writer.into_bytes()
}

/// A [`TILE_JOB`]'s output: the tile.
pub fn tile_output(bytes: &[u8]) -> Result<Tile, CodecError> {
    codec::decode(bytes)
}

/// The input of a [`WALK_JOB`]: the shortest walk over the recipe shared
/// as `recipe` from terrain-local `from` to `to` among `obstacles`, as
/// `options` walk.
pub fn walk_input(
    recipe: &str,
    (from, to): (Vec2, Vec2),
    obstacles: &[Obstacle],
    options: &NavOptions,
) -> Vec<u8> {
    let mut writer = Writer::new();
    writer
        .put(&recipe.to_owned())
        .put(&from)
        .put(&to)
        .put(&obstacles.to_vec())
        .put(options);
    writer.into_bytes()
}

/// A [`WALK_JOB`]'s output: the walk, `None` where none reaches.
pub fn walk_output(bytes: &[u8]) -> Result<Option<NavPath>, CodecError> {
    codec::decode(bytes)
}

/// The input of a [`BAKE_JOB`]: every bake of `layers` over `pyramid`.
pub fn bake_input(layers: &Layers, pyramid: &Pyramid) -> Vec<u8> {
    let mut writer = Writer::new();
    writer.put(layers).put(pyramid);
    writer.into_bytes()
}

/// A [`BAKE_JOB`]'s output: the layers with every bake done.
pub fn bake_output(bytes: &[u8]) -> Result<Layers, CodecError> {
    codec::decode(bytes)
}

/// A shared recipe as a worker holds it: its layers seen as its terrain
/// sees them, its pyramid and its compiled surface.
struct Recipe {
    layers: Arc<Layers>,
    pyramid: Pyramid,
    rules: Arc<BiomeRules>,
}

/// What a worker holds of terrains, and runs their jobs over.
#[derive(Default)]
pub struct TerrainWorker {
    stacks: HashMap<String, Arc<Layers>>,
    recipes: HashMap<String, Recipe>,
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

    /// Runs job `kind` on `input`; its output.
    pub fn run(&mut self, kind: &str, input: &[u8]) -> Result<Vec<u8>, String> {
        let invalid = |error: CodecError| error.to_string();
        let mut reader = Reader::new(input);
        match kind {
            TILE_JOB => {
                let recipe: String = reader.get().map_err(invalid)?;
                let key: TileKey = reader.get().map_err(invalid)?;
                reader.finish().map_err(invalid)?;
                let recipe = self.recipe(&recipe)?;
                if !recipe.pyramid.contains(key) {
                    return Err("tile outside its pyramid".into());
                }
                let mut generator = Generator::new(recipe.layers.clone(), &recipe.pyramid, key)
                    .with_surface(recipe.rules.clone());
                let mut unlimited = u64::MAX;
                let tile = generator
                    .advance(&mut unlimited)
                    .ok_or("the tile's heights are invalid")?;
                Ok(codec::encode(&tile))
            }
            WALK_JOB => {
                let recipe: String = reader.get().map_err(invalid)?;
                let from: Vec2 = reader.get().map_err(invalid)?;
                let to: Vec2 = reader.get().map_err(invalid)?;
                let obstacles: Vec<Obstacle> = reader.get().map_err(invalid)?;
                let options: NavOptions = reader.get().map_err(invalid)?;
                reader.finish().map_err(invalid)?;
                options
                    .validate()
                    .map_err(|(field, rule)| format!("{field} {rule}"))?;
                let recipe = self.recipe(&recipe)?;
                let mut search = WalkSearch::new(
                    recipe.layers.clone(),
                    &recipe.pyramid,
                    (from, to),
                    obstacles,
                    &options,
                );
                let walk = loop {
                    let mut unlimited = u64::MAX;
                    if let Some(found) = search.step(&mut unlimited) {
                        break found;
                    }
                };
                Ok(codec::encode(&walk))
            }
            BAKE_JOB => {
                let layers: Layers = reader.get().map_err(invalid)?;
                let pyramid: Pyramid = reader.get().map_err(invalid)?;
                reader.finish().map_err(invalid)?;
                Ok(codec::encode(&layers.bake_all(&pyramid)))
            }
            _ => Err(format!("unknown job kind {kind}")),
        }
    }

    fn recipe(&self, key: &str) -> Result<&Recipe, String> {
        self.recipes
            .get(key)
            .ok_or_else(|| format!("unknown recipe {key}"))
    }
}
