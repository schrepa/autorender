//! Terrain jobs: terrain work engines hand workers, as bytes
//! in and bytes out: a tile generated, a walk searched, a stack baked.
//! Workers hold once what many jobs read, shared by key: layer stacks with
//! their bakes, and the recipes generating a terrain's tiles from one. Jobs
//! run the same code in workers as engines without them run in place, so
//! their results are the same bits.
//!
//! Workers run jobs a slice at a time: [`TerrainWorker::start`]
//! one, [`TerrainWorker::step`] it within a budget until it is done, and
//! between slices take up a more urgent one or drop one its engine no longer
//! wants. A tile's objects of layers standing far away at its level may be
//! placed by a job of their own after it, as far tiles hold them by the tens
//! of thousands. A staged tile job hands out the tile's shape as a part,
//! [`TerrainWorker::take_part`], before the rest is done.
//! A tile may also generate in bands of rows on several workers at once,
//! joined by its engine, with its objects placed by a job
//! after.

use engine_math::collections::HashMap;
use std::sync::Arc;

use glam::Vec2;

use crate::codec::{self, CodecError, Reader, Writer, record};
use crate::{
    BiomeRules, Biomes, Generator, Layers, NavOptions, NavPath, Obstacle, Placed, Placing, Pyramid,
    Tile, TileBand, TileKey, WalkSearch,
};

/// The kinds of terrain jobs: a tile of a shared recipe, the objects
/// standing far away on one, a walk over one, and a stack's bakes.
pub const TILE_JOB: &str = "terrain_tile";
pub const FAR_JOB: &str = "terrain_far";
/// A band of a tile's rows, and the objects placed on a tile its bands
/// made.
pub const BAND_JOB: &str = "terrain_band";
pub const PLACE_JOB: &str = "terrain_place";
pub const WALK_JOB: &str = "terrain_walk";
pub const BAKE_JOB: &str = "terrain_bake";

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

/// The input of a [`TILE_JOB`]: tile `key` of the recipe shared as
/// `recipe`, with `placing`'s objects, its shape handed out
/// first where `staged`.
pub fn tile_input(recipe: &str, key: TileKey, placing: Placing, staged: bool) -> Vec<u8> {
    let mut writer = Writer::new();
    writer
        .put(&recipe.to_owned())
        .put(&key)
        .put(&placing)
        .put(&staged);
    writer.into_bytes()
}

/// A [`TILE_JOB`]'s output, and its part: the tile, or its shape alone.
pub fn tile_output(bytes: &[u8]) -> Result<Tile, CodecError> {
    codec::decode(bytes)
}

/// The input of a [`FAR_JOB`]: the objects of the recipe
/// shared as `recipe` standing far away on `tile`, one of its tiles.
pub fn far_input(recipe: &str, tile: &Tile) -> Vec<u8> {
    let mut writer = Writer::new();
    writer.put(&recipe.to_owned()).put(tile);
    writer.into_bytes()
}

/// The input of a [`BAND_JOB`]: stored rows `rows` (0 to
/// 259) of tile `key` of the recipe shared as `recipe`.
pub fn band_input(recipe: &str, key: TileKey, rows: std::ops::Range<u32>) -> Vec<u8> {
    let mut writer = Writer::new();
    writer
        .put(&recipe.to_owned())
        .put(&key)
        .put(&rows.start)
        .put(&rows.end);
    writer.into_bytes()
}

/// A [`BAND_JOB`]'s output: the band's rows, which
/// [`crate::merge_bands`] joins with the tile's other bands.
pub fn band_output(bytes: &[u8]) -> Result<TileBand, CodecError> {
    codec::decode(bytes)
}

/// The input of a [`PLACE_JOB`]: `placing`'s objects of the
/// recipe shared as `recipe` on `tile`, one of its tiles without them,
/// from layer and row `start` up to `end` (none for the
/// rest, with what authors stand on it). Its output is the objects, as a
/// [`FAR_JOB`]'s ([`far_output`]).
pub fn place_input(
    recipe: &str,
    tile: &Tile,
    placing: Placing,
    rows: ((u32, u32), Option<(u32, u32)>),
) -> Vec<u8> {
    let mut writer = Writer::new();
    writer
        .put(&recipe.to_owned())
        .put(tile)
        .put(&placing)
        .put(&rows.0.0)
        .put(&rows.0.1)
        .put(&rows.1);
    writer.into_bytes()
}

/// What a [`PLACE_JOB`]'s output depends on: its recipe,
/// tile, placing and rows.
pub fn place_identity(
    recipe: &str,
    key: TileKey,
    placing: Placing,
    rows: ((u32, u32), Option<(u32, u32)>),
) -> Vec<u8> {
    let mut writer = Writer::new();
    writer
        .put(&recipe.to_owned())
        .put(&key)
        .put(&placing)
        .put(&rows.0.0)
        .put(&rows.0.1)
        .put(&rows.1);
    writer.into_bytes()
}

/// What a [`FAR_JOB`]'s output depends on: its recipe and
/// tile, which that recipe generates.
pub fn far_identity(recipe: &str, key: TileKey) -> Vec<u8> {
    let mut writer = Writer::new();
    writer.put(&recipe.to_owned()).put(&key);
    writer.into_bytes()
}

/// A [`FAR_JOB`]'s output: the objects, which [`crate::merge_placed`] adds
/// to the tile's own.
pub fn far_output(bytes: &[u8]) -> Result<Vec<Placed>, CodecError> {
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
    materials: bool,
}

/// A job a worker started, to step until done.
enum Running {
    Tile(Generator),
    Far(Generator),
    Band(Generator),
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
        let mut reader = Reader::new(input);
        let running = match kind {
            TILE_JOB => {
                let recipe: String = reader.get().map_err(invalid)?;
                let key: TileKey = reader.get().map_err(invalid)?;
                let placing: Placing = reader.get().map_err(invalid)?;
                let staged: bool = reader.get().map_err(invalid)?;
                reader.finish().map_err(invalid)?;
                let recipe = self.recipe(&recipe)?;
                if !recipe.pyramid.contains(key) {
                    return Err("tile outside its pyramid".into());
                }
                Running::Tile(
                    Generator::new(recipe.layers.clone(), &recipe.pyramid, key)
                        .with_surface(recipe.rules.clone())
                        .with_placing(placing)
                        .with_staged(staged)
                        .with_materials(recipe.materials),
                )
            }
            FAR_JOB => {
                let recipe: String = reader.get().map_err(invalid)?;
                let tile: Tile = reader.get().map_err(invalid)?;
                reader.finish().map_err(invalid)?;
                let recipe = self.recipe(&recipe)?;
                if !recipe.pyramid.contains(tile.key()) {
                    return Err("tile outside its pyramid".into());
                }
                Running::Far(Generator::far(
                    tile,
                    recipe.layers.clone(),
                    recipe.rules.clone(),
                ))
            }
            PLACE_JOB => {
                let recipe: String = reader.get().map_err(invalid)?;
                let tile: Tile = reader.get().map_err(invalid)?;
                let placing: Placing = reader.get().map_err(invalid)?;
                let layer: u32 = reader.get().map_err(invalid)?;
                let row: u32 = reader.get().map_err(invalid)?;
                let end: Option<(u32, u32)> = reader.get().map_err(invalid)?;
                reader.finish().map_err(invalid)?;
                let recipe = self.recipe(&recipe)?;
                if !recipe.pyramid.contains(tile.key()) {
                    return Err("tile outside its pyramid".into());
                }
                Running::Far(Generator::placing_rows(
                    tile,
                    (recipe.layers.clone(), recipe.rules.clone()),
                    placing,
                    (
                        (layer as usize, row as usize),
                        end.map(|(layer, row)| (layer as usize, row as usize)),
                    ),
                ))
            }
            BAND_JOB => {
                let recipe: String = reader.get().map_err(invalid)?;
                let key: TileKey = reader.get().map_err(invalid)?;
                let first: u32 = reader.get().map_err(invalid)?;
                let end: u32 = reader.get().map_err(invalid)?;
                reader.finish().map_err(invalid)?;
                let recipe = self.recipe(&recipe)?;
                if !recipe.pyramid.contains(key) {
                    return Err("tile outside its pyramid".into());
                }
                let generator = Generator::band(
                    recipe.layers.clone(),
                    &recipe.pyramid,
                    key,
                    first as usize..end as usize,
                )
                .ok_or("rows outside the tile")?;
                Running::Band(
                    generator
                        .with_surface(recipe.rules.clone())
                        .with_materials(recipe.materials),
                )
            }
            WALK_JOB => {
                let recipe: String = reader.get().map_err(invalid)?;
                let from: Vec2 = reader.get().map_err(invalid)?;
                let to: Vec2 = reader.get().map_err(invalid)?;
                let obstacles: Vec<Obstacle> = reader.get().map_err(invalid)?;
                let options: NavOptions = reader.get().map_err(invalid)?;
                reader.finish().map_err(invalid)?;
                options.validate().map_err(|invalid| invalid.to_string())?;
                let recipe = self.recipe(&recipe)?;
                Running::Walk(Box::new(WalkSearch::new(
                    recipe.layers.clone(),
                    &recipe.pyramid,
                    (from, to),
                    obstacles,
                    &options,
                )))
            }
            BAKE_JOB => {
                let layers: Layers = reader.get().map_err(invalid)?;
                let pyramid: Pyramid = reader.get().map_err(invalid)?;
                reader.finish().map_err(invalid)?;
                Running::Bake(Box::new((layers, pyramid)))
            }
            _ => return Err(format!("unknown job kind {kind}")),
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
            Running::Far(generator) => match generator.advance(&mut budget) {
                Some(tile) => Ok(codec::encode(&tile.placed().to_vec())),
                None if generator.failed() => Err("the tile is invalid".into()),
                None => return Ok(None),
            },
            Running::Band(generator) => {
                let _ = generator.advance(&mut budget);
                match generator.take_band() {
                    Some(band) => Ok(codec::encode(&band)),
                    None if generator.failed() => Err("the band is invalid".into()),
                    None => return Ok(None),
                }
            }
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

    /// A part of job `id`'s output ready before the rest,
    /// taken once: a staged tile's shape.
    pub fn take_part(&mut self, id: u64) -> Option<Vec<u8>> {
        match self.running.get_mut(&id)? {
            Running::Tile(generator) => generator.take_shape().map(|shape| codec::encode(&shape)),
            Running::Far(_) | Running::Band(_) | Running::Walk(_) | Running::Bake(_) => None,
        }
    }

    /// Stops job `id`, if it runs.
    pub fn cancel(&mut self, id: u64) {
        self.running.remove(&id);
    }

    /// Generates a small fixed tile, so a new worker's tile
    /// code reaches its optimized tier before real tiles: what any tile
    /// runs, over a noise layer and a surface of two materials, one on
    /// slopes in noise.
    pub fn warm(&mut self) {
        let layers = Layers::new(vec![crate::Layer::Noise(crate::NoiseLayer {
            seed: 7,
            height: 400.0,
            wavelength: 1200.0,
            ridges: 0.5,
            ..crate::NoiseLayer::default()
        })]);
        let surface = Biomes {
            materials: vec!["ground".into(), "rock".into()],
            biomes: vec![crate::Biome {
                id: "warm".into(),
                fitness: crate::Mask::one(),
                layers: vec![
                    crate::BiomeLayer {
                        material: "ground".into(),
                        mask: crate::Mask::one(),
                        opacity: 1.0,
                    },
                    crate::BiomeLayer {
                        material: "rock".into(),
                        mask: crate::Mask::Multiply(vec![
                            crate::Mask::Slope([20.0, 40.0]),
                            crate::Mask::Noise(crate::NoiseMask {
                                wavelength: 40.0,
                                seed: 3,
                                octaves: 3,
                            }),
                        ]),
                        opacity: 1.0,
                    },
                ],
            }],
            assets: Vec::new(),
            placements: Vec::new(),
            placed: Vec::new(),
        };
        let (Ok(layers), Ok(rules), Ok(pyramid)) =
            (layers, surface.compile(), Pyramid::new([1, 1], 4.0))
        else {
            return;
        };
        let mut generator = Generator::new(Arc::new(layers), &pyramid, TileKey::new(0, 0, 0))
            .with_surface(Arc::new(rules));
        let mut budget = u64::MAX;
        let _ = generator.advance(&mut budget);
    }

    fn recipe(&self, key: &str) -> Result<&Recipe, String> {
        self.recipes
            .get(key)
            .ok_or_else(|| format!("unknown recipe {key}"))
    }
}
