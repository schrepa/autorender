//! Terrain jobs: terrain work engines hand workers, as bytes
//! in and bytes out: a tile generated whole, a tile's portals of a walk
//! graph, a stack baked, a terrain's lighting, a tile's water settled.
//! Workers hold once what many jobs read, shared by key: layer stacks with
//! their bakes, and the recipes generating a terrain's tiles from one. Jobs
//! run the same code wherever they run (a worker, a host without workers,
//! the cook), so their results are the same bits, and every job's output is
//! a record hosts keep and the cook ships: every field of a job is
//! something its output depends on.
//!
//! A tile's job, and its portals', carries the edits reaching its ground
//! alone, its recipe's stack none: an edit leaves every other tile's jobs,
//! and their records, as they were. A terrain's lighting reads a level's
//! heights alone, under edits or not.

use engine_math::collections::HashMap;
use std::sync::Arc;

use glam::Vec2;

use crate::codec::{self, Codec, CodecError, Reader, Writer, record};
use crate::lighting::{Ground, TerrainLighting, light};
use crate::{
    BiomeRules, Biomes, Gait, Generator, Layers, Placing, Pyramid, Stamp, SteadySolve,
    SteadyWater, Tile, TileKey, TilePortals,
};

/// The kinds of terrain jobs, which hosts carry by name: a tile of a
/// shared recipe, a tile's portals over a shared stack, a stack's bakes, a
/// terrain's lighting and a tile's settled water. Workers match every kind,
/// so a kind added without its work fails to build.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobKind {
    Tile,
    Portals,
    Bake,
    Lighting,
    Water,
}

impl JobKind {
    /// Every kind.
    pub const ALL: [Self; 5] = [
        Self::Tile,
        Self::Portals,
        Self::Bake,
        Self::Lighting,
        Self::Water,
    ];

    /// Its name, as hosts carry it and key the records they keep by.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Tile => "terrain_tile",
            Self::Portals => "terrain_portals",
            Self::Bake => "terrain_bake",
            Self::Lighting => "terrain_lighting",
            Self::Water => "terrain_water",
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

/// A [`JobKind::Portals`]: tile `key`'s portals of the walk graph of `gait`
/// over `pyramid` ([`TilePortals`]), its ground generated from the stack
/// shared as `stack` under `stamps`, the terrain's edits reaching its
/// ground, in order, as its tile's record holds it. Like every terrain job,
/// it is its own identity and its own input.
#[derive(Clone, Debug, PartialEq)]
pub struct PortalsJob {
    pub stack: String,
    pub pyramid: Pyramid,
    pub key: TileKey,
    pub stamps: Vec<Stamp>,
    pub gait: Gait,
}

record!(PortalsJob {
    stack,
    pyramid,
    key,
    stamps,
    gait,
});

impl PortalsJob {
    pub const KIND: JobKind = JobKind::Portals;

    /// What its output is kept by, and its input: the job.
    pub fn identity(&self) -> Vec<u8> {
        codec::encode(self)
    }
}

/// A [`JobKind::Portals`]'s output: the tile's portals.
pub fn portals_output(bytes: &[u8]) -> Result<TilePortals, CodecError> {
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

/// A [`JobKind::Lighting`]: the lighting ([`crate::lighting`]) of a
/// terrain's ground over `extent` meters along x and z, from `heights`,
/// `samples` along x and z `spacing` meters apart, row by row along z (its
/// finest level resident everywhere, at the steps tiles travel at), with a
/// sky map `sky` and a horizon map `horizons` texels a side. Its input is
/// the job; as it holds a whole level's heights, hosts keep its output by
/// that input's content hash, which engines compute.
#[derive(Clone, Debug, PartialEq)]
pub struct LightingJob {
    pub samples: [u32; 2],
    pub spacing: f32,
    pub extent: Vec2,
    pub heights: Box<[f32]>,
    pub sky: u32,
    pub horizons: u32,
}

impl Codec for LightingJob {
    fn write(&self, out: &mut Writer) {
        out.put(&self.samples)
            .put(&self.spacing)
            .put(&self.extent)
            .put(&self.sky)
            .put(&self.horizons);
        codec::put_grid(out, &self.heights, self.samples[0] as usize);
    }

    fn read(input: &mut Reader<'_>) -> Result<Self, CodecError> {
        let samples: [u32; 2] = input.get()?;
        let spacing = input.get()?;
        let extent = input.get()?;
        let sky = input.get()?;
        let horizons = input.get()?;
        let heights = codec::get_grid(input, samples[0] as usize)?;
        Ok(Self {
            samples,
            spacing,
            extent,
            heights,
            sky,
            horizons,
        })
    }
}

impl LightingJob {
    pub const KIND: JobKind = JobKind::Lighting;

    /// Its input: the job.
    pub fn input(&self) -> Vec<u8> {
        codec::encode(self)
    }

    /// Whether its grid holds at least two samples a side, as many
    /// heights, all numbers, and its maps a texel to the most a side.
    fn valid(&self) -> bool {
        let [columns, rows] = self.samples.map(|count| count as usize);
        columns >= 2
            && rows >= 2
            && self.heights.len() == columns * rows
            && self.heights.iter().all(|height| height.is_finite())
            && self.spacing.is_finite()
            && self.spacing > 0.0
            && self.extent.is_finite()
            && self.extent.x > 0.0
            && self.extent.y > 0.0
            && (1..=MAX_LIGHTING_TEXELS).contains(&self.sky)
            && (1..=MAX_LIGHTING_TEXELS).contains(&self.horizons)
    }
}

/// The most texels along a lighting map's side.
pub const MAX_LIGHTING_TEXELS: u32 = 4096;

/// A [`JobKind::Lighting`]'s output: the terrain's lighting.
pub fn lighting_output(bytes: &[u8]) -> Result<TerrainLighting, CodecError> {
    codec::decode(bytes)
}

/// A [`JobKind::Water`]: level-0 tile `key`'s water settled over the stack
/// shared as `stack` (with its bakes) and `pyramid` under `stamps`, the
/// terrain's edits reaching the ground it settles over, in order, in cells
/// `cell` meters across ([`SteadySolve`]). Like every terrain job, it is its
/// own identity and its own input.
#[derive(Clone, Debug, PartialEq)]
pub struct WaterJob {
    pub stack: String,
    pub pyramid: Pyramid,
    pub key: TileKey,
    pub stamps: Vec<Stamp>,
    pub cell: f32,
}

record!(WaterJob {
    stack,
    pyramid,
    key,
    stamps,
    cell,
});

impl WaterJob {
    pub const KIND: JobKind = JobKind::Water;

    /// What its output is kept by, and its input: the job.
    pub fn identity(&self) -> Vec<u8> {
        codec::encode(self)
    }
}

/// A [`JobKind::Water`]'s output: the tile's settled water.
pub fn water_output(bytes: &[u8]) -> Result<SteadyWater, CodecError> {
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
    Portals(Box<(Arc<Layers>, Pyramid, TileKey, Gait)>),
    Bake(Box<(Layers, Pyramid)>),
    Lighting(Box<LightingJob>),
    Water(Box<SteadySolve>),
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
                        .with_materials(recipe.materials),
                ))
            }
            JobKind::Portals => {
                let PortalsJob {
                    stack,
                    pyramid,
                    key,
                    stamps,
                    gait,
                } = reader.get().map_err(invalid)?;
                reader.finish().map_err(invalid)?;
                gait.validate(&pyramid)?;
                if !pyramid.contains(key) || u32::from(key.level) != gait.level {
                    return Err("portals of a tile outside its gait's level".into());
                }
                let stack = self
                    .stacks
                    .get(&stack)
                    .ok_or_else(|| format!("unknown stack {stack}"))?;
                let layers = if stamps.is_empty() {
                    stack.clone()
                } else {
                    Arc::new(
                        stack
                            .with_stamps(&stamps)
                            .map_err(|error| error.to_string())?,
                    )
                };
                Running::Portals(Box::new((layers, pyramid, key, gait)))
            }
            JobKind::Bake => {
                let BakeJob { layers, pyramid } = reader.get().map_err(invalid)?;
                reader.finish().map_err(invalid)?;
                Running::Bake(Box::new((layers, pyramid)))
            }
            JobKind::Lighting => {
                let job: LightingJob = reader.get().map_err(invalid)?;
                reader.finish().map_err(invalid)?;
                if !job.valid() {
                    return Err("lighting of an invalid grid".into());
                }
                Running::Lighting(Box::new(job))
            }
            JobKind::Water => {
                let WaterJob {
                    stack,
                    pyramid,
                    key,
                    stamps,
                    cell,
                } = reader.get().map_err(invalid)?;
                reader.finish().map_err(invalid)?;
                if key.level != 0 || !pyramid.contains(key) || !(cell.is_finite() && cell > 0.0) {
                    return Err("settled water of a tile outside its pyramid's finest level".into());
                }
                let layers = self
                    .stacks
                    .get(&stack)
                    .ok_or_else(|| format!("unknown stack {stack}"))?;
                let layers = if stamps.is_empty() {
                    layers.clone()
                } else {
                    Arc::new(
                        layers
                            .with_stamps(&stamps)
                            .map_err(|error| error.to_string())?,
                    )
                };
                Running::Water(Box::new(SteadySolve::new(layers, pyramid, key, cell)))
            }
        };
        self.running.insert(id, running);
        Ok(())
    }

    /// Works on job `id` for about `budget` work units (about one per
    /// sample; portals, bakes and lighting run whole): its output once done,
    /// `None` while work is left. A job that failed or finished is gone.
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
            Running::Portals(job) => {
                let (layers, pyramid, key, gait) = &**job;
                crate::nav::tile_portals(layers, pyramid, *key, gait)
                    .map(|portals| codec::encode(&portals))
            }
            Running::Bake(bake) => {
                let (layers, pyramid) = &**bake;
                Ok(codec::encode(&layers.bake_all(pyramid)))
            }
            Running::Lighting(job) => {
                let ground = Ground {
                    samples: job.samples,
                    spacing: job.spacing,
                    heights: &job.heights,
                };
                let extent = job.extent.to_array();
                Ok(codec::encode(&light(
                    &ground,
                    extent,
                    job.sky,
                    job.horizons,
                )))
            }
            Running::Water(solve) => match solve.advance(&mut budget)? {
                Some(water) => Ok(codec::encode(&water)),
                None => return Ok(None),
            },
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
