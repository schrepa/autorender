use engine_math::collections::HashMap;
use std::sync::Arc;

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
const SETTLED: u8 = 2;
const SUMMARY: u8 = 3;
const SHADING: u8 = 4;

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

pub fn share_shading(record: &[u8]) -> Vec<u8> {
    let mut shared = Vec::with_capacity(record.len() + 1);
    shared.push(SHADING);
    shared.extend_from_slice(record);
    shared
}

pub fn shading_key(record: &[u8; 32]) -> String {
    format!("shading-{}", hex(record))
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

#[derive(Clone, Debug, PartialEq)]
pub struct LightingJob {
    pub recipe: String,
    pub level: u32,
    pub tiles: Vec<(TileKey, Vec<Stamp>)>,
    pub sky: u32,
    pub horizons: u32,
}

record!(LightingJob {
    recipe,
    level,
    tiles,
    sky,
    horizons,
});

impl LightingJob {
    pub const KIND: JobKind = JobKind::Lighting;

    pub fn input(&self) -> Vec<u8> {
        codec::encode(self)
    }

    fn check(&self) -> Result<(), Error> {
        for (path, texels) in [("$.sky", self.sky), ("$.horizons", self.horizons)] {
            if !(1..=MAX_LIGHTING_TEXELS).contains(&texels) {
                return Err(Error::new(ErrorCode::OutOfRange, path)
                    .range(1.0, f64::from(MAX_LIGHTING_TEXELS))
                    .got(texels));
            }
        }
        Ok(())
    }
}

pub const MAX_LIGHTING_TEXELS: u32 = 4096;

pub fn lighting_output(bytes: &[u8]) -> Result<TerrainLighting, Error> {
    codec::decode(bytes)
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

#[derive(Clone, Debug, PartialEq)]
pub enum ShadingReads {
    Tiles(Vec<(TileKey, Vec<Stamp>)>),
    Shadings(Vec<(TileKey, [u8; 32])>),
}

codec::tagged!(ShadingReads {
    0 => Tiles(tiles),
    1 => Shadings(shadings),
});

#[derive(Clone, Debug, PartialEq)]
pub struct ShadingJob {
    pub recipe: String,
    pub key: TileKey,
    pub reads: ShadingReads,
}

record!(ShadingJob { recipe, key, reads });

impl ShadingJob {
    pub const KIND: JobKind = JobKind::Shading;

    pub fn identity(&self) -> Vec<u8> {
        codec::encode(self)
    }
}

pub fn shading_output(bytes: &[u8]) -> Result<TileShading, Error> {
    codec::decode(bytes)
}

#[derive(Clone, Debug, PartialEq)]
pub struct BounceJob {
    pub recipe: String,
    pub level: u32,
    pub tiles: Vec<(TileKey, Vec<Stamp>)>,
    pub probes: u32,
}

record!(BounceJob {
    recipe,
    level,
    tiles,
    probes,
});

impl BounceJob {
    pub const KIND: JobKind = JobKind::Bounce;

    pub fn input(&self) -> Vec<u8> {
        codec::encode(self)
    }

    fn check(&self) -> Result<(), Error> {
        if (2..=MAX_PROBES).contains(&self.probes) && self.probes.is_multiple_of(2) {
            Ok(())
        } else {
            Err(Error::new(ErrorCode::OutOfRange, "$.probes")
                .range(2.0, f64::from(MAX_PROBES))
                .got(self.probes))
        }
    }
}

pub fn bounce_output(bytes: &[u8]) -> Result<TerrainBounce, Error> {
    codec::decode(bytes)
}

#[derive(Clone, Debug, PartialEq)]
pub struct MacroJob {
    pub recipe: String,
    pub level: u32,
    pub size: u32,
}

record!(MacroJob {
    recipe,
    level,
    size,
});

impl MacroJob {
    pub const KIND: JobKind = JobKind::Macro;

    pub fn identity(&self) -> Vec<u8> {
        codec::encode(self)
    }

    fn check(&self) -> Result<(), Error> {
        if (MACRO_BLOCK..=MAX_MACRO_TEXELS).contains(&self.size)
            && self.size.is_multiple_of(MACRO_BLOCK)
        {
            Ok(())
        } else {
            Err(Error::new(ErrorCode::OutOfRange, "$.size")
                .range(f64::from(MACRO_BLOCK), f64::from(MAX_MACRO_TEXELS))
                .got(self.size))
        }
    }
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

struct GroundMaking {
    layers: Arc<Layers>,
    pyramid: Pyramid,
    level: u32,
    tiles: Vec<(TileKey, Vec<Stamp>)>,
    made: Vec<(TileKey, Tile)>,
    generator: Option<Generator>,
    next: usize,
    samples: [u32; 2],
    heights: Vec<f32>,
}

impl GroundMaking {
    fn new(recipe: &Recipe, level: u32, tiles: Vec<(TileKey, Vec<Stamp>)>) -> Result<Self, Error> {
        let pyramid = recipe.pyramid;
        if level >= pyramid.levels() {
            return Err(level_outside(level, &pyramid));
        }
        let [columns, rows] = pyramid.tiles(level);
        let stands = |key: TileKey, held: TileKey| {
            held.level >= key.level && pyramid.contains(held) && {
                let up = u32::from(held.level - key.level);
                key.x >> up == held.x && key.z >> up == held.z
            }
        };
        if tiles.len() != columns as usize * rows as usize
            || !pyramid
                .level_tiles(level)
                .zip(&tiles)
                .all(|(key, &(held, _))| stands(key, held))
        {
            return Err(Error::new(ErrorCode::InvalidValue, "$.tiles"));
        }
        let quads = TILE_QUADS as usize;
        let samples = [columns as usize * quads + 1, rows as usize * quads + 1];
        Ok(Self {
            layers: recipe.layers.clone(),
            pyramid,
            level,
            tiles,
            made: Vec::new(),
            generator: None,
            next: 0,
            samples: samples.map(|count| count as u32),
            heights: vec![0.0; samples[0] * samples[1]],
        })
    }

    fn advance(&mut self, budget: &mut u64) -> Result<bool, Error> {
        let columns = self.pyramid.tiles(self.level)[0] as usize;
        while let Some(&(held, _)) = self.tiles.get(self.next) {
            if let Some((_, tile)) = self.made.iter().find(|(made, _)| *made == held) {
                let key = TileKey::new(
                    self.level as u8,
                    (self.next % columns) as u16,
                    (self.next / columns) as u16,
                );
                fill(
                    (self.heights.as_mut_slice(), self.samples[0] as usize),
                    (key, self.level),
                    tile,
                );
                self.next += 1;
                continue;
            }
            if *budget == 0 {
                return Ok(false);
            }
            if self.generator.is_none() {
                let layers = stamped(&self.layers, &self.tiles[self.next].1)?;
                self.generator = Some(Generator::new(layers, &self.pyramid, held));
            }
            let Some(generator) = self.generator.as_mut() else {
                continue;
            };
            let Some(made) = generator.advance(budget) else {
                if generator.failed() {
                    return Err(invalid_heights());
                }
                return Ok(false);
            };
            self.generator = None;
            let tile: Tile = codec::decode(&codec::encode(&made))?;
            self.made.push((held, tile));
        }
        Ok(true)
    }

    fn ground(&self) -> Ground<'_> {
        Ground {
            samples: self.samples,
            spacing: self.pyramid.spacing(self.level),
            heights: &self.heights,
        }
    }

    fn extent(&self) -> [f32; 2] {
        self.pyramid.extent().to_array()
    }
}

fn fill((heights, columns): (&mut [f32], usize), (key, level): (TileKey, u32), tile: &Tile) {
    let quads = TILE_QUADS as usize;
    let first = [usize::from(key.x) * quads, usize::from(key.z) * quads];
    let held = tile.key();
    let coarser = u32::from(held.level) - level;
    let scale = 1.0 / (1u32 << coarser) as f32;
    let offset = [
        (u32::from(key.x) * TILE_QUADS) as f32 * scale - (u32::from(held.x) * TILE_QUADS) as f32,
        (u32::from(key.z) * TILE_QUADS) as f32 * scale - (u32::from(held.z) * TILE_QUADS) as f32,
    ];
    for z in 0..=quads {
        for x in 0..=quads {
            let height = if coarser == 0 {
                tile.height(x as i32, z as i32)
            } else {
                between(
                    tile,
                    offset[0] + x as f32 * scale,
                    offset[1] + z as f32 * scale,
                )
            };
            heights[(first[1] + z) * columns + first[0] + x] = height;
        }
    }
}

fn between(tile: &Tile, x: f32, z: f32) -> f32 {
    let (x0, z0) = (x.floor(), z.floor());
    let (tx, tz) = (x - x0, z - z0);
    let (x0, z0) = (x0 as i32, z0 as i32);
    let a = tile.height(x0, z0);
    let b = tile.height(x0 + 1, z0);
    let c = tile.height(x0, z0 + 1);
    let d = tile.height(x0 + 1, z0 + 1);
    let near = a + (b - a) * tx;
    let far = c + (d - c) * tx;
    near + (far - near) * tz
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

#[derive(Default)]
pub struct TerrainWorker {
    stacks: HashMap<String, Arc<Layers>>,
    recipes: HashMap<String, Recipe>,
    records: HashMap<String, Arc<[u8]>>,
    running: HashMap<u64, Running>,
}

impl TerrainWorker {
    pub fn new() -> Self {
        Self::default()
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

    fn shading(&self, job: ShadingJob) -> Result<ShadingMaking, Error> {
        let recipe = self.recipe(&job.recipe)?;
        let (pyramid, key) = (recipe.pyramid, job.key);
        let reads = TileShading::reads(&pyramid, key);
        if !pyramid.contains(key) || reads.is_empty() {
            return Err(outside(key));
        }
        match job.reads {
            ShadingReads::Tiles(tiles)
                if key.level == 1 && names(tiles.iter().map(|&(at, _)| at), &reads) =>
            {
                let surface = (recipe.layers.clone(), recipe.rules.clone());
                Ok(ShadingMaking::from_ground(surface, pyramid, key, tiles))
            }
            ShadingReads::Shadings(shadings)
                if key.level > 1 && names(shadings.iter().map(|&(at, _)| at), &reads) =>
            {
                let finer = shadings
                    .iter()
                    .map(|&(at, record)| self.shared_shading(&record, at))
                    .collect::<Result<Vec<_>, Error>>()?;
                Ok(ShadingMaking::from_finer(pyramid, key, finer))
            }
            ShadingReads::Tiles(_) | ShadingReads::Shadings(_) => {
                Err(Error::new(ErrorCode::InvalidValue, "$.reads"))
            }
        }
    }

    fn shared_shading(&self, record: &[u8; 32], key: TileKey) -> Result<TileShading, Error> {
        let name = shading_key(record);
        let shading: TileShading = match self.records.get(&name).map(|bytes| bytes.split_first()) {
            Some(Some((&SHADING, rest))) => codec::decode(rest)?,
            _ => return Err(unshared("$.reads", &name)),
        };
        if shading.key() != key {
            return Err(mismatched("$.reads", &name));
        }
        Ok(shading)
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
            macro_colors(layers, pyramid, job.level, job.size, causes).ok_or_else(invalid_heights)?
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

fn names(named: impl Iterator<Item = TileKey>, reads: &[TileKey]) -> bool {
    named.eq(reads.iter().copied())
}
