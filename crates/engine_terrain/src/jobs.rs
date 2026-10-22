//! Terrain jobs: terrain work engines hand workers, as bytes
//! in and bytes out: a tile generated whole, a tile's portals of a walk
//! graph, a stack baked, a terrain's lighting, bounce transfer and macro
//! colors, a tile's part of the water settling over the regions reaching it
//! (which the renderer settles on the GPU), a tile's water as settled water
//! stands it, and a terrain's far pages read into what uploads take.
//! Workers hold once what many jobs read, shared by key: layer stacks with
//! their bakes, the recipes generating a terrain's tiles from one, and the
//! records of settled water tiles' water reads, whole or their heads. Jobs
//! run the same code wherever they run (a worker, a host without workers,
//! the cook), so their results are the same bits, and every job's output
//! but a water's part and far pages read is a record hosts keep and the
//! cook ships: every field of a job is something its output depends on.
//!
//! A tile's job, and its portals', carries the edits reaching its ground
//! alone, its recipe's stack none: an edit leaves every other tile's jobs,
//! and their records, as they were. A terrain's lighting and its bounce
//! transfer read a level's heights, each tile generated as its record holds
//! it under the edits reaching it (the bounce transfer with its recipe's
//! surface), and hosts keep them by the keys of those tiles' records; its
//! macro colors read its recipe, without edits; a tile's water reads its
//! tile and the records of settled water around it, kept by their keys.

use engine_math::collections::HashMap;
use std::sync::Arc;

use glam::Vec2;

use crate::bounce::{MAX_PROBES, TerrainBounce, bounce};
use crate::codec::{self, CodecError, Reader, Writer, record};
use crate::lighting::{Ground, TerrainLighting, light};
use crate::macro_colors::{MACRO_BLOCK, MAX_MACRO_TEXELS, macro_colors};
use crate::steady_water::{PartMaking, SettledReads, Steadying};
use crate::{
    BiomeRules, Biomes, Gait, Generator, Layers, Placing, Pyramid, Stamp, SteadySummary,
    SteadyWater, TILE_QUADS, Tile, TileKey, TilePortals, TileWater, WaterPart,
};

/// The kinds of terrain jobs, which hosts carry by name: a tile of a
/// shared recipe, a tile's portals over a shared stack, a stack's bakes, a
/// terrain's lighting, its bounce transfer and its macro colors, a tile's
/// part of the water settling over the regions reaching it, a tile's water
/// as settled water stands it, and a terrain's far pages read into what
/// uploads take. Workers match every kind, so a kind added without its work
/// fails to build.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobKind {
    Tile,
    Portals,
    Bake,
    Lighting,
    Water,
    Bounce,
    Macro,
    Steadied,
    FarTexels,
}

impl JobKind {
    /// Every kind.
    pub const ALL: [Self; 9] = [
        Self::Tile,
        Self::Portals,
        Self::Bake,
        Self::Lighting,
        Self::Water,
        Self::Bounce,
        Self::Macro,
        Self::Steadied,
        Self::FarTexels,
    ];

    /// Its name, as hosts carry it and key the records they keep by.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Tile => "terrain_tile",
            Self::Portals => "terrain_portals",
            Self::Bake => "terrain_bake",
            Self::Lighting => "terrain_lighting",
            Self::Water => "terrain_water",
            Self::Bounce => "terrain_bounce",
            Self::Macro => "terrain_macro",
            Self::Steadied => "terrain_steadied",
            Self::FarTexels => "terrain_far_texels",
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

/// What shared bytes hold: a stack, a recipe naming one, a record of
/// settled water whole, or such a record's head ([`SteadyWater::head`]).
const STACK: u8 = 0;
const RECIPE: u8 = 1;
const SETTLED: u8 = 2;
const SUMMARY: u8 = 3;

/// `layers` (with their bakes) as workers share it.
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

/// The record of settled water `record` (its bytes) as workers share it
/// whole, for level-0 tiles' water to read ([`SteadiedJob`]), as
/// [`settled_key`].
pub fn share_settled(record: &[u8]) -> Vec<u8> {
    let mut shared = Vec::with_capacity(record.len() + 1);
    shared.push(SETTLED);
    shared.extend_from_slice(record);
    shared
}

/// A record of settled water's head (`head`, its bytes:
/// [`SteadyWater::head`]) as workers share it, for coarser tiles' water to
/// read its summary ([`SteadiedJob`]), as [`summary_key`].
pub fn share_summary(head: &[u8]) -> Vec<u8> {
    let mut shared = Vec::with_capacity(head.len() + 1);
    shared.push(SUMMARY);
    shared.extend_from_slice(head);
    shared
}

/// The key the record of settled water whose key's bytes are `record` is
/// shared whole as.
pub fn settled_key(record: &[u8; 32]) -> String {
    format!("settled-{}", hex(record))
}

/// The key the head of the record of settled water whose key's bytes are
/// `record` is shared as.
pub fn summary_key(record: &[u8; 32]) -> String {
    format!("summary-{}", hex(record))
}

/// `bytes` as lowercase hexadecimal digits.
fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
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

/// A [`JobKind::Lighting`]: the lighting ([`crate::lighting`]) of the
/// terrain the recipe shared as `recipe` generates, from its `level`'s
/// heights over the whole terrain (its finest pinned level, at the steps
/// tiles travel at), with a sky map `sky` and a horizon map `horizons`
/// texels a side. Each of the level's tiles, row by row, takes its heights
/// from the tile `tiles` names for it, generated under the edits named with
/// it as its record holds it: itself, or an ancestor standing in, read
/// between its samples, where surroundings leave its ground to their
/// terrain. Its input is the job; hosts keep its output by the keys of
/// those tiles' records, which engines find without its heights.
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

    /// Its input: the job.
    pub fn input(&self) -> Vec<u8> {
        codec::encode(self)
    }

    /// Whether its maps hold a texel to the most a side.
    fn valid(&self) -> bool {
        (1..=MAX_LIGHTING_TEXELS).contains(&self.sky)
            && (1..=MAX_LIGHTING_TEXELS).contains(&self.horizons)
    }
}

/// The most texels along a lighting map's side.
pub const MAX_LIGHTING_TEXELS: u32 = 4096;

/// A [`JobKind::Lighting`]'s output: the terrain's lighting.
pub fn lighting_output(bytes: &[u8]) -> Result<TerrainLighting, CodecError> {
    codec::decode(bytes)
}

/// A [`JobKind::Water`]: level-0 tile `key`'s part of the water settling
/// over the regions reaching it ([`WaterPart`]), its ground generated from
/// the stack shared as `stack` (with its bakes) over `pyramid` under
/// `stamps`, the terrain's edits reaching its ground, in order, on the
/// terrain's lattice of cells `cell` meters across. Its input is the job;
/// its output is no record: the regions it is part of settle on the GPU,
/// and their tiles' water is.
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

    /// Its input: the job.
    pub fn input(&self) -> Vec<u8> {
        codec::encode(self)
    }
}

/// A [`JobKind::Water`]'s output: the tile's part, its arrays holding its
/// corners and cells.
pub fn water_output(bytes: &[u8]) -> Result<WaterPart, CodecError> {
    let part: WaterPart = codec::decode(bytes)?;
    if !part.valid() {
        return Err(CodecError("invalid part of settling water"));
    }
    Ok(part)
}

/// A [`JobKind::Steadied`]: the water settled water stands at tile
/// `tile.key`'s samples, the tile as its record (`tile`) holds it, its
/// objects aside, on the terrain's lattice of cells `cell` meters across.
/// A level-0 tile's stands as the records of settled water `settled` names
/// say ([`Tile::steadied`]): its own and those of the level-0 tiles around
/// it holding water, each by its tile and its key's bytes, shared whole
/// ([`settled_key`]). A coarser tile's runs as their summaries say
/// ([`Tile::flowing`]): those of the level-0 tiles under it and around it,
/// each shared as its record's head ([`summary_key`]). Its input is the job;
/// hosts keep its output by the keys of its tile's record and of what it
/// reads, which engines find.
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

    /// Its input: the job.
    pub fn input(&self) -> Vec<u8> {
        codec::encode(self)
    }
}

/// A [`JobKind::Steadied`]'s output: the tile's water, none where none
/// stands on it.
pub fn steadied_output(bytes: &[u8]) -> Result<Option<TileWater>, CodecError> {
    codec::decode(bytes)
}

/// A [`JobKind::Bounce`]: the bounce transfer ([`crate::bounce`]) of the
/// terrain the recipe shared as `recipe` generates, from the heights its
/// lighting reads (its `level`'s, each tile from the one `tiles` names for
/// it, as a [`LightingJob`] reads them), with `probes` columns along each
/// side, its surface chosen by the recipe's rules and layers. Its input is
/// the job; hosts keep its output by the keys of those tiles' records, which
/// engines find.
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

    /// Its input: the job.
    pub fn input(&self) -> Vec<u8> {
        codec::encode(self)
    }

    /// Whether its probes are an even count from 2 to the most.
    fn valid(&self) -> bool {
        (2..=MAX_PROBES).contains(&self.probes) && self.probes.is_multiple_of(2)
    }
}

/// A [`JobKind::Bounce`]'s output: the terrain's bounce transfer.
pub fn bounce_output(bytes: &[u8]) -> Result<TerrainBounce, CodecError> {
    codec::decode(bytes)
}

/// A [`JobKind::Macro`]: the macro colors ([`crate::macro_colors`]) of the
/// terrain the recipe shared as `recipe` generates, `size` texels a side,
/// its shape read from the heights of its `level`'s tiles (its finest
/// pinned level, which its lighting reads too). Like a tile's job it is its
/// own identity and its own input. Its output is the map's RGBA8 texels,
/// rows along z.
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

    /// What its output is kept by, and its input: the job.
    pub fn identity(&self) -> Vec<u8> {
        codec::encode(self)
    }

    /// Whether its map's sides are whole blocks, at most
    /// [`MAX_MACRO_TEXELS`].
    fn valid(&self) -> bool {
        (MACRO_BLOCK..=MAX_MACRO_TEXELS).contains(&self.size) && self.size % MACRO_BLOCK == 0
    }
}

/// The record of a terrain's far pages, its runtime virtual texture's
/// coarse pages, which renderers bake on the GPU, read back and keep: of
/// `pages` (each one's mip and page along x and z) whose layers' RGBA8
/// texels `texels` holds, `side` a side, rows first, page after page and
/// layer after layer. It holds their count, the pages and `side`, then every
/// layer's channels as planes written as differences from what their
/// neighbours predict ([`codec::put_planes`]), for the cook's compression.
pub fn far_record(pages: &[(u8, u16, u16)], side: u32, texels: &[u8]) -> Vec<u8> {
    let mut out = Writer::new();
    out.put(&(pages.len() as u32));
    for &(mip, x, z) in pages {
        out.put(&mip).put(&x).put(&z);
    }
    out.put(&side);
    // Each layer's channels apart, as planes.
    let area = side as usize * side as usize;
    let mut planes = vec![0_u8; texels.len()];
    if area > 0 {
        for (layer, texels) in texels.chunks_exact(area * 4).enumerate() {
            for (at, texel) in texels.chunks_exact(4).enumerate() {
                for (channel, &value) in texel.iter().enumerate() {
                    planes[(layer * 4 + channel) * area + at] = value;
                }
            }
        }
    }
    codec::put_planes(&mut out, &planes, side as usize);
    out.into_bytes()
}

/// A [`JobKind::FarTexels`]: what uploads take of the far pages record
/// `record` ([`far_record`]), its input, which hosts fetched: every page's
/// layers' RGBA8 texels, rows first, page after page and layer after layer;
/// then, for [`far_pages`] to take off, each page's mip and page along x
/// and z, the texels a side and the pages' count, little-endian. Its output
/// is no record.
pub fn far_texels(record: &[u8]) -> Result<Vec<u8>, CodecError> {
    let invalid = CodecError("invalid far pages");
    let mut input = Reader::new(record);
    let count: u32 = input.get()?;
    let mut pages = Vec::with_capacity((count as usize).min(record.len()));
    for _ in 0..count {
        let mip: u8 = input.get()?;
        let x: u16 = input.get()?;
        let z: u16 = input.get()?;
        pages.push((mip, x, z));
    }
    let side: u32 = input.get()?;
    let area = side as usize * side as usize;
    if area == 0 {
        return Err(invalid);
    }
    let planes = codec::get_planes(&mut input, side as usize)?;
    input.finish()?;
    if planes.len() % (area * 4) != 0 {
        return Err(invalid);
    }
    let mut texels = vec![0_u8; planes.len()];
    for (layer, layer_texels) in texels.chunks_exact_mut(area * 4).enumerate() {
        for (at, texel) in layer_texels.chunks_exact_mut(4).enumerate() {
            for (channel, value) in texel.iter_mut().enumerate() {
                *value = planes[(layer * 4 + channel) * area + at];
            }
        }
    }
    texels.reserve(pages.len() * 5 + 8);
    for (mip, x, z) in pages {
        texels.push(mip);
        texels.extend_from_slice(&x.to_le_bytes());
        texels.extend_from_slice(&z.to_le_bytes());
    }
    texels.extend_from_slice(&side.to_le_bytes());
    texels.extend_from_slice(&count.to_le_bytes());
    Ok(texels)
}

/// Takes the pages, their side and their count off the end of what
/// reading far pages made ([`far_texels`]), leaving `texels` their texels
/// alone: the pages (each one's mip and page along x and z) and the texels
/// a side.
pub fn far_pages(texels: &mut Vec<u8>) -> Result<(Vec<(u8, u16, u16)>, u32), CodecError> {
    let invalid = CodecError("invalid far pages");
    let word = |bytes: &[u8]| u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let trailer = texels.len().checked_sub(8).ok_or(invalid)?;
    let side = word(&texels[trailer..]);
    let count = word(&texels[trailer + 4..]) as usize;
    let first = count
        .checked_mul(5)
        .and_then(|bytes| trailer.checked_sub(bytes))
        .ok_or(invalid)?;
    let pages = texels[first..trailer]
        .chunks_exact(5)
        .map(|page| {
            (
                page[0],
                u16::from_le_bytes([page[1], page[2]]),
                u16::from_le_bytes([page[3], page[4]]),
            )
        })
        .collect();
    texels.truncate(first);
    Ok((pages, side))
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
        stamped(&self.layers, stamps)
    }
}

/// `layers` under `stamps`, edits in order.
fn stamped(layers: &Arc<Layers>, stamps: &[Stamp]) -> Result<Arc<Layers>, String> {
    if stamps.is_empty() {
        return Ok(layers.clone());
    }
    layers
        .with_stamps(stamps)
        .map(Arc::new)
        .map_err(|error| error.to_string())
}

/// A level's heights over a whole terrain in the making, as its lighting
/// and its bounce transfer read them ([`Ground`]): from a recipe's layers
/// over its pyramid, each of the level's tiles, row by row, filled from the
/// tile standing in for it (itself, or an ancestor read between its
/// samples), generated under the edits reaching it a slice at a time, each
/// once, and read back as its record holds it; `samples` along x and z,
/// `spacing` meters apart, row by row along z.
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
    /// The heights of `recipe`'s `level`, each of its tiles, row by row,
    /// from the one `tiles` names for it with the edits reaching that one's
    /// ground; an error where the level is not the pyramid's or a tile named
    /// stands in for no tile of the level.
    fn new(recipe: &Recipe, level: u32, tiles: Vec<(TileKey, Vec<Stamp>)>) -> Result<Self, String> {
        let pyramid = recipe.pyramid;
        if level >= pyramid.levels() {
            return Err("heights of a level outside the pyramid".into());
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
            return Err("heights of tiles standing in for none of their level".into());
        }
        let quads = TILE_QUADS as usize;
        let samples = [
            columns as usize * quads + 1,
            rows as usize * quads + 1,
        ];
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

    /// Works for about `budget` units (a sample generated each): whether
    /// every tile of the level is filled; an error where a tile standing in
    /// does not generate or read back.
    fn advance(&mut self, budget: &mut u64) -> Result<bool, String> {
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
                    return Err("the ground's heights are invalid".into());
                }
                return Ok(false);
            };
            self.generator = None;
            // As its record holds it: heights at their steps of the grid.
            let tile: Tile =
                codec::decode(&codec::encode(&made)).map_err(|error| error.to_string())?;
            self.made.push((held, tile));
        }
        Ok(true)
    }

    /// The heights filled.
    fn ground(&self) -> Ground<'_> {
        Ground {
            samples: self.samples,
            spacing: self.pyramid.spacing(self.level),
            heights: &self.heights,
        }
    }

    /// The terrain's extent along x and z.
    fn extent(&self) -> [f32; 2] {
        self.pyramid.extent().to_array()
    }
}

/// Fills level-`level` tile `key`'s samples of `heights` (`columns` a
/// row) from `tile`, the tile standing in for it: its own samples, or an
/// ancestor's heights between its samples.
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

/// `tile`'s height at its fractional sample `(x, z)`, between the samples
/// around it.
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

/// A job a worker started, to step until done.
enum Running {
    Tile(Box<Generator>),
    Portals(Box<(Arc<Layers>, Pyramid, TileKey, Gait)>),
    Bake(Box<(Layers, Pyramid)>),
    Lighting(Box<(GroundMaking, LightingJob)>),
    Water(Box<PartMaking>),
    Bounce(Box<(GroundMaking, BounceJob, Arc<Layers>, Arc<BiomeRules>)>),
    Macro(Box<(Arc<Layers>, Pyramid, MacroJob)>),
    Steadied(Box<Steadying>),
    FarTexels(Box<Vec<u8>>),
}

/// What a worker holds of terrains, and runs their jobs over: stacks,
/// recipes, and records of settled water (whole or their heads) as shared.
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

    /// Holds the shared `bytes` as `key`: a stack, a recipe whose stack it
    /// already holds, or a record of settled water whole or its head, which
    /// it holds as shared until tiles' water reads them.
    pub fn share(&mut self, key: &str, bytes: impl Into<Arc<[u8]>>) -> Result<(), String> {
        let bytes: Arc<[u8]> = bytes.into();
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
            SETTLED | SUMMARY => {
                self.records.insert(key.to_owned(), Arc::clone(&bytes));
            }
            _ => return Err("unknown share".into()),
        }
        Ok(())
    }

    /// Drops what was shared as `key`.
    pub fn unshare(&mut self, key: &str) {
        self.stacks.remove(key);
        self.recipes.remove(key);
        self.records.remove(key);
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
    /// [`Self::step`] until done; another of the same id stops. A tile's
    /// water reads the records it names as they are shared now.
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
                Running::Portals(Box::new((stamped(stack, &stamps)?, pyramid, key, gait)))
            }
            JobKind::Bake => {
                let BakeJob { layers, pyramid } = reader.get().map_err(invalid)?;
                reader.finish().map_err(invalid)?;
                Running::Bake(Box::new((layers, pyramid)))
            }
            JobKind::Lighting => {
                let mut job: LightingJob = reader.get().map_err(invalid)?;
                reader.finish().map_err(invalid)?;
                if !job.valid() {
                    return Err("lighting of maps of an invalid size".into());
                }
                let recipe = self.recipe(&job.recipe)?;
                let making = GroundMaking::new(recipe, job.level, std::mem::take(&mut job.tiles))?;
                Running::Lighting(Box::new((making, job)))
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
                Running::Water(Box::new(PartMaking::new(
                    stamped(layers, &stamps)?,
                    pyramid,
                    key,
                    cell,
                )))
            }
            JobKind::Bounce => {
                let mut job: BounceJob = reader.get().map_err(invalid)?;
                reader.finish().map_err(invalid)?;
                if !job.valid() {
                    return Err("bounce of an invalid count of probes".into());
                }
                let recipe = self.recipe(&job.recipe)?;
                let making = GroundMaking::new(recipe, job.level, std::mem::take(&mut job.tiles))?;
                let (layers, rules) = (recipe.layers.clone(), recipe.rules.clone());
                Running::Bounce(Box::new((making, job, layers, rules)))
            }
            JobKind::Macro => {
                let job: MacroJob = reader.get().map_err(invalid)?;
                reader.finish().map_err(invalid)?;
                let recipe = self.recipe(&job.recipe)?;
                if !job.valid() || job.level >= recipe.pyramid.levels() {
                    return Err("macro colors of an invalid size or level".into());
                }
                Running::Macro(Box::new((recipe.layers.clone(), recipe.pyramid, job)))
            }
            JobKind::Steadied => {
                let job: SteadiedJob = reader.get().map_err(invalid)?;
                reader.finish().map_err(invalid)?;
                Running::Steadied(Box::new(self.steadying(&job)?))
            }
            JobKind::FarTexels => Running::FarTexels(Box::new(input.to_vec())),
        };
        self.running.insert(id, running);
        Ok(())
    }

    /// Works on job `id` for about `budget` work units (about one per
    /// sample; portals, bakes and macro colors run whole, as do lighting and
    /// bounce transfers once their tiles are generated, and far pages read):
    /// its output once done, `None` while work is left. A job that failed or
    /// finished is gone.
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
            Running::Lighting(held) => {
                let (making, job) = &mut **held;
                match making.advance(&mut budget) {
                    Ok(false) => return Ok(None),
                    Ok(true) => Ok(codec::encode(&light(
                        &making.ground(),
                        making.extent(),
                        job.sky,
                        job.horizons,
                    ))),
                    Err(error) => Err(error),
                }
            }
            Running::Water(part) => match part.advance(&mut budget)? {
                Some(part) => Ok(codec::encode(&part)),
                None => return Ok(None),
            },
            Running::Bounce(held) => {
                let (making, job, layers, rules) = &mut **held;
                match making.advance(&mut budget) {
                    Ok(false) => return Ok(None),
                    Ok(true) => Ok(codec::encode(&bounce(
                        &making.ground(),
                        making.extent(),
                        (layers.as_ref(), rules.as_ref()),
                        job.probes,
                    ))),
                    Err(error) => Err(error),
                }
            }
            Running::Macro(colors) => {
                let (layers, pyramid, job) = &**colors;
                macro_colors(layers, pyramid, job.level, job.size)
                    .ok_or_else(|| "the macro colors' heights are invalid".to_owned())
            }
            Running::Steadied(steadying) => match steadying.advance(&mut budget) {
                Ok(Some(water)) => Ok(codec::encode(&water)),
                Ok(None) => return Ok(None),
                Err(error) => Err(error),
            },
            Running::FarTexels(record) => {
                far_texels(record.as_slice()).map_err(|error| error.to_string())
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

    /// The making of `job`'s water, reading what it names as shared now:
    /// the shares may go once it started.
    fn steadying(&self, job: &SteadiedJob) -> Result<Steadying, String> {
        let recipe = self.recipe(&job.tile.recipe)?;
        let (key, cell) = (job.tile.key, job.cell);
        if !recipe.pyramid.contains(key) || !(cell.is_finite() && cell > 0.0) {
            return Err("settled water of a tile outside its pyramid".into());
        }
        let reads = if key.level == 0 {
            SettledReads::Records(
                job.settled
                    .iter()
                    .map(|&(at, record)| Ok((at, self.settled(&record, at, cell)?)))
                    .collect::<Result<_, String>>()?,
            )
        } else {
            SettledReads::Summaries(
                job.settled
                    .iter()
                    .map(|&(at, record)| Ok((at, self.summary(&record, at, cell)?)))
                    .collect::<Result<_, String>>()?,
            )
        };
        Ok(Steadying::new(
            recipe.stamped(&job.tile.stamps)?,
            recipe.pyramid,
            key,
            (cell, reads),
        ))
    }

    /// The record of settled water shared whole for the key whose bytes are
    /// `record` ([`settled_key`]): level-0 tile `key`'s, of cells `cell`
    /// meters across.
    fn settled(&self, record: &[u8; 32], key: TileKey, cell: f32) -> Result<SteadyWater, String> {
        let name = settled_key(record);
        let water: SteadyWater = match self.records.get(&name).map(|bytes| bytes.split_first()) {
            Some(Some((&SETTLED, rest))) => {
                codec::decode(rest).map_err(|error| error.to_string())?
            }
            _ => return Err(format!("unknown share {name}")),
        };
        if water.key != key || water.cell != cell {
            return Err(format!("share {name} holds another tile's settled water"));
        }
        Ok(water)
    }

    /// The summary of the record of settled water whose head is shared for
    /// the key whose bytes are `record` ([`summary_key`]): level-0 tile
    /// `key`'s, of cells `cell` meters across.
    fn summary(&self, record: &[u8; 32], key: TileKey, cell: f32) -> Result<SteadySummary, String> {
        let name = summary_key(record);
        let (head, _) = match self.records.get(&name).map(|bytes| bytes.split_first()) {
            Some(Some((&SUMMARY, rest))) => {
                SteadyWater::head(rest).map_err(|error| error.to_string())?
            }
            _ => return Err(format!("unknown share {name}")),
        };
        if head.key != key || head.cell != cell {
            return Err(format!("share {name} holds another tile's settled water"));
        }
        Ok(head.summary)
    }
}
