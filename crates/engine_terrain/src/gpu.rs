//! What a GPU pass reads to give the tiles shading with their own samples
//! their control values (level-0 tiles, and every tile of surroundings; a
//! terrain's coarser tiles take theirs from their shading, level 0's
//! averaged, [`crate::TileShading`]): the surface's rules as one program for
//! an interpreter shader, and what their masks read of the bakes (the
//! hydrology bake's images, and the erosions' where masks read them) and
//! roads, flattened into words; the meters to rivers' edges and the rivers'
//! beds it reads are each tile's own ([`Tile::river_texels`]). The pass
//! evaluates what [`BiomeRules::control`] evaluates at each sample, in the
//! same order, so its control values are the CPU's but where GPU arithmetic
//! rounds a value across a rule's threshold. [`sample_control`] gives the
//! CPU's at one sample, for gameplay.

use crate::hydrology::{FLOW_RANGE, Hydrology, LAKE_RANGE, WETNESS_RANGE};
use crate::layers::HeightImage;
use crate::path::PathProfile;
use crate::surface::{BiomeRules, Control, SampleInputs, SampleShape, erosion_row, erosions};
use crate::tile::RiverInputs;
use crate::{Erosion, HydrologySample, Layers, PathSample, TILE_STRIDE, Tile};

/// Opcodes of a surface program's masks, in postfix order: each leaf pushes
/// a value, [`op::INVERT`] and [`op::REMAP`] change the top one, and lists
/// pop their children's values and push them folded in order.
pub mod op {
    /// Followed by the value.
    pub const CONSTANT: u32 = 0;
    /// Each followed by the ramp's two ends.
    pub const HEIGHT: u32 = 1;
    pub const GRADIENT: u32 = 2;
    pub const CURVATURE: u32 = 3;
    pub const FLOW: u32 = 4;
    pub const WETNESS: u32 = 5;
    pub const LAKE: u32 = 6;
    pub const RIVER: u32 = 7;
    pub const PATH: u32 = 8;
    /// Followed by the frequency, seed and octaves.
    pub const NOISE: u32 = 9;
    /// Each followed by the number of children.
    pub const MULTIPLY: u32 = 10;
    pub const ADD: u32 = 11;
    pub const MIN: u32 = 12;
    pub const MAX: u32 = 13;
    pub const INVERT: u32 = 14;
    /// Followed by the ramp's two ends.
    pub const REMAP: u32 = 15;
    /// Each followed by the ramp's two ends.
    pub const GRAIN: u32 = 16;
    pub const POWER: u32 = 17;
    pub const FLOOD: u32 = 18;
    pub const WORN: u32 = 19;
    pub const DEPOSIT: u32 = 20;
    /// Followed by what the heading weighs the gradient by along x and z,
    /// then the ramp's two ends.
    pub const FACING: u32 = 21;
    /// Ends a mask's code.
    pub const END: u32 = 22;
}

/// Words of each biome's entry in a program's table
/// ([`header::PROGRAM`]): its fitness mask, its layer list and its wetness
/// rule.
pub const BIOME_WORDS: u32 = 3;

/// The words of [`SurfaceData`]'s header: where each section starts
/// (`u32::MAX` where it has none) and what the program needs.
pub mod header {
    /// The program: a table of each biome's fitness mask (`u32::MAX` for
    /// the first, which takes none), layer list and wetness rule
    /// (`u32::MAX` for none), [`super::BIOME_WORDS`] words a biome; each
    /// list its count, then each layer's material, opacity and mask
    /// (`u32::MAX` for the first); then the masks' code, each ending in
    /// [`super::op::END`].
    pub const PROGRAM: usize = 0;
    /// The most values the program's stack holds at once.
    pub const STACK: usize = 1;
    pub const BIOMES: usize = 2;
    /// The materials its control values index.
    pub const MATERIALS: usize = 3;
    /// Bit 0: masks read water; bit 1: they read roads; bit 2: they read
    /// erosion.
    pub const READS: usize = 4;
    /// The hydrology bake's images: its spacing, its three decode ranges,
    /// then flow's, wetness's and lakes' levels, each image its level
    /// count, then each level's width, height and first sample, then the
    /// samples (`f32`).
    pub const HYDROLOGY: usize = 5;
    /// The roads: their count, then where each begins; each its half
    /// width, reach, cell size, corner, columns, rows, and where its starts,
    /// segments, line points, distances along and spans begin, and how many
    /// spans.
    pub const PATHS: usize = 6;
    /// Surroundings' fade: its low and high corners and
    /// blend.
    pub const FADE: usize = 7;
    /// The biomes whose wetness rules give the ground's wetness: those
    /// authored, before the steep layer the program paints over them, where
    /// any has a rule; none where none has.
    pub const WET_BIOMES: usize = 8;
    /// The stack's erosions, where masks read them: their count, then where
    /// each begins; each the inverse of the meters between its samples,
    /// those meters and its change's decode range, then its change image
    /// as the hydrology section lays out each of its images: its level
    /// count, each level's width, height and first sample, then the samples
    /// (`f32`).
    pub const EROSION: usize = 9;
    pub const LEN: usize = 16;
}

/// The most values a program's stack holds and the most materials its
/// control values index, as the interpreter's arrays are sized: surfaces
/// needing more keep their control values on the CPU.
pub const MAX_STACK: u32 = 16;
pub const MAX_MATERIALS: u32 = 64;

/// A terrain's surface as a GPU pass reads it: its words,
/// and what places a tile's samples and picks the hydrology images' level
/// for its spacing.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceData {
    pub words: Vec<u32>,
    offset: [f32; 2],
    hydrology: Option<(f32, usize)>,
}

impl SurfaceData {
    /// `rules` over `layers` (with their bakes) for the GPU; `None` where
    /// the program needs more than the interpreter holds.
    pub fn new(rules: &BiomeRules, layers: &Layers) -> Option<Self> {
        let mut words = vec![u32::MAX; header::LEN];
        words[header::PROGRAM] = words.len() as u32;
        let (stack, biomes, materials) = rules.program(&mut words);
        if stack > MAX_STACK || materials > MAX_MATERIALS {
            return None;
        }
        words[header::STACK] = stack;
        words[header::BIOMES] = biomes;
        words[header::MATERIALS] = materials;
        words[header::WET_BIOMES] = rules.wet_biomes() as u32;
        let reads = rules.reads();
        words[header::READS] =
            u32::from(reads.water) | (u32::from(reads.path) << 1) | (u32::from(reads.erosion) << 2);
        if let Some(hydrology) = layers.hydrology() {
            words[header::HYDROLOGY] = words.len() as u32;
            write_hydrology(hydrology, &mut words);
        }
        let paths: Vec<&PathProfile> = layers.path_profiles().collect();
        if !paths.is_empty() {
            words[header::PATHS] = words.len() as u32;
            write_paths(&paths, &mut words);
        }
        // Only surfaces whose masks read erosion hold its images.
        let stages: Vec<&Erosion> = erosions(layers).collect();
        if reads.erosion && !stages.is_empty() {
            words[header::EROSION] = words.len() as u32;
            write_erosions(&stages, &mut words);
        }
        if let Some(fade) = layers.fade {
            words[header::FADE] = words.len() as u32;
            push_f32(
                &mut words,
                [fade.low.x, fade.low.y, fade.high.x, fade.high.y, fade.blend],
            );
        }
        Some(Self {
            words,
            offset: layers.offset.to_array(),
            hydrology: layers
                .hydrology()
                .map(|hydrology| (hydrology.spacing, hydrology.flow.levels())),
        })
    }

    /// The first sample (x and z) of tile `key` at `spacing` as its
    /// terrain's layers read it: a sample's x and z are
    /// these plus its column and row less one, times the spacing.
    pub fn first_sample(&self, key: crate::TileKey, spacing: f32) -> [i64; 2] {
        first_sample(self.offset, key, spacing)
    }

    /// The hydrology images' level masks read at samples `spacing` meters
    /// apart, as [`Hydrology::level_for`] picks it.
    pub fn level(&self, spacing: f32) -> u32 {
        self.hydrology.map_or(0, |(image, levels)| {
            crate::hydrology::image_level(image, levels, spacing) as u32
        })
    }
}

/// The first sample of tile `key` at `spacing` read through a stack's
/// offset `offset`, as generators place their samples.
fn first_sample(offset: [f32; 2], key: crate::TileKey, spacing: f32) -> [i64; 2] {
    let quads = i64::from(crate::TILE_QUADS);
    [
        i64::from(key.x) * quads + (offset[0] / spacing).round() as i64,
        i64::from(key.z) * quads + (offset[1] / spacing).round() as i64,
    ]
}

/// The control value of `tile`'s stored sample `(column, row)` (0 to 258,
/// samples -1 to 257) as generating it over `layers` with `rules` gives it:
/// what gameplay reads where tiles hold none, the GPU
/// giving them to rendering. The same inputs and evaluation as a tile's
/// rows, the meters to rivers' edges and the rivers' beds the tile's own,
/// so the same bits.
pub fn sample_control(
    tile: &Tile,
    (layers, rules): (&Layers, &BiomeRules),
    (column, row): (usize, usize),
) -> Control {
    let stride = TILE_STRIDE as usize;
    let spacing = tile.spacing();
    let heights = tile.heights();
    let first = first_sample(layers.offset.to_array(), tile.key(), spacing);
    let (up, down) = (row.saturating_sub(1), (row + 1).min(stride - 1));
    let (left, right) = (column.saturating_sub(1), (column + 1).min(stride - 1));
    let at = |r: usize, c: usize| heights[r * stride + c];
    let shape = SampleShape::of(
        [up, row, down].map(|r| [left, column, right].map(|c| at(r, c))),
        spacing,
    );
    let x = (first[0] + column as i64 - 1) as f32 * spacing;
    let z = (first[1] + row as i64 - 1) as f32 * spacing;
    let rivers = tile.river_inputs(column as i32 - 1, row as i32 - 1);
    control_with(
        (layers, rules),
        [x, z],
        (at(row, column), shape),
        (spacing, rivers),
        &mut Vec::new(),
    )
}

/// The control value `rules` give ground over `layers` at `(x, z)` in the
/// layers' frame, whose height there is `height` and whose smooth surface
/// there, through heights `spacing` meters apart, has `shape`: the inputs a
/// tile's sample takes, the meters to rivers' edges and the bed read from
/// the rivers there, evaluated as [`sample_control`] evaluates them.
/// `weights` is scratch space, reused across samples.
pub(crate) fn control_at(
    (layers, rules): (&Layers, &BiomeRules),
    [x, z]: [f32; 2],
    (height, shape): (f32, SampleShape),
    spacing: f32,
    weights: &mut Vec<(u8, f32)>,
) -> Control {
    let rivers = match layers.hydrology() {
        Some(water) if rules.reads().water => {
            RiverInputs::at(&water.rivers, [x, z], (height, spacing))
        }
        _ => RiverInputs::FAR,
    };
    control_with(
        (layers, rules),
        [x, z],
        (height, shape),
        (spacing, rivers),
        weights,
    )
}

/// [`control_at`] at samples `spacing` meters apart, with what masks read of
/// the rivers there, `rivers`, already read.
fn control_with(
    (layers, rules): (&Layers, &BiomeRules),
    [x, z]: [f32; 2],
    (height, shape): (f32, SampleShape),
    (spacing, rivers): (f32, RiverInputs),
    weights: &mut Vec<(u8, f32)>,
) -> Control {
    let reads = rules.reads();
    let shown = layers.shown(x, z);
    let water = match layers.hydrology() {
        Some(water) if shown > 0.0 && reads.water => rivers.with_maps(water, [x, z], spacing),
        _ => HydrologySample::default(),
    };
    // What the erosions changed there, as a tile's row reads it.
    let mut erosion = [0.0];
    if reads.erosion {
        erosion_row(layers, &[x], z, spacing, &mut erosion);
    }
    let inputs = SampleInputs {
        x,
        z,
        height,
        gradient: shape.gradient,
        curvature: shape.curvature,
        spacing,
        water: HydrologySample {
            flow: water.flow * shown,
            wetness: water.wetness * shown,
            lake: water.lake * shown,
            ..water
        },
        erosion: erosion[0] * shown,
        path: if shown > 0.0 && reads.path {
            layers.path_at(x, z)
        } else {
            PathSample::FAR
        },
    };
    rules.control(&inputs, weights)
}

fn push_f32(words: &mut Vec<u32>, values: impl IntoIterator<Item = f32>) {
    words.extend(values.into_iter().map(f32::to_bits));
}

/// The hydrology bake's images, as [`header::HYDROLOGY`] lays them out.
fn write_hydrology(hydrology: &Hydrology, words: &mut Vec<u32>) {
    push_f32(words, [hydrology.spacing]);
    push_f32(
        words,
        [
            FLOW_RANGE[0],
            FLOW_RANGE[1],
            WETNESS_RANGE[0],
            WETNESS_RANGE[1],
            LAKE_RANGE[0],
            LAKE_RANGE[1],
        ],
    );
    write_images(
        &[&hydrology.flow, &hydrology.wetness, &hydrology.lakes],
        words,
    );
}

/// The stack's erosions' change images, as [`header::EROSION`] lays them
/// out: each read at a point times the inverse of the meters between its
/// samples, as erosion reads it ([`Erosion::change_row`]).
fn write_erosions(stages: &[&Erosion], words: &mut Vec<u32>) {
    let table = words.len();
    words.push(stages.len() as u32);
    words.extend(std::iter::repeat_n(0, stages.len()));
    for (index, erosion) in stages.iter().enumerate() {
        words[table + 1 + index] = words.len() as u32;
        push_f32(
            words,
            [
                1.0 / erosion.spacing,
                erosion.spacing,
                erosion.range[0],
                erosion.range[1],
            ],
        );
        write_images(&[&erosion.change], words);
    }
}

/// `images`, as a section of the words lays them out: each one's table
/// first (its level count, then each level's width, height and first
/// sample), their samples after (`f32`).
fn write_images(images: &[&HeightImage], words: &mut Vec<u32>) {
    let mut tables = Vec::new();
    for image in images {
        tables.push(words.len());
        let levels = image.levels.len() + 1;
        words.push(levels as u32);
        words.extend(std::iter::repeat_n(0, 3 * levels));
    }
    for (image, table) in images.iter().zip(tables) {
        let mut entry = table + 1;
        let mut level = |words: &mut Vec<u32>, width: u32, height: u32| {
            words[entry] = width;
            words[entry + 1] = height;
            words[entry + 2] = words.len() as u32;
            entry += 3;
        };
        level(words, image.width, image.height);
        push_f32(words, image.samples.iter().map(|&sample| f32::from(sample)));
        for coarse in &image.levels {
            level(words, coarse.width, coarse.height);
            push_f32(words, coarse.samples.iter().copied());
        }
    }
}

/// The roads, as [`header::PATHS`] lays them out.
fn write_paths(paths: &[&PathProfile], words: &mut Vec<u32>) {
    let table = words.len();
    words.push(paths.len() as u32);
    words.extend(std::iter::repeat_n(0, paths.len()));
    for (index, path) in paths.iter().enumerate() {
        words[table + 1 + index] = words.len() as u32;
        let at = words.len();
        push_f32(
            words,
            [path.half, path.radius, path.cell, path.min.x, path.min.y],
        );
        words.push(path.columns as u32);
        words.push(path.rows as u32);
        words.extend([0; 6]);
        words[at + 7] = words.len() as u32;
        words.extend(path.starts.iter().copied());
        words[at + 8] = words.len() as u32;
        words.extend(path.segments.iter().copied());
        words[at + 9] = words.len() as u32;
        push_f32(words, path.line.iter().flat_map(|point| [point.x, point.y]));
        words[at + 10] = words.len() as u32;
        push_f32(words, path.along.iter().copied());
        words[at + 11] = words.len() as u32;
        let spans: Vec<[f32; 2]> = path.spans.iter().chain(&path.tunnels).copied().collect();
        push_f32(words, spans.iter().flat_map(|span| *span));
        words[at + 12] = spans.len() as u32;
    }
}
