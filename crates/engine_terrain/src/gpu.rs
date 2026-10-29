use crate::hydrology::{FLOW_RANGE, Hydrology, LAKE_RANGE, WETNESS_RANGE};
use crate::layer::{ChangeMap, TerrainLayer};
use crate::layers::HeightImage;
use crate::surface::{BiomeRules, Control, SampleInputs, SampleShape, erosion_row};
use crate::tile::RiverInputs;
use crate::{HydrologySample, Layers, PathSample, TILE_STRIDE, Tile};

pub mod op {
    pub const CONSTANT: u32 = 0;
    pub const HEIGHT: u32 = 1;
    pub const GRADIENT: u32 = 2;
    pub const CURVATURE: u32 = 3;
    pub const FLOW: u32 = 4;
    pub const WETNESS: u32 = 5;
    pub const LAKE: u32 = 6;
    pub const RIVER: u32 = 7;
    pub const PATH: u32 = 8;
    pub const NOISE: u32 = 9;
    pub const MULTIPLY: u32 = 10;
    pub const ADD: u32 = 11;
    pub const MIN: u32 = 12;
    pub const MAX: u32 = 13;
    pub const INVERT: u32 = 14;
    pub const REMAP: u32 = 15;
    pub const GRAIN: u32 = 16;
    pub const POWER: u32 = 17;
    pub const FLOOD: u32 = 18;
    pub const WORN: u32 = 19;
    pub const DEPOSIT: u32 = 20;
    pub const FACING: u32 = 21;
    pub const VALUE: u32 = 22;
    pub const END: u32 = 23;
}

pub const BIOME_WORDS: u32 = 3;

pub mod header {
    pub const PROGRAM: usize = 0;
    pub const STACK: usize = 1;
    pub const BIOMES: usize = 2;
    pub const MATERIALS: usize = 3;
    pub const READS: usize = 4;
    pub const HYDROLOGY: usize = 5;
    pub const PATHS: usize = 6;
    pub const FADE: usize = 7;
    pub const WET_BIOMES: usize = 8;
    pub const EROSION: usize = 9;
    pub const LEN: usize = 16;
}

pub const MAX_STACK: u32 = 16;
pub const MAX_MATERIALS: u32 = 64;

#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceData {
    pub words: Vec<u32>,
    offset: [f32; 2],
    hydrology: Option<(f32, usize)>,
}

impl SurfaceData {
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
        let paths: Vec<&dyn TerrainLayer> = (layers.layers().iter())
            .filter(|layer| layer.has_path())
            .map(|layer| &**layer)
            .collect();
        if !paths.is_empty() {
            words[header::PATHS] = words.len() as u32;
            write_paths(&paths, &mut words);
        }
        let changes: Vec<ChangeMap<'_>> = layers.changes().collect();
        if reads.erosion && !changes.is_empty() {
            words[header::EROSION] = words.len() as u32;
            write_erosions(&changes, &mut words);
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

    pub fn first_sample(&self, key: crate::TileKey, spacing: f32) -> [i64; 2] {
        first_sample(self.offset, key, spacing)
    }

    pub fn level(&self, spacing: f32) -> u32 {
        self.hydrology.map_or(0, |(image, levels)| {
            crate::hydrology::image_level(image, levels, spacing) as u32
        })
    }
}

fn first_sample(offset: [f32; 2], key: crate::TileKey, spacing: f32) -> [i64; 2] {
    let quads = i64::from(crate::TILE_QUADS);
    [
        i64::from(key.x) * quads + (offset[0] / spacing).round() as i64,
        i64::from(key.z) * quads + (offset[1] / spacing).round() as i64,
    ]
}

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

fn write_erosions(changes: &[ChangeMap<'_>], words: &mut Vec<u32>) {
    let table = words.len();
    words.push(changes.len() as u32);
    words.extend(std::iter::repeat_n(0, changes.len()));
    for (index, change) in changes.iter().enumerate() {
        words[table + 1 + index] = words.len() as u32;
        push_f32(
            words,
            [
                1.0 / change.spacing,
                change.spacing,
                change.range[0],
                change.range[1],
            ],
        );
        write_images(&[change.image], words);
    }
}

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

fn write_paths(paths: &[&dyn TerrainLayer], words: &mut Vec<u32>) {
    let table = words.len();
    words.push(paths.len() as u32);
    words.extend(std::iter::repeat_n(0, paths.len()));
    for (index, path) in paths.iter().enumerate() {
        words[table + 1 + index] = words.len() as u32;
        path.gpu_path(words);
    }
}
