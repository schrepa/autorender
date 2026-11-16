use super::surface::{BiomeRules, Reads, SampleInputs, erosion_row, published_row, rested};
use super::{Control, RiverInputs, SampleShape, Tile};
use crate::HydrologyBake;
use crate::hydrology::{FLOW_RANGE, LAKE_RANGE, WETNESS_RANGE};
use crate::layer::{ChangeMap, Published, TerrainLayer, maps};
use crate::{HeightImage, HydrologySample, Layers, PathSample, TILE_STRIDE};

pub const OP_CONSTANT: u32 = 0;
pub const OP_HEIGHT: u32 = 1;
pub const OP_GRADIENT: u32 = 2;
pub const OP_CURVATURE: u32 = 3;
pub const OP_FLOW: u32 = 4;
pub const OP_WETNESS: u32 = 5;
pub const OP_LAKE: u32 = 6;
pub const OP_RIVER: u32 = 7;
pub const OP_PATH: u32 = 8;
pub const OP_NOISE: u32 = 9;
pub const OP_MULTIPLY: u32 = 10;
pub const OP_ADD: u32 = 11;
pub const OP_MIN: u32 = 12;
pub const OP_MAX: u32 = 13;
pub const OP_INVERT: u32 = 14;
pub const OP_REMAP: u32 = 15;
pub const OP_GRAIN: u32 = 16;
pub const OP_POWER: u32 = 17;
pub const OP_FLOOD: u32 = 18;
pub const OP_WORN: u32 = 19;
pub const OP_DEPOSIT: u32 = 20;
pub const OP_FACING: u32 = 21;
pub const OP_VALUE: u32 = 22;
pub const OP_END: u32 = 23;
pub const OP_MAP: u32 = 24;
pub const OP_MAP_VALUE: u32 = 25;
pub const H_PROGRAM: u32 = 0;
pub const H_STACK: u32 = 1;
pub const H_BIOMES: u32 = 2;
pub const H_MATERIALS: u32 = 3;
pub const H_READS: u32 = 4;
pub const H_HYDROLOGY: u32 = 5;
pub const H_PATHS: u32 = 6;
pub const H_FADE: u32 = 7;
pub const H_WET_BIOMES: u32 = 8;
pub const H_EROSION: u32 = 9;
pub const H_MAPS: u32 = 10;
pub const H_SITE: u32 = 11;
pub const H_DEBRIS: u32 = 12;
pub const BIOME_WORDS: u32 = 3;
pub const LAYER_COVERS: u32 = 0x8000_0000;
pub const MAX_STACK: u32 = 16;
pub const MAX_MAPS: u32 = 8;
pub const MAX_MATERIALS: u32 = 64;

pub mod op {
    pub use super::{
        OP_ADD as ADD, OP_CONSTANT as CONSTANT, OP_CURVATURE as CURVATURE, OP_DEPOSIT as DEPOSIT,
        OP_END as END, OP_FACING as FACING, OP_FLOOD as FLOOD, OP_FLOW as FLOW,
        OP_GRADIENT as GRADIENT, OP_GRAIN as GRAIN, OP_HEIGHT as HEIGHT, OP_INVERT as INVERT,
        OP_LAKE as LAKE, OP_MAP as MAP, OP_MAP_VALUE as MAP_VALUE, OP_MAX as MAX, OP_MIN as MIN,
        OP_MULTIPLY as MULTIPLY, OP_NOISE as NOISE,
        OP_PATH as PATH, OP_POWER as POWER, OP_REMAP as REMAP, OP_RIVER as RIVER,
        OP_VALUE as VALUE, OP_WETNESS as WETNESS, OP_WORN as WORN,
    };
}

pub mod header {
    pub const PROGRAM: usize = super::H_PROGRAM as usize;
    pub const STACK: usize = super::H_STACK as usize;
    pub const BIOMES: usize = super::H_BIOMES as usize;
    pub const MATERIALS: usize = super::H_MATERIALS as usize;
    pub const READS: usize = super::H_READS as usize;
    pub const HYDROLOGY: usize = super::H_HYDROLOGY as usize;
    pub const PATHS: usize = super::H_PATHS as usize;
    pub const FADE: usize = super::H_FADE as usize;
    pub const WET_BIOMES: usize = super::H_WET_BIOMES as usize;
    pub const EROSION: usize = super::H_EROSION as usize;
    pub const MAPS: usize = super::H_MAPS as usize;
    pub const SITE: usize = super::H_SITE as usize;
    pub const DEBRIS: usize = super::H_DEBRIS as usize;
    pub const LEN: usize = 16;
}

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
        let site = layers.site();
        let reads = Reads {
            water: rules.reads().water || site.is_some(),
            erosion: rules.reads().erosion || site.is_some(),
            ..rules.reads()
        };
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
        if rules.users().next().is_some() {
            words[header::MAPS] = words.len() as u32;
            write_maps(rules.users(), layers, &mut words);
        }
        if let Some(rests) = layers.rests(maps::DEPOSIT) {
            words[header::DEBRIS] = words.len() as u32;
            push_f32(&mut words, [rests.slope[0], rests.slope[1], rests.gather]);
        }
        if let Some(site) = site {
            words[header::SITE] = words.len() as u32;
            push_f32(&mut words, site.ranges.into_iter().flatten());
            let names = [site.sun, site.soil, site.exposure];
            write_maps(names.into_iter(), layers, &mut words);
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

    pub fn offset(&self) -> [f32; 2] {
        self.offset
    }

    pub fn hydrology(&self) -> Option<(f32, usize)> {
        self.hydrology
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

pub fn control_at(
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
    let mut maps = [0.0; MAX_MAPS as usize];
    if rules.users().next().is_some() {
        let mut rows = Vec::new();
        let shape = (&[shape.gradient][..], &[shape.curvature][..]);
        published_row(layers, rules.users(), (&[x], z, spacing), shape, &mut rows);
        for (value, row) in maps.iter_mut().zip(&rows) {
            *value = row[0];
        }
    }
    let inputs = SampleInputs {
        maps,
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
        erosion: rested(
            erosion[0] * shown,
            layers.rests(maps::DEPOSIT),
            shape.gradient,
            shape.curvature,
        ),
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

fn write_hydrology(hydrology: &HydrologyBake, words: &mut Vec<u32>) {
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

fn write_maps<'n>(names: impl Iterator<Item = &'n str>, layers: &Layers, words: &mut Vec<u32>) {
    let names: Vec<&str> = names.collect();
    let table = words.len();
    words.push(names.len() as u32);
    words.extend(std::iter::repeat_n(u32::MAX, names.len()));
    for (index, name) in names.into_iter().enumerate() {
        let Some(Published { map, rests, .. }) =
            (layers.published()).find(|published| published.name == name)
        else {
            continue;
        };
        words[table + 1 + index] = words.len() as u32;
        push_f32(
            words,
            [1.0 / map.spacing, map.spacing, map.range[0], map.range[1]],
        );
        words.push(u32::from(rests.is_some()));
        let rests = rests.map_or([0.0; 3], |rests| {
            [rests.slope[0], rests.slope[1], rests.gather]
        });
        push_f32(words, rests);
        write_images(&[map.image], words);
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
