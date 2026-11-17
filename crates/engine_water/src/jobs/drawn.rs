use std::sync::Arc;

use bytemuck::Pod;
use engine_codec::{Codec, Reader, record};
use engine_core::{Error, ErrorCode, Job, JobCx, RecordKey};
use engine_heightfield::ErrorPyramid;
use engine_heightfield::rivers::Rivers;
use engine_heightfield::tile::{Footprint, PATCH_QUADS, PATCHES_PER_TILE, TILE_STRIDE, TileJob};
use glam::Vec2;

use super::gpu::{NO_WATER_SURFACE, WATER_BLOCK as BLOCK};
use super::tile_water::{TileWater, WaterMesh, rewatered};
use super::{Finished, kinds, outside, recipe};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Levels {
    pub discharge: Vec<f32>,
    pub half: Vec<f32>,
    pub depth: Vec<f32>,
    pub surface: Vec<f32>,
    pub speed: Vec<f32>,
}

record!(Levels {
    discharge,
    half,
    depth,
    surface,
    speed,
});

pub(crate) const SURFACE_BYTES: u32 = 8;
pub(crate) const INFO_BYTES: u32 = 8;
const BLOCK_TEXELS: usize = (BLOCK * BLOCK) as usize;
const SURFACES: usize = BLOCK_TEXELS * SURFACE_BYTES as usize;
pub(crate) const PATCH_BYTES: usize = SURFACES + BLOCK_TEXELS * INFO_BYTES as usize;
const SAMPLES: usize = (TILE_STRIDE * TILE_STRIDE) as usize;
const EVERY_PATCH: [u64; 4] = [u64::MAX; 4];
const REACH: usize = ((BLOCK + 1 - PATCH_QUADS) / 2) as usize;
const SPLINE: [f32; 3] = [1.0, 4.0, 1.0];

pub(crate) struct DrawnWater {
    pub(crate) record: RecordKey,
    pub(crate) errors: Arc<[u16]>,
    pub(crate) texels: Arc<[u8]>,
    pub(crate) changed: [u64; 4],
}

impl DrawnWater {
    pub(crate) fn changed(&self, patch: usize) -> bool {
        self.changed[patch / 64] & (1 << (patch % 64)) != 0
    }

    pub(crate) fn surfaces(&self, at: usize) -> &[u8] {
        &self.texels[at..at + SURFACES]
    }

    pub(crate) fn info(&self, at: usize) -> &[u8] {
        &self.texels[at + SURFACES..at + PATCH_BYTES]
    }
}

pub(crate) fn is_dry(record: &[u8]) -> Result<bool, Error> {
    Ok(!Reader::new(record).get::<bool>()?)
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FloodHead {
    pub(crate) tile: TileJob,
    pub(crate) flow: [u8; 32],
    pub(crate) running: u64,
    pub(crate) regions: Vec<[f32; 4]>,
    pub(crate) spacing: f32,
    pub(crate) wet: bool,
    pub(crate) held: [u64; 4],
    pub(crate) solids: Box<[Footprint]>,
}

record!(FloodHead {
    tile,
    flow,
    running,
    regions,
    spacing,
    wet,
    held,
    solids,
});

#[derive(Clone, Debug, PartialEq)]
struct DrawnHead {
    held: [u64; 4],
    solids: Box<[Footprint]>,
    errors: u64,
    changed: [u64; 4],
}

record!(DrawnHead {
    held,
    solids,
    errors,
    changed,
});

fn headed<T: Codec>(bytes: &[u8]) -> Result<(T, &[u8]), Error> {
    let mut reader = Reader::new(bytes);
    let head = reader.get()?;
    let at = bytes.len() - reader.left();
    Ok((head, &bytes[at..]))
}

fn take<'a>(bytes: &mut &'a [u8], count: usize) -> Result<&'a [u8], Error> {
    let Some((head, rest)) = bytes.split_at_checked(count) else {
        return Err(Error::new(ErrorCode::InvalidValue, "$.water_draw").got(count));
    };
    *bytes = rest;
    Ok(head)
}

fn words<T: Pod>(bytes: &[u8]) -> Vec<T> {
    let mut found = vec![T::zeroed(); bytes.len() / size_of::<T>()];
    let filled = found.len() * size_of::<T>();
    bytemuck::cast_slice_mut::<T, u8>(&mut found).copy_from_slice(&bytes[..filled]);
    found
}

fn take_words<T: Pod>(bytes: &mut &[u8], count: usize) -> Result<Vec<T>, Error> {
    Ok(words(take(bytes, count * size_of::<T>())?))
}

fn carried(mesh: &WaterMesh, heights: &[f32]) -> Vec<[f32; 2]> {
    let stride = TILE_STRIDE as usize;
    let wet = |at: usize| mesh.holds(at) && mesh.level(at) > heights[at];
    let mut texels: Vec<[f32; 2]> = (0..heights.len())
        .map(|at| {
            let level = if wet(at) { mesh.level(at) } else { NO_WATER_SURFACE };
            [level, heights[at]]
        })
        .collect();
    let mut reached: Vec<bool> = (0..heights.len()).map(wet).collect();
    for _ in 0..REACH {
        let ring: Vec<(usize, f32)> = (0..texels.len())
            .filter(|&at| !reached[at])
            .filter_map(|at| {
                let (x, z) = (at % stride, at / stride);
                let (mut sum, mut weight) = (0.0, 0.0);
                for (dz, along_z) in SPLINE.iter().enumerate() {
                    for (dx, along_x) in SPLINE.iter().enumerate() {
                        let (nx, nz) = ((x + dx).wrapping_sub(1), (z + dz).wrapping_sub(1));
                        if nx >= stride || nz >= stride || !reached[nz * stride + nx] {
                            continue;
                        }
                        let share = along_x * along_z;
                        sum += share * texels[nz * stride + nx][0];
                        weight += share;
                    }
                }
                (weight > 0.0).then(|| (at, sum / weight))
            })
            .collect();
        for (at, level) in ring {
            texels[at] = [level, engine_math::max(heights[at], level)];
            reached[at] = true;
        }
    }
    texels
}

fn patch_texels(
    (water, texels): (&TileWater, &[[f32; 2]]),
    (px, pz): (usize, usize),
    (surfaces, info): (&mut Vec<[f32; 2]>, &mut Vec<[u32; 2]>),
) {
    surfaces.clear();
    info.clear();
    let stride = TILE_STRIDE as usize;
    let side = BLOCK as usize;
    let quads = PATCH_QUADS as usize;
    let words = water.info();
    for z in 0..side {
        for x in 0..side {
            let (column, row) = (px * quads + x, pz * quads + z);
            let at = row * stride + column;
            surfaces.push(texels[at]);
            info.push(words[at]);
        }
    }
}

struct Painted {
    errors: Arc<[u16]>,
    texels: Vec<u8>,
    changed: [u64; 4],
}

fn paint(water: &TileWater, heights: &[f32], before: Option<&TileWater>) -> Painted {
    let patches = PATCHES_PER_TILE as usize;
    let mesh = water.mesh(heights);
    let holding: Vec<bool> = (0..mesh.levels().len()).map(|at| mesh.holds(at)).collect();
    let errors = ErrorPyramid::over(mesh.levels(), Some(&holding[..])).shared_halves();
    let drawn = carried(&mesh, heights);
    let was = before.map(|before| (before, carried(&before.mesh(heights), heights)));
    let mut surfaces: Vec<[f32; 2]> = Vec::with_capacity(BLOCK_TEXELS);
    let mut info: Vec<[u32; 2]> = Vec::with_capacity(BLOCK_TEXELS);
    let mut old_surfaces: Vec<[f32; 2]> = Vec::with_capacity(BLOCK_TEXELS);
    let mut old_info: Vec<[u32; 2]> = Vec::with_capacity(BLOCK_TEXELS);
    let mut texels = Vec::new();
    let mut changed = if was.is_some() { [0; 4] } else { EVERY_PATCH };
    for pz in 0..patches {
        for px in 0..patches {
            if !water.patch_holds_water(px as u32, pz as u32) {
                continue;
            }
            let patch = pz * patches + px;
            patch_texels((water, &drawn), (px, pz), (&mut surfaces, &mut info));
            let surface_bytes: &[u8] = bytemuck::cast_slice(&surfaces);
            let info_bytes: &[u8] = bytemuck::cast_slice(&info);
            texels.extend_from_slice(surface_bytes);
            texels.extend_from_slice(info_bytes);
            let Some((before, drawn_before)) = &was else {
                continue;
            };
            let same = before.patch_holds_water(px as u32, pz as u32) && {
                patch_texels(
                    (*before, drawn_before),
                    (px, pz),
                    (&mut old_surfaces, &mut old_info),
                );
                bytemuck::cast_slice::<[f32; 2], u8>(&old_surfaces) == surface_bytes
                    && bytemuck::cast_slice::<[u32; 2], u8>(&old_info) == info_bytes
            };
            if !same {
                changed[patch / 64] |= 1 << (patch % 64);
            }
        }
    }
    Painted {
        errors,
        texels,
        changed,
    }
}

fn rivers_over(base: &Rivers, running: Vec<(u32, Levels)>) -> Rivers {
    let mut whole = base.rivers().to_vec();
    for (river, levels) in running {
        if let Some(line) = whole.get_mut(river as usize) {
            line.discharge = levels.discharge;
            line.half = levels.half;
            line.depth = levels.depth;
            line.surface = levels.surface;
            line.speed = levels.speed;
        }
    }
    base.with_rivers(whole)
}

fn dry_output() -> Vec<u8> {
    engine_codec::encode(&None::<DrawnHead>)
}

fn drawn_output(water: &TileWater, painted: &Painted) -> Vec<u8> {
    let head = DrawnHead {
        held: water.holding(),
        solids: Box::from(water.solids()),
        errors: painted.errors.len() as u64,
        changed: painted.changed,
    };
    let mut output = engine_codec::encode(&Some(head));
    output.extend_from_slice(bytemuck::cast_slice(water.surfaces()));
    output.extend_from_slice(bytemuck::cast_slice(water.info()));
    output.extend_from_slice(bytemuck::cast_slice(&painted.errors));
    output.extend_from_slice(&painted.texels);
    output
}

pub(crate) struct DrawnParts {
    water: TileWater,
    errors: Arc<[u16]>,
    texels: Arc<[u8]>,
    changed: [u64; 4],
}

impl DrawnParts {
    pub(crate) fn keyed(self, record: RecordKey) -> (TileWater, DrawnWater) {
        let drawn = DrawnWater {
            record,
            errors: self.errors,
            texels: self.texels,
            changed: self.changed,
        };
        (self.water, drawn)
    }
}

pub(crate) fn landed(bytes: &[u8]) -> Result<Option<DrawnParts>, Error> {
    let (head, mut rest) = headed::<Option<DrawnHead>>(bytes)?;
    let Some(head) = head else {
        return Ok(None);
    };
    let surface = take_words::<f32>(&mut rest, SAMPLES)?;
    let info = take_words::<[u32; 2]>(&mut rest, SAMPLES)?;
    let errors = take_words::<u16>(&mut rest, head.errors as usize)?;
    let water = TileWater::from_parts(
        surface.into_boxed_slice(),
        info.into_boxed_slice(),
        head.held,
        head.solids,
    );
    Ok(Some(DrawnParts {
        water,
        errors: Arc::from(errors),
        texels: Arc::from(rest),
        changed: head.changed,
    }))
}

pub(crate) fn start_drawn(input: &[u8], _: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    let mut rest = input;
    let heights = take_words::<f32>(&mut rest, SAMPLES)?;
    let output = match engine_codec::decode::<Option<TileWater>>(rest)? {
        Some(water) => drawn_output(&water, &paint(&water, &heights, None)),
        None => dry_output(),
    };
    Ok(Box::new(Finished(Some(output))))
}

pub(crate) fn start_rewater(input: &[u8], cx: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    let (head, mut rest) = headed::<FloodHead>(input)?;
    let running = take(&mut rest, head.running as usize)?;
    let heights = take_words::<f32>(&mut rest, SAMPLES)?;
    let before = if head.wet {
        let surface = take_words::<f32>(&mut rest, SAMPLES)?;
        let info = take_words::<[u32; 2]>(&mut rest, SAMPLES)?;
        Some(TileWater::from_parts(
            surface.into_boxed_slice(),
            info.into_boxed_slice(),
            head.held,
            head.solids,
        ))
    } else {
        None
    };
    let kinds = kinds(cx);
    let recipe = recipe(cx, head.tile.recipe, &kinds)?;
    let key = head.tile.key;
    if !recipe.pyramid.contains(key) {
        return Err(outside(key));
    }
    let layers = recipe.stamped(&head.tile.stamps, &kinds)?;
    let Some(hydrology) = layers.hydrology() else {
        return Ok(Box::new(Finished(Some(dry_output()))));
    };
    let rivers = if running.is_empty() {
        Arc::clone(&hydrology.rivers)
    } else {
        let held = RecordKey::shared(&[head.tile.recipe, head.flow].concat()).hash();
        cx.held(held, || {
            let levels: Vec<(u32, Levels)> = engine_codec::decode(running)?;
            Ok(rivers_over(&hydrology.rivers, levels))
        })?
    };
    let regions: Vec<(Vec2, Vec2)> = head
        .regions
        .iter()
        .map(|&[x0, z0, x1, z1]| (Vec2::new(x0, z0), Vec2::new(x1, z1)))
        .collect();
    let made = rewatered(
        (key, head.spacing, &heights),
        before.as_ref(),
        (&layers, &recipe.pyramid),
        &rivers,
        &regions,
    );
    let output = match made {
        Some(water) => drawn_output(&water, &paint(&water, &heights, before.as_ref())),
        None => dry_output(),
    };
    Ok(Box::new(Finished(Some(output))))
}
