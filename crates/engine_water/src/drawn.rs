use std::sync::Arc;

use bevy_ecs::prelude::*;
use bytemuck::Pod;
use engine_codec::{Codec, Reader, record};
use engine_core::{
    Error, ErrorCode, FrameTime, Job, JobCx, JobExchange, JobRef, VIEW_JOBS, WorkClock,
    job_priority,
};
use engine_heightfield::tile::{
    Footprint, PATCH_QUADS, PATCHES_PER_TILE, TILE_STRIDE, TileJob, bed_at,
};
use engine_heightfield::{ErrorPyramid, TileKey};
use engine_places::ground::GroundWork;
use glam::Vec2;

use crate::flow::{Levels, rivers_over};
use crate::gpu::{NO_WATER_SURFACE, WATER_BLOCK as BLOCK};
use crate::jobs::{DRAWN, Finished, kinds, outside, recipe};
use crate::shallow_water::roughness;
use crate::state::{Base, Drawing, HeldWater, WaterState};
use crate::tile_water::rewatered;
use crate::work::Work;
use crate::{TileWater, WaterMesh};

pub(crate) const SURFACE_BYTES: u32 = 8;
pub(crate) const INFO_BYTES: u32 = 16;
const BLOCK_TEXELS: usize = (BLOCK * BLOCK) as usize;
const SURFACES: usize = BLOCK_TEXELS * SURFACE_BYTES as usize;
pub(crate) const PATCH_BYTES: usize = SURFACES + BLOCK_TEXELS * INFO_BYTES as usize;
const SAMPLES: usize = (TILE_STRIDE * TILE_STRIDE) as usize;
const EVERY_PATCH: [u64; 4] = [u64::MAX; 4];

pub(crate) struct DrawnWater {
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

pub(crate) struct DrawAsk {
    pub(crate) terrain: Entity,
    pub(crate) key: TileKey,
    pub(crate) stamp: engine_core::Stamp,
    pub(crate) record: Arc<[u8]>,
    pub(crate) drawing: Drawing,
}

pub(crate) fn is_dry(record: &[u8]) -> Result<bool, Error> {
    Ok(!Reader::new(record).get::<bool>()?)
}

#[derive(Clone, Debug, PartialEq)]
struct DrawHead {
    spacing: f32,
    beds: u64,
}

record!(DrawHead { spacing, beds });

#[derive(Clone, Debug, PartialEq)]
struct FloodHead {
    tile: TileJob,
    flow: [u8; 32],
    running: u64,
    regions: Vec<[f32; 4]>,
    spacing: f32,
    beds: u64,
    wet: bool,
    held: [u64; 4],
    solids: Box<[Footprint]>,
}

record!(FloodHead {
    tile,
    flow,
    running,
    regions,
    spacing,
    beds,
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

fn headed<'a, T: Codec>(bytes: &'a [u8]) -> Result<(T, &'a [u8]), Error> {
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

struct Ground<'a> {
    heights: &'a [f32],
    beds: Option<&'a [u16]>,
    spacing: f32,
}

fn patch_texels(
    ground: &Ground<'_>,
    (water, mesh): (&TileWater, &WaterMesh),
    (px, pz): (usize, usize),
    (surfaces, info): (&mut Vec<[f32; 2]>, &mut Vec<[u32; 4]>),
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
            let local = Vec2::new(column as f32 - 1.0, row as f32 - 1.0) * ground.spacing;
            let level = mesh.level(at);
            surfaces.push([
                if mesh.holds(at) {
                    level
                } else {
                    NO_WATER_SURFACE
                },
                ground.heights[at],
            ]);
            let [packed, state] = words[at];
            let bed = ground
                .beds
                .and_then(|beds| bed_at(beds, ground.heights, ground.spacing, local));
            info.push([
                packed,
                state,
                level.to_bits(),
                roughness(bed.as_ref()).to_bits(),
            ]);
        }
    }
}

struct Painted {
    errors: Arc<[u16]>,
    texels: Vec<u8>,
    changed: [u64; 4],
}

fn paint(water: &TileWater, ground: &Ground<'_>, before: Option<&TileWater>) -> Painted {
    let patches = PATCHES_PER_TILE as usize;
    let mesh = water.mesh(ground.heights);
    let holding: Vec<bool> = (0..mesh.levels().len()).map(|at| mesh.holds(at)).collect();
    let errors = ErrorPyramid::over(mesh.levels(), Some(&holding[..])).shared_halves();
    let was = before.map(|before| (before, before.mesh(ground.heights)));
    let mut surfaces: Vec<[f32; 2]> = Vec::with_capacity(BLOCK_TEXELS);
    let mut info: Vec<[u32; 4]> = Vec::with_capacity(BLOCK_TEXELS);
    let mut old_surfaces: Vec<[f32; 2]> = Vec::with_capacity(BLOCK_TEXELS);
    let mut old_info: Vec<[u32; 4]> = Vec::with_capacity(BLOCK_TEXELS);
    let mut texels = Vec::new();
    let mut changed = if was.is_some() { [0; 4] } else { EVERY_PATCH };
    for pz in 0..patches {
        for px in 0..patches {
            if !water.patch_holds_water(px as u32, pz as u32) {
                continue;
            }
            let patch = pz * patches + px;
            patch_texels(
                ground,
                (water, &mesh),
                (px, pz),
                (&mut surfaces, &mut info),
            );
            let surface_bytes: &[u8] = bytemuck::cast_slice(&surfaces);
            let info_bytes: &[u8] = bytemuck::cast_slice(&info);
            texels.extend_from_slice(surface_bytes);
            texels.extend_from_slice(info_bytes);
            let Some((before, mesh)) = &was else {
                continue;
            };
            let same = before.patch_holds_water(px as u32, pz as u32) && {
                patch_texels(
                    ground,
                    (*before, mesh),
                    (px, pz),
                    (&mut old_surfaces, &mut old_info),
                );
                bytemuck::cast_slice::<[f32; 2], u8>(&old_surfaces) == surface_bytes
                    && bytemuck::cast_slice::<[u32; 4], u8>(&old_info) == info_bytes
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

fn landed(bytes: &[u8]) -> Result<Option<(TileWater, DrawnWater)>, Error> {
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
    let drawn = DrawnWater {
        errors: Arc::from(errors),
        texels: Arc::from(rest),
        changed: head.changed,
    };
    Ok(Some((water, drawn)))
}

pub(crate) fn start_drawn(input: &[u8], _: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    let (head, mut rest) = headed::<DrawHead>(input)?;
    let heights = take_words::<f32>(&mut rest, SAMPLES)?;
    let beds = take_words::<u16>(&mut rest, head.beds as usize)?;
    let output = match engine_codec::decode::<Option<TileWater>>(rest)? {
        Some(water) => {
            let ground = Ground {
                heights: &heights,
                beds: (!beds.is_empty()).then_some(&beds[..]),
                spacing: head.spacing,
            };
            drawn_output(&water, &paint(&water, &ground, None))
        }
        None => dry_output(),
    };
    Ok(Box::new(Finished(Some(output))))
}

pub(crate) fn start_rewater(input: &[u8], cx: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    let (head, mut rest) = headed::<FloodHead>(input)?;
    let running = take(&mut rest, head.running as usize)?;
    let heights = take_words::<f32>(&mut rest, SAMPLES)?;
    let beds = take_words::<u16>(&mut rest, head.beds as usize)?;
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
        cx.held(head.flow, || {
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
        Some(water) => {
            let ground = Ground {
                heights: &heights,
                beds: (!beds.is_empty()).then_some(&beds[..]),
                spacing: head.spacing,
            };
            drawn_output(&water, &paint(&water, &ground, before.as_ref()))
        }
        None => dry_output(),
    };
    Ok(Box::new(Finished(Some(output))))
}

pub(crate) fn draw_input(held: &HeldWater, record: &[u8]) -> Vec<u8> {
    let beds = held.face.river_beds.as_deref().unwrap_or(&[]);
    let head = DrawHead {
        spacing: held.face.spacing,
        beds: beds.len() as u64,
    };
    let mut input = engine_codec::encode(&head);
    input.extend_from_slice(bytemuck::cast_slice(&held.face.heights));
    input.extend_from_slice(bytemuck::cast_slice(beds));
    input.extend_from_slice(record);
    input
}

pub(crate) fn flood_input(
    held: &HeldWater,
    tile: TileJob,
    (flow, running): ([u8; 32], &[u8]),
    regions: Vec<[f32; 4]>,
) -> Vec<u8> {
    let beds = held.face.river_beds.as_deref().unwrap_or(&[]);
    let water = held.water.as_deref();
    let head = FloodHead {
        tile,
        flow,
        running: running.len() as u64,
        regions,
        spacing: held.face.spacing,
        beds: beds.len() as u64,
        wet: water.is_some(),
        held: water.map_or([0; 4], TileWater::holding),
        solids: water.map_or(&[][..], TileWater::solids).into(),
    };
    let mut input = engine_codec::encode(&head);
    input.extend_from_slice(running);
    input.extend_from_slice(bytemuck::cast_slice(&held.face.heights));
    input.extend_from_slice(bytemuck::cast_slice(beds));
    if let Some(water) = water {
        input.extend_from_slice(bytemuck::cast_slice(water.surfaces()));
        input.extend_from_slice(bytemuck::cast_slice(water.info()));
    }
    input
}

pub(crate) fn draw(
    mut exchange: ResMut<JobExchange>,
    time: Res<FrameTime>,
    (clock, ground): (Res<WorkClock>, Res<GroundWork>),
    mut state: ResMut<WaterState>,
) {
    let state = &mut *state;
    let work = Work::until(&clock, &ground, state.work_end);
    for ask in std::mem::take(&mut state.draw_asks) {
        let Some(held) = state
            .terrains
            .iter_mut()
            .find(|terrain| terrain.entity == ask.terrain)
            .and_then(|terrain| terrain.held.get_mut(&ask.key))
            .filter(|held| held.face.stamp == ask.stamp)
        else {
            continue;
        };
        let job = exchange.submit(
            JobRef::from(&DRAWN),
            draw_input(held, &ask.record),
            job_priority(VIEW_JOBS, 0),
        );
        held.drawing = Some((job, ask.drawing));
        state.flights.push((job, ask.terrain, ask.key));
    }
    let mut landed_any = false;
    let flights = std::mem::take(&mut state.flights);
    for (job, terrain, key) in flights {
        if landed_any && work.expired() {
            state.flights.push((job, terrain, key));
            continue;
        }
        let Some(output) = exchange.take(job) else {
            state.flights.push((job, terrain, key));
            continue;
        };
        landed_any = true;
        let Some(terrain) = state
            .terrains
            .iter_mut()
            .find(|held| held.entity == terrain)
        else {
            continue;
        };
        let Some(held) = terrain.held.get_mut(&key) else {
            continue;
        };
        if !held.drawing.as_ref().is_some_and(|(at, _)| *at == job) {
            continue;
        }
        let Some((_, drawing)) = held.drawing.take() else {
            continue;
        };
        let made = output.and_then(|bytes| {
            landed(&bytes).map_err(|error| {
                format!("{}: its output does not decode: {error}", DRAWN.name())
            })
        });
        let frame = time.frame;
        match (made, drawing) {
            (Ok(made), Drawing::Base) => {
                held.base = Base::Made;
                held.hold(made);
                held.watered = None;
            }
            (Ok(made), Drawing::Given(record)) => {
                held.hold(made);
                held.watered = None;
                held.given = Some(record);
            }
            (Ok(made), Drawing::Flood(to)) => {
                if made.is_some() {
                    state.rewatered_tiles += 1;
                    held.hold(made);
                }
                held.watered = to;
                terrain.flow_moved = true;
            }
            (Err(error), Drawing::Base) => {
                held.base = Base::Failed(frame);
                state.failures.record(frame, error);
            }
            (Err(error), Drawing::Given(_)) => {
                terrain.settled.water_failed(key, frame);
                state.failures.record(frame, error);
            }
            (Err(error), Drawing::Flood(to)) => {
                held.watered = to;
                terrain.flow_moved = true;
                state.failures.record(frame, error);
            }
        }
        terrain.touch(key);
    }
}

pub(crate) fn stop_draws(mut exchange: ResMut<JobExchange>, mut state: ResMut<WaterState>) {
    for (job, _, _) in std::mem::take(&mut state.flights) {
        exchange.cancel(job);
    }
    state.draw_asks.clear();
}
