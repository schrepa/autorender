//! Bakes of a layer stack: layers whose
//! results read the heights the layers below them give, over a whole
//! pyramid level where rivers and lakes form and what water wears away, or
//! along a road. A level's bake generates every tile of its level from the
//! layers below it, gathers their samples into one grid, then computes:
//! hydrology at once, then its rivers' water from the finest ground along
//! them a point at a time, erosion a round at a time. A
//! path's bake reads the ground along its line a point at a time. Terrains
//! load no tiles until their bakes are done; tools bake at once.

use std::collections::HashMap;
use std::sync::Arc;

use glam::Vec2;

use crate::erosion::{Erosion, ErosionLayer, erode_round, finish};
use crate::hydrology::{FALL_SLOPE, Hydrology, HydrologyLayer, bake};
use crate::layers::{Layer, Layers};
use crate::path::{PathLayer, PathProfile, RIVER_DEPTH};
use crate::rivers::{BANK_OFFSET, RiverLine, Rivers, across, profile};
use crate::{Generator, Pyramid, TILE_QUADS, TILE_SAMPLES, TILE_STRIDE, TileKey};

/// What a layer bakes.
#[derive(Clone, Debug, PartialEq)]
pub enum BakeKind {
    Hydrology(HydrologyLayer),
    Erosion(ErosionLayer),
    Path(Arc<PathLayer>),
}

impl BakeKind {
    /// The level whose tiles it gathers; none for paths.
    fn level(&self) -> Option<u32> {
        match self {
            Self::Hydrology(layer) => Some(layer.level),
            Self::Erosion(layer) => Some(layer.level),
            Self::Path(_) => None,
        }
    }
}

/// What a bake found.
#[derive(Clone, Debug, PartialEq)]
pub enum Baked {
    Hydrology(Hydrology),
    Erosion(Erosion),
    Path(PathProfile),
}

/// A bake in progress: the layers below it generate every tile of its
/// level, whose samples gather into one grid, then it computes.
pub struct TerrainBake {
    index: usize,
    kind: BakeKind,
    level: u32,
    below: Arc<Layers>,
    pending: Vec<TileKey>,
    job: Option<(TileKey, Generator)>,
    width: usize,
    heights: Vec<f32>,
    // Erosion: the heights it started from, and rounds done.
    original: Option<Vec<f32>>,
    rounds: u32,
    // Paths: the points the ground is read at, the meters between them, and
    // the ground and any water standing on it read so far.
    profile: Vec<glam::Vec2>,
    step: f32,
    ground: Vec<f32>,
    water: Vec<Option<f32>>,
    // Hydrology: its rivers' lines and the finest ground read along them.
    rivers: Option<RiverJob>,
}

/// A hydrology bake's rivers being read: what the bake
/// found, the rivers' lines, and the finest ground read at their points so
/// far (at each line's middle and its banks).
struct RiverJob {
    hydrology: Option<Hydrology>,
    lines: Vec<RiverLine>,
    ground: Vec<Vec<[f32; 3]>>,
}

impl RiverJob {
    /// Reads the finest ground of `below` along the rivers within `budget`,
    /// the hydrology's channels carved into it; then sets every river's
    /// water, upstream rivers first, a river leaving a confluence starting
    /// no higher than those meeting there.
    fn advance(&mut self, below: &Layers, pyramid: &Pyramid, budget: &mut u64) -> Option<Baked> {
        let spacing = pyramid.spacing(0);
        let hydrology = self.hydrology.as_ref()?;
        let level = hydrology.level_for(spacing);
        let cost = below.height_cost() * 3;
        for (line, ground) in self.lines.iter().zip(&mut self.ground) {
            while ground.len() < line.points.len() {
                if *budget == 0 {
                    return None;
                }
                let index = ground.len();
                let point = line.points[index];
                let side = across(&line.points, index) * (line.half[index] + BANK_OFFSET);
                let read = |at: Vec2| {
                    let sample = [
                        (at.x / spacing).round() as i64,
                        (at.y / spacing).round() as i64,
                    ];
                    below.sample_height(pyramid, 0, sample) - hydrology.carve_at(at.x, at.y, level)
                };
                ground.push([read(point), read(point + side), read(point - side)]);
                *budget = budget.saturating_sub(cost);
            }
        }
        let key = |point: Vec2| [point.x.to_bits(), point.y.to_bits()];
        let mut ends: HashMap<[u32; 2], f32> = HashMap::new();
        let mut rivers = Vec::with_capacity(self.lines.len());
        for (line, ground) in self.lines.iter().zip(&self.ground) {
            let incoming = ends.get(&key(line.points[0])).copied();
            let river = profile(line, ground, incoming, FALL_SLOPE);
            let end = key(line.points[line.points.len() - 1]);
            let last = river.surface[river.surface.len() - 1];
            ends.entry(end)
                .and_modify(|level| *level = engine_math::min(*level, last))
                .or_insert(last);
            rivers.push(river);
        }
        let mut hydrology = self.hydrology.take()?;
        hydrology.rivers = Arc::new(Rivers::new(rivers));
        Some(Baked::Hydrology(hydrology))
    }
}

impl TerrainBake {
    /// The bake of layer `index` of `layers` (as [`Layers::unbaked`] names
    /// it) over `pyramid`.
    pub fn new(layers: &Layers, (index, kind): (usize, BakeKind), pyramid: &Pyramid) -> Self {
        let (level, pending, width, height) = match kind.level() {
            Some(level) => {
                let level = level.min(pyramid.levels() - 1);
                let [x, z] = pyramid.tiles(level);
                (
                    level,
                    pyramid.level_tiles(level).collect(),
                    x as usize * TILE_QUADS as usize + 1,
                    z as usize * TILE_QUADS as usize + 1,
                )
            }
            None => (0, Vec::new(), 0, 0),
        };
        let (profile, step) = match &kind {
            BakeKind::Path(layer) => layer.profile_points(),
            _ => (Vec::new(), 0.0),
        };
        Self {
            index,
            kind,
            level,
            below: Arc::new(layers.below(index)),
            pending,
            job: None,
            width,
            heights: vec![0.0; width * height],
            original: None,
            rounds: 0,
            ground: Vec::with_capacity(profile.len()),
            water: Vec::with_capacity(profile.len()),
            profile,
            step,
            rivers: None,
        }
    }

    /// The baked layer's index in its stack.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Generates the level's tiles, then computes, within `budget` work
    /// units (about one a sample); what it found once done.
    pub fn advance(&mut self, pyramid: &Pyramid, budget: &mut u64) -> Option<Baked> {
        if let Some(job) = &mut self.rivers {
            return job.advance(&self.below, pyramid, budget);
        }
        if let BakeKind::Path(layer) = &self.kind {
            // The ground at each profile point's nearest level-0 sample: the
            // profile averages it anyway.
            let spacing = pyramid.spacing(0);
            let cost = self.below.height_cost();
            while let Some(&point) = self.profile.get(self.ground.len()) {
                if *budget == 0 {
                    return None;
                }
                let sample = [
                    (point.x / spacing).round() as i64,
                    (point.y / spacing).round() as i64,
                ];
                let ground = self.below.sample_height(pyramid, 0, sample);
                // Rivers and lakes the hydrology bake found
                // there, on its own grid: streams narrower than its samples
                // can run below the finest ground's detail.
                let water = self.below.hydrology().and_then(|hydrology| {
                    hydrology
                        .water_at(point.x, point.y)
                        .filter(|&(_, depth)| depth > RIVER_DEPTH)
                        .map(|(surface, _)| surface)
                });
                self.ground.push(ground);
                self.water.push(water);
                *budget = budget.saturating_sub(cost);
            }
            return Some(Baked::Path(PathProfile::new(
                layer,
                self.step,
                &self.ground,
                &self.water,
            )));
        }
        while *budget > 0 {
            let Some((key, job)) = (match &mut self.job {
                Some(job) => Some(job),
                None => match self.pending.pop() {
                    Some(key) => Some(
                        &mut *self
                            .job
                            .insert((key, Generator::new(self.below.clone(), pyramid, key))),
                    ),
                    None => None,
                },
            }) else {
                let computed = self.compute(pyramid.spacing(self.level), budget);
                if computed.is_some() {
                    return computed;
                }
                // A hydrology bake reads its rivers' ground next.
                return match &mut self.rivers {
                    Some(job) => job.advance(&self.below, pyramid, budget),
                    None => None,
                };
            };
            let key = *key;
            if let Some(tile) = job.advance(budget) {
                self.job = None;
                // The tile's samples without their border.
                let (stride, samples) = (TILE_STRIDE as usize, TILE_SAMPLES as usize);
                let quads = TILE_QUADS as usize;
                let first = [usize::from(key.x) * quads, usize::from(key.z) * quads];
                for row in 0..samples {
                    let from = (row + 1) * stride + 1;
                    let to = (first[1] + row) * self.width + first[0];
                    self.heights[to..to + samples]
                        .copy_from_slice(&tile.heights()[from..from + samples]);
                }
            }
        }
        None
    }

    /// Computes from the gathered grid, whose samples lie `spacing` meters
    /// apart: hydrology at once, erosion a round per grid's worth of
    /// `budget`.
    fn compute(&mut self, spacing: f32, budget: &mut u64) -> Option<Baked> {
        match &self.kind {
            BakeKind::Path(_) => None,
            &BakeKind::Hydrology(layer) => {
                // Its rivers' water follows from the finest ground along
                // them, read a point at a time.
                let (hydrology, lines) = bake(&self.heights, self.width, spacing, &layer);
                let ground = lines
                    .iter()
                    .map(|line| Vec::with_capacity(line.points.len()))
                    .collect();
                self.rivers = Some(RiverJob {
                    hydrology: Some(hydrology),
                    lines,
                    ground,
                });
                None
            }
            &BakeKind::Erosion(layer) => {
                let original = self.original.get_or_insert_with(|| self.heights.clone());
                let cost = self.heights.len() as u64 * 8;
                while self.rounds < layer.rounds {
                    if *budget == 0 {
                        return None;
                    }
                    erode_round(&mut self.heights, self.width, spacing, &layer);
                    self.rounds += 1;
                    *budget = budget.saturating_sub(cost);
                }
                Some(Baked::Erosion(finish(
                    original,
                    &self.heights,
                    self.width,
                    spacing,
                    self.rounds,
                )))
            }
        }
    }
}

impl Layers {
    /// The first layer whose bake is not done, by index, and what it bakes:
    /// tiles wait for it.
    pub fn unbaked(&self) -> Option<(usize, BakeKind)> {
        self.layers()
            .iter()
            .enumerate()
            .find_map(|(index, layer)| match layer {
                Layer::Hydrology(stage) if stage.baked.is_none() => {
                    Some((index, BakeKind::Hydrology(stage.layer)))
                }
                Layer::Erosion(stage) if stage.baked.is_none() => {
                    Some((index, BakeKind::Erosion(stage.layer)))
                }
                Layer::Path(stage) if stage.baked.is_none() => {
                    Some((index, BakeKind::Path(stage.layer.clone())))
                }
                _ => None,
            })
    }

    /// This stack with layer `index` baked as `baked` found.
    pub fn with_baked(&self, index: usize, baked: Baked) -> Self {
        let mut layers = self.clone();
        match (layers.layer_mut(index), baked) {
            (Some(Layer::Hydrology(stage)), Baked::Hydrology(found)) => {
                stage.baked = Some(Arc::new(found));
            }
            (Some(Layer::Erosion(stage)), Baked::Erosion(found)) => {
                stage.baked = Some(Arc::new(found));
            }
            (Some(Layer::Path(stage)), Baked::Path(found)) => {
                stage.baked = Some(Arc::new(found));
            }
            _ => {}
        }
        layers
    }

    /// This stack with every bake done at once, as tools that write terrain
    /// files need it; engines bake within frame budgets.
    pub fn bake_all(&self, pyramid: &Pyramid) -> Self {
        let mut layers = self.clone();
        while let Some((index, kind)) = layers.unbaked() {
            let mut bake = TerrainBake::new(&layers, (index, kind), pyramid);
            let mut budget = u64::MAX;
            let baked = bake
                .advance(pyramid, &mut budget)
                .expect("an unlimited budget finishes the bake");
            layers = layers.with_baked(index, baked);
        }
        layers
    }
}
