//! Bakes of a layer stack (decisions 066, 068 and 069): layers whose
//! results read the heights the layers below them give, over a whole
//! pyramid level where rivers and lakes form and what water wears away, or
//! along a road. A level's bake generates every tile of its level from the
//! layers below it, gathers their samples into one grid, then computes:
//! hydrology at once, erosion a round at a time. A path's bake reads the
//! ground along its line a point at a time. Terrains load no tiles until
//! their bakes are done; tools bake at once.

use std::sync::Arc;

use crate::erosion::{Erosion, ErosionLayer, erode_round, finish};
use crate::hydrology::{Hydrology, HydrologyLayer, bake};
use crate::layers::{Layer, Layers};
use crate::path::{PathLayer, PathProfile};
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
    // Paths: the points the ground is read at, the meters between them and
    // the ground read so far.
    profile: Vec<glam::Vec2>,
    step: f32,
    ground: Vec<f32>,
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
            profile,
            step,
        }
    }

    /// The baked layer's index in its stack.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Generates the level's tiles, then computes, within `budget` work
    /// units (about one a sample); what it found once done.
    pub fn advance(&mut self, pyramid: &Pyramid, budget: &mut u64) -> Option<Baked> {
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
                self.ground
                    .push(self.below.sample_height(pyramid, 0, sample));
                *budget = budget.saturating_sub(cost);
            }
            return Some(Baked::Path(PathProfile::new(
                layer,
                self.step,
                &self.ground,
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
                return self.compute(pyramid.spacing(self.level), budget);
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
            &BakeKind::Hydrology(layer) => Some(Baked::Hydrology(bake(
                &self.heights,
                self.width,
                spacing,
                &layer,
            ))),
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
