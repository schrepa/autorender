//! Bakes of a layer stack: layers whose
//! results read the heights the layers below them give, over a whole
//! pyramid level where rivers and lakes form and what water wears away, or
//! along a road. A level's bake generates every tile of its level from the
//! layers below it, gathers their samples into one grid, then computes:
//! hydrology at once, then its rivers' water from the finest ground along
//! them a point at a time, erosion a round at a time. A
//! path's bake reads the ground along its line a point at a time. Terrains
//! load no tiles until their bakes are done; tools bake at once.
//!
//! Each bake is a [`Bake`] in a module of its own: its layer's settings,
//! what it finds and its work in progress. A stack holds a baked layer as
//! a [`Stage`] of its bake, and [`TerrainBake`] runs whichever bake a
//! stack waits on. Every match on a layer, a bake's kind or what it found
//! names each case: a new kind of bake that one forgot fails to build
//! instead of never baking, or baking forever.
#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod erosion;
mod hydrology;
mod path;

use std::fmt::Debug;
use std::sync::Arc;

use crate::erosion::{Erosion, ErosionLayer};
use crate::hydrology::{Hydrology, HydrologyLayer};
use crate::layers::{Layer, Layers};
use crate::path::{PathLayer, PathProfile};
use crate::{Generator, Pyramid, TILE_QUADS, TILE_SAMPLES, TILE_STRIDE, TileKey};

pub use erosion::ErosionBake;
pub use hydrology::HydrologyBake;
pub use path::PathBake;

/// A layer whose result reads the heights the layers below it give,
/// baked once over the stack below it, within budgets.
pub trait Bake {
    /// Its layer's settings, as a stack holds them, and what its bake
    /// finds.
    type Layer: Clone + Debug + PartialEq;
    type Found: Clone + Debug + PartialEq;
    /// Its bake in progress.
    type Work;

    /// Starts baking `layer` over `pyramid`.
    fn start(layer: &Self::Layer, pyramid: &Pyramid) -> Self::Work;

    /// Bakes within `budget` work units (about one a sample) over `below`,
    /// the layers before it; what it found once done.
    fn advance(
        work: &mut Self::Work,
        below: &Arc<Layers>,
        pyramid: &Pyramid,
        budget: &mut u64,
    ) -> Option<Self::Found>;
}

/// A baked layer of a stack: its settings, and what its bake found once
/// done.
#[derive(Clone, Debug, PartialEq)]
pub struct Stage<B: Bake> {
    pub layer: B::Layer,
    pub baked: Option<Arc<B::Found>>,
}

/// What a layer bakes.
#[derive(Clone, Debug, PartialEq)]
pub enum BakeKind {
    Hydrology(HydrologyLayer),
    Erosion(ErosionLayer),
    Path(Arc<PathLayer>),
}

/// What a bake found.
#[derive(Clone, Debug, PartialEq)]
pub enum Baked {
    Hydrology(Hydrology),
    Erosion(Erosion),
    Path(PathProfile),
}

/// The bake a stack waits on, in progress: its layer's index, the layers
/// below it and its work.
pub struct TerrainBake {
    index: usize,
    below: Arc<Layers>,
    work: Work,
}

/// Each bake's work in progress; a level bake's holds a tile's generator.
enum Work {
    Hydrology(Box<<HydrologyBake as Bake>::Work>),
    Erosion(Box<<ErosionBake as Bake>::Work>),
    Path(<PathBake as Bake>::Work),
}

impl TerrainBake {
    /// The bake of layer `index` of `layers` (as [`Layers::unbaked`] names
    /// it) over `pyramid`.
    pub fn new(layers: &Layers, (index, kind): (usize, BakeKind), pyramid: &Pyramid) -> Self {
        let work = match &kind {
            BakeKind::Hydrology(layer) => {
                Work::Hydrology(Box::new(HydrologyBake::start(layer, pyramid)))
            }
            BakeKind::Erosion(layer) => Work::Erosion(Box::new(ErosionBake::start(layer, pyramid))),
            BakeKind::Path(layer) => Work::Path(PathBake::start(layer, pyramid)),
        };
        Self {
            index,
            below: Arc::new(layers.below(index)),
            work,
        }
    }

    /// The baked layer's index in its stack.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Bakes within `budget` work units (about one a sample); what it found
    /// once done.
    pub fn advance(&mut self, pyramid: &Pyramid, budget: &mut u64) -> Option<Baked> {
        let below = &self.below;
        match &mut self.work {
            Work::Hydrology(work) => {
                HydrologyBake::advance(work, below, pyramid, budget).map(Baked::Hydrology)
            }
            Work::Erosion(work) => {
                ErosionBake::advance(work, below, pyramid, budget).map(Baked::Erosion)
            }
            Work::Path(work) => PathBake::advance(work, below, pyramid, budget).map(Baked::Path),
        }
    }
}

/// A level bake's gathering: the layers below it generate every tile of
/// its level, one at a time, and their samples (without their borders)
/// gather into one grid `width` samples wide.
struct Gather {
    level: u32,
    pending: Vec<TileKey>,
    job: Option<(TileKey, Generator)>,
    width: usize,
    heights: Vec<f32>,
}

impl Gather {
    /// The gathering of `pyramid`'s level `level` (its coarsest, past it).
    fn new(level: u32, pyramid: &Pyramid) -> Self {
        let level = level.min(pyramid.levels() - 1);
        let [x, z] = pyramid.tiles(level);
        let width = x as usize * TILE_QUADS as usize + 1;
        let height = z as usize * TILE_QUADS as usize + 1;
        Self {
            level,
            pending: pyramid.level_tiles(level).collect(),
            job: None,
            width,
            heights: vec![0.0; width * height],
        }
    }

    /// Generates the level's tiles from `below` within `budget`; whether
    /// every one is gathered, which it finds only while budget remains.
    fn advance(&mut self, below: &Arc<Layers>, pyramid: &Pyramid, budget: &mut u64) -> bool {
        while *budget > 0 {
            let Some((key, job)) = (match &mut self.job {
                Some(job) => Some(job),
                None => match self.pending.pop() {
                    Some(key) => Some(
                        &mut *self
                            .job
                            .insert((key, Generator::new(below.clone(), pyramid, key))),
                    ),
                    None => None,
                },
            }) else {
                return true;
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
        false
    }
}

impl Layers {
    /// The first layer whose bake is not done, by index, and what it bakes:
    /// tiles wait for it.
    pub fn unbaked(&self) -> Option<(usize, BakeKind)> {
        self.layers().iter().enumerate().find_map(|(index, layer)| {
            let kind = match layer {
                Layer::Hydrology(stage) => stage
                    .baked
                    .is_none()
                    .then_some(BakeKind::Hydrology(stage.layer)),
                Layer::Erosion(stage) => stage
                    .baked
                    .is_none()
                    .then_some(BakeKind::Erosion(stage.layer)),
                Layer::Path(stage) => stage
                    .baked
                    .is_none()
                    .then(|| BakeKind::Path(stage.layer.clone())),
                Layer::Noise(_) | Layer::Heightmap(_) | Layer::Stamp(_) => None,
            };
            kind.map(|kind| (index, kind))
        })
    }

    /// This stack with layer `index` baked as `baked` found: the layer
    /// [`Self::unbaked`] named, of the bake's kind.
    #[must_use]
    pub fn with_baked(&self, index: usize, baked: Baked) -> Self {
        let mut layers = self.clone();
        let layer = layers
            .layer_mut(index)
            .expect("a bake's layer is in the stack it baked");
        match (layer, baked) {
            (Layer::Hydrology(stage), Baked::Hydrology(found)) => {
                stage.baked = Some(Arc::new(found));
            }
            (Layer::Erosion(stage), Baked::Erosion(found)) => {
                stage.baked = Some(Arc::new(found));
            }
            (Layer::Path(stage), Baked::Path(found)) => {
                stage.baked = Some(Arc::new(found));
            }
            (Layer::Hydrology(_), Baked::Erosion(_) | Baked::Path(_))
            | (Layer::Erosion(_), Baked::Hydrology(_) | Baked::Path(_))
            | (Layer::Path(_), Baked::Hydrology(_) | Baked::Erosion(_))
            | (
                Layer::Noise(_) | Layer::Heightmap(_) | Layer::Stamp(_),
                Baked::Hydrology(_) | Baked::Erosion(_) | Baked::Path(_),
            ) => unreachable!("a bake's result goes to the layer of its kind that it baked"),
        }
        layers
    }

    /// This stack with every bake done at once, as tools that write terrain
    /// files need it; engines bake within frame budgets.
    #[must_use]
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
