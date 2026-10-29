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

pub trait Bake {
    type Layer: Clone + Debug + PartialEq;
    type Found: Clone + Debug + PartialEq;
    type Work;

    fn start(layer: &Self::Layer, pyramid: &Pyramid) -> Self::Work;

    fn advance(
        work: &mut Self::Work,
        below: &Arc<Layers>,
        pyramid: &Pyramid,
        budget: &mut u64,
    ) -> Option<Self::Found>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct Stage<B: Bake> {
    pub layer: B::Layer,
    pub baked: Option<Arc<B::Found>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum BakeKind {
    Hydrology(HydrologyLayer),
    Erosion(ErosionLayer),
    Path(Arc<PathLayer>),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Baked {
    Hydrology(Hydrology),
    Erosion(Erosion),
    Path(PathProfile),
}

pub struct TerrainBake {
    index: usize,
    below: Arc<Layers>,
    work: Work,
}

enum Work {
    Hydrology(Box<<HydrologyBake as Bake>::Work>),
    Erosion(Box<<ErosionBake as Bake>::Work>),
    Path(<PathBake as Bake>::Work),
}

impl TerrainBake {
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

    pub fn index(&self) -> usize {
        self.index
    }

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

struct Gather {
    level: u32,
    pending: Vec<TileKey>,
    job: Option<(TileKey, Generator)>,
    width: usize,
    heights: Vec<f32>,
}

impl Gather {
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
                Layer::Noise(_)
                | Layer::Heightmap(_)
                | Layer::Stamp(_)
                | Layer::Rivers(_)
                | Layer::Steps(_)
                | Layer::Hummocks(_) => None,
            };
            kind.map(|kind| (index, kind))
        })
    }

    #[must_use]
    pub fn with_baked(&self, index: usize, baked: Baked) -> Self {
        let mut layers = self.clone();
        let layer = layers.layer_mut(index).expect("bake_layer_in_stack");
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
                Layer::Noise(_)
                | Layer::Heightmap(_)
                | Layer::Stamp(_)
                | Layer::Rivers(_)
                | Layer::Steps(_)
                | Layer::Hummocks(_),
                Baked::Hydrology(_) | Baked::Erosion(_) | Baked::Path(_),
            ) => unreachable!("bake_matches_layer_kind"),
        }
        layers
    }

    #[must_use]
    pub fn bake_all(&self, pyramid: &Pyramid) -> Self {
        let mut layers = self.clone();
        while let Some((index, kind)) = layers.unbaked() {
            let mut bake = TerrainBake::new(&layers, (index, kind), pyramid);
            let mut budget = u64::MAX;
            let baked = bake
                .advance(pyramid, &mut budget)
                .expect("unlimited_budget_finishes");
            layers = layers.with_baked(index, baked);
        }
        layers
    }
}
