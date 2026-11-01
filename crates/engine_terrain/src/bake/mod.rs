#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

use std::sync::Arc;

use crate::layer::{LayerBake, TerrainLayer};
use crate::Layers;
use crate::{Generator, Pyramid, TILE_QUADS, TILE_SAMPLES, TILE_STRIDE, TileKey};

pub struct TerrainBake {
    index: usize,
    below: Arc<Layers>,
    work: Box<dyn LayerBake>,
}

impl TerrainBake {
    pub fn new(layers: &Layers, index: usize, pyramid: &Pyramid) -> Option<Self> {
        let work = layers.layers().get(index)?.bake(pyramid)?;
        Some(Self {
            index,
            below: Arc::new(layers.below(index)),
            work,
        })
    }

    pub fn index(&self) -> usize {
        self.index
    }

    pub fn advance(
        &mut self,
        pyramid: &Pyramid,
        budget: &mut u64,
    ) -> Option<Arc<dyn TerrainLayer>> {
        self.work.advance(&self.below, pyramid, budget)
    }
}

pub struct Gather {
    pub level: u32,
    pending: Vec<TileKey>,
    job: Option<(TileKey, Generator)>,
    pub width: usize,
    pub heights: Vec<f32>,
}

impl Gather {
    pub fn new(level: u32, pyramid: &Pyramid) -> Self {
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

    pub fn advance(&mut self, below: &Arc<Layers>, pyramid: &Pyramid, budget: &mut u64) -> bool {
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

#[must_use]
pub fn bake_all(below: &Layers, pyramid: &Pyramid) -> Layers {
    let mut layers = below.clone();
    while let Some(index) = layers.unbaked() {
        let mut bake = TerrainBake::new(&layers, index, pyramid).expect("unbaked_layer_bakes");
        let mut budget = u64::MAX;
        let baked = bake
            .advance(pyramid, &mut budget)
            .expect("unlimited_budget_finishes");
        layers = layers.with_baked(index, baked);
    }
    layers
}
