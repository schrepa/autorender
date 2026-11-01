#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

use std::sync::Arc;

use engine_heightfield::TileHeights;

use crate::layer::{LayerBake, TerrainLayer};
use crate::{Generator, Layers, Pyramid, TileKey};

pub struct TerrainBake {
    index: usize,
    below: Arc<Layers>,
    work: Box<dyn LayerBake>,
}

impl TerrainBake {
    pub fn new(layers: &Layers, index: usize, pyramid: &Pyramid) -> Option<Self> {
        let work = layers.layers().get(index)?.bake(pyramid, start_tile)?;
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

struct Heights(Generator);

impl TileHeights for Heights {
    fn advance(&mut self, budget: &mut u64) -> Option<Box<[f32]>> {
        self.0.advance(budget).map(|tile| Box::from(tile.heights()))
    }
}

pub fn start_tile(below: Arc<Layers>, pyramid: &Pyramid, key: TileKey) -> Box<dyn TileHeights> {
    Box::new(Heights(Generator::new(below, pyramid, key)))
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
