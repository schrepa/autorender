use std::sync::Arc;

use crate::grid::{Pyramid, TILE_QUADS, TILE_SAMPLES, TILE_STRIDE, TileKey};
use crate::layers::Layers;

pub type TileHeights = fn(Arc<Layers>, &Pyramid, TileKey) -> Box<[f32]>;

pub struct Gather {
    pub level: u32,
    pub width: usize,
    pub heights: Vec<f32>,
}

impl Gather {
    pub fn new(level: u32, below: &Arc<Layers>, pyramid: &Pyramid, tiles: TileHeights) -> Self {
        let level = level.min(pyramid.levels() - 1);
        let [x, z] = pyramid.tiles(level);
        let width = x as usize * TILE_QUADS as usize + 1;
        let height = z as usize * TILE_QUADS as usize + 1;
        let mut heights = vec![0.0; width * height];
        let keys: Vec<TileKey> = pyramid.level_tiles(level).collect();
        let (stride, samples) = (TILE_STRIDE as usize, TILE_SAMPLES as usize);
        let quads = TILE_QUADS as usize;
        for key in keys.into_iter().rev() {
            let tile = tiles(below.clone(), pyramid, key);
            let first = [usize::from(key.x) * quads, usize::from(key.z) * quads];
            for row in 0..samples {
                let from = (row + 1) * stride + 1;
                let to = (first[1] + row) * width + first[0];
                heights[to..to + samples].copy_from_slice(&tile[from..from + samples]);
            }
        }
        Self {
            level,
            width,
            heights,
        }
    }
}
