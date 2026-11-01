use std::sync::Arc;

use crate::grid::{Pyramid, TILE_QUADS, TILE_SAMPLES, TILE_STRIDE, TileKey};
use crate::layers::Layers;

pub trait TileHeights: Send {
    fn advance(&mut self, budget: &mut u64) -> Option<Box<[f32]>>;
}

pub type StartTile = fn(Arc<Layers>, &Pyramid, TileKey) -> Box<dyn TileHeights>;

pub struct Gather {
    pub level: u32,
    pending: Vec<TileKey>,
    job: Option<(TileKey, Box<dyn TileHeights>)>,
    start: StartTile,
    pub width: usize,
    pub heights: Vec<f32>,
}

impl Gather {
    pub fn new(level: u32, pyramid: &Pyramid, start: StartTile) -> Self {
        let level = level.min(pyramid.levels() - 1);
        let [x, z] = pyramid.tiles(level);
        let width = x as usize * TILE_QUADS as usize + 1;
        let height = z as usize * TILE_QUADS as usize + 1;
        Self {
            level,
            pending: pyramid.level_tiles(level).collect(),
            job: None,
            start,
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
                            .insert((key, (self.start)(below.clone(), pyramid, key))),
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
                    self.heights[to..to + samples].copy_from_slice(&tile[from..from + samples]);
                }
            }
        }
        false
    }
}
