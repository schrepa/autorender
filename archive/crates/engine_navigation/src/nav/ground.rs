use super::*;

#[derive(Clone, Debug, PartialEq)]
pub struct WalkGround {
    pub(super) key: TileKey,
    pub(super) spacing: f32,
    pub(super) heights: Box<[f32]>,
    water: Option<Box<[f32]>>,
    holes: Option<Box<[u64]>>,
}

record!(WalkGround {
    key,
    spacing,
    heights,
    water,
    holes,
});

pub const WALK_SAMPLES: usize = SIDE * SIDE;
pub const HOLE_WORDS: usize = WALK_SAMPLES.div_ceil(64);

impl WalkGround {
    pub fn new(
        (key, spacing): (TileKey, f32),
        heights: Box<[f32]>,
        holes: Option<Box<[u64]>>,
        water: Option<Box<[f32]>>,
    ) -> Self {
        Self {
            key,
            spacing,
            heights,
            water,
            holes,
        }
    }

    pub fn key(&self) -> TileKey {
        self.key
    }

    pub fn fits(&self) -> bool {
        self.heights.len() == WALK_SAMPLES
            && self
                .water
                .as_ref()
                .is_none_or(|water| water.len() == WALK_SAMPLES)
            && self
                .holes
                .as_ref()
                .is_none_or(|holes| holes.len() == HOLE_WORDS)
    }

    pub(super) fn stands(&self, index: usize, max_water: f32) -> bool {
        let hole = self
            .holes
            .as_ref()
            .is_some_and(|bits| bits[index / 64] & (1 << (index % 64)) != 0);
        let wades = self.water.as_ref().is_none_or(|water| {
            let depth = water[index] - self.heights[index];
            depth.is_nan() || depth <= max_water
        });
        !hole && wades
    }

    pub(super) fn standing(&self, max_water: f32) -> Vec<bool> {
        (0..SIDE * SIDE)
            .map(|index| self.stands(index, max_water))
            .collect()
    }
}
