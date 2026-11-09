use super::*;

#[derive(Clone, Debug, PartialEq)]
pub struct WalkGround {
    pub(super) key: TileKey,
    pub(super) spacing: f32,
    pub(super) heights: Box<[f32]>,
    water: Option<Box<[f32]>>,
    holes: Option<Box<[u64]>>,
}

impl WalkGround {
    pub fn new(grid: &GroundGrid, water: Option<Box<[f32]>>, pyramid: &Pyramid) -> Self {
        let key = grid.key;
        Self {
            key,
            spacing: pyramid.spacing(u32::from(key.level)),
            heights: grid.heights.clone(),
            water,
            holes: grid.holes.clone(),
        }
    }

    pub fn key(&self) -> TileKey {
        self.key
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
