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
        let spacing = pyramid.spacing(u32::from(key.level));
        let mut heights = grid.heights.clone();
        let first = pyramid.first_sample(key);
        lay_bridges(&grid.bridges, first, spacing, &mut heights);
        Self {
            key,
            spacing,
            heights,
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

fn lay_bridges(bridges: &[GroundBridge], first: Sample, spacing: f32, heights: &mut [f32]) {
    let corner = position(first, spacing);
    let last = Vec2::splat((SIDE - 1) as f32);
    for bridge in bridges.iter().filter(|bridge| !bridge.tunnel) {
        let half = bridge.width * 0.5;
        for pair in bridge.deck.windows(2) {
            let (a, b) = (
                Vec2::new(pair[0].x, pair[0].z),
                Vec2::new(pair[1].x, pair[1].z),
            );
            let low = (a.min(b) - Vec2::splat(half) - corner) / spacing;
            let high = (a.max(b) + Vec2::splat(half) - corner) / spacing;
            if high.x < 0.0 || high.y < 0.0 || low.x > last.x || low.y > last.y {
                continue;
            }
            let low = low.floor().clamp(Vec2::ZERO, last);
            let high = high.ceil().clamp(Vec2::ZERO, last);
            let along = b - a;
            let run = max(along.length_squared(), 1.0e-6);
            for row in low.y as usize..=high.y as usize {
                for column in low.x as usize..=high.x as usize {
                    let sample = [first[0] + column as i64, first[1] + row as i64];
                    let point = position(sample, spacing);
                    let t = min(max((point - a).dot(along) / run, 0.0), 1.0);
                    if point.distance(a + along * t) <= half {
                        heights[row * SIDE + column] = pair[0].y + (pair[1].y - pair[0].y) * t;
                    }
                }
            }
        }
    }
}
