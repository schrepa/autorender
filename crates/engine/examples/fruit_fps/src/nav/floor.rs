use std::collections::BTreeMap;

use glam::{Vec2, Vec3};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Carved {
    pub positions: Vec<Vec3>,
    pub polygons: Vec<[u32; 4]>,
}

fn cuts(low: f32, high: f32, inside: impl Iterator<Item = f32>) -> Vec<f32> {
    let mut found: Vec<f32> = inside.filter(|&v| v > low && v < high).collect();
    found.push(low);
    found.push(high);
    found.sort_by(f32::total_cmp);
    found.dedup();
    found
}

pub fn carve(min: Vec2, max: Vec2, height: f32, holes: &[(Vec2, Vec2)]) -> Carved {
    let holes: Vec<(Vec2, Vec2)> = holes
        .iter()
        .copied()
        .filter(|(low, high)| high.x > min.x && low.x < max.x && high.y > min.y && low.y < max.y)
        .collect();
    let xs = cuts(min.x, max.x, holes.iter().flat_map(|(l, h)| [l.x, h.x]));
    let zs = cuts(min.y, max.y, holes.iter().flat_map(|(l, h)| [l.y, h.y]));
    let open = |i: usize, j: usize| {
        let middle = Vec2::new((xs[i] + xs[i + 1]) * 0.5, (zs[j] + zs[j + 1]) * 0.5);
        !holes
            .iter()
            .any(|(l, h)| middle.x > l.x && middle.x < h.x && middle.y > l.y && middle.y < h.y)
    };
    let mut carved = Carved::default();
    let mut rising: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    for row in 0..zs.len() {
        let runs = if row + 1 < zs.len() {
            runs(xs.len() - 1, |i| open(i, row))
        } else {
            Vec::new()
        };
        let ended: Vec<(usize, usize)> = rising
            .keys()
            .copied()
            .filter(|key| !runs.contains(key))
            .collect();
        for key in ended {
            if let Some(start) = rising.remove(&key) {
                carved.push(xs[key.0], zs[start], xs[key.1], zs[row], height);
            }
        }
        for run in runs {
            rising.entry(run).or_insert(row);
        }
    }
    carved
}

fn runs(cells: usize, open: impl Fn(usize) -> bool) -> Vec<(usize, usize)> {
    let mut found = Vec::new();
    let mut index = 0;
    while index < cells {
        if open(index) {
            let start = index;
            while index < cells && open(index) {
                index += 1;
            }
            found.push((start, index));
        } else {
            index += 1;
        }
    }
    found
}

impl Carved {
    fn push(&mut self, x0: f32, z0: f32, x1: f32, z1: f32, height: f32) {
        let base = self.positions.len() as u32;
        self.positions.extend([
            Vec3::new(x0, height, z0),
            Vec3::new(x1, height, z0),
            Vec3::new(x1, height, z1),
            Vec3::new(x0, height, z1),
        ]);
        self.polygons.push([base, base + 1, base + 2, base + 3]);
    }
}
