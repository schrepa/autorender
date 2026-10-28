use super::*;
use search::{Flow, Stage};

struct Block {
    key: TileKey,
    first: Sample,
    ground: Arc<WalkGround>,
    stands: Arc<[bool]>,
    cost: Vec<u32>,
    from: Vec<u8>,
}

pub(super) struct Corridor {
    tiles: Vec<TileKey>,
    samples: [Sample; 2],
    blocks: Vec<Block>,
    index: BTreeMap<TileKey, usize>,
    open: Buckets,
    widened: bool,
}

impl Corridor {
    pub(super) fn new(tiles: Vec<TileKey>, samples: [Sample; 2]) -> Self {
        let mut unique = Vec::with_capacity(tiles.len());
        for key in tiles {
            if !unique.contains(&key) {
                unique.push(key);
            }
        }
        Self {
            tiles: unique,
            samples,
            blocks: Vec::new(),
            index: BTreeMap::new(),
            open: Buckets::default(),
            widened: false,
        }
    }

    fn locate(&self, sample: Sample, [columns, rows]: [u32; 2]) -> Option<(usize, usize)> {
        if sample[0] < 0 || sample[1] < 0 {
            return None;
        }
        let x = (sample[0] / QUADS).min(i64::from(columns) - 1);
        let z = (sample[1] / QUADS).min(i64::from(rows) - 1);
        let (column, row) = (sample[0] - x * QUADS, sample[1] - z * QUADS);
        if column >= SIDE as i64 || row >= SIDE as i64 {
            return None;
        }
        let level = self.blocks.first()?.key.level;
        let block = *self.index.get(&TileKey::new(level, x as u16, z as u16))?;
        Some((block, row as usize * SIDE + column as usize))
    }

    fn stands(&self, sample: Sample, tiles: [u32; 2]) -> bool {
        self.locate(sample, tiles)
            .is_some_and(|(block, index)| self.blocks[block].stands[index])
    }

    fn height(&self, sample: Sample, tiles: [u32; 2]) -> Option<f32> {
        self.locate(sample, tiles)
            .map(|(block, index)| self.blocks[block].ground.heights[index])
    }

    fn id(block: usize, index: usize) -> u32 {
        (block * SIDE * SIDE + index) as u32
    }

    fn sample_of(&self, id: u32) -> (usize, usize, Sample) {
        let (block, index) = (id as usize / (SIDE * SIDE), id as usize % (SIDE * SIDE));
        let first = self.blocks[block].first;
        let sample = [
            first[0] + (index % SIDE) as i64,
            first[1] + (index / SIDE) as i64,
        ];
        (block, index, sample)
    }
}

impl WalkSearch {
    fn build(&mut self, corridor: &mut Corridor, data: &dyn WalkData) -> bool {
        let grounds: Vec<Option<Arc<WalkGround>>> = corridor
            .tiles
            .clone()
            .into_iter()
            .map(|key| self.ground(key, data))
            .collect();
        if !self.wants.is_empty() {
            return false;
        }
        corridor.blocks.clear();
        corridor.index.clear();
        for ground in grounds.into_iter().flatten() {
            let key = ground.key();
            corridor.index.insert(key, corridor.blocks.len());
            corridor.blocks.push(Block {
                key,
                first: self.pyramid.first_sample(key),
                stands: self.standing(&ground),
                ground,
                cost: vec![UNREACHED; SIDE * SIDE],
                from: vec![u8::MAX; SIDE * SIDE],
            });
        }
        let tiles = self.pyramid.tiles(self.gait.level);
        let [start, goal] = corridor.samples;
        corridor.open.restart(octile(start, goal));
        if let Some((block, index)) = corridor.locate(start, tiles) {
            corridor.blocks[block].cost[index] = 0;
            corridor
                .open
                .push(octile(start, goal), Corridor::id(block, index));
        }
        true
    }

    pub(super) fn corridor(
        &mut self,
        mut corridor: Box<Corridor>,
        budget: &mut u64,
        data: &dyn WalkData,
    ) -> Flow {
        if corridor.blocks.is_empty() && !self.build(&mut corridor, data) {
            return Flow::Stay(Stage::Corridor(corridor));
        }
        let tiles = self.pyramid.tiles(self.gait.level);
        let goal = corridor.samples[1];
        while *budget > 0 {
            let Some((estimate, id)) = corridor.open.pop() else {
                if corridor.widened {
                    return Flow::Found(None);
                }
                self.widen(&mut corridor);
                return Flow::Next(Stage::Corridor(corridor));
            };
            let (block, index, sample) = corridor.sample_of(id);
            let cost = corridor.blocks[block].cost[index];
            if u64::from(cost) + octile(sample, goal) != estimate {
                continue;
            }
            if sample == goal {
                return Flow::Next(Stage::Pull(Box::new(Pull::new(corridor, tiles))));
            }
            *budget = budget.saturating_sub(EXPAND_COST);
            self.searched += 1;
            let height = corridor.blocks[block].ground.heights[index];
            for (direction, &(dx, dz)) in NEIGHBORS.iter().enumerate() {
                let next = [sample[0] + dx, sample[1] + dz];
                let Some((to, at)) = corridor
                    .locate(next, tiles)
                    .filter(|&(to, at)| corridor.blocks[to].stands[at])
                else {
                    continue;
                };
                let diagonal = direction >= 4;
                if diagonal
                    && !(corridor.stands([sample[0] + dx, sample[1]], tiles)
                        && corridor.stands([sample[0], sample[1] + dz], tiles))
                {
                    continue;
                }
                let target = &mut corridor.blocks[to];
                let Some(step) = self
                    .stepper
                    .cost(height, target.ground.heights[at], diagonal)
                else {
                    continue;
                };
                let reached = cost.saturating_add(step);
                if reached < target.cost[at] {
                    target.cost[at] = reached;
                    target.from[at] = direction as u8;
                    corridor.open.push(
                        u64::from(reached) + octile(next, goal),
                        Corridor::id(to, at),
                    );
                }
            }
        }
        Flow::Stay(Stage::Corridor(corridor))
    }

    fn widen(&self, corridor: &mut Corridor) {
        let level = self.pyramid.tiles(self.gait.level);
        let mut tiles = corridor.tiles.clone();
        for &key in &corridor.tiles {
            for near in graph::neighbors(key, level) {
                if !tiles.contains(&near) {
                    tiles.push(near);
                }
            }
        }
        corridor.tiles = tiles;
        corridor.blocks.clear();
        corridor.widened = true;
    }
}

pub(super) struct Pull {
    corridor: Box<Corridor>,
    tiles: [u32; 2],
    cells: Vec<Sample>,
    anchors: Vec<Sample>,
    index: usize,
}

impl Pull {
    fn new(corridor: Box<Corridor>, tiles: [u32; 2]) -> Self {
        let [start, goal] = corridor.samples;
        let mut cells = vec![goal];
        let mut at = goal;
        while at != start {
            let Some((block, index)) = corridor.locate(at, tiles) else {
                break;
            };
            let Some(&(dx, dz)) = NEIGHBORS.get(usize::from(corridor.blocks[block].from[index]))
            else {
                break;
            };
            at = [at[0] - dx, at[1] - dz];
            cells.push(at);
        }
        cells.reverse();
        Self {
            corridor,
            tiles,
            anchors: vec![cells[0]],
            cells,
            index: 0,
        }
    }

    fn stands_at(&self, point: Vec2, spacing: f32) -> bool {
        let cell = (point / spacing).round();
        self.corridor
            .stands([cell.x as i64, cell.y as i64], self.tiles)
    }

    fn ground(&self, point: Vec2, spacing: f32) -> Option<f32> {
        let cell = point / spacing;
        let (x0, z0) = (cell.x.floor(), cell.y.floor());
        let (fx, fz) = (cell.x - x0, cell.y - z0);
        let (x0, z0) = (x0 as i64, z0 as i64);
        let height = |dx: i64, dz: i64| self.corridor.height([x0 + dx, z0 + dz], self.tiles);
        match [height(0, 0), height(1, 0), height(0, 1), height(1, 1)] {
            [Some(a), Some(b), Some(c), Some(d)] => {
                let low = a + (b - a) * fx;
                let up = c + (d - c) * fx;
                Some(low + (up - low) * fz)
            }
            _ => {
                let nearest = cell.round();
                self.corridor
                    .height([nearest.x as i64, nearest.y as i64], self.tiles)
            }
        }
    }
}

impl WalkSearch {
    fn straight(&self, pull: &Pull, (a, b): (usize, usize), budget: &mut u64) -> bool {
        let spacing = self.spacing;
        let (a, b) = (
            position(pull.cells[a], spacing),
            position(pull.cells[b], spacing),
        );
        let steps = max((a.distance(b) / (0.5 * spacing)).ceil(), 1.0) as usize;
        *budget = budget.saturating_sub(steps as u64);
        let step = a.distance(b) / steps as f32;
        let Some(mut last) = pull.ground(a, spacing) else {
            return false;
        };
        (1..=steps).all(|index| {
            let point = a.lerp(b, index as f32 / steps as f32);
            let Some(height) = pull.ground(point, spacing) else {
                return false;
            };
            let ok = pull.stands_at(point, spacing)
                && (height - last).abs() <= self.gait.max_slope * step + 1.0e-3;
            last = height;
            ok
        })
    }

    fn reach_from(&self, pull: &Pull, budget: &mut u64) -> usize {
        let index = pull.index;
        let last = (index + PULL_REACH).min(pull.cells.len() - 1);
        let mut seen = index + 1;
        let mut unseen = None;
        let mut stride = 2;
        while index + stride <= last {
            if self.straight(pull, (index, index + stride), budget) {
                seen = index + stride;
                stride *= 2;
            } else {
                unseen = Some(index + stride);
                break;
            }
        }
        if unseen.is_none() && seen < last {
            if self.straight(pull, (index, last), budget) {
                return last;
            }
            unseen = Some(last);
        }
        if let Some(mut high) = unseen {
            while high - seen > 1 {
                let middle = (seen + high) / 2;
                if self.straight(pull, (index, middle), budget) {
                    seen = middle;
                } else {
                    high = middle;
                }
            }
        }
        seen
    }

    pub(super) fn pull(&mut self, mut pull: Box<Pull>, budget: &mut u64) -> Flow {
        while pull.index + 1 < pull.cells.len() {
            if *budget == 0 {
                return Flow::Stay(Stage::Pull(pull));
            }
            let reach = self.reach_from(&pull, budget);
            pull.anchors.push(pull.cells[reach]);
            pull.index = reach;
        }
        let spacing = self.spacing;
        let anchors: Vec<Vec2> = pull
            .anchors
            .iter()
            .map(|&anchor| position(anchor, spacing))
            .collect();
        let mut last = pull.ground(anchors[0], spacing).unwrap_or_default();
        let mut points: Vec<Vec3> = vec![anchors[0].extend(last)];
        for pair in anchors.windows(2) {
            let steps = max((pair[0].distance(pair[1]) / FOLLOW).ceil(), 1.0) as usize;
            for step in 1..=steps {
                let point = pair[0].lerp(pair[1], step as f32 / steps as f32);
                last = pull.ground(point, spacing).unwrap_or(last);
                points.push(point.extend(last));
            }
        }
        let length = points
            .windows(2)
            .map(|pair| pair[0].distance(pair[1]))
            .sum();
        Flow::Found(Some(NavPath {
            points: points
                .iter()
                .map(|point| [point.x, point.z, point.y])
                .collect(),
            length,
            level: self.gait.level,
            searched: self.searched,
        }))
    }
}
