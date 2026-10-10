//! Navigation over a generated terrain: the samples of a
//! pyramid level an agent can stand on (not holes, not under water deeper
//! than it wades, not within its radius of a placed obstacle) and the
//! shortest walk between two points across them, each step between
//! neighboring samples no steeper than it climbs. Roads' bridges carry
//! walks at their decks' height over the water and ground below them.
//! A* walks the eight
//! neighbors of each sample without cutting corners; the walk is then
//! pulled taut wherever a straight stretch stays walkable, and follows the
//! ground in steps of a few meters. Walks longer than a level's sample
//! budget walk coarser levels and say which. A [`WalkSearch`] does the
//! same work in budgeted steps, so agents' walks spread
//! over frames.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;

use crate::MAX_LEVEL;
use engine_math::{max, min};
use engine_settings::settings;
use glam::Vec2;
use serde::{Deserialize, Serialize};

use crate::layers::{Layer, PathStage};
use crate::{Generator, Layers, Pyramid, TILE_QUADS, TILE_SAMPLES, TILE_STRIDE, TileKey};

settings! {
    /// How an agent walks.
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(default, deny_unknown_fields)]
    pub struct NavOptions {
        /// The finest pyramid level it walks.
        pub level: u32 = 0 => range(0, MAX_LEVEL),
        /// The steepest step it climbs: rise per meter.
        pub max_slope: f32 = 0.8 => range(0.01, 10.0),
        /// Its radius, by which obstacles widen, meters.
        pub radius: f32 = 0.5 => range(0.0, 100.0),
        /// The deepest water it wades, meters.
        pub max_water: f32 = 0.6 => range(0.0, 100.0),
        /// The meters around the walk's ends it may stray.
        pub margin: f32 = 96.0 => range(0.0, 10_000.0),
        /// The most samples a walk searches, beyond which it walks a coarser
        /// level.
        pub max_samples: u32 = 1 << 20 => at_least(1024),
    }
}

/// A placed obstacle's footprint: its terrain-local center and radius.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Obstacle {
    pub center: Vec2,
    pub radius: f32,
}

/// A walk: its points (terrain-local, following the ground), its length
/// (meters), the level it walked and the samples it searched.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NavPath {
    pub points: Vec<[f32; 3]>,
    pub length: f32,
    pub level: u32,
    pub searched: usize,
}

/// Meters between the points a walk follows the ground by.
const FOLLOW: f32 = 4.0;
/// Samples around an end searched for ground to stand on.
const SNAP: i64 = 8;
/// The work, in samples, of checking whether a sample is walkable and of
/// expanding a searched sample.
const MASK_COST: u64 = 4;
const EXPAND_COST: u64 = 16;

/// The shortest walk across `layers`' terrain from terrain-local `from` to
/// `to` among `obstacles`, as `options` walk; `None` where none reaches.
pub fn walk(
    layers: &Layers,
    pyramid: &Pyramid,
    from: Vec2,
    to: Vec2,
    obstacles: &[Obstacle],
    options: &NavOptions,
) -> Option<NavPath> {
    let mut search = WalkSearch::new(
        Arc::new(layers.clone()),
        pyramid,
        (from, to),
        obstacles.to_vec(),
        options,
    );
    loop {
        let mut unlimited = u64::MAX;
        if let Some(found) = search.step(&mut unlimited) {
            return found;
        }
    }
}

/// A walk being searched: it generates the ground's tiles
/// one at a time, marks the samples an agent stands on a band of rows at a
/// time and expands the search's open samples a batch at a time, each
/// [`Self::step`] doing about its budget's samples of work, so walks spread
/// over frames.
pub struct WalkSearch {
    layers: Arc<Layers>,
    pyramid: Pyramid,
    from: Vec2,
    to: Vec2,
    obstacles: Vec<Obstacle>,
    options: NavOptions,
    level: u32,
    first: [i64; 2],
    width: usize,
    height: usize,
    spacing: f32,
    heights: Vec<f32>,
    stage: Stage,
    searched: usize,
}

enum Stage {
    /// Generating `tiles`, the next with its generator.
    Heights {
        tiles: Vec<TileKey>,
        next: usize,
        generator: Option<Box<Generator>>,
    },
    /// Marking the samples an agent stands on, from `row` on.
    Mask {
        walkable: Vec<bool>,
        row: usize,
    },
    /// A* over the eight neighbors, never cutting a corner.
    Search(Box<Search>),
    Done,
}

struct Search {
    walkable: Vec<bool>,
    start: usize,
    goal: usize,
    cost: Vec<f32>,
    parent: Vec<u32>,
    open: BinaryHeap<Reverse<(Ordered, usize)>>,
}

impl WalkSearch {
    /// A search across `layers`' terrain from terrain-local `ends.0` to
    /// `ends.1` among `obstacles`, as `options` walk: on the finest level
    /// whose samples around the ends fit its budget.
    pub fn new(
        layers: Arc<Layers>,
        pyramid: &Pyramid,
        (from, to): (Vec2, Vec2),
        obstacles: Vec<Obstacle>,
        options: &NavOptions,
    ) -> Self {
        let low = from.min(to) - Vec2::splat(options.margin);
        let high = from.max(to) + Vec2::splat(options.margin);
        let mut level = options.level.min(pyramid.levels() - 1);
        while level + 1 < pyramid.levels() {
            let size = (high - low) / pyramid.spacing(level);
            if size.x * size.y <= options.max_samples as f32 {
                break;
            }
            level += 1;
        }
        let spacing = pyramid.spacing(level);
        let [tiles_x, tiles_z] = pyramid.tiles(level);
        let quads = i64::from(TILE_QUADS);
        let last = [i64::from(tiles_x) * quads, i64::from(tiles_z) * quads];
        let first = [
            ((low.x / spacing).floor() as i64).clamp(0, last[0]),
            ((low.y / spacing).floor() as i64).clamp(0, last[1]),
        ];
        let end = [
            ((high.x / spacing).ceil() as i64).clamp(first[0], last[0]),
            ((high.y / spacing).ceil() as i64).clamp(first[1], last[1]),
        ];
        let (width, height) = (
            (end[0] - first[0] + 1) as usize,
            (end[1] - first[1] + 1) as usize,
        );
        let tile_of = |sample: i64, tiles: u32| (sample / quads).min(i64::from(tiles) - 1);
        let mut tiles = Vec::new();
        for z in tile_of(first[1], tiles_z)..=tile_of(end[1], tiles_z) {
            for x in tile_of(first[0], tiles_x)..=tile_of(end[0], tiles_x) {
                tiles.push(TileKey {
                    level: level as u8,
                    x: x as u16,
                    z: z as u16,
                });
            }
        }
        Self {
            layers,
            pyramid: *pyramid,
            from,
            to,
            obstacles,
            options: *options,
            level,
            first,
            width,
            height,
            spacing,
            heights: vec![0.0; width * height],
            stage: Stage::Heights {
                tiles,
                next: 0,
                generator: None,
            },
            searched: 0,
        }
    }

    /// The level the search walks.
    pub fn level(&self) -> u32 {
        self.level
    }

    /// Samples searched so far.
    pub fn searched(&self) -> usize {
        self.searched
    }

    /// Advances the search by about `budget` samples of work, taking what it
    /// does from `budget`: the walk once it is found (`Some(None)` where
    /// none reaches), otherwise `None` to continue another time.
    pub fn step(&mut self, budget: &mut u64) -> Option<Option<NavPath>> {
        while *budget > 0 {
            let stage = std::mem::replace(&mut self.stage, Stage::Done);
            self.stage = match stage {
                Stage::Heights {
                    tiles,
                    next,
                    mut generator,
                } => {
                    let Some(&key) = tiles.get(next) else {
                        self.lay_bridges();
                        self.stage = Stage::Mask {
                            walkable: vec![false; self.width * self.height],
                            row: 0,
                        };
                        continue;
                    };
                    let tile = generator
                        .get_or_insert_with(|| {
                            Box::new(Generator::new(self.layers.clone(), &self.pyramid, key))
                        })
                        .advance(budget);
                    match tile {
                        Some(tile) => {
                            self.copy_tile(key, tile.heights());
                            Stage::Heights {
                                tiles,
                                next: next + 1,
                                generator: None,
                            }
                        }
                        None => Stage::Heights {
                            tiles,
                            next,
                            generator,
                        },
                    }
                }
                Stage::Mask { mut walkable, row } => {
                    let rows = (*budget / (self.width as u64 * MASK_COST)).max(1) as usize;
                    let until = (row + rows).min(self.height);
                    self.mask_rows(&mut walkable, row..until);
                    *budget =
                        budget.saturating_sub((until - row) as u64 * self.width as u64 * MASK_COST);
                    if until < self.height {
                        Stage::Mask {
                            walkable,
                            row: until,
                        }
                    } else {
                        self.mask_obstacles(&mut walkable);
                        match self.begin_search(walkable) {
                            Some(search) => Stage::Search(Box::new(search)),
                            None => return Some(None),
                        }
                    }
                }
                Stage::Search(mut search) => {
                    if self.expand(&mut search, budget) {
                        return Some(self.finish(&search));
                    }
                    Stage::Search(search)
                }
                Stage::Done => return Some(None),
            };
        }
        None
    }

    fn position(&self, sample: usize) -> Vec2 {
        Vec2::new(
            (self.first[0] + (sample % self.width) as i64) as f32,
            (self.first[1] + (sample / self.width) as i64) as f32,
        ) * self.spacing
    }

    fn corner(&self) -> Vec2 {
        Vec2::new(self.first[0] as f32, self.first[1] as f32) * self.spacing
    }

    /// The sample at terrain-local `point`, if the grid holds it.
    fn at(&self, point: Vec2) -> Option<usize> {
        let cell = ((point - self.corner()) / self.spacing).round();
        (cell.x >= 0.0
            && cell.y >= 0.0
            && (cell.x as usize) < self.width
            && (cell.y as usize) < self.height)
            .then(|| cell.y as usize * self.width + cell.x as usize)
    }

    /// Copies tile `key`'s `heights` over the grid's samples it holds.
    fn copy_tile(&mut self, key: TileKey, heights: &[f32]) {
        let quads = i64::from(TILE_QUADS);
        let (stride, samples) = (i64::from(TILE_STRIDE), i64::from(TILE_SAMPLES));
        let (tx, tz) = (i64::from(key.x), i64::from(key.z));
        let end = [
            self.first[0] + self.width as i64 - 1,
            self.first[1] + self.height as i64 - 1,
        ];
        let rows = (self.first[1] - tz * quads).max(0)..=(end[1] - tz * quads).min(samples - 1);
        let columns = (self.first[0] - tx * quads).max(0)..=(end[0] - tx * quads).min(samples - 1);
        for row in rows {
            let to = (tz * quads + row - self.first[1]) as usize * self.width;
            for column in columns.clone() {
                self.heights[to + (tx * quads + column - self.first[0]) as usize] =
                    heights[((row + 1) * stride + column + 1) as usize];
            }
        }
    }

    /// Lays roads' bridge decks over the grid's ground:
    /// samples within a deck's width read its height, so
    /// walks cross at it, above the water and ground under it. Tunnels
    /// stay below the ground they bore through.
    fn lay_bridges(&mut self) {
        let corner = self.corner();
        let (width, height, spacing) = (self.width, self.height, self.spacing);
        let last = Vec2::new((width - 1) as f32, (height - 1) as f32);
        for layer in self.layers.layers() {
            let Layer::Path(PathStage {
                baked: Some(profile),
                ..
            }) = layer
            else {
                continue;
            };
            for bridge in profile.bridges().iter().filter(|bridge| !bridge.tunnel) {
                let half = bridge.width * 0.5;
                for pair in bridge.deck.windows(2) {
                    let (a, b) = (
                        Vec2::new(pair[0].x, pair[0].z),
                        Vec2::new(pair[1].x, pair[1].z),
                    );
                    let low = ((a.min(b) - Vec2::splat(half) - corner) / spacing)
                        .floor()
                        .clamp(Vec2::ZERO, last);
                    let high = ((a.max(b) + Vec2::splat(half) - corner) / spacing)
                        .ceil()
                        .clamp(Vec2::ZERO, last);
                    let along = b - a;
                    let run = max(along.length_squared(), 1.0e-6);
                    for row in low.y as usize..=high.y as usize {
                        for column in low.x as usize..=high.x as usize {
                            let sample = row * width + column;
                            let point = self.position(sample);
                            let t = min(max((point - a).dot(along) / run, 0.0), 1.0);
                            if point.distance(a + along * t) <= half {
                                self.heights[sample] = pair[0].y + (pair[1].y - pair[0].y) * t;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Marks `rows`' samples an agent stands on: not holes, not under water
    /// deeper than it wades. Water stands as deep as its surface rises over
    /// the ground the walk reads there, not over the ground
    /// the hydrology bake read: roads, stamps and edits after the bake fill
    /// over its streams and dig below them.
    fn mask_rows(&self, walkable: &mut [bool], rows: std::ops::Range<usize>) {
        let water = self.layers.hydrology();
        let samples = rows.start * self.width..rows.end * self.width;
        for (stands, sample) in walkable[samples.clone()].iter_mut().zip(samples) {
            let point = self.position(sample);
            let ground = self.heights[sample];
            *stands = !self.layers.hole_at(point.x, point.y)
                && !water
                    .and_then(|water| water.water_at(point.x, point.y))
                    .is_some_and(|(surface, _)| surface - ground > self.options.max_water);
        }
    }

    /// Unmarks the samples within the agent's radius of an obstacle.
    fn mask_obstacles(&self, walkable: &mut [bool]) {
        let first = self.corner();
        let (width, height, spacing) = (self.width, self.height, self.spacing);
        for obstacle in &self.obstacles {
            let reach = obstacle.radius + self.options.radius;
            let lowest = ((obstacle.center - Vec2::splat(reach) - first) / spacing).floor();
            let highest = ((obstacle.center + Vec2::splat(reach) - first) / spacing).ceil();
            if highest.x < 0.0
                || highest.y < 0.0
                || lowest.x >= width as f32
                || lowest.y >= height as f32
            {
                continue;
            }
            for row in max(lowest.y, 0.0) as usize..=min(highest.y, (height - 1) as f32) as usize {
                for column in
                    max(lowest.x, 0.0) as usize..=min(highest.x, (width - 1) as f32) as usize
                {
                    let sample = row * width + column;
                    if self.position(sample).distance(obstacle.center) <= reach {
                        walkable[sample] = false;
                    }
                }
            }
        }
    }

    /// The nearest sample to stand on around `point`.
    fn snap(&self, walkable: &[bool], point: Vec2) -> Option<usize> {
        let center = self.at(point)?;
        let (column, row) = ((center % self.width) as i64, (center / self.width) as i64);
        let mut best: Option<(f32, usize)> = None;
        for dz in -SNAP..=SNAP {
            for dx in -SNAP..=SNAP {
                let (x, z) = (column + dx, row + dz);
                if x < 0 || z < 0 || x >= self.width as i64 || z >= self.height as i64 {
                    continue;
                }
                let sample = z as usize * self.width + x as usize;
                let distance = self.position(sample).distance(point);
                if walkable[sample] && best.is_none_or(|(nearest, _)| distance < nearest) {
                    best = Some((distance, sample));
                }
            }
        }
        best.map(|(_, sample)| sample)
    }

    /// The search from the ground nearest the start to that nearest the
    /// goal; `None` without ground at either end.
    fn begin_search(&self, walkable: Vec<bool>) -> Option<Search> {
        if self.width < 2 || self.height < 2 {
            return None;
        }
        let start = self.snap(&walkable, self.from)?;
        let goal = self.snap(&walkable, self.to)?;
        let samples = self.width * self.height;
        let mut cost = vec![f32::INFINITY; samples];
        cost[start] = 0.0;
        let mut open = BinaryHeap::new();
        open.push(Reverse((
            Ordered(self.position(start).distance(self.position(goal))),
            start,
        )));
        Some(Search {
            walkable,
            start,
            goal,
            cost,
            parent: vec![u32::MAX; samples],
            open,
        })
    }

    /// Expands open samples until `budget` runs out; whether the search is
    /// over (the goal reached, or nothing left open).
    fn expand(&mut self, search: &mut Search, budget: &mut u64) -> bool {
        let (width, height, spacing) = (self.width, self.height, self.spacing);
        let goal_point = self.position(search.goal);
        while *budget > 0 {
            let Some(Reverse((_, sample))) = search.open.pop() else {
                return true;
            };
            if sample == search.goal {
                return true;
            }
            *budget = budget.saturating_sub(EXPAND_COST);
            self.searched += 1;
            let (column, row) = ((sample % width) as i64, (sample / width) as i64);
            for (dx, dz) in [
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ] {
                let (x, z) = (column + dx, row + dz);
                if x < 0 || z < 0 || x >= width as i64 || z >= height as i64 {
                    continue;
                }
                let next = z as usize * width + x as usize;
                if !search.walkable[next] {
                    continue;
                }
                let diagonal = dx != 0 && dz != 0;
                if diagonal
                    && !(search.walkable[row as usize * width + x as usize]
                        && search.walkable[z as usize * width + column as usize])
                {
                    continue;
                }
                let run = if diagonal {
                    spacing * std::f32::consts::SQRT_2
                } else {
                    spacing
                };
                let rise = (self.heights[next] - self.heights[sample]).abs();
                if rise > self.options.max_slope * run {
                    continue;
                }
                let reached = search.cost[sample] + run + rise;
                if reached < search.cost[next] {
                    search.cost[next] = reached;
                    search.parent[next] = sample as u32;
                    search.open.push(Reverse((
                        Ordered(reached + self.position(next).distance(goal_point)),
                        next,
                    )));
                }
            }
        }
        false
    }

    /// The ground's height at terrain-local `point`, between its samples.
    fn ground(&self, point: Vec2) -> f32 {
        let (width, height) = (self.width, self.height);
        let cell = ((point - self.corner()) / self.spacing).clamp(
            Vec2::ZERO,
            Vec2::new((width - 1) as f32, (height - 1) as f32),
        );
        let (x0, z0) = (
            min(cell.x.floor(), (width - 2) as f32) as usize,
            min(cell.y.floor(), (height - 2) as f32) as usize,
        );
        let (fx, fz) = (cell.x - x0 as f32, cell.y - z0 as f32);
        let value = |x: usize, z: usize| self.heights[z * width + x];
        let low = value(x0, z0) + (value(x0 + 1, z0) - value(x0, z0)) * fx;
        let up = value(x0, z0 + 1) + (value(x0 + 1, z0 + 1) - value(x0, z0 + 1)) * fx;
        low + (up - low) * fz
    }

    /// Whether walking straight from `a` to `b` stays on ground the agent
    /// stands on and climbs.
    fn straight(&self, walkable: &[bool], a: Vec2, b: Vec2) -> bool {
        let steps = max((a.distance(b) / (0.5 * self.spacing)).ceil(), 1.0) as usize;
        let step = a.distance(b) / steps as f32;
        let mut last = self.ground(a);
        (1..=steps).all(|index| {
            let point = a.lerp(b, index as f32 / steps as f32);
            let height = self.ground(point);
            let ok = self.at(point).is_some_and(|sample| walkable[sample])
                && (height - last).abs() <= self.options.max_slope * step + 1.0e-3;
            last = height;
            ok
        })
    }

    /// The walk the finished `search` found: pulled taut, following the
    /// ground every few meters; `None` where the goal was not reached.
    fn finish(&mut self, search: &Search) -> Option<NavPath> {
        self.stage = Stage::Done;
        if !search.cost[search.goal].is_finite() {
            return None;
        }
        let mut cells = vec![search.goal];
        while let Some(&last) = cells.last() {
            let previous = search.parent[last];
            if previous == u32::MAX || last == search.start {
                break;
            }
            cells.push(previous as usize);
        }
        cells.reverse();
        // Pulled taut: from each anchor, straight to the farthest cell it sees.
        let mut anchors = vec![self.position(cells[0])];
        let mut index = 0;
        while index + 1 < cells.len() {
            let mut reach = index + 1;
            for candidate in (index + 2..cells.len()).rev() {
                if self.straight(
                    &search.walkable,
                    self.position(cells[index]),
                    self.position(cells[candidate]),
                ) {
                    reach = candidate;
                    break;
                }
            }
            anchors.push(self.position(cells[reach]));
            index = reach;
        }
        // Following the ground every few meters.
        let mut points = vec![anchors[0].extend(self.ground(anchors[0]))];
        for pair in anchors.windows(2) {
            let steps = max((pair[0].distance(pair[1]) / FOLLOW).ceil(), 1.0) as usize;
            for step in 1..=steps {
                let point = pair[0].lerp(pair[1], step as f32 / steps as f32);
                points.push(point.extend(self.ground(point)));
            }
        }
        let length = points
            .windows(2)
            .map(|pair| pair[0].distance(pair[1]))
            .sum();
        Some(NavPath {
            points: points
                .iter()
                .map(|point| [point.x, point.z, point.y])
                .collect(),
            length,
            level: self.level,
            searched: self.searched,
        })
    }
}

/// A float the heap orders.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Ordered(f32);

impl Eq for Ordered {}

impl PartialOrd for Ordered {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Ordered {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

crate::codec::record!(NavOptions {
    level,
    max_slope,
    radius,
    max_water,
    margin,
    max_samples,
});
crate::codec::record!(Obstacle { center, radius });
crate::codec::record!(NavPath {
    points,
    length,
    level,
    searched,
});
