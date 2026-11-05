use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Want {
    Ground(TileKey),
    Portals(TileKey),
}

pub trait WalkData {
    fn ground(&self, key: TileKey) -> Option<Arc<WalkGround>>;
    fn portals(&self, key: TileKey) -> Option<Arc<TilePortals>>;
}

pub struct WalkSearch {
    pub(super) pyramid: Pyramid,
    pub(super) gait: Gait,
    pub(super) spacing: f32,
    pub(super) stepper: Stepper,
    ends: [Vec2; 2],
    pub(super) tiles: Option<[TileKey; 2]>,
    obstacles: BTreeMap<TileKey, Vec<GroundObstacle>>,
    clearance: f32,
    stands: BTreeMap<TileKey, Arc<[bool]>>,
    stage: Stage,
    pub(super) wants: Vec<Want>,
    ahead: Vec<TileKey>,
    listed: BTreeSet<TileKey>,
    pub(super) searched: usize,
}

pub(super) enum Stage {
    Ends,
    Reach(Box<Reach>),
    Graph(Box<Graph>),
    Corridor(Box<Corridor>),
    Pull(Box<Pull>),
    Done,
}

pub(super) enum Flow {
    Stay(Stage),
    Next(Stage),
    Found(Option<NavPath>),
}

impl WalkSearch {
    pub fn new(
        pyramid: &Pyramid,
        (from, to): (Vec2, Vec2),
        obstacles: Vec<GroundObstacle>,
        options: &NavOptions,
    ) -> Self {
        let gait = options.gait(pyramid);
        let spacing = pyramid.spacing(gait.level);
        let tiles = pyramid
            .tile_at(gait.level, from)
            .zip(pyramid.tile_at(gait.level, to))
            .map(|(start, goal)| [start, goal]);
        let ahead = tiles.map_or_else(Vec::new, |_| line_tiles(pyramid, gait.level, (from, to)));
        Self {
            pyramid: *pyramid,
            gait,
            spacing,
            stepper: Stepper::new(spacing, gait.max_slope),
            ends: [from, to],
            tiles,
            obstacles: bin(obstacles, options.radius, pyramid, gait.level),
            clearance: options.radius,
            stands: BTreeMap::new(),
            stage: Stage::Ends,
            wants: Vec::new(),
            listed: ahead.iter().copied().collect(),
            ahead,
            searched: 0,
        }
    }

    pub fn gait(&self) -> Gait {
        self.gait
    }

    pub fn searched(&self) -> usize {
        self.searched
    }

    pub fn wants(&self) -> &[Want] {
        &self.wants
    }

    pub fn ahead(&self) -> &[TileKey] {
        &self.ahead
    }

    pub fn step(&mut self, budget: &mut u64, data: &dyn WalkData) -> Option<Option<NavPath>> {
        self.wants.clear();
        loop {
            if *budget == 0 {
                return None;
            }
            let stage = std::mem::replace(&mut self.stage, Stage::Done);
            let flow = match stage {
                Stage::Ends => self.snap_ends(data),
                Stage::Reach(reach) => self.reach(reach, budget, data),
                Stage::Graph(graph) => self.graph(graph, budget, data),
                Stage::Corridor(corridor) => self.corridor(corridor, budget, data),
                Stage::Pull(pull) => self.pull(pull, budget),
                Stage::Done => Flow::Found(None),
            };
            match flow {
                Flow::Stay(stage) => {
                    self.stage = stage;
                    return None;
                }
                Flow::Next(stage) => self.stage = stage,
                Flow::Found(found) => return Some(found),
            }
        }
    }

    pub(super) fn ground(&mut self, key: TileKey, data: &dyn WalkData) -> Option<Arc<WalkGround>> {
        let ground = data.ground(key);
        if ground.is_none() && !self.wants.contains(&Want::Ground(key)) {
            self.wants.push(Want::Ground(key));
        }
        ground
    }

    pub(super) fn portals(
        &mut self,
        key: TileKey,
        data: &dyn WalkData,
    ) -> Option<Arc<TilePortals>> {
        let portals = data.portals(key);
        if portals.is_none() && !self.wants.contains(&Want::Portals(key)) {
            self.wants.push(Want::Portals(key));
        }
        portals
    }

    pub(super) fn around(&mut self, key: TileKey, data: &dyn WalkData) {
        for near in graph::neighbors(key, self.pyramid.tiles(self.gait.level)) {
            if !self.listed.contains(&near) && data.portals(near).is_none() {
                self.listed.insert(near);
                self.ahead.push(near);
            }
        }
    }

    pub(super) fn standing(&mut self, ground: &WalkGround) -> Arc<[bool]> {
        if let Some(stands) = self.stands.get(&ground.key) {
            return stands.clone();
        }
        let mut stands = ground.standing(self.gait.max_water);
        let first = self.pyramid.first_sample(ground.key);
        let corner = position(first, self.spacing);
        let last = (SIDE - 1) as f32;
        for obstacle in self.obstacles.get(&ground.key).into_iter().flatten() {
            let (center, reach) = center_and_reach(obstacle, self.clearance);
            let lowest = ((center - reach - corner) / self.spacing).floor();
            let highest = ((center + reach - corner) / self.spacing).ceil();
            if highest.x < 0.0 || highest.y < 0.0 || lowest.x > last || lowest.y > last {
                continue;
            }
            for row in max(lowest.y, 0.0) as usize..=min(highest.y, last) as usize {
                for column in max(lowest.x, 0.0) as usize..=min(highest.x, last) as usize {
                    let sample = [first[0] + column as i64, first[1] + row as i64];
                    if blocks(obstacle, position(sample, self.spacing), self.clearance) {
                        stands[row * SIDE + column] = false;
                    }
                }
            }
        }
        let stands: Arc<[bool]> = stands.into();
        self.stands.insert(ground.key, stands.clone());
        stands
    }

    fn snap_ends(&mut self, data: &dyn WalkData) -> Flow {
        let Some([start, goal]) = self.tiles else {
            return Flow::Found(None);
        };
        let grounds = [self.ground(start, data), self.ground(goal, data)];
        let held = [self.portals(start, data), self.portals(goal, data)];
        let ([Some(from), Some(to)], [Some(_), Some(_)]) = (grounds, held) else {
            return Flow::Stay(Stage::Ends);
        };
        let samples = [self.snap(&from, self.ends[0]), self.snap(&to, self.ends[1])];
        let [Some(from), Some(to)] = samples else {
            return Flow::Found(None);
        };
        Flow::Next(Stage::Reach(Box::new(Reach {
            samples: [from, to],
            turn: 0,
            flood: None,
            costs: [Vec::new(), Vec::new()],
            direct: None,
        })))
    }

    fn snap(&mut self, ground: &WalkGround, point: Vec2) -> Option<Sample> {
        let stands = self.standing(ground);
        let first = self.pyramid.first_sample(ground.key);
        let center = (point / self.spacing).round();
        let (column, row) = (center.x as i64 - first[0], center.y as i64 - first[1]);
        let mut best: Option<(f32, Sample)> = None;
        for dz in -SNAP..=SNAP {
            for dx in -SNAP..=SNAP {
                let (x, z) = (column + dx, row + dz);
                if x < 0 || z < 0 || x >= SIDE as i64 || z >= SIDE as i64 {
                    continue;
                }
                let sample = [first[0] + x, first[1] + z];
                let distance = position(sample, self.spacing).distance(point);
                if stands[z as usize * SIDE + x as usize]
                    && best.is_none_or(|(nearest, _)| distance < nearest)
                {
                    best = Some((distance, sample));
                }
            }
        }
        best.map(|(_, sample)| sample)
    }

    fn local(&self, key: TileKey, sample: Sample) -> usize {
        let first = self.pyramid.first_sample(key);
        (sample[1] - first[1]) as usize * SIDE + (sample[0] - first[0]) as usize
    }

    fn reach(&mut self, mut reach: Box<Reach>, budget: &mut u64, data: &dyn WalkData) -> Flow {
        let Some(tiles) = self.tiles else {
            return Flow::Found(None);
        };
        while reach.turn < 2 {
            let key = tiles[reach.turn];
            let (Some(ground), Some(portals)) = (self.ground(key, data), self.portals(key, data))
            else {
                return Flow::Stay(Stage::Reach(reach));
            };
            let stands = self.standing(&ground);
            let start = self.local(key, reach.samples[reach.turn]);
            let goal = self.local(key, reach.samples[1]);
            let same = reach.turn == 0 && tiles[0] == tiles[1];
            let flood = reach.flood.get_or_insert_with(|| {
                let mut targets: Vec<usize> = portals.portals().iter().map(Portal::index).collect();
                if same {
                    targets.push(goal);
                }
                let mut flood = Flood::default();
                flood.start(start, &targets);
                flood
            });
            if !flood.step(
                &ground,
                &stands,
                &self.stepper,
                (budget, &mut self.searched),
            ) {
                return Flow::Stay(Stage::Reach(reach));
            }
            reach.costs[reach.turn] = portals
                .portals()
                .iter()
                .map(|portal| flood.cost[portal.index()])
                .collect();
            if same {
                reach.direct = Some(flood.cost[goal]).filter(|&cost| cost != UNREACHED);
            }
            reach.flood = None;
            reach.turn += 1;
        }
        let reaches = |costs: &[u32]| costs.iter().any(|&cost| cost != UNREACHED);
        if reach.direct.is_none() && !(reaches(&reach.costs[0]) && reaches(&reach.costs[1])) {
            return Flow::Found(None);
        }
        Flow::Next(Stage::Graph(Box::new(Graph::new(*reach))))
    }
}

pub(super) struct Reach {
    pub(super) samples: [Sample; 2],
    turn: usize,
    flood: Option<Flood>,
    pub(super) costs: [Vec<u32>; 2],
    pub(super) direct: Option<u32>,
}

fn line_tiles(pyramid: &Pyramid, level: u32, (from, to): (Vec2, Vec2)) -> Vec<TileKey> {
    const MOST: usize = 256;
    let size = pyramid.tile_size(level);
    let [columns, rows] = pyramid.tiles(level);
    let steps = max((from.distance(to) / (0.5 * size)).ceil(), 1.0) as usize;
    let mut tiles = Vec::new();
    for step in 0..=steps {
        let Some(key) = pyramid.tile_at(level, from.lerp(to, step as f32 / steps as f32)) else {
            continue;
        };
        for (dx, dz) in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)] {
            let (x, z) = (i64::from(key.x) + dx, i64::from(key.z) + dz);
            if x < 0 || z < 0 || x >= i64::from(columns) || z >= i64::from(rows) {
                continue;
            }
            let near = TileKey::new(key.level, x as u16, z as u16);
            if tiles.len() < MOST && !tiles.contains(&near) {
                tiles.push(near);
            }
        }
    }
    tiles
}

fn bin(
    obstacles: Vec<GroundObstacle>,
    radius: f32,
    pyramid: &Pyramid,
    level: u32,
) -> BTreeMap<TileKey, Vec<GroundObstacle>> {
    let (size, spacing) = (pyramid.tile_size(level), pyramid.spacing(level));
    let [columns, rows] = pyramid.tiles(level);
    let mut bins: BTreeMap<TileKey, Vec<GroundObstacle>> = BTreeMap::new();
    for obstacle in obstacles {
        let (center, reach) = center_and_reach(&obstacle, radius);
        if !(reach.is_finite() && center.is_finite()) {
            continue;
        }
        let low = ((center - (reach + Vec2::splat(spacing))) / size).floor();
        let high = ((center + reach) / size).floor();
        let span = |low: f32, high: f32, count: u32| {
            let last = (count - 1) as f32;
            max(low, 0.0) as u16..=min(high, last) as u16
        };
        if high.x < 0.0 || high.y < 0.0 {
            continue;
        }
        for z in span(low.y, high.y, rows) {
            for x in span(low.x, high.x, columns) {
                bins.entry(TileKey::new(level as u8, x, z))
                    .or_default()
                    .push(obstacle);
            }
        }
    }
    bins
}

fn center_and_reach(obstacle: &GroundObstacle, clearance: f32) -> (Vec2, Vec2) {
    match *obstacle {
        GroundObstacle::Circle { center, radius } => (center, Vec2::splat(radius + clearance)),
        GroundObstacle::Box {
            center,
            half_extents,
            unit_axis,
        } => {
            let (along, across) = (unit_axis.x.abs(), unit_axis.y.abs());
            let reach = Vec2::new(
                along * half_extents.x + across * half_extents.y,
                across * half_extents.x + along * half_extents.y,
            );
            (center, reach + Vec2::splat(clearance))
        }
    }
}

fn blocks(obstacle: &GroundObstacle, at: Vec2, clearance: f32) -> bool {
    match *obstacle {
        GroundObstacle::Circle { center, radius } => at.distance(center) <= radius + clearance,
        GroundObstacle::Box {
            center,
            half_extents,
            unit_axis,
        } => {
            let to = at - center;
            let beyond = Vec2::new(
                max(to.dot(unit_axis).abs() - half_extents.x, 0.0),
                max(to.dot(unit_axis.perp()).abs() - half_extents.y, 0.0),
            );
            beyond.length() <= clearance
        }
    }
}

pub(super) fn tiles_holding(pyramid: &Pyramid, level: u32, sample: Sample) -> Vec<TileKey> {
    let [columns, rows] = pyramid.tiles(level);
    let along = |at: i64, count: u32| -> Vec<u16> {
        let count = i64::from(count);
        let tile = (at / QUADS).min(count - 1);
        let mut tiles = vec![tile];
        if at % QUADS == 0 && at > 0 && tile == at / QUADS {
            tiles.insert(0, tile - 1);
        }
        tiles
            .into_iter()
            .filter(|&tile| (0..count).contains(&tile))
            .map(|tile| tile as u16)
            .collect()
    };
    let (xs, zs) = (along(sample[0], columns), along(sample[1], rows));
    zs.iter()
        .flat_map(|&z| xs.iter().map(move |&x| TileKey::new(level as u8, x, z)))
        .collect()
}
