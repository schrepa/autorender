use engine_math::collections::HashSet;

use super::*;
use search::{Flow, Reach, Stage, tiles_holding};

const START: u32 = 0;
const GOAL: u32 = 1;
const WALLED: usize = 1024;

struct Node {
    at: Sample,
    cost: u64,
    parent: u32,
    via: Option<TileKey>,
    closed: bool,
}

struct Walled {
    seen: HashSet<Sample>,
    open: BinaryHeap<Reverse<(u64, u32)>>,
    samples: Vec<Sample>,
    next: Option<Sample>,
}

impl Walled {
    fn open(&mut self, sample: Sample, start: Sample) {
        if self.seen.insert(sample) {
            self.open
                .push(Reverse((octile(sample, start), self.samples.len() as u32)));
            self.samples.push(sample);
        }
    }
}

pub(super) struct Graph {
    reach: Reach,
    walled: Option<Walled>,
    nodes: Vec<Node>,
    index: HashMap<Sample, u32>,
    open: BinaryHeap<Reverse<(u64, u32)>>,
    next: Option<u32>,
}

impl Graph {
    pub(super) fn new(reach: Reach) -> Self {
        let node = |at: Sample, cost: u64| Node {
            at,
            cost,
            parent: START,
            via: None,
            closed: false,
        };
        let [start, goal] = reach.samples;
        let mut open = BinaryHeap::new();
        open.push(Reverse((octile(start, goal), START)));
        let walled = reach.direct.is_none().then(|| Walled {
            seen: HashSet::default(),
            open: BinaryHeap::new(),
            samples: Vec::new(),
            next: None,
        });
        Self {
            nodes: vec![node(start, 0), node(goal, u64::MAX)],
            reach,
            walled,
            index: HashMap::default(),
            open,
            next: None,
        }
    }

    fn goal(&self) -> Sample {
        self.reach.samples[1]
    }

    fn node(&mut self, at: Sample) -> u32 {
        let next = self.nodes.len() as u32;
        let id = *self.index.entry(at).or_insert(next);
        if id == next {
            self.nodes.push(Node {
                at,
                cost: u64::MAX,
                parent: START,
                via: None,
                closed: false,
            });
        }
        id
    }

    fn relax(&mut self, id: u32, cost: u64, parent: u32, via: TileKey) {
        let goal = self.goal();
        let node = &mut self.nodes[id as usize];
        if node.closed || cost >= node.cost {
            return;
        }
        node.cost = cost;
        node.parent = parent;
        node.via = Some(via);
        let estimate = if id == GOAL { 0 } else { octile(node.at, goal) };
        self.open.push(Reverse((cost + estimate, id)));
    }

    fn take(&mut self) -> Option<u32> {
        if let Some(next) = self.next.take() {
            return Some(next);
        }
        let goal = self.goal();
        while let Some(Reverse((estimate, id))) = self.open.pop() {
            let node = &self.nodes[id as usize];
            let current = if id == GOAL {
                node.cost
            } else {
                node.cost + octile(node.at, goal)
            };
            if !node.closed && estimate == current {
                return Some(id);
            }
        }
        None
    }

    fn tiles(&self) -> Vec<TileKey> {
        let mut tiles = Vec::new();
        let mut id = GOAL;
        while id != START {
            let node = &self.nodes[id as usize];
            tiles.extend(node.via);
            id = node.parent;
        }
        tiles.reverse();
        tiles
    }
}

enum Seen {
    Walled,
    Open,
    Waiting,
}

impl WalkSearch {
    pub(super) fn graph(
        &mut self,
        mut graph: Box<Graph>,
        budget: &mut u64,
        data: &dyn WalkData,
    ) -> Flow {
        let Some([start_tile, goal_tile]) = self.tiles else {
            return Flow::Found(None);
        };
        match self.walled(&mut graph, budget, data) {
            Seen::Walled => return Flow::Found(None),
            Seen::Waiting => return Flow::Stay(Stage::Graph(graph)),
            Seen::Open => {}
        }
        while *budget > 0 {
            let Some(id) = graph.take() else {
                return Flow::Found(None);
            };
            if id == GOAL {
                let mut tiles = graph.tiles();
                for sample in graph.reach.samples {
                    tiles.push(self.owner(sample));
                }
                return Flow::Next(Stage::Corridor(Box::new(Corridor::new(
                    tiles,
                    graph.reach.samples,
                ))));
            }
            let at = graph.nodes[id as usize].at;
            let holding = if id == START {
                vec![start_tile]
            } else {
                tiles_holding(&self.pyramid, self.gait.level, at)
            };
            let held: Vec<Arc<TilePortals>> = holding
                .into_iter()
                .filter_map(|key| self.portals(key, data))
                .collect();
            if !self.wants.is_empty() {
                graph.next = Some(id);
                return Flow::Stay(Stage::Graph(graph));
            }
            graph.nodes[id as usize].closed = true;
            *budget = budget.saturating_sub(NODE_COST);
            self.searched += 1;
            let cost = graph.nodes[id as usize].cost;
            if id == START {
                self.around(start_tile, data);
                self.leave_start(&mut graph, &held[0]);
                continue;
            }
            for portals in &held {
                self.around(portals.key(), data);
                let first = self.pyramid.first_sample(portals.key());
                let local = [(at[0] - first[0]) as usize, (at[1] - first[1]) as usize];
                for from in portals.at(local) {
                    for (to, portal) in portals.portals().iter().enumerate() {
                        let Some(step) = portals.cost(from, to).filter(|_| to != from) else {
                            continue;
                        };
                        let [x, z] = portal.sample();
                        let next = graph.node([first[0] + x as i64, first[1] + z as i64]);
                        graph.relax(next, cost + u64::from(step), id, portals.key());
                    }
                    if portals.key() == goal_tile
                        && let Some(&last) = graph.reach.costs[1].get(from)
                        && last != UNREACHED
                    {
                        graph.relax(GOAL, cost + u64::from(last), id, goal_tile);
                    }
                }
            }
        }
        Flow::Stay(Stage::Graph(graph))
    }

    fn walled(&mut self, graph: &mut Graph, budget: &mut u64, data: &dyn WalkData) -> Seen {
        let (Some([start_tile, goal_tile]), Some(mut walled)) = (self.tiles, graph.walled.take())
        else {
            return Seen::Open;
        };
        let start = graph.reach.samples[0];
        if walled.seen.is_empty() {
            let Some(portals) = self.portals(goal_tile, data) else {
                graph.walled = Some(walled);
                return Seen::Waiting;
            };
            let first = self.pyramid.first_sample(goal_tile);
            for (portal, &cost) in portals.portals().iter().zip(&graph.reach.costs[1]) {
                if cost != UNREACHED {
                    let [x, z] = portal.sample();
                    walled.open([first[0] + x as i64, first[1] + z as i64], start);
                }
            }
        }
        while *budget > 0 {
            if walled.seen.len() > WALLED {
                return Seen::Open;
            }
            let at = match walled.next.take() {
                Some(at) => at,
                None => match walled.open.pop() {
                    Some(Reverse((_, index))) => walled.samples[index as usize],
                    None => return Seen::Walled,
                },
            };
            let held: Vec<Arc<TilePortals>> = tiles_holding(&self.pyramid, self.gait.level, at)
                .into_iter()
                .filter_map(|key| self.portals(key, data))
                .collect();
            if !self.wants.is_empty() {
                walled.next = Some(at);
                graph.walled = Some(walled);
                return Seen::Waiting;
            }
            *budget = budget.saturating_sub(NODE_COST);
            self.searched += 1;
            for portals in &held {
                self.around(portals.key(), data);
                let first = self.pyramid.first_sample(portals.key());
                let local = [(at[0] - first[0]) as usize, (at[1] - first[1]) as usize];
                for from in portals.at(local) {
                    if portals.key() == start_tile && graph.reach.costs[0][from] != UNREACHED {
                        return Seen::Open;
                    }
                    for (to, portal) in portals.portals().iter().enumerate() {
                        if to != from && portals.cost(from, to).is_some() {
                            let [x, z] = portal.sample();
                            walled.open([first[0] + x as i64, first[1] + z as i64], start);
                        }
                    }
                }
            }
        }
        graph.walled = Some(walled);
        Seen::Waiting
    }

    fn leave_start(&self, graph: &mut Graph, portals: &TilePortals) {
        let first = self.pyramid.first_sample(portals.key());
        for (index, portal) in portals.portals().iter().enumerate() {
            let cost = graph.reach.costs[0][index];
            if cost == UNREACHED {
                continue;
            }
            let [x, z] = portal.sample();
            let node = graph.node([first[0] + x as i64, first[1] + z as i64]);
            graph.relax(node, u64::from(cost), START, portals.key());
        }
        if let Some(direct) = graph.reach.direct {
            graph.relax(GOAL, u64::from(direct), START, portals.key());
        }
    }

    fn owner(&self, sample: Sample) -> TileKey {
        let [columns, rows] = self.pyramid.tiles(self.gait.level);
        let x = (sample[0] / QUADS).min(i64::from(columns) - 1);
        let z = (sample[1] / QUADS).min(i64::from(rows) - 1);
        TileKey::new(self.gait.level as u8, x as u16, z as u16)
    }
}

pub(super) fn neighbors(key: TileKey, [columns, rows]: [u32; 2]) -> Vec<TileKey> {
    NEIGHBORS
        .iter()
        .filter_map(|&(dx, dz)| {
            let (x, z) = (i64::from(key.x) + dx, i64::from(key.z) + dz);
            (x >= 0 && z >= 0 && x < i64::from(columns) && z < i64::from(rows))
                .then(|| TileKey::new(key.level, x as u16, z as u16))
        })
        .collect()
}
