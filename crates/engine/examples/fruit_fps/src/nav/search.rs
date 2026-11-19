use std::cmp::Ordering;
use std::collections::BinaryHeap;

use glam::Vec3;

use super::{NavMesh, PolygonId, Spot, flat};

#[derive(Clone, Debug, PartialEq)]
pub struct Costs {
    factors: Vec<f32>,
}

impl Costs {
    pub fn new(mesh: &NavMesh) -> Costs {
        Costs {
            factors: vec![1.0; mesh.polygons.len()],
        }
    }

    pub fn set(&mut self, polygon: PolygonId, factor: f32) {
        if let Some(slot) = self.factors.get_mut(polygon as usize) {
            *slot = factor;
        }
    }

    pub fn get(&self, polygon: PolygonId) -> f32 {
        self.factors
            .get(polygon as usize)
            .copied()
            .unwrap_or(f32::INFINITY)
    }

    pub fn scale_disc(&mut self, mesh: &NavMesh, center: Vec3, radius: f32, factor: f32) {
        let middle = flat(center);
        let reach = glam::Vec2::splat(radius);
        for id in mesh.grid.gather(middle - reach, middle + reach) {
            let polygon = mesh.polygon(id);
            let level = polygon.low <= center.y + radius && polygon.high >= center.y - radius;
            if level && mesh.closest_in(polygon, middle).1 <= radius {
                self.set(id, self.get(id) * factor);
            }
        }
    }

    fn lowest(&self) -> f32 {
        self.factors
            .iter()
            .copied()
            .filter(|f| f.is_finite())
            .fold(f32::INFINITY, super::lesser)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Path {
    pub corners: Vec<Vec3>,
    pub expanded: u32,
}

struct Route {
    polygons: Vec<PolygonId>,
    crossings: Vec<u32>,
    expanded: u32,
}

#[derive(Clone, Copy)]
struct Entry {
    estimate: f32,
    walked: f32,
    node: u32,
}

impl PartialEq for Entry {
    fn eq(&self, other: &Entry) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Entry {}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Entry) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Entry {
    fn cmp(&self, other: &Entry) -> Ordering {
        other
            .estimate
            .total_cmp(&self.estimate)
            .then(other.node.cmp(&self.node))
    }
}

const START: u32 = u32::MAX;
const STEPS: u32 = 4;

struct Frontier<'a> {
    mesh: &'a NavMesh,
    costs: &'a Costs,
    start: Vec3,
    goal: Spot,
    radius: f32,
    lowest: f32,
    best: Vec<f32>,
    parent: Vec<(u32, PolygonId)>,
    heap: BinaryHeap<Entry>,
}

impl Frontier<'_> {
    fn goal_node(&self) -> u32 {
        self.mesh.portals.len() as u32 * STEPS
    }

    fn position(&self, node: u32) -> Vec3 {
        self.mesh
            .portals
            .get((node / STEPS) as usize)
            .map_or(self.goal.point, |portal| match node % STEPS {
                3 => portal.toward_line(self.start, self.goal.point, self.radius),
                step => portal.step(step, self.radius),
            })
    }

    fn relax(&mut self, node: u32, walked: f32, from: u32, via: PolygonId) {
        if walked < self.best[node as usize] {
            self.best[node as usize] = walked;
            self.parent[node as usize] = (from, via);
            let estimate = walked + self.lowest * self.position(node).distance(self.goal.point);
            self.heap.push(Entry {
                estimate,
                walked,
                node,
            });
        }
    }

    fn spread(&mut self, from: u32, at: Vec3, walked: f32, polygon: PolygonId, factor: f32) {
        if polygon == self.goal.polygon {
            let goal = self.goal_node();
            self.relax(
                goal,
                walked + factor * at.distance(self.goal.point),
                from,
                polygon,
            );
        }
        let links: Vec<u32> = self
            .mesh
            .polygon(polygon)
            .links
            .iter()
            .flatten()
            .copied()
            .collect();
        for portal in links {
            let crossing = self.mesh.portals[portal as usize];
            if (from != START && portal == from / STEPS) || !crossing.passable(self.radius) {
                continue;
            }
            for step in 0..STEPS {
                let node = portal * STEPS + step;
                let walked = walked + factor * at.distance(self.position(node));
                self.relax(node, walked, from, polygon);
            }
        }
    }
}

impl NavMesh {
    pub fn path(&self, from: Spot, to: Spot, radius: f32, costs: &Costs) -> Option<Path> {
        let route = self.route(from, to, radius, costs)?;
        let gates = self.gates(from, to, &route.polygons, &route.crossings, radius);
        Some(Path {
            corners: super::funnel::corners(&gates, radius),
            expanded: route.expanded,
        })
    }

    fn factor(&self, costs: &Costs, polygon: PolygonId, start: PolygonId, lowest: f32) -> f32 {
        let factor = costs.get(polygon);
        if polygon == start && !factor.is_finite() {
            lowest
        } else {
            factor
        }
    }

    fn route(&self, from: Spot, to: Spot, radius: f32, costs: &Costs) -> Option<Route> {
        let lowest = costs.lowest();
        if self.island(from.polygon) != self.island(to.polygon) || !lowest.is_finite() {
            return None;
        }
        if from.polygon == to.polygon {
            return Some(Route {
                polygons: vec![from.polygon],
                crossings: Vec::new(),
                expanded: 0,
            });
        }
        let start_factor = self.factor(costs, from.polygon, from.polygon, lowest);
        let nodes = self.portals.len() * STEPS as usize + 1;
        let mut frontier = Frontier {
            mesh: self,
            costs,
            start: from.point,
            goal: to,
            radius,
            lowest,
            best: vec![f32::INFINITY; nodes],
            parent: vec![(START, from.polygon); nodes],
            heap: BinaryHeap::new(),
        };
        frontier.spread(START, from.point, 0.0, from.polygon, start_factor);
        let goal = frontier.goal_node();
        let mut expanded = 0;
        while let Some(entry) = frontier.heap.pop() {
            if entry.walked > frontier.best[entry.node as usize] {
                continue;
            }
            if entry.node == goal {
                return Some(self.trace_back(&frontier, expanded));
            }
            expanded += 1;
            let portal = self.portals[(entry.node / STEPS) as usize];
            let at = frontier.position(entry.node);
            for side in portal.polygons {
                let factor = self.factor(frontier.costs, side, from.polygon, lowest);
                if factor.is_finite() {
                    frontier.spread(entry.node, at, entry.walked, side, factor);
                }
            }
        }
        None
    }

    fn trace_back(&self, frontier: &Frontier<'_>, expanded: u32) -> Route {
        let mut polygons = Vec::new();
        let mut crossings = Vec::new();
        let mut node = frontier.goal_node();
        loop {
            let (previous, via) = frontier.parent[node as usize];
            polygons.push(via);
            if previous == START {
                break;
            }
            crossings.push(previous / STEPS);
            node = previous;
        }
        polygons.reverse();
        crossings.reverse();
        let mut route = Route {
            polygons: Vec::with_capacity(polygons.len()),
            crossings: Vec::with_capacity(crossings.len()),
            expanded,
        };
        route.polygons.push(polygons[0]);
        for (index, &portal) in crossings.iter().enumerate() {
            let next = polygons[index + 1];
            if route.polygons.last() != Some(&next) {
                route.polygons.push(next);
                route.crossings.push(portal);
            }
        }
        route
    }
}
