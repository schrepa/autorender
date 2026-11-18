use glam::{Vec2, Vec3};

use super::{NavMesh, PolygonId, Spot, flat, lesser, segments_distance};

impl NavMesh {
    pub fn locate(&self, point: Vec3, height: f32, distance: f32) -> Option<Spot> {
        let at = flat(point);
        let reach = Vec2::splat(distance);
        let mut best: Option<(f32, f32, Spot)> = None;
        for id in self.grid.gather(at - reach, at + reach) {
            let polygon = self.polygon(id);
            let (near, away) = self.closest_in(polygon, at);
            if away > distance {
                continue;
            }
            let lifted = polygon.lift(near);
            let rise = (lifted.y - point.y).abs();
            if rise > height {
                continue;
            }
            let better =
                best.is_none_or(|(a, r, _)| away.total_cmp(&a).then(rise.total_cmp(&r)).is_lt());
            if better {
                best = Some((
                    away,
                    rise,
                    Spot {
                        polygon: id,
                        point: lifted,
                    },
                ));
            }
        }
        best.map(|(_, _, spot)| spot)
    }

    pub fn walk_to(&self, from: Spot, to: Vec3) -> Result<Spot, Vec3> {
        self.trace(from, to, &mut Vec::new())
    }

    pub fn raycast(&self, from: Spot, to: Vec3) -> Option<Vec3> {
        self.walk_to(from, to).err()
    }

    fn trace(&self, from: Spot, to: Vec3, visited: &mut Vec<PolygonId>) -> Result<Spot, Vec3> {
        let start = flat(from.point);
        let along = flat(to) - start;
        let mut current = from.polygon;
        for _ in 0..=self.polygons.len() {
            visited.push(current);
            let polygon = self.polygon(current);
            let mut exit = f32::INFINITY;
            let mut leaving = Vec::with_capacity(polygon.corners.len());
            for edge in 0..polygon.corners.len() {
                let (a, b) = self.edge(polygon, edge);
                let side = b - a;
                let outward = Vec2::new(side.y, -side.x);
                let speed = outward.dot(along);
                if speed > 0.0 {
                    leaving.push(edge);
                    exit = lesser(exit, outward.dot(a - start) / speed);
                }
            }
            if exit >= 1.0 {
                return Ok(Spot {
                    polygon: current,
                    point: polygon.lift(flat(to)),
                });
            }
            let leave = start + along * exit;
            let Some(edge) = self.edge_holding(polygon, &leaving, leave) else {
                return Err(polygon.lift(leave));
            };
            match polygon.links[edge] {
                Some(portal) => current = self.portals[portal as usize].other(current),
                None => return Err(polygon.lift(leave)),
            }
        }
        Err(from.point)
    }

    fn edge_holding(
        &self,
        polygon: &super::Polygon,
        edges: &[usize],
        point: Vec2,
    ) -> Option<usize> {
        edges.iter().copied().min_by(|&x, &y| {
            let (a, b) = self.edge(polygon, x);
            let (p, q) = self.edge(polygon, y);
            let dx = segments_distance(point, point, a, b);
            let dy = segments_distance(point, point, p, q);
            dx.total_cmp(&dy)
                .then(polygon.links[y].is_some().cmp(&polygon.links[x].is_some()))
                .then(x.cmp(&y))
        })
    }

    pub fn has_room(&self, from: Spot, to: Vec3, radius: f32) -> bool {
        let mut near = Vec::new();
        if self.trace(from, to, &mut near).is_err() {
            return false;
        }
        near.sort_unstable();
        near.dedup();
        let a = flat(from.point);
        let b = flat(to);
        let mut queue = near.clone();
        while let Some(id) = queue.pop() {
            let polygon = self.polygon(id);
            for edge in 0..polygon.corners.len() {
                let (p, q) = self.edge(polygon, edge);
                match polygon.links[edge] {
                    None => {
                        if segments_distance(a, b, p, q) < radius - self.weld {
                            return false;
                        }
                    }
                    Some(portal) => {
                        let other = self.portals[portal as usize].other(id);
                        let close = segments_distance(a, b, p, q) < radius;
                        if close && near.binary_search(&other).is_err() {
                            let at = near.partition_point(|&n| n < other);
                            near.insert(at, other);
                            queue.push(other);
                        }
                    }
                }
            }
        }
        true
    }

    pub fn distance_to_edge(&self, at: Spot) -> f32 {
        let point = flat(at.point);
        let mut best = f32::INFINITY;
        let mut seen = vec![at.polygon];
        let mut queue = vec![at.polygon];
        while let Some(id) = queue.pop() {
            let polygon = self.polygon(id);
            for edge in 0..polygon.corners.len() {
                let (p, q) = self.edge(polygon, edge);
                let away = segments_distance(point, point, p, q);
                match polygon.links[edge] {
                    None => best = lesser(best, away),
                    Some(portal) => {
                        let other = self.portals[portal as usize].other(id);
                        if away < best && seen.binary_search(&other).is_err() {
                            let index = seen.partition_point(|&n| n < other);
                            seen.insert(index, other);
                            queue.push(other);
                        }
                    }
                }
            }
        }
        best
    }

    pub fn random_point(&self, seed: u64, filter: impl Fn(PolygonId) -> bool) -> Option<Spot> {
        let mut state = seed;
        let chosen: Vec<PolygonId> = (0..self.polygons.len() as PolygonId)
            .filter(|&id| filter(id))
            .collect();
        let areas: Vec<f32> = chosen.iter().map(|&id| self.polygon(id).area).collect();
        let polygon = *chosen.get(pick(&areas, unit(&mut state))?)?;
        let points = self.polygon_points(polygon);
        let fan: Vec<f32> = (1..points.len() - 1)
            .map(|i| {
                let (a, b, c) = (flat(points[0]), flat(points[i]), flat(points[i + 1]));
                (b - a).perp_dot(c - a).abs() * 0.5
            })
            .collect();
        let triangle = pick(&fan, unit(&mut state))? + 1;
        let (r1, r2) = (unit(&mut state), unit(&mut state));
        let s = r1.sqrt();
        let (a, b, c) = (points[0], points[triangle], points[triangle + 1]);
        let point = a * (1.0 - s) + b * (s * (1.0 - r2)) + c * (s * r2);
        Some(Spot { polygon, point })
    }
}

fn pick(weights: &[f32], unit: f32) -> Option<usize> {
    let total: f32 = weights.iter().sum();
    if weights.is_empty() || total <= 0.0 {
        return None;
    }
    let mut left = unit * total;
    for (index, &weight) in weights.iter().enumerate() {
        if left < weight {
            return Some(index);
        }
        left -= weight;
    }
    weights.iter().rposition(|&w| w > 0.0)
}

fn unit(state: &mut u64) -> f32 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    (z >> 40) as f32 / (1u64 << 24) as f32
}
