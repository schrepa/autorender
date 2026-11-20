use glam::{Vec2, Vec3};

use super::{NavMesh, Spot, flat, segments_distance};

impl NavMesh {
    pub fn locate(&self, point: Vec3, height: f32, distance: f32) -> Option<Spot> {
        let at = flat(point);
        let reach = Vec2::splat(distance);
        let mut best: Option<(f32, f32, Spot)> = None;
        self.nearby(at - reach, at + reach, |id, polygon| {
            let (near, away) = self.closest_in(polygon, at);
            if away > distance {
                return;
            }
            let lifted = polygon.lift(near);
            let rise = (lifted.y - point.y).abs();
            if rise > height {
                return;
            }
            let better = best.is_none_or(|(a, r, spot)| {
                away.total_cmp(&a)
                    .then(rise.total_cmp(&r))
                    .then(id.cmp(&spot.polygon))
                    .is_lt()
            });
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
        });
        best.map(|(_, _, spot)| spot)
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
                    None => best = engine_math::min(best, away),
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
}
