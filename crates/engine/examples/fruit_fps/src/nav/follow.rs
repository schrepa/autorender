use glam::Vec3;

use super::{NavMesh, NavNumbers, Path, Spot, flat, segment_distance};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Steer {
    Toward { point: Vec3, corner: usize },
    Arrived,
    Replan,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Follower {
    next: usize,
    anchor: Vec3,
}

impl Follower {
    pub fn new(path: &Path) -> Follower {
        Follower {
            next: path.corners.len().saturating_sub(1).min(1),
            anchor: path.corners.first().copied().unwrap_or(Vec3::ZERO),
        }
    }

    pub fn next_corner(&self) -> usize {
        self.next
    }

    pub fn steer(
        &mut self,
        mesh: &NavMesh,
        path: &Path,
        at: Spot,
        radius: f32,
        numbers: &NavNumbers,
    ) -> Steer {
        let corners = &path.corners;
        let Some(last) = corners.len().checked_sub(1) else {
            return Steer::Replan;
        };
        let here = flat(at.point);
        while self.next < last {
            let away = here.distance(flat(corners[self.next]));
            let reached = away <= mesh.weld
                || (away <= numbers.arrive_m && mesh.has_room(at, corners[self.next + 1], radius));
            if !reached {
                break;
            }
            self.anchor = corners[self.next];
            self.next += 1;
        }
        let goal = corners[last];
        if self.next >= last && here.distance(flat(goal)) <= numbers.arrive_m {
            return Steer::Arrived;
        }
        self.next = self.next.min(last);
        let target = corners[self.next];
        if segment_distance(here, flat(self.anchor), flat(target)) > numbers.replan_m {
            return Steer::Replan;
        }
        let limit = last.min(self.next + numbers.lookahead_corners);
        let mut reach = self.next;
        for (corner, &point) in corners
            .iter()
            .enumerate()
            .take(limit + 1)
            .skip(self.next + 1)
        {
            if !mesh.has_room(at, point, radius) {
                break;
            }
            reach = corner;
        }
        if reach != self.next {
            self.anchor = at.point;
            self.next = reach;
        }
        Steer::Toward {
            point: corners[self.next],
            corner: self.next,
        }
    }
}
