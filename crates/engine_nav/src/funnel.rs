use glam::{Vec2, Vec3};

use super::{NavMesh, PolygonId, Spot, flat, turn};

#[derive(Clone, Copy, Debug)]
pub(super) struct Gate {
    left: Vec3,
    right: Vec3,
    left_wall: Option<Vec3>,
    right_wall: Option<Vec3>,
}

#[derive(Clone, Copy, Debug)]
struct Corner {
    point: Vec3,
    wall: Option<Vec3>,
}

impl NavMesh {
    pub(super) fn gates(
        &self,
        from: Spot,
        to: Spot,
        polygons: &[PolygonId],
        crossings: &[u32],
        radius: f32,
    ) -> Vec<Gate> {
        let mut gates = Vec::with_capacity(crossings.len() + 2);
        gates.push(Gate::point(from.point));
        for (index, &crossing) in crossings.iter().enumerate() {
            let portal = self.portals[crossing as usize];
            let (a, b) = portal.ends(radius);
            let wall_a = portal.walls[0].then_some(portal.a);
            let wall_b = portal.walls[1].then_some(portal.b);
            gates.push(if portal.polygons[0] == polygons[index] {
                Gate {
                    left: b,
                    right: a,
                    left_wall: wall_b,
                    right_wall: wall_a,
                }
            } else {
                Gate {
                    left: a,
                    right: b,
                    left_wall: wall_a,
                    right_wall: wall_b,
                }
            });
        }
        gates.push(Gate::point(to.point));
        gates
    }
}

impl Gate {
    fn point(point: Vec3) -> Gate {
        Gate {
            left: point,
            right: point,
            left_wall: None,
            right_wall: None,
        }
    }
}

pub(super) fn corners(gates: &[Gate], radius: f32) -> Vec<Vec3> {
    round(&string_pull(gates), radius)
}

fn push_corner(corners: &mut Vec<Corner>, point: Vec3, wall: Option<Vec3>) {
    if corners.last().map(|c| c.point) != Some(point) {
        corners.push(Corner { point, wall });
    }
}

fn string_pull(gates: &[Gate]) -> Vec<Corner> {
    let mut corners = Vec::new();
    let (Some(first), Some(last)) = (gates.first(), gates.last()) else {
        return corners;
    };
    push_corner(&mut corners, first.left, None);
    let (mut apex, mut left, mut right) = (first.left, first.left, first.left);
    let (mut left_at, mut right_at) = (0, 0);
    let mut index = 1;
    while index < gates.len() {
        let gate = gates[index];
        let a = flat(apex);
        if turn(a, flat(right), flat(gate.right)) >= 0.0 {
            if apex == right || turn(a, flat(left), flat(gate.right)) <= 0.0 {
                right = gate.right;
                right_at = index;
            } else {
                apex = left;
                push_corner(&mut corners, apex, gates[left_at].left_wall);
                (right, right_at) = (apex, left_at);
                index = left_at + 1;
                continue;
            }
        }
        if turn(a, flat(left), flat(gate.left)) <= 0.0 {
            if apex == left || turn(a, flat(right), flat(gate.left)) >= 0.0 {
                left = gate.left;
                left_at = index;
            } else {
                apex = right;
                push_corner(&mut corners, apex, gates[right_at].right_wall);
                (left, left_at) = (apex, right_at);
                index = right_at + 1;
                continue;
            }
        }
        index += 1;
    }
    push_corner(&mut corners, last.left, None);
    corners
}

#[derive(Clone, Copy, Debug)]
struct Bend {
    center: Vec2,
    point: Vec2,
    radius: f32,
    side: f32,
    height: f32,
}

type Link = Option<(Vec2, Vec2)>;

fn bends(corners: &[Corner], radius: f32) -> Vec<Bend> {
    let mut bends = Vec::with_capacity(corners.len());
    for (index, corner) in corners.iter().enumerate() {
        let point = flat(corner.point);
        let bend = match (
            corner.wall,
            index.checked_sub(1).map(|i| corners[i]),
            corners.get(index + 1),
        ) {
            (Some(_), Some(previous), Some(next)) => {
                turn(flat(previous.point), point, flat(next.point)).signum()
            }
            _ => 0.0,
        };
        let center = corner
            .wall
            .filter(|_| bend != 0.0 && radius > 0.0)
            .map(flat);
        bends.push(Bend {
            center: center.unwrap_or(point),
            point,
            radius: if center.is_some() { radius } else { 0.0 },
            side: if center.is_some() { bend } else { 0.0 },
            height: corner.point.y,
        });
    }
    bends
}

fn round(corners: &[Corner], radius: f32) -> Vec<Vec3> {
    let mut bends = bends(corners, radius);
    for _ in 0..corners.len() {
        let links: Vec<Link> = bends
            .windows(2)
            .map(|w| common_tangent(w[0], w[1]))
            .collect();
        let broken = (1..bends.len().saturating_sub(1)).find_map(|index| {
            let bend = bends[index];
            if bend.radius <= 0.0 {
                return None;
            }
            match (links[index - 1], links[index]) {
                (Some((_, enter)), Some((leave, _))) => {
                    let first = (enter - bend.center) / bend.radius;
                    let second = (leave - bend.center) / bend.radius;
                    let reversed =
                        bend.side * first.perp_dot(second) < 0.0 && first.dot(second) > 0.0;
                    reversed.then_some((index, true))
                }
                _ => Some((index, false)),
            }
        });
        match broken {
            Some((index, true)) => {
                bends.remove(index);
            }
            Some((index, false)) => {
                let bend = &mut bends[index];
                (bend.center, bend.radius, bend.side) = (bend.point, 0.0, 0.0);
            }
            None => return emit(&bends, &links),
        }
    }
    corners.iter().map(|c| c.point).collect()
}

fn emit(bends: &[Bend], links: &[Link]) -> Vec<Vec3> {
    let mut out: Vec<Vec3> = Vec::with_capacity(bends.len() * 2);
    for (index, bend) in bends.iter().enumerate() {
        let enter = index.checked_sub(1).and_then(|i| links[i]).map(|l| l.1);
        let leave = links.get(index).copied().flatten().map(|l| l.0);
        let mut points = Vec::new();
        match (enter, leave) {
            (Some(enter), Some(leave)) if bend.radius > 0.0 => {
                let first = (enter - bend.center) / bend.radius;
                let second = (leave - bend.center) / bend.radius;
                arc(
                    bend.center,
                    first,
                    second,
                    bend.radius,
                    bend.side,
                    0,
                    &mut points,
                );
            }
            _ => points.push(bend.center),
        }
        for point in points {
            let lifted = Vec3::new(point.x, bend.height, point.y);
            if out.last() != Some(&lifted) {
                out.push(lifted);
            }
        }
    }
    out
}

fn common_tangent(from: Bend, to: Bend) -> Link {
    let offset = to.center - from.center;
    let distance = offset.length();
    let shift = to.radius * to.side - from.radius * from.side;
    if distance <= shift.abs() || distance <= 0.0 {
        return None;
    }
    let toward = offset / distance;
    let along = (distance * distance - shift * shift).sqrt();
    let heading = toward * (along / distance) - toward.perp() * (shift / distance);
    let left = heading.perp();
    Some((
        from.center - left * (from.side * from.radius),
        to.center - left * (to.side * to.radius),
    ))
}

fn arc(
    wall: Vec2,
    first: Vec2,
    second: Vec2,
    radius: f32,
    side: f32,
    depth: u32,
    out: &mut Vec<Vec2>,
) {
    let sum = first + second;
    let spin = side * first.perp_dot(second);
    let quarter = first.dot(second) >= 0.0 && spin >= 0.0;
    let length = sum.length_squared();
    if (quarter || depth >= 3) && length > 0.0 {
        out.push(wall + sum * (2.0 * radius / length));
        return;
    }
    let middle = if length <= 0.0 {
        first.perp() * side
    } else if spin >= 0.0 {
        sum / length.sqrt()
    } else {
        -sum / length.sqrt()
    };
    arc(wall, first, middle, radius, side, depth + 1, out);
    arc(wall, middle, second, radius, side, depth + 1, out);
}
