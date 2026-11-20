use engine::math;
use engine_solids::{Capsule, Solids, Work};
use glam::{Quat, Vec3};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Opening {
    pub corners: [Vec3; 4],
    pub normal: Vec3,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Probe {
    pub column_m: f32,
    pub radius_m: f32,
    pub height_m: f32,
    pub step_m: f32,
}

const SIDES: [(Vec3, Vec3); 4] = [
    (Vec3::X, Vec3::NEG_Z),
    (Vec3::NEG_X, Vec3::Z),
    (Vec3::Z, Vec3::X),
    (Vec3::NEG_Z, Vec3::NEG_X),
];

pub fn openings(
    solids: &Solids,
    (centre, rotation, half): (Vec3, Quat, Vec3),
    probe: Probe,
) -> Vec<Opening> {
    let mut found = Vec::new();
    let rise = Vec3::Y * half.y;
    for (axis, across) in SIDES {
        let normal = rotation * axis;
        let along = rotation * across;
        let span = half.dot(across.abs());
        let face = centre + normal * half.dot(axis.abs());
        let columns = math::max((2.0 * span / probe.column_m).ceil(), 1.0) as u32;
        let width = 2.0 * span / columns as f32;
        let at = |column: u32| face + along * (width * column as f32 - span);
        let mut start = None;
        for column in 0..=columns {
            let middle = at(column) + along * (width * 0.5);
            let open = column < columns && passable(solids, (middle, normal, half.y), probe);
            match (open, start) {
                (true, None) => start = Some(column),
                (false, Some(first)) => {
                    let (a, b) = (at(first), at(column));
                    found.push(Opening {
                        corners: [a - rise, b - rise, b + rise, a + rise],
                        normal,
                    });
                    start = None;
                }
                _ => {}
            }
        }
    }
    found
}

fn passable(solids: &Solids, (middle, normal, high): (Vec3, Vec3, f32), probe: Probe) -> bool {
    let clear = probe.radius_m + probe.step_m;
    let inner = middle + Vec3::Y * high - normal * clear;
    let Some(ground) = solids.ground_under(inner, 2.0 * high + probe.step_m) else {
        return false;
    };
    let capsule = Capsule {
        feet: Vec3::new(inner.x, ground.point.y + probe.step_m, inner.z),
        height: probe.height_m - probe.step_m,
        radius: probe.radius_m,
    };
    !solids.capsule_overlaps(capsule, &mut Work::default())
        && solids
            .capsule_sweep(capsule, normal * (2.0 * clear), &mut Work::default())
            .is_none()
}
