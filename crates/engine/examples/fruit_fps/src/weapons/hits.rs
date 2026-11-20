use engine_math::min;
use glam::Vec3;

use super::numbers::{GunNumbers, HitShapeNumbers, MeleeNumbers};
use super::recoil::{Aim, forward};

pub trait Blocker {
    fn first_hit(&self, origin: Vec3, direction: Vec3, max: f32) -> Option<f32>;
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord, Hash)]
pub struct CharacterId(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BodyPart {
    Head,
    Body,
    Leg,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Capsule {
    pub a: Vec3,
    pub b: Vec3,
    pub radius: f32,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct HitShapes {
    pub character: CharacterId,
    pub facing: Vec3,
    pub head: Sphere,
    pub torso: Capsule,
    pub legs: [Capsule; 2],
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct HitJoints {
    pub head: Vec3,
    pub spine: Vec3,
    pub hip: Vec3,
    pub feet: [Vec3; 2],
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Struck {
    Nothing,
    World {
        distance: f32,
    },
    Character {
        character: CharacterId,
        part: BodyPart,
        distance: f32,
    },
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Vitals {
    pub health: f32,
    pub armor: f32,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Wound {
    pub damage: f32,
    pub health_lost: f32,
    pub armor_lost: f32,
    pub after: Vitals,
    pub killed: bool,
    pub headshot: bool,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MeleeHit {
    pub character: CharacterId,
    pub part: BodyPart,
    pub distance: f32,
    pub from_behind: bool,
}

pub fn hit_shapes(
    numbers: &HitShapeNumbers,
    character: CharacterId,
    head_radius_m: f32,
    scale: f32,
    joints: HitJoints,
    facing: Vec3,
) -> HitShapes {
    let neck = (joints.head - joints.spine).normalize_or_zero();
    let leg = |foot: Vec3| Capsule {
        a: joints.hip,
        b: foot,
        radius: numbers.leg_radius_m * scale,
    };
    HitShapes {
        character,
        facing,
        head: Sphere {
            center: joints.head + neck * (numbers.head_lift_m * scale),
            radius: head_radius_m * scale,
        },
        torso: Capsule {
            a: joints.hip,
            b: joints.spine,
            radius: numbers.torso_radius_m * scale,
        },
        legs: joints.feet.map(leg),
    }
}

impl HitShapes {
    fn nearest(&self, origin: Vec3, direction: Vec3, max: f32) -> Option<(BodyPart, f32)> {
        let parts = [
            (
                BodyPart::Head,
                ray_sphere(origin, direction, self.head, max),
            ),
            (
                BodyPart::Body,
                ray_capsule(origin, direction, self.torso, max),
            ),
            (
                BodyPart::Leg,
                ray_capsule(origin, direction, self.legs[0], max),
            ),
            (
                BodyPart::Leg,
                ray_capsule(origin, direction, self.legs[1], max),
            ),
        ];
        let mut best: Option<(BodyPart, f32)> = None;
        for (part, distance) in parts {
            if let Some(distance) = distance
                && best.is_none_or(|(_, nearest)| distance < nearest)
            {
                best = Some((part, distance));
            }
        }
        best
    }
}

pub fn trace(
    origin: Vec3,
    direction: Vec3,
    max: f32,
    shooter: Option<CharacterId>,
    blocker: &impl Blocker,
    targets: &[HitShapes],
) -> Struck {
    let wall = blocker.first_hit(origin, direction, max);
    let reach = wall.unwrap_or(max);
    let mut best: Option<(CharacterId, BodyPart, f32)> = None;
    for shapes in targets {
        if Some(shapes.character) == shooter {
            continue;
        }
        let Some((part, distance)) = shapes.nearest(origin, direction, reach) else {
            continue;
        };
        let walled = wall.is_some_and(|wall| distance >= wall);
        if !walled && best.is_none_or(|(_, _, nearest)| distance < nearest) {
            best = Some((shapes.character, part, distance));
        }
    }
    match (best, wall) {
        (Some((character, part, distance)), _) => Struck::Character {
            character,
            part,
            distance,
        },
        (None, Some(distance)) => Struck::World { distance },
        (None, None) => Struck::Nothing,
    }
}

pub fn gun_damage(gun: &GunNumbers, part: BodyPart, distance: f32) -> f32 {
    let Some(band) = gun.damage.iter().find(|band| distance <= band.to_m) else {
        return 0.0;
    };
    match part {
        BodyPart::Head => band.head,
        BodyPart::Body => band.body,
        BodyPart::Leg => band.leg,
    }
}

pub fn wound(victim: Vitals, damage: f32, part: BodyPart, armor_absorb: f32) -> Wound {
    let armor_lost = min(damage * armor_absorb, victim.armor);
    let health_lost = min(damage - armor_lost, victim.health);
    let after = Vitals {
        health: victim.health - health_lost,
        armor: victim.armor - armor_lost,
    };
    Wound {
        damage,
        health_lost,
        armor_lost,
        after,
        killed: after.health <= 0.0,
        headshot: part == BodyPart::Head,
    }
}

pub fn melee(
    melee: &MeleeNumbers,
    aim: Aim,
    shooter: CharacterId,
    blocker: &impl Blocker,
    targets: &[HitShapes],
) -> Option<MeleeHit> {
    let direction = forward(aim.yaw_radians, aim.pitch_radians);
    let struck = trace(
        aim.eye,
        direction,
        melee.range_m,
        Some(shooter),
        blocker,
        targets,
    );
    let (character, part, distance) = match struck {
        Struck::Character {
            character,
            part,
            distance,
        } => (character, part, distance),
        Struck::World { .. } | Struck::Nothing => return None,
    };
    let victim = targets
        .iter()
        .find(|shapes| shapes.character == character)?;
    let flat = |v: Vec3| Vec3::new(v.x, 0.0, v.z);
    Some(MeleeHit {
        character,
        part,
        distance,
        from_behind: flat(victim.facing).dot(flat(direction)) > 0.0,
    })
}

pub fn melee_damage(melee: &MeleeNumbers, hit: &MeleeHit, heavy: bool) -> f32 {
    match (heavy, hit.from_behind) {
        (false, false) => melee.damage_front,
        (false, true) => melee.damage_back,
        (true, false) => melee.heavy_front,
        (true, true) => melee.heavy_back,
    }
}

fn ray_sphere(origin: Vec3, direction: Vec3, sphere: Sphere, max: f32) -> Option<f32> {
    let offset = origin - sphere.center;
    let along = offset.dot(direction);
    let outside = offset.dot(offset) - sphere.radius * sphere.radius;
    if outside > 0.0 && along > 0.0 {
        return None;
    }
    let discriminant = along * along - outside;
    if discriminant < 0.0 {
        return None;
    }
    let distance = engine_math::max(-along - discriminant.sqrt(), 0.0);
    (distance <= max).then_some(distance)
}

fn ray_capsule(origin: Vec3, direction: Vec3, capsule: Capsule, max: f32) -> Option<f32> {
    let ends = [capsule.a, capsule.b].map(|center| {
        ray_sphere(
            origin,
            direction,
            Sphere {
                center,
                radius: capsule.radius,
            },
            max,
        )
    });
    [
        ray_cylinder(origin, direction, capsule, max),
        ends[0],
        ends[1],
    ]
    .into_iter()
    .flatten()
    .reduce(min)
}

fn ray_cylinder(origin: Vec3, direction: Vec3, capsule: Capsule, max: f32) -> Option<f32> {
    let axis = capsule.b - capsule.a;
    let offset = origin - capsule.a;
    let axis_square = axis.dot(axis);
    let axis_along = axis.dot(direction);
    let axis_offset = axis.dot(offset);
    let quadratic = axis_square - axis_along * axis_along;
    let linear = axis_square * offset.dot(direction) - axis_offset * axis_along;
    let constant = axis_square * offset.dot(offset)
        - axis_offset * axis_offset
        - capsule.radius * capsule.radius * axis_square;
    if constant < 0.0 && axis_offset > 0.0 && axis_offset < axis_square {
        return Some(0.0);
    }
    if quadratic <= 0.0 {
        return None;
    }
    let discriminant = linear * linear - quadratic * constant;
    if discriminant < 0.0 {
        return None;
    }
    let distance = (-linear - discriminant.sqrt()) / quadratic;
    let height = axis_offset + distance * axis_along;
    (distance >= 0.0 && distance <= max && height > 0.0 && height < axis_square).then_some(distance)
}
