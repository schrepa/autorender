use std::collections::BTreeMap;

use engine::prelude::*;
use engine_core::SceneContext;
use engine_core::math::max;
use glam::{Mat3, Quat, Vec3};

use super::drops::{drawn_of, ground_node};
use super::{Memories, PropShown, PuppetBrick, PuppetStats};
use crate::abilities::{AbilityBook, Delivery, DrawKind, PropPose, PropState};
use crate::play::powers::Powers;
use crate::play::puppet::{PropsLook, PuppetFile};

#[derive(Resource, Default)]
pub(super) struct Spins {
    held: BTreeMap<u32, f32>,
}

type Reads<'w, 's> = (
    (
        Res<'w, FixedTime>,
        Option<Res<'w, Powers>>,
        Res<'w, Memories>,
    ),
    Option<Res<'w, SceneContext>>,
    Query<'w, 's, &'static PuppetFile>,
    Query<'w, 's, (Entity, &'static PropShown)>,
);

type Writes<'w, 's> = (
    (ResMut<'w, PuppetStats>, ResMut<'w, Spins>),
    Spawner<'w, 's, PuppetBrick>,
    Moves<'w, 's, PuppetBrick>,
);

fn standing(normal: Vec3, facing: Vec3) -> Quat {
    let z = normal.normalize_or(Vec3::Y);
    let y = (facing - z * facing.dot(z)).normalize_or(z.any_orthonormal_vector());
    Quat::from_mat3(&Mat3::from_cols(y.cross(z), y, z)).normalize()
}

fn placed(at: Vec3, rotation: Quat, scale: Vec3) -> Transform {
    Transform {
        translation: at,
        rotation,
        scale,
    }
}

fn lift_of(book: &AbilityBook, prop: &PropState) -> f32 {
    match book.ability(prop.ability).delivery {
        Delivery::Place(place) => place.lift_m,
        Delivery::Caster | Delivery::Throw(_) => 0.0,
    }
}

fn wall_tiles(look: &PropsLook, (base, axes, half): (Vec3, Mat3, Vec3)) -> Vec<Transform> {
    let width = 2.0 * half.x;
    let count = max((width / max(look.tile_m, 0.01)).round(), 1.0);
    let rotation =
        Quat::from_mat3(&Mat3::from_cols(axes.x_axis, -axes.z_axis, axes.y_axis)).normalize();
    let scale = Vec3::new(
        width / count / max(look.tile_m, 0.01),
        1.0,
        2.0 * half.y / max(look.wall_height_m, 0.01),
    );
    (0..count as u32)
        .map(|tile| {
            let along = -half.x + (tile as f32 + 0.5) * width / count;
            placed(base + axes.x_axis * along, rotation, scale)
        })
        .collect()
}

fn falling(
    book: &AbilityBook,
    prop: &PropState,
    left_ticks: Option<u32>,
    target: Vec3,
) -> Vec<Transform> {
    let Some(drop) = book.ability(prop.ability).drop else {
        return Vec::new();
    };
    let Some(left) = left_ticks.filter(|left| *left <= drop.ticks) else {
        return Vec::new();
    };
    let share = 1.0 - left as f32 / max(drop.ticks as f32, 1.0);
    let height = drop.height_m * (1.0 - share * share);
    vec![placed(
        target + Vec3::Y * height,
        standing(Vec3::Y, Vec3::Z),
        Vec3::ONE,
    )]
}

fn pieces(
    (look, book): (&PropsLook, &AbilityBook),
    prop: &PropState,
    (angle, left_ticks): (f32, Option<u32>),
) -> Vec<Transform> {
    match prop.pose {
        PropPose::Flying {
            centre,
            velocity_per_tick,
            ..
        } => {
            let axis = velocity_per_tick
                .cross(Vec3::Y)
                .try_normalize()
                .unwrap_or(Vec3::X);
            let rotation = Quat::from_axis_angle(axis, angle) * standing(Vec3::Y, Vec3::Z);
            vec![placed(centre, rotation, Vec3::ONE)]
        }
        PropPose::Placed {
            contact,
            normal,
            facing,
        } => {
            let at = contact + normal * lift_of(book, prop);
            vec![placed(at, standing(normal, facing), Vec3::ONE)]
        }
        PropPose::Wall {
            base,
            axes,
            half_extents,
        } => wall_tiles(look, (base, axes, half_extents)),
        PropPose::Wire {
            a,
            b,
            normal_a,
            normal_b,
        } => {
            let lift = lift_of(book, prop);
            vec![
                placed(a + normal_a * lift, standing(normal_a, b - a), Vec3::ONE),
                placed(b + normal_b * lift, standing(normal_b, a - b), Vec3::ONE),
            ]
        }
        PropPose::Falling { target, .. } => falling(book, prop, left_ticks, target),
    }
}

fn spun(spins: &mut Spins, look: &PropsLook, prop: &PropState, delta: f32) -> f32 {
    let angle = spins.held.entry(prop.effect.0).or_insert(0.0);
    if let PropPose::Flying { resting: false, .. } = prop.pose {
        *angle = (*angle + look.spin_radians_per_second * delta).rem_euclid(std::f32::consts::TAU);
    }
    *angle
}

pub(super) fn show(
    ((fixed, powers, memories), scene, files, shown): Reads,
    ((mut stats, mut spins), mut spawner, mut moves): Writes,
) {
    let Some(book) = files.iter().find_map(PuppetFile::book) else {
        return;
    };
    let look = &book.table.props;
    let abilities = memories
        .abilities
        .as_ref()
        .and_then(|(_, book)| book.clone());
    let props = powers
        .as_deref()
        .map_or_else(Vec::new, |powers| powers.effects.props());
    let marks: BTreeMap<u32, u32> = powers
        .as_deref()
        .map_or_else(Vec::new, Powers::drawables)
        .into_iter()
        .filter(|drawn| drawn.kind == DrawKind::StrikeMark)
        .filter_map(|drawn| Some((drawn.effect.0, drawn.lifetime?.saturating_sub(drawn.age))))
        .collect();
    let existing: BTreeMap<(u32, u32), Entity> = shown
        .iter()
        .map(|(entity, prop)| ((prop.effect, prop.part), entity))
        .collect();
    let mut wanted = Vec::new();
    let mut skipped: Vec<String> = Vec::new();
    for prop in &props {
        let Some(abilities) = abilities.as_deref() else {
            skipped.push(String::from("(ability numbers not read)"));
            continue;
        };
        let Some(key) = abilities.ability(prop.ability).prop.as_deref() else {
            continue;
        };
        let Ok(drawn) = drawn_of(scene.as_deref(), (key, key)) else {
            skipped.push(key.to_owned());
            continue;
        };
        let angle = spun(&mut spins, look, prop, fixed.delta_seconds);
        let left = marks.get(&prop.effect.0).copied();
        for (part, at) in pieces((look, abilities), prop, (angle, left))
            .into_iter()
            .enumerate()
        {
            let id = (prop.effect.0, u32::try_from(part).unwrap_or(u32::MAX));
            wanted.push(id);
            match existing.get(&id) {
                Some(entity) => {
                    if let Some(mut transform) = moves.get_mut(*entity) {
                        transform.set_if_neq(at);
                    }
                }
                None => {
                    let (effect, part) = id;
                    spawner.spawn((at, ground_node(), drawn.clone(), PropShown { effect, part }));
                }
            }
        }
    }
    for (id, entity) in &existing {
        if !wanted.contains(id) {
            spawner.despawn(*entity);
        }
    }
    spins
        .held
        .retain(|effect, _| props.iter().any(|prop| prop.effect.0 == *effect));
    let count = u32::try_from(wanted.len()).unwrap_or(u32::MAX);
    if stats.props != count {
        stats.props = count;
    }
    let missed = u32::try_from(skipped.len()).unwrap_or(u32::MAX);
    if stats.props_skipped != missed {
        stats.props_skipped = missed;
    }
    skipped.sort();
    skipped.dedup();
    skipped.truncate(4);
    if stats.props_skipped_keys != skipped {
        stats.props_skipped_keys = skipped;
    }
}
