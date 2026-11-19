use std::collections::BTreeMap;

use engine::prelude::*;
use engine_core::math::max;
use engine_core::{AssetLibrary, SceneContext, Stats};
use glam::{Mat3, Quat, Vec3};

use super::drops::{drawn_of, ground_node};
use super::{InHandStats, Memories, PropInHand, PropShown, PropStats, PuppetBrick, PuppetStats};
use crate::abilities::{AbilityBook, Delivery, DrawKind, PropPose, PropState};
use crate::play::actors::Actor;
use crate::play::figures::{Held, Offset};
use crate::play::powers::Powers;
use crate::play::puppet::{Holding, InHand, PropTrim, PropsLook, PuppetFile};
use crate::play::view::Watched;

#[derive(Resource, Default)]
pub(super) struct Spins {
    held: BTreeMap<u32, f32>,
    facing: BTreeMap<u32, Vec3>,
}

type Reads<'w, 's> = (
    (
        Res<'w, FixedTime>,
        Option<Res<'w, Powers>>,
        Res<'w, Memories>,
    ),
    Option<Res<'w, SceneContext>>,
    Query<'w, 's, &'static PuppetFile>,
    Query<'w, 's, (Entity, &'static PropShown, &'static Drawn)>,
    (
        Option<Res<'w, AssetLibrary>>,
        Option<Res<'w, Watched>>,
        Query<'w, 's, &'static Actor>,
    ),
    (Option<Res<'w, Holding>>, InHands<'w, 's>),
);

type InHands<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static PropInHand,
        &'static Drawn,
        &'static Held,
        &'static Node,
    ),
>;

struct Shelf<'a> {
    scene: Option<&'a SceneContext>,
    library: Option<&'a AssetLibrary>,
    trim: &'a PropTrim,
    viewer: Option<usize>,
}

type Writes<'w, 's> = (
    (Stats<'w, PuppetStats>, ResMut<'w, Spins>),
    Spawner<'w, 's, PuppetBrick>,
    Moves<'w, 's, PuppetBrick>,
);

fn standing(normal: Vec3, facing: Vec3) -> Quat {
    let up = normal.normalize_or(Vec3::Y);
    let ahead = (facing - up * facing.dot(up)).normalize_or(up.any_orthonormal_vector());
    Quat::from_mat3(&Mat3::from_cols(up.cross(ahead), up, ahead)).normalize()
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

fn radius_of(book: &AbilityBook, prop: &PropState) -> f32 {
    match book.ability(prop.ability).delivery {
        Delivery::Throw(throw) => throw.radius_m,
        Delivery::Caster | Delivery::Place(_) => 0.0,
    }
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

fn flying(
    spins: &mut Spins,
    prop: &PropState,
    (centre, velocity): (Vec3, Vec3),
    radius: f32,
) -> Transform {
    let facing = spins.facing.entry(prop.effect.0).or_insert(Vec3::Z);
    if let Some(ahead) = Vec3::new(velocity.x, 0.0, velocity.z).try_normalize() {
        *facing = ahead;
    }
    let angle = spins.held.get(&prop.effect.0).copied().unwrap_or(0.0);
    let rotation = (standing(Vec3::Y, *facing) * Quat::from_rotation_x(angle)).normalize();
    placed(centre - rotation * (Vec3::Y * radius), rotation, Vec3::ONE)
}

fn pieces(
    (spins, book): (&mut Spins, &AbilityBook),
    prop: &PropState,
    left_ticks: Option<u32>,
) -> Vec<Transform> {
    match prop.pose {
        PropPose::Flying {
            centre,
            velocity_per_tick,
            ..
        } => vec![flying(
            spins,
            prop,
            (centre, velocity_per_tick),
            radius_of(book, prop),
        )],
        PropPose::Placed {
            contact,
            normal,
            facing,
        } => {
            let at = contact + normal * lift_of(book, prop);
            vec![placed(at, standing(normal, facing), Vec3::ONE)]
        }
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

fn spun(spins: &mut Spins, look: &PropsLook, prop: &PropState, delta: f32) {
    let angle = spins.held.entry(prop.effect.0).or_insert(0.0);
    *angle = match prop.pose {
        PropPose::Flying { resting: false, .. } => {
            (*angle + look.spin_radians_per_second * delta).rem_euclid(std::f32::consts::TAU)
        }
        PropPose::Flying { resting: true, .. }
        | PropPose::Placed { .. }
        | PropPose::Wire { .. }
        | PropPose::Falling { .. } => 0.0,
    };
}

fn bounds_of(library: Option<&AssetLibrary>, drawn: &Drawn) -> Option<(Vec3, Vec3)> {
    let bounds = library?.mesh(drawn.geometry?)?.bounds();
    Some((bounds.min(), bounds.max()))
}

fn largest((low, high): (Vec3, Vec3)) -> f32 {
    let extent = high - low;
    max(max(extent.x, extent.y), extent.z)
}

fn fitted(held: &InHand, bounds: (Vec3, Vec3)) -> Offset {
    let (low, high) = bounds;
    let scale = held.size_m / max(largest(bounds), 1.0e-4);
    let base = Vec3::new((low.x + high.x) * 0.5, low.y, (low.z + high.z) * 0.5);
    let turn = Quat::from_array(held.seat.rotation).normalize();
    Offset {
        translation: (Vec3::from_array(held.seat.translation) - turn * (base * scale)).to_array(),
        rotation: held.seat.rotation,
        scale,
    }
}

fn trimmed(mut drawn: Drawn, trim: &PropTrim, ally: Option<bool>) -> Drawn {
    let color = match ally {
        Some(true) | None => trim.ally,
        Some(false) => trim.enemy,
    };
    drawn.values.floats = vec![
        0.0,
        0.0,
        0.0,
        trim.strength,
        color[0],
        color[1],
        color[2],
        0.0,
    ];
    drawn
}

fn row(key: &str, part: u32, at: &Transform, ally: Option<bool>) -> PropStats {
    PropStats {
        key: key.to_owned(),
        part,
        position: at.translation.to_array(),
        up: (at.rotation * Vec3::Y).to_array(),
        scale: at.scale.to_array(),
        ally,
    }
}

pub(super) fn show(
    (
        (fixed, powers, memories),
        scene,
        files,
        shown,
        (library, watched, actors),
        (holding, in_hand),
    ): Reads,
    ((mut stats, mut spins), mut spawner, mut moves): Writes,
) {
    let viewer = watched
        .as_deref()
        .and_then(|watched| watched.entity)
        .and_then(|entity| actors.get(entity).ok())
        .map(|actor| actor.team.index());
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
    let existing: BTreeMap<(u32, u32), (Entity, &Drawn)> = shown
        .iter()
        .map(|(entity, prop, drawn)| ((prop.effect, prop.part), (entity, drawn)))
        .collect();
    let first = holding.as_deref().and_then(|holding| holding.held.as_ref());
    let hands: Vec<(&InHand, bool)> = first
        .map(|held| (held, true))
        .into_iter()
        .chain(
            memories
                .held
                .values()
                .filter_map(|memory| memory.hand.as_ref())
                .map(|held| (held, false)),
        )
        .filter(|(held, _)| held.holds(fixed.tick))
        .collect();
    let shelf = Shelf {
        scene: scene.as_deref(),
        library: library.as_deref(),
        trim: &look.trim,
        viewer,
    };
    let in_hands = in_hand_shown(&hands, &shelf, &in_hand, &mut spawner);
    stats.edit(|stats| stats.in_hand = in_hands);
    let mut wanted = Vec::new();
    let mut rows = Vec::new();
    let mut skipped: Vec<String> = Vec::new();
    for prop in &props {
        let Some(abilities) = abilities.as_deref() else {
            skipped.push(String::from("(ability numbers not read)"));
            continue;
        };
        let Some(key) = abilities.ability(prop.ability).prop.as_deref() else {
            continue;
        };
        let ally = viewer.map(|team| team == prop.team);
        let made = (prop.owner.0, prop.ability.slot.index(), prop.born);
        if hands.iter().any(|(held, _)| held.hides(made)) {
            continue;
        }
        let Ok(drawn) = drawn_of(scene.as_deref(), (key, key)) else {
            skipped.push(key.to_owned());
            continue;
        };
        let drawn = trimmed(drawn, &look.trim, ally);
        spun(&mut spins, look, prop, fixed.delta_seconds);
        let left = marks.get(&prop.effect.0).copied();
        for (part, at) in pieces((&mut spins, abilities), prop, left)
            .into_iter()
            .enumerate()
        {
            let id = (prop.effect.0, u32::try_from(part).unwrap_or(u32::MAX));
            wanted.push(id);
            rows.push(row(key, id.1, &at, ally));
            match existing.get(&id) {
                Some((entity, was)) => {
                    if **was != drawn {
                        spawner.insert(*entity, drawn.clone());
                    }
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
    for (id, (entity, _)) in &existing {
        if !wanted.contains(id) {
            spawner.despawn(*entity);
        }
    }
    let live = |effect: &u32| props.iter().any(|prop| prop.effect.0 == *effect);
    spins.held.retain(|effect, _| live(effect));
    spins.facing.retain(|effect, _| live(effect));
    stats.edit(|stats| {
        stats.props = rows;
        stats.props_skipped = u32::try_from(skipped.len()).unwrap_or(u32::MAX);
        skipped.sort();
        skipped.dedup();
        skipped.truncate(4);
        stats.props_skipped_keys = skipped;
    });
}

fn in_hand_shown(
    hands: &[(&InHand, bool)],
    shelf: &Shelf<'_>,
    existing: &InHands,
    spawner: &mut Spawner<PuppetBrick>,
) -> Vec<InHandStats> {
    let mut kept = Vec::new();
    let mut rows = Vec::new();
    for &(held, first_person) in hands {
        let Ok(drawn) = drawn_of(shelf.scene, (&held.prop, &held.prop)) else {
            continue;
        };
        let ally = shelf.viewer.map(|team| team == held.team);
        let drawn = trimmed(drawn, shelf.trim, ally);
        let bounds = bounds_of(shelf.library, &drawn);
        let seat = bounds.map(|bounds| fitted(held, bounds));
        let wanted = Held::on(held.holder, held.socket.clone(), seat.unwrap_or(held.seat));
        let found = existing
            .iter()
            .find(|(_, mark, ..)| mark.holder == held.holder);
        let shown = found.is_some() && seat.is_some();
        match found {
            Some((entity, _, was_drawn, was_held, node)) => {
                kept.push(entity);
                if *was_drawn != drawn {
                    spawner.insert(entity, drawn);
                }
                if *was_held != wanted {
                    spawner.insert(entity, wanted);
                }
                if node.visible != shown {
                    spawner.insert(
                        entity,
                        Node {
                            visible: shown,
                            ..*node
                        },
                    );
                }
            }
            None => {
                spawner.spawn((
                    Transform::default(),
                    Node {
                        visible: false,
                        ..ground_node()
                    },
                    drawn,
                    wanted,
                    PropInHand {
                        holder: held.holder,
                    },
                ));
            }
        }
        rows.push(InHandStats {
            owner: held.owner,
            first_person,
            prop: held.prop.clone(),
            joint: held.socket.clone(),
            size_m: held.size_m,
            extent_m: bounds.map(largest),
            scale: seat.map(|seat| seat.scale),
            shown,
            cast_tick: held.cast,
            release_tick: held.release,
        });
    }
    for (entity, ..) in existing {
        if !kept.contains(&entity) {
            spawner.despawn(entity);
        }
    }
    rows
}
