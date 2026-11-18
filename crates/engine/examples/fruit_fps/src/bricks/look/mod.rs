engine::brick!(look);

use engine::glam::{Quat, Vec3};
use engine::math;
use engine::prelude::*;

use crate::names::{FAR, NEAR, PITCH_LIMIT_DEGREES};
use crate::play::actors::{Body, Human, Wish};
use crate::play::numbers::{Numbers, SpectateNumbers};
use crate::play::view::{Controls, CorpseView, ViewOffsets, Watch, Watched};

#[derive(
    Component,
    Clone,
    Copy,
    Default,
    PartialEq,
    serde::Serialize,
    serde::Deserialize,
    schemars::JsonSchema,
)]
#[serde(deny_unknown_fields)]
pub struct Look {}

stats! {
    #[section("look")]
    pub struct LookStats {
        pub yaw: f32,
        pub pitch: f32,
        pub position: [f32; 3],
        pub fov: f32,
    }
}

pub struct LookBrick;

impl Brick for LookBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .join::<SceneComponents>(
                ComponentKind::of::<Look>("look").schema(engine_core::schema_of::<Look>),
            )
            .moves::<Look>(Pace::Frame)
            .stats::<LookStats>()
            .systems(Schedule::FrameStart, lens)
            .systems(Schedule::FrameStart, place);
    }
}

fn lens_of(fov_degrees: f32, zoom: f32) -> Camera {
    let base = fov_degrees.to_radians();
    let zoom = if zoom > 0.0 { zoom } else { 1.0 };
    Camera {
        near: NEAR,
        far: FAR,
        projection: Projection::Perspective {
            vertical_fov_radians: 2.0 * ((base * 0.5).tan() / zoom).atan(),
        },
    }
}

fn lens(
    controls: Res<Controls>,
    eyes: Query<Entity, (With<Look>, Without<Camera>)>,
    mut spawner: Spawner<LookBrick>,
) {
    for eye in &eyes {
        spawner.insert(eye, lens_of(controls.fov_degrees, 1.0));
    }
}

struct Shot {
    position: Vec3,
    rotation: Quat,
    yaw: f32,
    pitch: f32,
    zoom: f32,
}

fn first_person(alpha: f32, (body, wish, offsets): (&Body, &Wish, &ViewOffsets)) -> Shot {
    let limit = PITCH_LIMIT_DEGREES.to_radians();
    let yaw = wish.yaw_radians + offsets.yaw_radians();
    let pitch = math::clamp(wish.pitch_radians + offsets.pitch_radians(), -limit, limit);
    Shot {
        position: body.eye_before.lerp(body.eye, alpha),
        rotation: Quat::from_rotation_y(yaw) * Quat::from_rotation_x(pitch),
        yaw,
        pitch,
        zoom: offsets.zoom,
    }
}

fn corpse_shot(corpse: &CorpseView, spectate: &SpectateNumbers, ticks: f32) -> Shot {
    let progress = math::clamp(ticks / spectate.corpse_seconds, 0.0, 1.0);
    let eased = progress * progress * (3.0 - 2.0 * progress);
    let (sin, cos) = math::sin_cos(corpse.yaw_radians);
    let behind = Vec3::new(sin, 0.0, cos);
    let position =
        corpse.eye + behind * (spectate.pull_back_m * eased) + Vec3::Y * (spectate.rise_m * eased);
    let toward = corpse.eye - position;
    let flat = toward.with_y(0.0).length();
    let pitch = if flat > 0.0 {
        math::atan2(toward.y, flat)
    } else {
        0.0
    };
    Shot {
        position,
        rotation: Quat::from_rotation_y(corpse.yaw_radians) * Quat::from_rotation_x(pitch),
        yaw: corpse.yaw_radians,
        pitch,
        zoom: 1.0,
    }
}

type Seen<'a> = (&'a Body, &'a Wish, &'a ViewOffsets);

#[derive(SystemParam)]
struct Viewing<'w, 's> {
    frame: Res<'w, FrameTime>,
    tick: Res<'w, FixedTime>,
    numbers: Res<'w, Numbers>,
    controls: Res<'w, Controls>,
    watched: Option<Res<'w, Watched>>,
    actors: Query<'w, 's, Seen<'static>>,
    humans: Query<'w, 's, Seen<'static>, With<Human>>,
}

fn shot_of(view: &Viewing<'_, '_>) -> Option<Shot> {
    let alpha = view.frame.alpha;
    let watched = view.watched.as_deref().copied().unwrap_or_default();
    match (watched.watch, watched.corpse, watched.entity) {
        (Watch::Corpse, Some(corpse), _) => {
            let spectate = &view.numbers.get()?.spectate;
            let ticks = (view.tick.tick.saturating_sub(corpse.tick)) as f32 + alpha;
            Some(corpse_shot(
                &corpse,
                spectate,
                ticks * view.tick.delta_seconds,
            ))
        }
        (_, _, Some(entity)) => view
            .actors
            .get(entity)
            .ok()
            .map(|seen| first_person(alpha, seen)),
        (_, _, None) => view
            .humans
            .iter()
            .next()
            .map(|seen| first_person(alpha, seen)),
    }
}

fn place(
    view: Viewing<'_, '_>,
    mut eyes: Query<(Entity, &mut Camera), With<Look>>,
    mut moves: Moves<LookBrick>,
    mut stats: ResMut<LookStats>,
) {
    let Some(shot) = shot_of(&view) else {
        return;
    };
    let lens = lens_of(view.controls.fov_degrees, shot.zoom);
    for (eye, mut camera) in &mut eyes {
        camera.set_if_neq(lens);
        if let Some(mut transform) = moves.get_mut(eye) {
            let placed = Transform {
                translation: shot.position,
                rotation: shot.rotation,
                scale: transform.scale,
            };
            transform.set_if_neq(placed);
        }
    }
    let fov = match lens.projection {
        Projection::Perspective {
            vertical_fov_radians,
        } => vertical_fov_radians,
        Projection::Orthographic { .. } | Projection::Fitted { .. } => 0.0,
    };
    stats.set_if_neq(LookStats {
        yaw: shot.yaw,
        pitch: shot.pitch,
        position: shot.position.to_array(),
        fov,
    });
}
