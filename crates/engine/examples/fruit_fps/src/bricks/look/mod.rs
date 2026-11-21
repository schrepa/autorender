engine::brick!(look);

use engine::glam::{Quat, Vec3};
use engine::math;
use engine::prelude::*;
use engine_core::{DevSetting, Poses, Stats};

use crate::names::{FAR, NEAR, PITCH_LIMIT_DEGREES};
use crate::play::actors::{Actor, Body, Eye, Human, Wish};
use crate::play::camera::TestCamera;
use crate::play::lobby::LobbyView;
use crate::play::numbers::{Numbers, SpectateNumbers};
use crate::play::turns::eased;
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

#[derive(
    Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
)]
#[serde(deny_unknown_fields)]
pub struct FreeCamera {
    pub position: [f32; 3],
    pub target: [f32; 3],
    pub fov_degrees: f32,
}

#[derive(
    Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
)]
#[serde(deny_unknown_fields)]
pub struct FollowCamera {
    pub slot: u32,
    pub offset: [f32; 3],
    pub look_height_m: f32,
    pub fov_degrees: f32,
}

settings! {
    #[dev]
    #[section("look_test")]
    #[serde(deny_unknown_fields)]
    pub struct LookTest {
        pub free: Option<FreeCamera> = None => any,
        pub follow: Option<FollowCamera> = None => any,
        pub hide_hud: bool = false => any,
    }
}

stats! {
    #[section("look")]
    pub struct LookStats {
        pub yaw: f32,
        pub pitch: f32,
        pub position: [f32; 3],
        pub fov: f32,
        pub free: bool,
        pub follow: Option<u32>,
        pub lobby: bool,
        pub test_refusal: Option<String>,
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
            .resource::<TestCamera>()
            .dev(|dev| {
                dev.settings::<LookTest>().stats::<LookStats>();
            })
            .systems(Schedule::Presentation, lens)
            .systems(Schedule::Presentation, place);
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
            fit: FovFit::MinSpan,
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

fn drawn_at(poses: &Poses<'_>, node: Entity) -> Option<Vec3> {
    poses.drawn_pose(node).map(|pose| pose.w_axis.truncate())
}

fn first_person(poses: &Poses<'_>, (eye, wish, offsets): Seen<'_>) -> Option<Shot> {
    let limit = PITCH_LIMIT_DEGREES.to_radians();
    let yaw = wish.yaw_radians + offsets.yaw_radians();
    let pitch = math::clamp(wish.pitch_radians + offsets.pitch_radians(), -limit, limit);
    Some(Shot {
        position: drawn_at(poses, eye.node)?,
        rotation: Quat::from_rotation_y(yaw) * Quat::from_rotation_x(pitch),
        yaw,
        pitch,
        zoom: offsets.zoom,
    })
}

fn corpse_shot(corpse: &CorpseView, spectate: &SpectateNumbers, ticks: f32) -> Shot {
    let eased = eased(ticks / spectate.corpse_seconds);
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

fn aimed(position: Vec3, target: Vec3) -> Option<(Quat, f32, f32)> {
    let toward = (target - position).try_normalize()?;
    let flat = toward.with_y(0.0).length();
    let yaw = math::atan2(-toward.x, -toward.z);
    let pitch = math::atan2(toward.y, flat);
    Some((
        Quat::from_rotation_y(yaw) * Quat::from_rotation_x(pitch),
        yaw,
        pitch,
    ))
}

fn fov_checked(fov_degrees: f32, path: &str) -> Result<f32, String> {
    if fov_degrees.is_finite() && fov_degrees > 1.0 && fov_degrees < 170.0 {
        Ok(fov_degrees)
    } else {
        Err(format!(
            "{path}.fov_degrees: {fov_degrees} is outside 1..170"
        ))
    }
}

fn aimed_shot(position: Vec3, target: Vec3, path: &str) -> Result<Shot, String> {
    let (rotation, yaw, pitch) = aimed(position, target)
        .ok_or_else(|| format!("{path}: the camera stands on its target"))?;
    Ok(Shot {
        position,
        rotation,
        yaw,
        pitch,
        zoom: 1.0,
    })
}

fn followed(
    follow: &FollowCamera,
    poses: &Poses<'_>,
    actors: &Query<'_, '_, Slotted<'static>>,
) -> Result<Shot, String> {
    let (_, eye, body, wish) = actors
        .iter()
        .find(|(actor, ..)| actor.slot == follow.slot)
        .ok_or_else(|| format!("$.look_test.follow.slot: no actor in slot {}", follow.slot))?;
    let drawn_eye = drawn_at(poses, eye.node).ok_or_else(|| {
        format!(
            "$.look_test.follow.slot: actor in slot {} has no drawn pose",
            follow.slot
        )
    })?;
    let feet = drawn_eye - (body.eye - body.feet);
    let turn = Quat::from_rotation_y(wish.yaw_radians);
    let position = feet + turn * Vec3::from_array(follow.offset);
    aimed_shot(
        position,
        feet + Vec3::Y * follow.look_height_m,
        "$.look_test.follow",
    )
}

fn test_shot(view: &Viewing<'_, '_>) -> Option<Result<(Shot, f32), String>> {
    let test = view.test.get()?;
    if let Some(follow) = test.follow.as_ref() {
        return Some(
            fov_checked(follow.fov_degrees, "$.look_test.follow")
                .and_then(|fov| followed(follow, &view.poses, &view.slots).map(|shot| (shot, fov))),
        );
    }
    let free = test.free.as_ref()?;
    Some(
        fov_checked(free.fov_degrees, "$.look_test.free").and_then(|fov| {
            aimed_shot(
                Vec3::from_array(free.position),
                Vec3::from_array(free.target),
                "$.look_test.free",
            )
            .map(|shot| (shot, fov))
        }),
    )
}

fn lobby_shot(view: &Viewing<'_, '_>) -> Option<(Shot, f32)> {
    let (position, target, fov) = view.lobby.as_deref()?.shot(view.frame.real_elapsed)?;
    let shot = aimed_shot(position, target, "$.lobby").ok()?;
    Some((shot, fov))
}

type Seen<'a> = (&'a Eye, &'a Wish, &'a ViewOffsets);

type Slotted<'a> = (&'a Actor, &'a Eye, &'a Body, &'a Wish);

#[derive(SystemParam)]
struct Viewing<'w, 's> {
    frame: Res<'w, FrameTime>,
    tick: Res<'w, FixedTime>,
    numbers: Res<'w, Numbers>,
    controls: Res<'w, Controls>,
    poses: Poses<'w>,
    watched: Option<Res<'w, Watched>>,
    lobby: Option<Res<'w, LobbyView>>,
    test: DevSetting<'w, LookTest>,
    actors: Query<'w, 's, Seen<'static>>,
    slots: Query<'w, 's, Slotted<'static>>,
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
            .and_then(|seen| first_person(&view.poses, seen)),
        (_, _, None) => view
            .humans
            .iter()
            .next()
            .and_then(|seen| first_person(&view.poses, seen)),
    }
}

fn place(
    view: Viewing<'_, '_>,
    mut eyes: Query<(Entity, &mut Camera), With<Look>>,
    mut moves: Moves<LookBrick>,
    (mut stats, mut outside): (Stats<LookStats>, ResMut<TestCamera>),
) {
    let tested = test_shot(&view);
    let test_refusal = tested
        .as_ref()
        .and_then(|tested| tested.as_ref().err().cloned());
    let tested = tested.and_then(Result::ok);
    let free = tested.is_some();
    let hud_hidden = view.test.get().is_some_and(|test| test.hide_hud);
    let lobby = lobby_shot(&view).filter(|_| tested.is_none());
    let in_lobby = lobby.is_some();
    let (shot, fov_degrees) = match tested.or(lobby) {
        Some((shot, fov)) => (Some(shot), fov),
        None => (shot_of(&view), view.controls.fov_degrees),
    };
    outside.set_if_neq(TestCamera {
        outside: free,
        hud_hidden: hud_hidden && free,
    });
    let Some(shot) = shot else {
        return;
    };
    let lens = lens_of(fov_degrees, shot.zoom);
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
            ..
        } => vertical_fov_radians,
        Projection::Orthographic { .. } | Projection::Fitted { .. } => 0.0,
    };
    stats.show(|| LookStats {
        yaw: shot.yaw,
        pitch: shot.pitch,
        position: shot.position.to_array(),
        fov,
        free,
        follow: view
            .test
            .get()
            .and_then(|test| test.follow)
            .filter(|_| free)
            .map(|follow| follow.slot),
        lobby: in_lobby,
        test_refusal,
    });
}
