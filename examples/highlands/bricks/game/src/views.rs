use bevy_ecs::prelude::*;
use engine_core::{Camera, MoveInput, Projection, Transform, math};
use engine_places::ground::Ground;
use engine_places::physics::RigidBody;
use glam::{Vec2, Vec3};

use crate::HighlandsSettings;

const EYE_HEIGHT: f32 = 1.7;

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct BodiesView(Option<Camera>);

#[derive(Default)]
pub(crate) struct Views {
    seen: Option<HighlandsSettings>,
    standing: Option<StandingEye>,
    awaiting: bool,
}

#[derive(Clone, Copy)]
struct StandingEye {
    x: f32,
    z: f32,
    direction: Vec3,
    eye: Vec3,
}

impl StandingEye {
    fn new(ground: &Ground, [x, z, yaw, pitch]: [f32; 4]) -> Self {
        let (yaw_sin, yaw_cos) = math::sin_cos(yaw.to_radians());
        let (pitch_sin, pitch_cos) = math::sin_cos(pitch.to_radians());
        let ground = ground
            .sample(Vec2::new(x, z))
            .map_or(0.0, |ground| ground.height);
        Self {
            x,
            z,
            direction: Vec3::new(yaw_sin * pitch_cos, pitch_sin, -yaw_cos * pitch_cos),
            eye: Vec3::new(x, ground + EYE_HEIGHT, z),
        }
    }

    fn camera(&self) -> Camera {
        Camera {
            position: self.eye,
            target: self.eye + self.direction,
            near: 0.2,
            far: 20_000.0,
            projection: Projection::Perspective {
                vertical_fov_radians: 60.0_f32.to_radians(),
            },
        }
    }
}

pub(crate) fn watch(bodies: Query<&Transform, With<RigidBody>>, mut view: ResMut<BodiesView>) {
    view.set_if_neq(BodiesView(bodies_view(&bodies)));
}

pub(crate) fn look(
    (settings, ground, input): (Res<HighlandsSettings>, Res<Ground>, Res<MoveInput>),
    bodies: Res<BodiesView>,
    mut camera: ResMut<Camera>,
    mut views: Local<Views>,
) {
    let views = &mut *views;
    if settings.is_changed() {
        let seen = views.seen.replace(settings.clone());
        if seen.as_ref().map(|seen| seen.stand) != Some(settings.stand) {
            views.standing = settings.stand.map(|place| StandingEye::new(&ground, place));
            if let Some(standing) = views.standing {
                *camera = standing.camera();
            }
        }
        if seen.as_ref().map(|seen| seen.watch_bodies) != Some(settings.watch_bodies) {
            watch_bodies(views, settings.watch_bodies, bodies.0, &mut camera);
        }
    }
    hold_standing_eye(views, (&ground, &input), &mut camera);
    if views.awaiting && bodies.0.is_some() {
        watch_bodies(views, true, bodies.0, &mut camera);
    }
}

fn watch_bodies(views: &mut Views, on: bool, bodies: Option<Camera>, camera: &mut ResMut<Camera>) {
    if !on {
        views.awaiting = false;
        return;
    }
    match bodies {
        Some(view) => {
            views.awaiting = false;
            views.standing = None;
            **camera = view;
        }
        None => views.awaiting = true,
    }
}

fn hold_standing_eye(
    views: &mut Views,
    (ground, input): (&Ground, &MoveInput),
    camera: &mut ResMut<Camera>,
) {
    let Some(standing) = views.standing else {
        return;
    };
    let steered = [
        input.move_forward,
        input.move_right,
        input.move_up,
        input.look_yaw,
        input.look_pitch,
    ]
    .into_iter()
    .any(|amount| amount != 0.0);
    if steered || camera.position != standing.eye {
        views.standing = None;
        return;
    }
    let Some(found) = ground.sample(Vec2::new(standing.x, standing.z)) else {
        return;
    };
    let held = StandingEye {
        eye: Vec3::new(standing.x, found.height + EYE_HEIGHT, standing.z),
        ..standing
    };
    **camera = held.camera();
    views.standing = Some(held);
}

fn bodies_view(bodies: &Query<&Transform, With<RigidBody>>) -> Option<Camera> {
    let at: Vec<Vec3> = bodies.iter().map(|transform| transform.translation).collect();
    if at.is_empty() {
        return None;
    }
    let middle = at.iter().copied().sum::<Vec3>() / at.len() as f32;
    let reach = at
        .iter()
        .map(|point| point.distance(middle))
        .fold(4.0, math::max);
    Some(Camera {
        position: middle + Vec3::new(1.0, 0.8, 1.2) * (reach * 1.6 + 6.0),
        target: middle,
        far: 20_000.0,
        ..Camera::default()
    })
}
