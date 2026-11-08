use bevy_ecs::prelude::*;
use engine::prelude::{Commands, World};
use engine_core::{
    Camera, Input, Player, Projection, Transform, main_camera, math, place_camera,
};
use engine_places_simulation::ground::Ground;
use engine_places_simulation::physics::RigidBody;
use glam::{Vec2, Vec3};

const EYE_HEIGHT: f32 = 1.7;
const STEERING: [&str; 4] = [
    "fly_camera.move",
    "fly_camera.rise",
    "fly_camera.look",
    "fly_camera.turn",
];

pub(crate) type Placed = (Transform, Camera);

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct BodiesView(Option<Placed>);

#[derive(Resource, Default)]
pub(crate) struct Views {
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
    fn new(ground: Option<&Ground>, [x, z, yaw, pitch]: [f32; 4]) -> Self {
        let (yaw_sin, yaw_cos) = math::sin_cos(yaw.to_radians());
        let (pitch_sin, pitch_cos) = math::sin_cos(pitch.to_radians());
        let ground = ground
            .and_then(|ground| ground.point(Vec2::new(x, z)).solid())
            .map_or(0.0, |ground| ground.height);
        Self {
            x,
            z,
            direction: Vec3::new(yaw_sin * pitch_cos, pitch_sin, -yaw_cos * pitch_cos),
            eye: Vec3::new(x, ground + EYE_HEIGHT, z),
        }
    }

    fn placed(&self) -> Placed {
        (
            Transform::looking_at(self.eye, self.eye + self.direction),
            Camera {
                near: 0.2,
                far: 20_000.0,
                projection: Projection::Perspective {
                    vertical_fov_radians: 60.0_f32.to_radians(),
                },
            },
        )
    }
}

impl Views {
    pub(crate) fn stand(&mut self, ground: Option<&Ground>, place: [f32; 4]) -> Placed {
        let standing = StandingEye::new(ground, place);
        self.standing = Some(standing);
        standing.placed()
    }

    pub(crate) fn watch_bodies(&mut self, on: bool, bodies: Option<Placed>) -> Option<Placed> {
        if !on {
            self.awaiting = false;
            return None;
        }
        match bodies {
            Some(view) => {
                self.awaiting = false;
                self.standing = None;
                Some(view)
            }
            None => {
                self.awaiting = true;
                None
            }
        }
    }

    pub(crate) fn reset(&mut self) {
        self.standing = None;
        self.awaiting = false;
    }
}

pub(crate) fn show(commands: &mut Commands, (at, lens): Placed) {
    commands.queue(move |world: &mut World| {
        if let Some(camera) = main_camera(world) {
            place_camera(world, camera, at, lens);
        }
    });
}

pub(crate) fn watch(bodies: Query<&Transform, With<RigidBody>>, mut view: ResMut<BodiesView>) {
    view.set_if_neq(BodiesView(bodies_view(&bodies)));
}

pub(crate) fn look(
    (ground, input, bodies): (Option<Res<Ground>>, Res<Input>, Res<BodiesView>),
    mut views: ResMut<Views>,
    mut commands: Commands,
) {
    let views = &mut *views;
    if views.awaiting
        && bodies.0.is_some()
        && let Some(view) = views.watch_bodies(true, bodies.0)
    {
        show(&mut commands, view);
    }
    let steered = STEERING
        .into_iter()
        .filter_map(|name| input.find(name))
        .any(|verb| input.value2(verb, Player::ONE) != Vec2::ZERO);
    hold_standing_eye(views, (ground.as_deref(), steered), &mut commands);
}

fn hold_standing_eye(
    views: &mut Views,
    (ground, steered): (Option<&Ground>, bool),
    commands: &mut Commands,
) {
    let Some(standing) = views.standing else {
        return;
    };
    if steered {
        views.standing = None;
        return;
    }
    let Some(found) =
        ground.and_then(|ground| ground.point(Vec2::new(standing.x, standing.z)).solid())
    else {
        return;
    };
    let held = StandingEye {
        eye: Vec3::new(standing.x, found.height + EYE_HEIGHT, standing.z),
        ..standing
    };
    if held.eye != standing.eye {
        show(commands, held.placed());
    }
    views.standing = Some(held);
}

pub(crate) fn bodies_view(bodies: &Query<&Transform, With<RigidBody>>) -> Option<Placed> {
    let at: Vec<Vec3> = bodies
        .iter()
        .map(|transform| transform.translation)
        .collect();
    if at.is_empty() {
        return None;
    }
    let middle = at.iter().copied().sum::<Vec3>() / at.len() as f32;
    let reach = at
        .iter()
        .map(|point| point.distance(middle))
        .fold(4.0, math::max);
    let position = middle + Vec3::new(1.0, 0.8, 1.2) * (reach * 1.6 + 6.0);
    Some((
        Transform::looking_at(position, middle),
        Camera {
            far: 20_000.0,
            ..Camera::default()
        },
    ))
}
