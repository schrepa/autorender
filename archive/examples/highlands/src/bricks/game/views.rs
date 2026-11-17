use bevy_ecs::prelude::*;
use engine::prelude::{Commands, World};
use engine_core::{Camera, Transform, main_camera, math, place_camera};
use engine_places_simulation::physics::RigidBody;
use glam::Vec3;

pub(crate) type Placed = (Transform, Camera);

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct BodiesView(Option<Placed>);

#[derive(Resource, Default)]
pub(crate) struct Views {
    awaiting: bool,
}

impl Views {
    pub(crate) fn watch_bodies(&mut self, on: bool, bodies: Option<Placed>) -> Option<Placed> {
        if !on {
            self.awaiting = false;
            return None;
        }
        self.awaiting = bodies.is_none();
        bodies
    }

    pub(crate) fn reset(&mut self) {
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

pub(crate) fn look(bodies: Res<BodiesView>, mut views: ResMut<Views>, mut commands: Commands) {
    if views.awaiting
        && bodies.0.is_some()
        && let Some(view) = views.watch_bodies(true, bodies.0)
    {
        show(&mut commands, view);
    }
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
