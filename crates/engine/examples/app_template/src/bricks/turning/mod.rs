include!(concat!(env!("OUT_DIR"), "/bricks/turning.rs"));

use bevy_ecs::prelude::*;
use engine_core::{Brick, Bricks, FixedTime, Schedule, Source, Transform};
use glam::Quat;

#[derive(Component)]
pub struct Turning;

pub struct TurningBrick;

impl Brick for TurningBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks.systems(Schedule::FixedSimulation, turn);
    }
}

fn turn(time: Res<FixedTime>, mut cubes: Query<&mut Transform, With<Turning>>) {
    for mut transform in &mut cubes {
        transform.rotation = Quat::from_rotation_y(time.delta_seconds) * transform.rotation;
    }
}
