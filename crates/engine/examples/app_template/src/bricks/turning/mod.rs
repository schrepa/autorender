include!(concat!(env!("OUT_DIR"), "/bricks/turning.rs"));

use bevy_ecs::prelude::*;
use engine_core::{Brick, Bricks, FixedTime, Moves, Pace, Schedule, Source};
use glam::Quat;

#[derive(Component)]
pub struct Turning;

pub struct TurningBrick;

impl Brick for TurningBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .moves::<Turning>(Pace::Tick)
            .systems(Schedule::FixedSimulation, turn);
    }
}

fn turn(time: Res<FixedTime>, mut cubes: Moves<TurningBrick>) {
    for (_, mut transform) in cubes.iter_mut() {
        transform.rotation = Quat::from_rotation_y(time.delta_seconds) * transform.rotation;
    }
}
