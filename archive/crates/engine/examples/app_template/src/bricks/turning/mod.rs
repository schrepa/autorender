engine::brick!(turning);

use engine::prelude::*;

#[derive(Component, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Turning {}

pub struct TurningBrick;

impl Brick for TurningBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .join::<SceneComponents>(ComponentKind::of::<Turning>("turning"), None)
            .moves::<Turning>(Pace::Tick)
            .systems(Schedule::FixedSimulation, turn);
    }
}

fn turn(time: Res<FixedTime>, mut cubes: Moves<TurningBrick>) {
    for (_, mut transform) in cubes.iter_mut() {
        transform.rotation = Quat::from_rotation_y(time.delta_seconds) * transform.rotation;
    }
}
