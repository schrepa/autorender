include!(concat!(env!("OUT_DIR"), "/brick.rs"));

use bevy_ecs::prelude::*;
use engine_core::{
    Brick, Bricks, ComponentKind, FixedTime, InterpolateTransform, MoveInput, SceneComponents,
    Schedule, Source, Transform, stats,
};
use engine_places::navigation::NavAgent;
use engine_places::physics::{
    Collider, Collision, CollisionLayers, QueryFilter, SensorFilter, ShapeCastOptions,
};
use glam::Vec3;

stats! {
    #[section("player_walk_stats")]
    #[derive(Copy)]
    pub struct PlayerWalkStats {
        pub collision_errors: u32,
    }
}

#[derive(Component, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
#[require(InterpolateTransform)]
#[serde(deny_unknown_fields)]
struct Walker {
    #[serde(skip)]
    vertical_speed: f32,
}

pub struct PlayerWalkBrick;

impl Brick for PlayerWalkBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .stats::<PlayerWalkStats>()
            .join::<SceneComponents>(ComponentKind::of::<Walker>("walker"), None)
            .systems(Schedule::FixedMove, walk);
    }
}

const WORLD_LAYER: u32 = 1;

fn walk(
    time: Res<FixedTime>,
    input: Res<MoveInput>,
    collision: Res<Collision>,
    mut stats: ResMut<PlayerWalkStats>,
    mut walkers: Query<(Entity, &Collider, &mut Transform, &mut Walker), Without<NavAgent>>,
) {
    let dt = time.delta_seconds;
    for (entity, collider, mut transform, mut walker) in &mut walkers {
        walker.vertical_speed -= 20.0 * dt;
        let horizontal =
            Vec3::new(input.move_right, 0.0, -input.move_forward).normalize_or_zero() * dt * 6.5;
        let displacement = Vec3::new(horizontal.x, walker.vertical_speed * dt, horizontal.z);
        for axis in [1, 0, 2] {
            if displacement[axis] == 0.0 {
                continue;
            }
            let mut delta = Vec3::ZERO;
            delta[axis] = displacement[axis];
            let result = collision.0.cast_shape(
                collider.shape(),
                transform.matrix(),
                delta,
                QueryFilter {
                    layers: CollisionLayers {
                        memberships: collider.layers().memberships,
                        filter: collider.layers().filter & WORLD_LAYER,
                    },
                    sensors: SensorFilter::Exclude,
                    exclude: Some(entity),
                },
                ShapeCastOptions {
                    stop_at_penetration: false,
                    target_distance: 0.0,
                },
            );
            match result {
                Ok(hit) => {
                    transform.translation += delta * hit.as_ref().map_or(1.0, |hit| hit.fraction);
                    if hit.is_some() && axis == 1 {
                        walker.vertical_speed = 0.0;
                    }
                }
                Err(_) => {
                    stats.collision_errors = stats.collision_errors.saturating_add(1);
                    walker.vertical_speed = 0.0;
                }
            }
        }
    }
}
