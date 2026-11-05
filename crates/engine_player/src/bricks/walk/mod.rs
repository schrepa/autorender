include!(concat!(env!("OUT_DIR"), "/bricks/walk.rs"));

use bevy_ecs::prelude::*;
use engine_core::{
    Brick, Bricks, ComponentKind, Filled, FixedTime, InterpolateTransform, MoveInput,
    SceneComponents, Schedule, Source, Transform, stats,
};
use engine_places::navigation::NavAgent;
use engine_places::physics::{Collider, Collision, CollisionLayers, QueryFilter, SensorFilter};
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
    collides: Filled<Collision>,
    mut stats: ResMut<PlayerWalkStats>,
    mut walkers: Query<(Entity, &Collider, &mut Transform, &mut Walker), Without<NavAgent>>,
) {
    if !collides.on() {
        return;
    }
    let dt = time.delta_seconds;
    for (entity, collider, mut transform, mut walker) in &mut walkers {
        walker.vertical_speed -= 20.0 * dt;
        let horizontal =
            Vec3::new(input.move_right, 0.0, -input.move_forward).normalize_or_zero() * dt * 6.5;
        let displacement = Vec3::new(horizontal.x, walker.vertical_speed * dt, horizontal.z);
        let result = collision.0.slide(
            collider.shape(),
            transform.matrix(),
            displacement,
            QueryFilter {
                layers: CollisionLayers {
                    memberships: collider.layers().memberships,
                    filter: collider.layers().filter & WORLD_LAYER,
                },
                sensors: SensorFilter::Exclude,
                exclude: Some(entity),
            },
        );
        match result {
            Ok(slid) => {
                transform.translation = slid.position;
                let speed = walker.vertical_speed;
                if slid
                    .hits
                    .iter()
                    .flatten()
                    .any(|hit| hit.normal.y * speed < 0.0)
                {
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
