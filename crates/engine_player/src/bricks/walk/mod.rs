include!(concat!(env!("OUT_DIR"), "/bricks/walk.rs"));

use bevy_ecs::prelude::*;
use engine_core::{
    Binding, BindingSource, Brick, Bricks, ComponentKind, ContextAsk, Contexts, Filled, FixedTime,
    Input, Interest, Moves, Pace, Player, SceneComponents, Schedule, Source, Stick, Verb, VerbId,
    VerbShape, Verbs, stats,
};
use engine_core::math::clamp;
use engine_places_simulation::navigation::NavAgent;
use engine_places_simulation::physics::{
    Collider, Collision, CollisionLayers, QueryFilter, SensorFilter,
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
#[serde(deny_unknown_fields)]
struct Walker {
    #[serde(skip)]
    vertical_speed: f32,
}

const CONTEXT: &str = "walk";

const MOVE: Verb = Verb {
    name: "walk.move",
    shape: VerbShape::Value2,
    bindings: &[
        Binding::new("walk.move", BindingSource::keys("KeyD", "KeyA", "KeyW", "KeyS")),
        Binding::new(
            "walk.move",
            BindingSource::keys("ArrowRight", "ArrowLeft", "ArrowUp", "ArrowDown"),
        ),
        Binding::new("walk.move", BindingSource::Stick(Stick::Left)).dead_zone(0.15),
    ],
};

#[derive(Resource, Default)]
struct Walking(Option<(VerbId, Interest<ContextAsk>)>);

pub struct PlayerWalkBrick;

impl Brick for PlayerWalkBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .stats::<PlayerWalkStats>()
            .join::<SceneComponents>(ComponentKind::of::<Walker>("walker"), None)
            .join::<Verbs>(MOVE, None)
            .resource::<Walking>()
            .moves::<Walker>(Pace::Tick)
            .systems(Schedule::FixedMove, walk);
    }
}

const WORLD_LAYER: u32 = 1;

fn steered(
    input: &Input,
    contexts: &mut Contexts,
    walking: &mut Walking,
    active: bool,
) -> Option<Vec3> {
    if !active {
        walking.0 = None;
        return None;
    }
    let verb = match &walking.0 {
        Some((verb, _)) => *verb,
        None => {
            let verb = input.find(MOVE.name)?;
            walking.0 = Some((verb, contexts.ask(CONTEXT, CONTEXT, false)));
            verb
        }
    };
    let moved = input.value2(verb, Player::ONE);
    Some(Vec3::new(clamp(moved.x, -1.0, 1.0), 0.0, -clamp(moved.y, -1.0, 1.0)).normalize_or_zero())
}

fn walk(
    (time, input, collision): (Res<FixedTime>, Res<Input>, Res<Collision>),
    (mut contexts, mut walking): (ResMut<Contexts>, ResMut<Walking>),
    collides: Filled<Collision>,
    mut stats: ResMut<PlayerWalkStats>,
    mut moves: Moves<PlayerWalkBrick>,
    mut walkers: Query<(Entity, &Collider, &mut Walker), Without<NavAgent>>,
) {
    let active = collides.on() && !walkers.is_empty();
    let Some(heading) = steered(&input, &mut contexts, &mut walking, active) else {
        return;
    };
    let dt = time.delta_seconds;
    for (entity, collider, mut walker) in &mut walkers {
        let Some(mut transform) = moves.get_mut(entity) else {
            continue;
        };
        walker.vertical_speed -= 20.0 * dt;
        let horizontal = heading * dt * 6.5;
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
