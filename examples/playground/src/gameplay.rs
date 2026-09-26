use engine::prelude::*;

#[derive(Component)]
struct Player {
    spawn: Vec3,
    vertical_speed: f32,
    grounded: bool,
}
#[derive(Component)]
struct Pickup {
    base_height: f32,
    phase: f32,
}

/// Score is gameplay data, independent of DOM and rendering.
#[derive(Resource, Default)]
pub struct GameProgress {
    pub collected: u32,
    pub total: u32,
    pub jumps: u32,
    /// Failed bounded movement queries stop motion and remain observable.
    pub collision_errors: u32,
}

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GameplaySystems {
    Movement,
    Collection,
    Animation,
}

pub struct PlaygroundPlugin;
impl Plugin for PlaygroundPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<CollisionPlugin>() {
            app.add_plugins(CollisionPlugin);
        }
        app.init_resource::<GameProgress>();
        app.configure_sets(
            FixedSimulation,
            (GameplaySystems::Movement, GameplaySystems::Animation).chain(),
        );
        app.add_systems(
            FixedSimulation,
            (
                move_player.in_set(GameplaySystems::Movement),
                animate.in_set(GameplaySystems::Animation),
            ),
        );
        app.add_systems(
            FixedPostSimulation,
            collect
                .in_set(GameplaySystems::Collection)
                .after(CollisionSystems::Detect),
        );
    }
}

pub(crate) fn attach(world: &mut World, instance: &SceneInstance) {
    let mut progress = GameProgress::default();
    for (_, entity) in instance.entities() {
        let tags = world
            .get::<SceneEntity>(entity)
            .expect("scene metadata exists")
            .tags
            .clone();
        let position = world
            .get::<Transform>(entity)
            .expect("scene has transforms")
            .translation;
        let mut entity = world.entity_mut(entity);
        for tag in &tags {
            match tag.as_str() {
                "player" => {
                    entity.insert(InterpolateTransform);
                    entity.insert(Player {
                        spawn: position,
                        vertical_speed: 0.0,
                        grounded: true,
                    });
                }
                "pickup" => {
                    entity.insert(InterpolateTransform);
                    entity.insert(Pickup {
                        base_height: position.y,
                        phase: progress.total as f32,
                    });
                    progress.total += 1;
                }
                _ => {}
            }
        }
    }
    world.insert_resource(progress);
}

pub(crate) fn reset_player(world: &mut World, entity: Entity) {
    let Some(player) = world.get::<Player>(entity) else {
        return;
    };
    let spawn = player.spawn;
    if world.get::<Transform>(entity).is_some() {
        edit_transform(world, entity, |transform| transform.translation = spawn)
            .expect("the player transform exists at this exclusive boundary");
    }
    if let Some(mut player) = world.get_mut::<Player>(entity) {
        player.vertical_speed = 0.0;
        player.grounded = true;
    }
    // A teleport is a game decision; the engine resets this presented subtree.
    reset_interpolation(world, entity);
}

// Layer meanings are game policy; the engine only evaluates symmetric bit masks.
const WORLD_LAYER: u32 = 1;

fn move_player(
    time: Res<FixedTime>,
    input: Res<InputState>,
    actions: Res<ActionInput>,
    mut progress: ResMut<GameProgress>,
    mut collisions: ResMut<CollisionWorld>,
    mut players: Query<(Entity, &Collider, &mut Transform, &mut Player)>,
) {
    let dt = time.delta_seconds;
    for (entity, collider, mut transform, mut player) in &mut players {
        if actions.just_pressed(ActionId::JUMP) && player.grounded {
            player.vertical_speed = 7.5;
            player.grounded = false;
            progress.jumps += 1;
        }
        player.grounded = false;
        player.vertical_speed -= 20.0 * dt;
        let horizontal =
            Vec3::new(input.move_right, 0.0, -input.move_forward).normalize_or_zero() * dt * 6.5;
        let displacement = Vec3::new(horizontal.x, player.vertical_speed * dt, horizontal.z);
        // Axis order and grounding are this game's controller policy. The engine
        // supplies continuous shape casts against arbitrary authored colliders.
        for axis in [1, 0, 2] {
            if displacement[axis] == 0.0 {
                continue;
            }
            let mut delta = Vec3::ZERO;
            delta[axis] = displacement[axis];
            let result = collisions.cast_shape(
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
                    if let Some(hit) = hit
                        && axis == 1
                    {
                        player.grounded = delta.y < 0.0 && hit.normal.y > 0.5;
                        player.vertical_speed = 0.0;
                    }
                }
                Err(_) => {
                    // An incomplete query never authorizes movement through geometry.
                    progress.collision_errors = progress.collision_errors.saturating_add(1);
                    player.vertical_speed = 0.0;
                }
            }
        }
    }
}

fn collect(
    mut commands: Commands,
    mut progress: ResMut<GameProgress>,
    events: Res<engine::collision::SensorEvents>,
    players: Query<(), With<Player>>,
    pickups: Query<(), With<Pickup>>,
) {
    if !events.complete() {
        return;
    }
    for event in events.events() {
        if event.kind != engine::collision::SensorEventKind::Enter {
            continue;
        }
        let (a, b) = (event.pair.first, event.pair.second);
        let pickup = if players.contains(a) && pickups.contains(b) {
            Some(b)
        } else if players.contains(b) && pickups.contains(a) {
            Some(a)
        } else {
            None
        };
        if let Some(entity) = pickup {
            commands.entity(entity).despawn();
            progress.collected += 1;
        }
    }
}

fn animate(time: Res<FixedTime>, mut pickups: Query<(&Pickup, &mut Transform)>) {
    let time = (time.elapsed_seconds % 10000.0) as f32;
    for (pickup, mut transform) in &mut pickups {
        transform.translation.y = pickup.base_height + math::sin(time * 2.0 + pickup.phase) * 0.18;
        transform.rotation = Quat::from_rotation_y(time * 0.8 + pickup.phase);
    }
}
