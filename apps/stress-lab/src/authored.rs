use engine::ErrorCode;
use engine::prelude::*;
use engine::scene::{SceneError, SceneFiles};

use crate::FrameMetrics;

pub struct AuthoredScene {
    instance: Option<SceneInstance>,
    player: Option<Entity>,
}

impl AuthoredScene {
    pub fn new(
        source: &str,
        files: &SceneFiles,
        without: &[Feature],
    ) -> Result<(Engine, Self), SceneError> {
        let mut engine = Engine::default();
        engine
            .app
            .add_plugins(EnginePlugins::without(without.iter().copied()))
            .add_plugins(PlayerPlugin);
        engine.app.init_resource::<FrameMetrics>();
        let mut scene = Self {
            instance: None,
            player: None,
        };
        scene.reload(&mut engine, source, files)?;
        Ok((engine, scene))
    }

    pub fn reload(
        &mut self,
        engine: &mut Engine,
        source: &str,
        files: &SceneFiles,
    ) -> Result<(), SceneError> {
        let document = SceneDocument::from_json(source)?;
        let expanded = document.expanded_entities()?;
        let players: Vec<_> = expanded
            .iter()
            .filter(|object| object.tags.iter().any(|tag| tag == "player"))
            .collect();
        if players.len() > 1 {
            return Err(SceneError {
                code: ErrorCode::GameRule,
                path: "$.entities".into(),
                message: "at most one entity may be tagged player".into(),
            });
        }
        if let Some(player) = players.first()
            && (player.parent.is_some() || player.collider.is_none())
        {
            return Err(SceneError {
                code: ErrorCode::GameRule,
                path: "$.entities".into(),
                message: format!(
                    "expanded entity {}: the player walks with a root transform and an authored collider",
                    player.id
                ),
            });
        }
        let player_id = players.first().map(|player| player.id.clone());
        let instance = document.instantiate_with(engine.app.world_mut(), files)?;
        if let Some(previous) = self.instance.take() {
            previous.despawn(engine.app.world_mut());
        }
        self.player = player_id.and_then(|id| instance.entity(&id));
        if let Some(entity) = self.player {
            attach_player(engine.app.world_mut(), entity);
        }
        self.instance = Some(instance);
        engine.reset_clock();
        Ok(())
    }

    pub fn revision(&self) -> u64 {
        self.instance.as_ref().map_or(0, SceneInstance::revision)
    }

    pub fn scene_id(&self) -> &str {
        self.instance.as_ref().map_or("", SceneInstance::id)
    }

    pub fn instance(&self) -> Option<&SceneInstance> {
        self.instance.as_ref()
    }

    pub fn progress(&self, engine: &Engine) -> serde_json::Value {
        let world = engine.app.world();
        let player = self
            .player
            .and_then(|entity| world.get::<Transform>(entity))
            .map(|transform| transform.translation.to_array());
        let assets = world.resource::<AssetLibrary>();
        serde_json::json!({
            "scene_id": self.scene_id(), "revision": self.revision(),
            "frame": world.resource::<FrameTime>().frame, "player": player,
            "cpu_meshes": assets.mesh_count(), "cpu_materials": assets.material_count(),
            "cpu_asset_bytes": assets.resident_bytes(),
            "cpu_textures": assets.texture_count(), "cpu_texture_bytes": assets.texture_bytes(),
        })
    }

    pub fn reset_player(&self, engine: &mut Engine) {
        if let Some(entity) = self.player {
            let world = engine.app.world_mut();
            if let Some(spawn) = world.get::<Player>(entity).map(|player| player.spawn) {
                edit_transform(world, entity, |transform| transform.translation = spawn)
                    .expect("the player's transform exists at this exclusive boundary");
                if let Some(mut player) = world.get_mut::<Player>(entity) {
                    player.vertical_speed = 0.0;
                }
                reset_interpolation(world, entity);
            }
        }
        engine.reset_clock();
    }
}

#[derive(Resource, Default, Clone, Copy, Debug, serde::Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct PlayerWalk {
    pub collision_errors: u32,
}

#[derive(Component)]
struct Player {
    spawn: Vec3,
    vertical_speed: f32,
}

fn attach_player(world: &mut World, entity: Entity) {
    let spawn = world
        .get::<Transform>(entity)
        .expect("scene entities have transforms")
        .translation;
    world.entity_mut(entity).insert((
        InterpolateTransform,
        Player {
            spawn,
            vertical_speed: 0.0,
        },
    ));
}

struct PlayerPlugin;

impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<CollisionPlugin>() {
            return;
        }
        app.insert_resource(crate::scene::stress_collision_budgets())
            .init_resource::<PlayerWalk>()
            .add_systems(FixedSimulation, walk);
    }
}

const WORLD_LAYER: u32 = 1;

fn walk(
    time: Res<FixedTime>,
    input: Res<InputState>,
    collisions: Res<CollisionWorld>,
    mut errors: ResMut<PlayerWalk>,
    mut players: Query<(Entity, &Collider, &mut Transform, &mut Player)>,
) {
    let dt = time.delta_seconds;
    for (entity, collider, mut transform, mut player) in &mut players {
        player.vertical_speed -= 20.0 * dt;
        let horizontal =
            Vec3::new(input.move_right, 0.0, -input.move_forward).normalize_or_zero() * dt * 6.5;
        let displacement = Vec3::new(horizontal.x, player.vertical_speed * dt, horizontal.z);
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
                    if hit.is_some() && axis == 1 {
                        player.vertical_speed = 0.0;
                    }
                }
                Err(_) => {
                    errors.collision_errors = errors.collision_errors.saturating_add(1);
                    player.vertical_speed = 0.0;
                }
            }
        }
    }
}
