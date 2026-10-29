use engine::collision::CollisionConfig;
use engine::prelude::*;
use engine::scene::SceneFiles;
use engine::{Error, ErrorCode, Expected};

pub struct ScenePlayer {
    instance: Option<SceneInstance>,
    player: Option<Entity>,
    view: Camera,
    without: Vec<&'static str>,
}

impl ScenePlayer {
    pub fn new(
        source: &str,
        files: &SceneFiles,
        without: &[&'static str],
    ) -> Result<(Engine, Self), Error> {
        let mut engine = Engine::default();
        engine
            .app
            .add_bricks(engine::bricks::standard().without(without.iter().copied()))
            .add_plugins(PlayerPlugin);
        let mut player = Self {
            instance: None,
            player: None,
            view: Camera::default(),
            without: without.to_vec(),
        };
        player.load(&mut engine, source, files)?;
        Ok((engine, player))
    }

    fn load(
        &mut self,
        engine: &mut Engine,
        source: &str,
        files: &SceneFiles,
    ) -> Result<(), Error> {
        let document = SceneDocument::from_json(source)?;
        let expanded = document.expanded_entities()?;
        let players: Vec<_> = expanded
            .iter()
            .filter(|object| object.tags.iter().any(|tag| tag == "player"))
            .collect();
        if players.len() > 1 {
            return Err(Error::new(ErrorCode::GameRule, "$.entities")
                .at_most(1)
                .got(players.len()));
        }
        if let Some(player) = players.first()
            && (player.parent.is_some() || player.collider.is_none())
        {
            let expected = if player.parent.is_some() {
                Expected::Absent
            } else {
                Expected::Present
            };
            return Err(Error::new(ErrorCode::GameRule, "$.entities")
                .expected(expected)
                .got(&player.id));
        }
        let player_id = players.first().map(|player| player.id.clone());
        let world = engine.app.world_mut();
        let instance = document.instantiate_with(world, files)?;
        if let Some(previous) = self.instance.take() {
            previous.despawn(world);
        }
        self.player = player_id.and_then(|id| instance.entity(&id));
        if let Some(entity) = self.player {
            attach_player(world, entity);
        }
        self.instance = Some(instance);
        self.view = *world.resource::<Camera>();
        align_fly_camera(world, &self.view);
        engine.reset_clock();
        Ok(())
    }

    pub fn progress(&self, engine: &Engine) -> serde_json::Value {
        let world = engine.app.world();
        let player = self
            .player
            .and_then(|entity| world.get::<Transform>(entity))
            .map(|transform| transform.translation.to_array());
        let assets = world.resource::<AssetLibrary>();
        let scene = self.instance.as_ref();
        serde_json::json!({
            "scene_id": scene.map_or("", SceneInstance::id),
            "revision": scene.map_or(0, SceneInstance::revision),
            "frame": world.resource::<FrameTime>().frame, "player": player,
            "cpu_meshes": assets.mesh_count(), "cpu_materials": assets.material_count(),
            "cpu_asset_bytes": assets.resident_bytes(),
            "cpu_textures": assets.texture_count(), "cpu_texture_bytes": assets.texture_bytes(),
        })
    }
}

impl Game for ScenePlayer {
    fn reset_camera(&mut self, engine: &mut Engine) {
        let world = engine.app.world_mut();
        match self.player {
            Some(entity) => reset_player(world, entity),
            None => {
                world.insert_resource(self.view);
                align_fly_camera(world, &self.view);
            }
        }
        engine.reset_clock();
    }

    fn scene(&self) -> Option<&SceneInstance> {
        self.instance.as_ref()
    }

    fn content(&self, _engine: &Engine) -> Option<Vec<u8>> {
        let mut content = crate::SOURCE.to_vec();
        if let Some(scene) = self.scene() {
            content.extend(scene.id().as_bytes());
            content.extend(scene.revision().to_le_bytes());
        }
        for brick in &self.without {
            content.extend(brick.as_bytes());
            content.push(0);
        }
        Some(content)
    }

    fn reload_scene(
        &mut self,
        engine: &mut Engine,
        source: &str,
        files: &SceneFiles,
    ) -> Result<(), Error> {
        self.load(engine, source, files)
    }

    fn inspectors(inspectors: &mut Inspectors) {
        inspectors.resource::<PlayerWalk>(
            include_str!("generated/inspectors.json"),
            "PlayerWalk",
            |walk| serde_json::to_value(walk).expect("walk counts serialize"),
        );
    }
}

pub fn bricks(names: &str, path: &str) -> engine::Result<Vec<&'static str>> {
    engine::bricks::standard_names(names, path)
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

fn reset_player(world: &mut World, entity: Entity) {
    let Some(spawn) = world.get::<Player>(entity).map(|player| player.spawn) else {
        return;
    };
    edit_transform(world, entity, |transform| transform.translation = spawn)
        .expect("the player's transform exists at this exclusive boundary");
    if let Some(mut player) = world.get_mut::<Player>(entity) {
        player.vertical_speed = 0.0;
    }
    reset_interpolation(world, entity);
}

fn align_fly_camera(world: &mut World, camera: &Camera) {
    if let Some(mut controller) = world.get_resource_mut::<FlyCameraController>() {
        controller.align_to(camera);
    }
}

struct PlayerPlugin;

impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(FlyCameraPlugin)
            .configure_sets(Presentation, CameraSystems.run_if(flies));
        if !app
            .world()
            .get_resource::<engine::bricks::BrickRegistry>()
            .is_some_and(|registry| registry.has("collision"))
        {
            return;
        }
        app.insert_resource(CollisionConfig {
            max_colliders: Some(10_000),
            max_candidate_pairs: Some(100_000),
            max_sensor_pairs: Some(16_384),
            max_query_candidates: Some(10_000),
            max_query_hits: Some(1024),
        })
        .init_resource::<PlayerWalk>()
        .add_systems(FixedSimulation, walk);
    }
}

fn flies(input: Res<InputState>, players: Query<(), With<Player>>) -> bool {
    players.is_empty()
        && [
            input.move_forward,
            input.move_right,
            input.move_up,
            input.look_yaw,
            input.look_pitch,
        ]
        .iter()
        .any(|&value| value != 0.0)
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
