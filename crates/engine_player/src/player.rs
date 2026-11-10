use engine::prelude::*;
use engine::scene::SceneFiles;
use engine::{Error, ErrorCode, Expected};
use engine_core::{ChangeOperation, ChangeRequest, ComponentChange, Each};
use serde_json::json;

#[cfg(feature = "dev")]
const SMOKES: &[Smoke] = &crate::smokes::SMOKES;

#[cfg(not(feature = "dev"))]
const SMOKES: &[Smoke] = &[];

const WALKER: &str = "walker";

const COLLISION: &str = "collision";

const COLLISION_LIMITS: [(&str, u32); 5] = [
    ("collision.max_colliders", 10_000),
    ("collision.max_candidate_pairs", 100_000),
    ("collision.max_sensor_pairs", 16_384),
    ("collision.max_query_candidates", 10_000),
    ("collision.max_query_hits", 1024),
];

const FLIES: &str = "fly_camera.enabled";

#[derive(Clone, Copy)]
struct Player {
    entity: Entity,
    spawn: Vec3,
}

#[derive(Resource, Default)]
pub struct ScenePlayer {
    player: Option<Player>,
}

impl ScenePlayer {
    pub fn build(engine: &mut Engine) -> Result<Self, Error> {
        let world = engine.app.world_mut();
        if engine::host::has_setting(world, FLIES) {
            engine::host::apply_setting(world, FLIES, &json!(false))?;
        }
        Ok(Self::default())
    }
}

impl Game for ScenePlayer {
    const SMOKES: &'static [Smoke] = SMOKES;

    fn load_scene(
        engine: &mut Engine,
        document: &SceneDocument,
        files: &SceneFiles,
        _arrival: Arrival,
    ) -> Result<SceneInstance, Error> {
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
            && (player.parent.is_some() || !player.components.contains_key("collider"))
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
        let limited = engine::host::has_setting(world, COLLISION);
        let (instance, player) = document.instantiate_then(world, files, |world, entities| {
            let player = match player_id.as_ref().and_then(|id| entities.get(id)) {
                Some(&entity) => Some(attach_player(world, entity)?),
                None => None,
            };
            let mut settings: Vec<(String, serde_json::Value)> = Vec::new();
            if limited {
                settings.extend(
                    COLLISION_LIMITS
                        .iter()
                        .map(|&(key, limit)| (key.to_owned(), json!(limit))),
                );
            }
            settings.push((FLIES.to_owned(), json!(player.is_none())));
            engine::host::apply_settings(world, &settings)?;
            Ok(player)
        })?;
        world.resource_mut::<ScenePlayer>().player = player;
        engine.reset_clock();
        Ok(instance)
    }

    fn reset_camera(engine: &mut Engine, scene: &LoadedScene) {
        let world = engine.app.world_mut();
        let player = world.resource::<ScenePlayer>().player;
        match player {
            Some(player) => reset_player(world, player),
            None => scene.place_camera(world),
        }
        engine.reset_clock();
    }

    fn content(_engine: &Engine, scene: &LoadedScene) -> Option<Vec<u8>> {
        let mut content = crate::SOURCE.hash().to_vec();
        if let Some(scene) = scene.instance() {
            content.extend(scene.id().as_bytes());
            content.extend(scene.revision().to_le_bytes());
        }
        Some(content)
    }
}

fn attach_player(world: &mut World, entity: Entity) -> Result<Player, Error> {
    let spawn = world
        .get::<Transform>(entity)
        .expect("scene_entity_has_transform")
        .translation;
    walks(world, entity)?;
    Ok(Player { entity, spawn })
}

fn walks(world: &mut World, entity: Entity) -> Result<(), Error> {
    if engine_core::scene_component(world, WALKER).is_none() {
        return Ok(());
    }
    let id = world
        .get::<SceneEntity>(entity)
        .map(|authored| authored.authored_id.clone())
        .ok_or_else(|| Error::new(ErrorCode::GameRule, "$.entities").expected(Expected::Present))?;
    let change = ComponentChange {
        entity: id,
        component: WALKER.to_owned(),
        data: json!({}),
    };
    let changed = engine::host::call::<ChangeOperation>(
        world,
        ChangeRequest {
            changes: vec![change],
        },
    )?;
    match changed.rows.into_iter().next() {
        Some(Each::Refused(refused)) => {
            let code = serde_json::from_value(json!(refused.reason)).unwrap_or(ErrorCode::GameRule);
            Err(Error::new(code, refused.path))
        }
        Some(Each::Done(_)) | None => Ok(()),
    }
}

fn reset_player(world: &mut World, player: Player) {
    let Some(&held) = world.get::<Transform>(player.entity) else {
        return;
    };
    engine_core::send_ask(
        world,
        Teleport {
            node: player.entity,
            to: Transform {
                translation: player.spawn,
                ..held
            },
        },
    );
    walks(world, player.entity).expect("player_walks");
}
