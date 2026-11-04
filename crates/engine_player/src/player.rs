use engine::prelude::*;
use engine::scene::SceneFiles;
use engine::{Error, ErrorCode, Expected};
use serde_json::json;

const WALKER: &str = "walker";

const COLLISION: &str = "collision";

const COLLISION_LIMITS: [(&str, u32); 5] = [
    ("collision.max_colliders", 10_000),
    ("collision.max_candidate_pairs", 100_000),
    ("collision.max_sensor_pairs", 16_384),
    ("collision.max_query_candidates", 10_000),
    ("collision.max_query_hits", 1024),
];

pub(crate) const FLIES: &str = "fly_camera.enabled";

const STEERED: &str = "fly_camera.steered_only";

#[derive(Clone, Copy)]
struct Player {
    entity: Entity,
    spawn: Vec3,
}

pub struct ScenePlayer {
    instance: Option<SceneInstance>,
    player: Option<Player>,
    view: Camera,
}

impl ScenePlayer {
    pub fn new(source: &str, files: &SceneFiles, engine: &mut Engine) -> Result<Self, Error> {
        let mut player = Self {
            instance: None,
            player: None,
            view: Camera::default(),
        };
        player.load(engine, source, files)?;
        Ok(player)
    }

    fn load(&mut self, engine: &mut Engine, source: &str, files: &SceneFiles) -> Result<(), Error> {
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
        let instance = document.instantiate_with(world, files)?;
        if let Some(previous) = self.instance.take() {
            previous.despawn(world);
        }
        self.player = match player_id.and_then(|id| instance.entity(&id)) {
            Some(entity) => Some(attach_player(world, entity)?),
            None => None,
        };
        self.instance = Some(instance);
        self.view = *world.resource::<Camera>();
        if engine::host::has_setting(world, COLLISION) {
            for (key, limit) in COLLISION_LIMITS {
                engine::host::apply_setting(world, key, &json!(limit))?;
            }
        }
        engine::host::apply_setting(world, FLIES, &json!(self.player.is_none()))?;
        engine::host::apply_setting(world, STEERED, &json!(true))?;
        engine.reset_clock();
        Ok(())
    }

    pub fn progress(&self, engine: &Engine) -> serde_json::Value {
        let world = engine.app.world();
        let player = self
            .player
            .and_then(|player| world.get::<Transform>(player.entity))
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
            Some(player) => reset_player(world, player),
            None => {
                world.insert_resource(self.view);
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
}

pub fn listed(
    without: &str,
    path: &str,
) -> Result<impl FnOnce(BrickList) -> BrickList + use<>, Error> {
    let everything = engine_standard::everything(BrickList::default());
    let names: Vec<&'static str> = everything
        .listed()
        .iter()
        .map(|listed| listed.package())
        .collect();
    let left_out = without
        .split(',')
        .filter(|name| !name.is_empty())
        .map(|name| {
            names
                .iter()
                .copied()
                .find(|&listed| listed == name)
                .ok_or_else(|| {
                    Error::new(ErrorCode::InvalidArguments, path)
                        .got(name)
                        .variants(names.iter().copied())
                })
        })
        .collect::<Result<Vec<&'static str>, Error>>()?;
    Ok(move |list: BrickList| {
        let mut list = crate::bricks(list);
        for package in left_out {
            list = list.without_package(package);
        }
        list
    })
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
    match engine_core::scene_component(world, WALKER) {
        Some(walker) => walker.read(world, entity, &json!({})),
        None => Ok(()),
    }
}

fn reset_player(world: &mut World, player: Player) {
    if world.get_entity(player.entity).is_err() {
        return;
    }
    edit_transform(world, player.entity, |transform| {
        transform.translation = player.spawn
    })
    .expect("player_has_transform");
    walks(world, player.entity).expect("player_walks");
    reset_interpolation(world, player.entity);
}
