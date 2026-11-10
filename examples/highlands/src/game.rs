use engine::prelude::*;
use engine::scene::SceneFiles;
use engine::Error;
use engine_fly_camera::FlyCamera;
use serde_json::json;

use crate::land::{self, RockMaps};
use crate::scene;

const ROCK_FOLDER: &str = "highlands/";

const ROCK_FILES: [&str; 6] = [
    "highlands/rocks/boulder_albedo.ktx2",
    "highlands/rocks/boulder_normal.ktx2",
    "highlands/rocks/boulder_orm.ktx2",
    "highlands/rocks/cliff_albedo.ktx2",
    "highlands/rocks/cliff_normal.ktx2",
    "highlands/rocks/cliff_orm.ktx2",
];

#[derive(Resource)]
pub struct Highlands {
    rocks: RockMaps,
    land: bool,
}

impl Highlands {
    pub fn build(engine: &mut Engine) -> engine::Result<Self> {
        let rocks = engine
            .app
            .world()
            .get_resource::<GameFiles>()
            .map(rocks)
            .unwrap_or_default();
        Ok(Self { rocks, land: false })
    }
}

fn rocks(files: &GameFiles) -> RockMaps {
    ROCK_FILES
        .iter()
        .filter_map(|path| {
            let bytes = files.bytes(path)?;
            if engine::host::image_format(bytes).map(|format| format.name) != Some("ktx2-rgba8") {
                return None;
            }
            let image = engine::host::decode_image(path, bytes).ok()?;
            Some((path.strip_prefix(ROCK_FOLDER)?.to_owned(), image))
        })
        .collect()
}

fn provide_land(world: &mut World, arrival: Arrival) -> engine::Result<()> {
    let rocks = {
        let mut highlands = world.resource_mut::<Highlands>();
        if highlands.land {
            return Ok(());
        }
        match arrival {
            Arrival::Opening => std::mem::take(&mut highlands.rocks),
            Arrival::Reload => RockMaps::default(),
        }
    };
    land::provide(world, &rocks)?;
    world.resource_mut::<Highlands>().land = true;
    Ok(())
}

fn report_failure(event: &str, error: &Error) {
    let record = json!({
        "event": event,
        "error": error,
    });
    eprintln!("{record}");
}

impl Game for Highlands {
    const FILES: &'static [&'static str] = &ROCK_FILES;

    #[cfg(feature = "dev")]
    const SMOKES: &'static [Smoke] = &crate::smokes::SMOKES;

    fn load_scene(
        engine: &mut Engine,
        document: &SceneDocument,
        files: &SceneFiles,
        arrival: Arrival,
    ) -> Result<SceneInstance, Error> {
        let world = engine.app.world_mut();
        let highlands = engine::scene::parse_json::<SceneDocument>(scene::SCENE)?;
        match arrival {
            Arrival::Opening => {
                provide_land(world, arrival)?;
                scene::furnish(world, document, files, true)
            }
            Arrival::Reload if document.id == highlands.id => {
                provide_land(world, arrival)?;
                scene::furnish(world, document, files, false)
            }
            Arrival::Reload => {
                let instance = document.instantiate_with(world, files)?;
                world
                    .section::<FlyCamera>()
                    .set(|fly| fly.enabled = false)?;
                Ok(instance)
            }
        }
    }

    fn reset_camera(engine: &mut Engine, scene: &LoadedScene) {
        let world = engine.app.world_mut();
        if let Err(error) = land::control(world, json!("reset")) {
            report_failure("highlands_reset_failed", &error);
        }
        scene.place_camera(world);
    }

    fn content(_engine: &Engine, _scene: &LoadedScene) -> Option<Vec<u8>> {
        Some(crate::SOURCE.hash().to_vec())
    }
}
