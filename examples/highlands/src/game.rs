use std::sync::Arc;

use engine::Error;
use engine::host::Image;
use engine::prelude::*;
use engine::scene::SceneFiles;
use engine_fly_camera::FlyCamera;
use serde_json::json;

use crate::bricks::game::HighlandsControl;
use crate::jobs::photographs::{self, Photographs};
use crate::land::{self, GROUND_FILES, GroundFiles, RockMaps, WaterMaps};
use crate::scene;

const ROCK_FOLDER: &str = "highlands/";

const FOAM_FILE: &str = "highlands/water_foam.png";
const RIPPLE_FILE: &str = "highlands/water_ripples.png";

const GAME_FILES: [&str; 8] = [
    "highlands/rocks/boulder_albedo.ktx2",
    "highlands/rocks/boulder_normal.ktx2",
    "highlands/rocks/boulder_orm.ktx2",
    "highlands/rocks/cliff_albedo.ktx2",
    "highlands/rocks/cliff_normal.ktx2",
    "highlands/rocks/cliff_orm.ktx2",
    FOAM_FILE,
    RIPPLE_FILE,
];

const FILE_COUNT: usize = GAME_FILES.len() + GROUND_FILES.len() + photographs::FILES.len();

const ALL_FILES: [&str; FILE_COUNT] = {
    let mut files = [""; FILE_COUNT];
    let mut at = 0;
    while at < files.len() {
        files[at] = if at < GAME_FILES.len() {
            GAME_FILES[at]
        } else if at < GAME_FILES.len() + GROUND_FILES.len() {
            GROUND_FILES[at - GAME_FILES.len()]
        } else {
            photographs::FILES[at - GAME_FILES.len() - GROUND_FILES.len()]
        };
        at += 1;
    }
    files
};

#[derive(Resource)]
pub struct Highlands {
    rocks: RockMaps,
    water: WaterMaps,
    grounds: GroundFiles,
    land: bool,
}

impl Highlands {
    pub fn build(engine: &mut Engine) -> engine::Result<Self> {
        let files = engine.app.world().get_resource::<GameFiles>();
        let photographs = files.map(Photographs::read).unwrap_or_default();
        let highlands = Self {
            rocks: files.map(rocks).unwrap_or_default(),
            water: files.map(water).unwrap_or_default(),
            grounds: files.map(grounds).unwrap_or_default(),
            land: false,
        };
        engine.app.world_mut().insert_resource(photographs);
        Ok(highlands)
    }
}

fn water(files: &GameFiles) -> WaterMaps {
    let image = |path: &str| -> Option<Image> {
        engine::host::decode_image(path, files.bytes(path)?).ok()
    };
    WaterMaps {
        foam: image(FOAM_FILE),
        ripples: image(RIPPLE_FILE),
    }
}

fn grounds(files: &GameFiles) -> GroundFiles {
    GROUND_FILES
        .iter()
        .filter_map(|&path| Some((path, Arc::from(files.bytes(path)?))))
        .collect()
}

fn rocks(files: &GameFiles) -> RockMaps {
    GAME_FILES
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
    let (rocks, water, grounds) = {
        let mut highlands = world.resource_mut::<Highlands>();
        if highlands.land {
            return Ok(());
        }
        let rocks = match arrival {
            Arrival::Opening => std::mem::take(&mut highlands.rocks),
            Arrival::Reload => RockMaps::default(),
        };
        (
            rocks,
            std::mem::take(&mut highlands.water),
            std::mem::take(&mut highlands.grounds),
        )
    };
    land::provide(world, &rocks, water, &grounds)?;
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
    const FILES: &'static [&'static str] = &ALL_FILES;

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
        if let Err(error) = land::control(world, HighlandsControl::Reset) {
            report_failure("highlands_reset_failed", &error);
        }
        scene.place_camera(world);
    }

    fn content(_engine: &Engine, _scene: &LoadedScene) -> Option<Vec<u8>> {
        Some(crate::SOURCE.hash().to_vec())
    }
}
