use engine::prelude::*;
use engine::scene::SceneFiles;
use engine::{Error, ErrorCode};
use serde_json::json;

use crate::bricks::game::HighlandsControl;
use crate::land;
use crate::options::CONTROLS;
use crate::scene::{self, set};
use crate::{Options, RockMaps};

pub struct Highlands {
    options: Options,
    scene: Option<SceneInstance>,
    start: Option<(Transform, Camera)>,
    land: bool,
}

impl Highlands {
    pub fn build(
        engine: &mut Engine,
        options: Options,
        rocks: &RockMaps,
        source: &str,
    ) -> engine::Result<Self> {
        let document: SceneDocument = engine::scene::parse_json(source)?;
        let world = engine.app.world_mut();
        land::provide(world, rocks)?;
        let mut highlands = Self::new(options);
        highlands.land = true;
        highlands.furnish(world, &document, true)?;
        Ok(highlands)
    }

    fn new(options: Options) -> Self {
        Self {
            options,
            scene: None,
            start: None,
            land: false,
        }
    }

    fn furnish(
        &mut self,
        world: &mut World,
        document: &SceneDocument,
        flying: bool,
    ) -> engine::Result<()> {
        let (instance, ()) = document.instantiate_then(world, &SceneFiles::default(), |world, _| {
            scene::furnish(world, document, flying)
        })?;
        if let Some(previous) = self.scene.replace(instance) {
            previous.despawn(world);
        }
        self.start = scene::start(world);
        watch_bodies(world, &self.options)
    }

    pub fn control(&mut self, engine: &mut Engine, name: &str, value: &str) -> engine::Result<()> {
        if !CONTROLS.contains(&name) {
            return Err(Error::new(ErrorCode::InvalidValue, "$.control")
                .got(name)
                .variants(CONTROLS));
        }
        let mut options = self.options.clone();
        options.set(name, value)?;
        watch_bodies(engine.app.world_mut(), &options)?;
        self.options = options;
        Ok(())
    }

    fn load(&mut self, world: &mut World, source: &str) -> engine::Result<()> {
        let document = SceneDocument::from_json(source)?;
        if document.id == engine::scene::parse_json::<SceneDocument>(scene::SCENE)?.id {
            if !self.land {
                land::provide(world, &RockMaps::default())?;
                self.land = true;
            }
            return self.furnish(world, &document, false);
        }
        let instance = document.instantiate(world)?;
        if let Some(previous) = self.scene.replace(instance) {
            previous.despawn(world);
        }
        set(world, "fly_camera.enabled", json!(false))
    }
}

engine::__if_dev! {
    impl Highlands {
        #[cfg(all(feature = "dev", not(target_arch = "wasm32")))]
        pub(crate) fn played(
            options: Options,
            source: Option<&str>,
            engine: &mut Engine,
        ) -> engine::Result<Self> {
            let mut highlands = Self::new(options);
            if let Some(source) = source {
                highlands.load(engine.app.world_mut(), source)?;
            }
            Ok(highlands)
        }
    }
}

fn watch_bodies(world: &mut World, options: &Options) -> engine::Result<()> {
    land::control(
        world,
        HighlandsControl::WatchBodies {
            on: options.watch_bodies,
        },
    )
}

fn report_failure(event: &str, error: &Error) {
    let record = json!({
        "event": event,
        "error": error,
    });
    eprintln!("{record}");
}

impl Game for Highlands {
    fn reset_camera(&mut self, engine: &mut Engine) {
        let world = engine.app.world_mut();
        if let Err(error) = land::control(world, HighlandsControl::Reset) {
            report_failure("highlands_reset_failed", &error);
        }
        if let Some((at, lens)) = self.start
            && let Some(camera) = main_camera(world)
        {
            place_camera(world, camera, at, lens);
        }
    }

    fn scene(&self) -> Option<&SceneInstance> {
        self.scene.as_ref()
    }

    fn content(&self, _engine: &Engine) -> Option<Vec<u8>> {
        Some(crate::SOURCE.hash().to_vec())
    }

    fn reload_scene(
        &mut self,
        engine: &mut Engine,
        source: &str,
        _files: &SceneFiles,
    ) -> Result<(), Error> {
        self.load(engine.app.world_mut(), source)
    }
}
