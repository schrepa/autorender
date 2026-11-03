use engine::prelude::*;
use engine::scene::SceneFiles;
use engine::{Error, ErrorCode};
use serde_json::{Value, json};

use crate::land;
use crate::options::{CONTROLS, WATCH_BODIES};
use crate::scene::{self, Furnished, set};
use crate::{Options, RockMaps};

const APPLIED: [&str; 4] = ["lod", "effects", "stand", WATCH_BODIES];

pub struct Highlands {
    options: Options,
    scene: Option<SceneInstance>,
    start: Option<Camera>,
    reflections: Option<Value>,
}

impl Highlands {
    pub fn build(engine: &mut Engine, options: Options, rocks: &RockMaps) -> engine::Result<Self> {
        let world = engine.app.world_mut();
        let furnished = scene::furnish(world, &options, rocks, true)?;
        let mut highlands = Self::new(options.clone());
        highlands.furnished(world, furnished, &options)?;
        Ok(highlands)
    }

    fn new(options: Options) -> Self {
        Self {
            options,
            scene: None,
            start: None,
            reflections: None,
        }
    }

    fn furnished(
        &mut self,
        world: &mut World,
        furnished: Furnished,
        options: &Options,
    ) -> engine::Result<()> {
        let Furnished {
            instance,
            start,
            reflections,
        } = furnished;
        self.scene = Some(instance);
        self.start = Some(start);
        self.reflections = reflections;
        for name in APPLIED {
            self.apply(world, options, name)?;
        }
        Ok(())
    }

    pub fn control(&mut self, engine: &mut Engine, name: &str, value: &str) -> engine::Result<()> {
        if !CONTROLS.contains(&name) {
            return Err(Error::new(ErrorCode::InvalidValue, "$.control")
                .got(name)
                .variants(CONTROLS));
        }
        let mut options = self.options.clone();
        options.set(name, value)?;
        self.apply(engine.app.world_mut(), &options, name)?;
        self.options = options;
        Ok(())
    }

    fn apply(&self, world: &mut World, options: &Options, name: &str) -> engine::Result<()> {
        match name {
            "lod" | "impostors" => {
                set(world, "detail.enabled", json!(options.lod))?;
                set(world, "impostors.enabled", json!(options.impostors))
            }
            "effects" => scene::set_effects(world, &options.effects, self.reflections.as_ref()),
            "stand" => options
                .stand
                .as_deref()
                .map_or(Ok(()), |place| scene::stand(world, place)),
            WATCH_BODIES => {
                land::control(world, json!({"watch_bodies": {"on": options.watch_bodies}}))
            }
            _ => Ok(()),
        }
    }

    fn load(&mut self, world: &mut World, source: &str) -> engine::Result<()> {
        if let Some(instance) = self.scene.take() {
            instance.despawn(world);
        }
        if source != scene::SCENE {
            self.scene = Some(SceneDocument::from_json(source)?.instantiate(world)?);
            return set(world, "fly_camera.enabled", json!(false));
        }
        let options = Options {
            rocks: false,
            ..Options::default()
        };
        let furnished = scene::furnish(world, &options, &RockMaps::default(), false)?;
        self.furnished(world, furnished, &options)
    }
}

engine::__if_dev! {
    impl Highlands {
        #[cfg(any(feature = "dev", target_arch = "wasm32"))]
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
        if let Err(error) = land::control(world, json!("reset")) {
            report_failure("highlands_reset_failed", &error);
        }
        if let Some(start) = self.start {
            world.insert_resource(start);
        }
    }

    fn scene(&self) -> Option<&SceneInstance> {
        self.scene.as_ref()
    }

    fn content(&self, _engine: &Engine) -> Option<Vec<u8>> {
        let options = &self.options;
        let mut content = crate::SOURCE.hash().to_vec();
        content.extend(options.bodies.to_le_bytes());
        content.extend(options.effects.as_bytes());
        content.push(0);
        content.extend([u8::from(options.lod), u8::from(options.impostors)]);
        Some(content)
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
