use engine::prelude::*;
use engine::scene::SceneFiles;
use engine::{Error, ErrorCode};
use serde_json::{Value, json};

use crate::options::{CONTROLS, WATCH_BODIES};
use crate::scene::{self, Furnished, set};
use crate::{Options, RockMaps};

const FREEZE_DETAIL: ActionId = ActionId::new(2).expect("action_index_valid");

const APPLIED: [&str; 4] = ["lod", "effects", "stand", WATCH_BODIES];

pub struct Highlands {
    options: Options,
    freeze_held: bool,
    scene: Option<SceneInstance>,
    start: Option<Camera>,
}

impl Highlands {
    pub fn build(engine: &mut Engine, options: Options, rocks: &RockMaps) -> engine::Result<Self> {
        let world = engine.app.world_mut();
        let Furnished { instance, start } = scene::furnish(world, &options, rocks, true)?;
        for name in APPLIED {
            apply(world, &options, name)?;
        }
        Ok(Self {
            scene: Some(instance),
            start: Some(start),
            ..Self::new(options)
        })
    }

    fn new(options: Options) -> Self {
        Self {
            options,
            freeze_held: false,
            scene: None,
            start: None,
        }
    }

    pub fn control(&mut self, engine: &mut Engine, name: &str, value: &str) -> engine::Result<()> {
        if !CONTROLS.contains(&name) {
            return Err(Error::new(ErrorCode::InvalidValue, "$.control")
                .got(name)
                .variants(CONTROLS));
        }
        let mut options = self.options.clone();
        options.set(name, value)?;
        apply(engine.app.world_mut(), &options, name)?;
        self.options = options;
        Ok(())
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
        let Furnished { instance, start } =
            scene::furnish(world, &options, &RockMaps::default(), false)?;
        for name in APPLIED {
            apply(world, &options, name)?;
        }
        self.scene = Some(instance);
        self.start = Some(start);
        Ok(())
    }
}

engine::__if_dev! {
    impl Highlands {
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

fn apply(world: &mut World, options: &Options, name: &str) -> engine::Result<()> {
    match name {
        "lod" | "impostors" => {
            set(world, "detail.enabled", json!(options.lod))?;
            set(world, "impostors.enabled", json!(options.impostors))
        }
        "effects" => scene::set_effects(world, &options.effects),
        "stand" => options
            .stand
            .as_deref()
            .map_or(Ok(()), |place| scene::stand(world, place)),
        WATCH_BODIES => set(world, "highlands.watch_bodies", json!(options.watch_bodies)),
        _ => Ok(()),
    }
}

fn freeze_detail(world: &mut World) -> engine::Result<()> {
    let view = engine::host::run_operation(world, "view", json!({}))?;
    let freeze = view["frozen_at"].is_null();
    engine::host::run_operation(world, "view", json!({ "freeze": freeze })).map(drop)
}

fn report_failure(event: &str, error: &Error) {
    let record = json!({
        "event": event,
        "error": error,
    });
    eprintln!("{record}");
}

impl Game for Highlands {
    const ACTIONS: &'static [(&'static str, u32)] = &[("Space", 0), ("KeyL", 2)];

    fn before_frame(&mut self, engine: &mut Engine) {
        let world = engine.app.world_mut();
        let held = world.resource::<ActionInput>().pressed(FREEZE_DETAIL);
        if held
            && !self.freeze_held
            && let Err(error) = freeze_detail(world)
        {
            report_failure("highlands_freeze_failed", &error);
        }
        self.freeze_held = held;
    }

    fn reset_camera(&mut self, engine: &mut Engine) {
        let world = engine.app.world_mut();
        let cleared = set(world, "highlands.stand", Value::Null)
            .and_then(|()| set(world, "highlands.watch_bodies", json!(false)));
        if let Err(error) = cleared {
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
