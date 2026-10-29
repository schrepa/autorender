use engine::host::TerrainDebug;
use engine::prelude::*;
use engine::{Error, ErrorCode};

use crate::options::{CONTROLS, WATCH_BODIES};
use crate::{Options, RockMaps, scene, views};

const FREEZE_DETAIL: ActionId = ActionId::new(2).expect("action_index_valid");

const APPLIED: [&str; 4] = ["lod", "effects", "stand", WATCH_BODIES];

pub struct Highlands {
    options: Options,
    freeze_held: bool,
    scene: Option<SceneInstance>,
}

impl Highlands {
    pub fn build(options: Options, rocks: &RockMaps) -> engine::Result<(Engine, Self)> {
        let (mut engine, instance) = scene::create(options.bodies, rocks)?;
        for name in APPLIED {
            apply(engine.app.world_mut(), &options, name)?;
        }
        Ok((
            engine,
            Self {
                scene: Some(instance),
                ..Self::new(options)
            },
        ))
    }

    pub(crate) fn new(options: Options) -> Self {
        Self {
            options,
            freeze_held: false,
            scene: None,
        }
    }

    pub(crate) fn furnish(
        app: &mut App,
        options: &Options,
        rocks: &RockMaps,
    ) -> engine::Result<SceneInstance> {
        let instance = scene::furnish(app, options.bodies, rocks, false)?;
        for name in APPLIED {
            apply(app.world_mut(), options, name)?;
        }
        Ok(instance)
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
}

fn apply(world: &mut World, options: &Options, name: &str) -> engine::Result<()> {
    match name {
        "lod" | "impostors" => {
            if let Some(mut detail) = world.get_resource_mut::<LevelOfDetail>() {
                detail.enabled = options.lod;
                detail.impostors = options.impostors;
            }
            Ok(())
        }
        "effects" => scene::set_effects(world, &options.effects),
        "stand" => options
            .stand
            .as_deref()
            .map_or(Ok(()), |place| views::stand(world, place)),
        WATCH_BODIES => {
            views::watch_bodies(world, options.watch_bodies);
            Ok(())
        }
        _ => Ok(()),
    }
}

impl Game for Highlands {
    const ACTIONS: &'static [(&'static str, u32)] = &[("Space", 0), ("KeyL", 2)];

    fn before_frame(&mut self, engine: &mut Engine) {
        let world = engine.app.world_mut();
        let held = world.resource::<ActionInput>().pressed(FREEZE_DETAIL);
        if held && !self.freeze_held {
            let eye = world.resource::<Camera>().eye();
            let mut debug = world
                .get_resource::<TerrainDebug>()
                .copied()
                .unwrap_or_default();
            debug.freeze = match debug.freeze {
                Some(_) => None,
                None => Some(eye),
            };
            world.insert_resource(debug);
        }
        self.freeze_held = held;
    }

    fn reset_camera(&mut self, engine: &mut Engine) {
        views::reset(engine.app.world_mut());
    }

    fn scene(&self) -> Option<&SceneInstance> {
        self.scene.as_ref()
    }

    fn content(&self, _engine: &Engine) -> Option<Vec<u8>> {
        let options = &self.options;
        let mut content = crate::SOURCE.to_vec();
        content.extend(options.bodies.to_le_bytes());
        content.extend(options.effects.as_bytes());
        content.push(0);
        content.extend([u8::from(options.lod), u8::from(options.impostors)]);
        Some(content)
    }
}
