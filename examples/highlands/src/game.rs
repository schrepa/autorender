use engine::host::TerrainDebug;
use engine::prelude::*;
use engine::{Error, ErrorCode};

use crate::options::CONTROLS;
use crate::views::set_view;
use crate::{Options, RockMaps, scene};

const FREEZE_DETAIL: ActionId = ActionId::new(2).expect("an action's index");

pub struct Highlands {
    options: Options,
    freeze_held: bool,
}

impl Highlands {
    pub fn build(options: Options, rocks: &RockMaps) -> engine::Result<(Engine, Self)> {
        let mut engine = scene::create(options.bodies, rocks)?;
        for name in ["lod", "look", "effects", "view"] {
            apply(&mut engine, &options, name)?;
        }
        Ok((
            engine,
            Self {
                options,
                freeze_held: false,
            },
        ))
    }

    pub fn control(&mut self, engine: &mut Engine, name: &str, value: &str) -> engine::Result<()> {
        if !CONTROLS.contains(&name) {
            return Err(Error::new(
                ErrorCode::InvalidValue,
                "$.control",
                CONTROLS.join("|"),
            ));
        }
        let mut options = self.options.clone();
        options.set(name, value)?;
        apply(engine, &options, name)?;
        self.options = options;
        Ok(())
    }
}

fn apply(engine: &mut Engine, options: &Options, name: &str) -> engine::Result<()> {
    match name {
        "lod" | "impostors" => {
            if let Some(mut detail) = engine.app.world_mut().get_resource_mut::<LevelOfDetail>() {
                detail.enabled = options.lod;
                detail.impostors = options.impostors;
            }
            Ok(())
        }
        "look" => scene::set_look(engine, &options.look),
        "effects" => scene::set_effects(engine, &options.effects, &options.look),
        "view" => options
            .view
            .as_deref()
            .map_or(Ok(()), |view| set_view(engine, view)),
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
        set_view(engine, "overview").expect("known view");
    }

    fn content(&self, _engine: &Engine) -> Option<Vec<u8>> {
        let options = &self.options;
        let mut content = crate::SOURCE.to_vec();
        content.extend(options.bodies.to_le_bytes());
        for choice in [&options.look, &options.effects] {
            content.extend(choice.as_bytes());
            content.push(0);
        }
        content.extend([u8::from(options.lod), u8::from(options.impostors)]);
        Some(content)
    }
}
