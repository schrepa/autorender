use bevy_ecs::resource::Resource;
use serde::{Serialize, de::DeserializeOwned};

use crate::{Error, SceneContext};

pub trait Section:
    Resource + Clone + Default + PartialEq + Serialize + DeserializeOwned + schemars::JsonSchema
{
    const KEY: &'static str;

    fn check(&self) -> Result<(), Error> {
        Ok(())
    }

    fn check_scene(&self, _: &SceneContext) -> Result<(), Error> {
        Ok(())
    }

    fn schema() -> schemars::Schema {
        schemars::schema_for!(Self)
    }
}

pub trait Switch: Section {
    const DEFAULT: bool;

    fn on(&self) -> bool;
}

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, serde::Deserialize, schemars::JsonSchema,
)]
pub struct Cut {
    pub asked: u32,
    pub kept: u32,
    pub cut: u32,
}

impl Cut {
    pub fn of(asked: u32, kept: u32) -> Self {
        let kept = kept.min(asked);
        Self {
            asked,
            kept,
            cut: asked - kept,
        }
    }
}
