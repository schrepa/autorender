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
