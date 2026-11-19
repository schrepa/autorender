use serde::{Serialize, de::DeserializeOwned};

use crate::{Error, SceneContext};

pub trait Section:
    Clone
    + Default
    + PartialEq
    + Serialize
    + DeserializeOwned
    + schemars::JsonSchema
    + Send
    + Sync
    + 'static
{
    const KEY: &'static str;

    fn check(&self) -> Result<(), Error> {
        Ok(())
    }

    fn check_scene(&self, _: &SceneContext) -> Result<(), Error> {
        Ok(())
    }
}

pub trait Switch: Section {
    const DEFAULT: bool;

    fn on(&self) -> bool;
}
