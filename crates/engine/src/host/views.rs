use bevy_ecs::prelude::World;
use engine_core::{Error, Operation};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneViewControl {
    pub name: String,
}

impl Operation for SceneViewControl {
    const NAME: &'static str = "scene_view";
    const CONTROL: bool = true;
    const SHOWS: &'static str = "Camera";
    type Request = Self;

    fn run(world: &mut World, request: Self) -> Result<Value, Error> {
        show_view(world, &request.name, "$.name")?;
        Ok(json!({ "temporary": true, "view": request.name }))
    }

    #[cfg(feature = "schema")]
    fn schema() -> schemars::Schema {
        schemars::schema_for!(SceneViewControl)
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneLookControl {
    pub name: String,
}

impl Operation for SceneLookControl {
    const NAME: &'static str = "scene_look";
    const CONTROL: bool = true;
    const SHOWS: &'static str = "Environment";
    type Request = Self;

    fn run(world: &mut World, request: Self) -> Result<Value, Error> {
        show_look(world, &request.name, "$.name")?;
        Ok(json!({ "temporary": true, "look": request.name }))
    }

    #[cfg(feature = "schema")]
    fn schema() -> schemars::Schema {
        schemars::schema_for!(SceneLookControl)
    }
}

pub fn show_view(world: &mut World, name: &str, path: &str) -> Result<(), Error> {
    engine_scene::show_view(world, name, path)?;
    Ok(())
}

pub fn show_look(world: &mut World, name: &str, path: &str) -> Result<(), Error> {
    engine_scene::show_look(world, name, path)
}
