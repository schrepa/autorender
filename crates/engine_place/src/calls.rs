use serde_json::Value;

pub const SCENE_SCHEMA: &str = "scene.schema.json";

pub fn schema_of<T: schemars::JsonSchema>() -> Value {
    schemars::schema_for!(T).to_value()
}
