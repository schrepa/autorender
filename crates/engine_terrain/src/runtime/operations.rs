use super::*;
use engine_core::bounds;

#[derive(Serialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum TerrainKey {
    Authored(String),
    Entity(u64),
}

pub fn terrain_key(entity: Entity) -> TerrainKey {
    TerrainKey::Entity(entity.to_bits())
}

pub(super) fn out_of_range(invalid: InvalidSetting, path: impl Into<String>) -> Error {
    Error {
        code: ErrorCode::OutOfRange,
        path: path.into(),
        ..Error::from(invalid)
    }
}

pub(super) fn coordinate(value: f64, path: &str) -> Result<f32, Error> {
    match Setting::check(&value, "value", bounds::COORDINATE) {
        Ok(()) => Ok(value as f32),
        Err(invalid) => Err(out_of_range(invalid, path)),
    }
}

pub(super) fn named_terrain(name: Option<&str>) -> Result<(), Error> {
    match name {
        Some(name) => Err(Error::new(ErrorCode::UnknownValue, "$.terrain").got(name)),
        None => Ok(()),
    }
}

pub(super) fn reply(result: impl Serialize) -> serde_json::Value {
    serde_json::to_value(result).expect("results_serialize")
}
