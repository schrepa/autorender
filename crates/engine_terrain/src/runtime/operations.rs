use super::*;
use engine_core::{SceneIds, bounds};

#[derive(Serialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum TerrainKey {
    Authored(String),
    Entity(u64),
}

pub fn terrain_key(ids: &SceneIds<'_, '_>, entity: Entity) -> TerrainKey {
    ids.id(entity).map_or_else(
        || TerrainKey::Entity(entity.to_bits()),
        |id| TerrainKey::Authored(id.to_owned()),
    )
}

pub(super) fn out_of_range(invalid: InvalidSetting, path: impl Into<String>) -> Error {
    Error {
        code: ErrorCode::OutOfRange,
        path: path.into(),
        ..Error::from(invalid)
    }
}

pub(super) fn coordinate(value: f64, path: &str) -> Result<f32, Error> {
    within(value, bounds::COORDINATE, path).map(|value| value as f32)
}

pub(super) fn within(value: f64, rule: Rule, path: &str) -> Result<f64, Error> {
    match Setting::check(&value, "value", rule) {
        Ok(()) => Ok(value),
        Err(invalid) => Err(out_of_range(invalid, path)),
    }
}

pub(super) fn terrain_entity(
    scenes: &Query<'_, '_, (Entity, &mut TerrainScene)>,
    ids: &SceneIds<'_, '_>,
    name: Option<&str>,
) -> Result<Entity, Error> {
    let built = |entity: Entity| {
        scenes
            .get(entity)
            .is_ok_and(|(_, scene)| scene.built.is_some())
    };
    match name {
        Some(name) => ids
            .entity(name)
            .filter(|&entity| built(entity))
            .ok_or_else(|| Error::new(ErrorCode::UnknownTerrain, "$.terrain").got(name)),
        None => scenes
            .iter()
            .find(|(_, scene)| scene.built.is_some())
            .map(|(entity, _)| entity)
            .ok_or_else(|| Error::new(ErrorCode::UnknownTerrain, "$.terrain")),
    }
}

pub(super) fn reply(result: impl Serialize) -> serde_json::Value {
    serde_json::to_value(result).expect("results_serialize")
}
