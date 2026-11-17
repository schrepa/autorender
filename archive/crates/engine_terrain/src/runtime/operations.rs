use super::*;
use engine_core::SceneIds;

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

const COORDINATE: Rule = Rule::Range(-1.0e6, 1.0e6);

pub(super) fn coordinate(value: f64, path: &str) -> Result<f32, Error> {
    within(value, COORDINATE, path).map(|value| value as f32)
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
        Some(name) => match ids.named(name, "$.terrain") {
            Err(error) if error.code == ErrorCode::EntityRemoved => Err(error),
            Ok(entity) if built(entity) => Ok(entity),
            Ok(_) | Err(_) => Err(Error::new(UNKNOWN_TERRAIN, "$.terrain").got(name)),
        },
        None => scenes
            .iter()
            .find(|(_, scene)| scene.built.is_some())
            .map(|(entity, _)| entity)
            .ok_or_else(|| Error::new(UNKNOWN_TERRAIN, "$.terrain")),
    }
}
