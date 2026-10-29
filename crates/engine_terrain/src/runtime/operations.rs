use super::*;
use engine_core::{Error, ErrorCode, Operation, TerrainDebug};
#[cfg(feature = "schema")]
use engine_settings::{rule::same, stated};

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct TerrainControl {
    #[serde(default)]
    #[cfg_attr(feature = "schema", schemars(transform = stated(same(TerrainSettings::rule_of("subdivision_error")))))]
    pub subdivision_error: Option<f32>,
    #[serde(default)]
    #[cfg_attr(feature = "schema", schemars(transform = stated(same(TerrainSettings::rule_of("pixel_error")))))]
    pub pixel_error: Option<f32>,
    #[serde(default)]
    pub freeze: Option<bool>,
}

#[derive(Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct TerrainControlResult {
    pub temporary: bool,
    pub terrain_changed: bool,
    pub frozen_at: Option<[f32; 3]>,
    pub subdivision_error: Option<f32>,
    pub pixel_error: Option<f32>,
}

fn terrain_setting(value: f32, name: &'static str) -> Result<f32, Error> {
    let rule = TerrainSettings::rule_of(name).expect("terrain settings declare it");
    Setting::check(&value, name, rule)
        .map(|()| value)
        .map_err(|invalid| {
            Error::new(
                ErrorCode::OutOfRange,
                format!("$.{name}"),
                invalid.to_string(),
            )
        })
}

impl Operation for TerrainControl {
    const NAME: &'static str = "terrain";
    const CONTROL: bool = true;
    type Request = Self;

    fn run(world: &mut World, request: Self) -> Result<serde_json::Value, Error> {
        let subdivision_error = request
            .subdivision_error
            .map(|value| terrain_setting(value, "subdivision_error"))
            .transpose()?;
        let pixel_error = request
            .pixel_error
            .map(|value| terrain_setting(value, "pixel_error"))
            .transpose()?;
        if subdivision_error.is_some() || pixel_error.is_some() {
            let mut settings = world
                .get_resource::<TerrainSettings>()
                .copied()
                .unwrap_or_default();
            if let Some(subdivision_error) = subdivision_error {
                settings.subdivision_error = subdivision_error;
            }
            if let Some(pixel_error) = pixel_error {
                settings.pixel_error = pixel_error;
            }
            world.insert_resource(settings);
        }
        let eye = world.resource::<Camera>().eye();
        let mut debug = world
            .get_resource::<TerrainDebug>()
            .copied()
            .unwrap_or_default();
        if let Some(freeze) = request.freeze {
            debug.freeze = freeze.then_some(debug.freeze.unwrap_or(eye));
        }
        world.insert_resource(debug);
        let settings = world.get_resource::<TerrainSettings>();
        let result = TerrainControlResult {
            temporary: true,
            terrain_changed: true,
            frozen_at: debug.freeze.map(|eye| eye.to_array()),
            subdivision_error: settings.map(|settings| settings.subdivision_error),
            pixel_error: settings.map(|settings| settings.pixel_error),
        };
        Ok(serde_json::to_value(result).expect("results serialize"))
    }

    #[cfg(feature = "schema")]
    fn schema() -> schemars::Schema {
        schemars::schema_for!(Self)
    }
}

#[derive(Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(untagged)]
pub enum TerrainKey {
    Authored(String),
    Entity(u64),
}

pub fn terrain_key(world: &World, entity: Entity) -> TerrainKey {
    engine_core::authored_id(world, entity)
        .map_or_else(|| TerrainKey::Entity(entity.to_bits()), TerrainKey::Authored)
}
