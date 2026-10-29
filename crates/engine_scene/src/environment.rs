use std::collections::BTreeMap;

use engine_core::{
    Atmosphere, BrickRegistry, Error, SettingsEntry, bevy_ecs::prelude::World, glam::Vec3,
};
use engine_settings::ErrorCode;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum AtmospherePreset {
    Day,
    #[default]
    GoldenHour,
    Dusk,
    Night,
}

impl AtmospherePreset {
    pub fn atmosphere(self) -> Atmosphere {
        match self {
            Self::Day => Atmosphere::day(),
            Self::GoldenHour => Atmosphere::golden_hour(),
            Self::Dusk => Atmosphere::dusk(),
            Self::Night => Atmosphere::night(),
        }
    }
}

pub trait AuthoredValue: Copy {
    type Authored: Copy;
    fn authored(self) -> Self::Authored;
    fn from_authored(value: Self::Authored) -> Self;
}

impl AuthoredValue for f32 {
    type Authored = f32;

    fn authored(self) -> f32 {
        self
    }

    fn from_authored(value: f32) -> Self {
        value
    }
}

impl AuthoredValue for Vec3 {
    type Authored = [f32; 3];

    fn authored(self) -> [f32; 3] {
        self.to_array()
    }

    fn from_authored(value: [f32; 3]) -> Self {
        Vec3::from_array(value)
    }
}

macro_rules! scene_atmosphere {
    ($($(#[$doc:meta])* $field:ident: $ty:ident, $rule:ident $(($($arg:expr),*))?;)*) => {
        #[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
        #[cfg_attr(
            feature = "schema",
            derive(schemars::JsonSchema),
            schemars(transform = engine_core::state_atmosphere_rules)
        )]
        #[serde(default, deny_unknown_fields)]
        pub struct SceneAtmosphere {
            pub preset: AtmospherePreset,
            $(
                $(#[$doc])*
                #[serde(skip_serializing_if = "Option::is_none")]
                pub $field: Option<<$ty as AuthoredValue>::Authored>,
            )*
        }

        impl From<&Atmosphere> for SceneAtmosphere {
            fn from(atmosphere: &Atmosphere) -> Self {
                Self {
                    preset: AtmospherePreset::GoldenHour,
                    $($field: Some(atmosphere.$field.authored()),)*
                }
            }
        }

        impl SceneAtmosphere {
            pub fn atmosphere(&self) -> Atmosphere {
                let mut atmosphere = self.preset.atmosphere();
                $(if let Some(value) = self.$field {
                    atmosphere.$field = AuthoredValue::from_authored(value);
                })*
                atmosphere
            }
        }
    };
}

engine_core::atmosphere_fields!(scene_atmosphere);

impl SceneAtmosphere {
    pub fn resolve(&self, path: &str) -> Result<Atmosphere, Error> {
        let atmosphere = self.atmosphere();
        atmosphere
            .validate()
            .map_err(|invalid| invalid.error(ErrorCode::InvalidEnvironment, path))?;
        Ok(atmosphere)
    }
}

const ATMOSPHERE: &str = "atmosphere";

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SceneEnvironment {
    pub sections: BTreeMap<String, Value>,
}

impl SceneEnvironment {
    pub fn current(world: &World) -> Self {
        let atmosphere = world
            .get_resource::<Atmosphere>()
            .copied()
            .unwrap_or_default();
        let mut sections = BTreeMap::new();
        let atmosphere = serde_json::to_value(SceneAtmosphere::from(&atmosphere))
            .expect("atmosphere_serializes");
        sections.insert(ATMOSPHERE.to_owned(), atmosphere);
        for section in registered(world) {
            sections.insert(section.key.to_owned(), section.current(world));
        }
        Self { sections }
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.sections
            .iter()
            .filter(|(_, value)| !value.is_null())
            .map(|(key, _)| key.as_str())
    }

    pub fn ordered_keys(&self, world: &World) -> Vec<String> {
        let mut keys: Vec<String> = Vec::new();
        if self.value(ATMOSPHERE).is_some() {
            keys.push(ATMOSPHERE.to_owned());
        }
        for section in registered(world) {
            if self.value(section.key).is_some() {
                keys.push(section.key.to_owned());
            }
        }
        for key in self.keys() {
            if !keys.iter().any(|known| known == key) {
                keys.push(key.to_owned());
            }
        }
        keys
    }

    pub fn value(&self, key: &str) -> Option<&Value> {
        self.sections.get(key).filter(|value| !value.is_null())
    }

    pub fn atmosphere(&self) -> Option<SceneAtmosphere> {
        self.value(ATMOSPHERE)
            .and_then(|value| engine_core::parse_value(value.clone()).ok())
    }

    pub fn opt_in_sections(world: &World) -> Vec<&'static str> {
        registered(world)
            .filter(|section| section.opt_in)
            .map(|section| section.key)
            .collect()
    }

    pub fn validate(&self, path: &str) -> Result<(), Error> {
        if let Some(value) = self.value(ATMOSPHERE) {
            let path = format!("{path}.{ATMOSPHERE}");
            let atmosphere: SceneAtmosphere = engine_core::parse_value_at(value.clone(), &path)?;
            atmosphere.resolve(&path)?;
        }
        Ok(())
    }

    pub fn check(&self, world: &World, path: &str) -> Result<(), Error> {
        self.validate(path)?;
        for key in self.keys() {
            if key == ATMOSPHERE {
                continue;
            }
            let at = format!("{path}.{key}");
            let section = section_of(world, key, &at)?;
            section.check(self.sections[key].clone(), &at)?;
        }
        Ok(())
    }

    pub fn apply_sections(&self, world: &mut World, path: &str) -> Result<(), Error> {
        self.check(world, path)?;
        if let Some(atmosphere) = self.atmosphere() {
            world.insert_resource(atmosphere.atmosphere());
        }
        for key in self.keys() {
            if key == ATMOSPHERE {
                continue;
            }
            let at = format!("{path}.{key}");
            let section = *section_of(world, key, &at)?;
            section.insert(world, self.sections[key].clone(), &at)?;
        }
        Ok(())
    }

    pub(crate) fn apply_all(&self, world: &mut World, path: &str) -> Result<(), Error> {
        world.insert_resource(
            self.atmosphere()
                .map_or_else(Atmosphere::default, |atmosphere| atmosphere.atmosphere()),
        );
        let sections: Vec<SettingsEntry> = registered(world).copied().collect();
        for section in sections {
            match self.value(section.key) {
                Some(value) => {
                    section.insert(world, value.clone(), &format!("{path}.{}", section.key))?;
                }
                None if section.opt_in => section.remove(world),
                None => section.reset(world),
            }
        }
        Ok(())
    }
}

fn registered(world: &World) -> impl Iterator<Item = &SettingsEntry> {
    world
        .get_resource::<BrickRegistry>()
        .into_iter()
        .flat_map(BrickRegistry::sections)
}

fn section_of<'w>(world: &'w World, key: &str, path: &str) -> Result<&'w SettingsEntry, Error> {
    world
        .get_resource::<BrickRegistry>()
        .and_then(|registry| registry.section(key))
        .ok_or_else(|| {
            Error::new(ErrorCode::MissingPlugin, path)
                .got(key)
                .variants(
                    std::iter::once(ATMOSPHERE)
                        .chain(registered(world).map(|section| section.key)),
                )
        })
}

#[cfg(feature = "schema")]
impl schemars::JsonSchema for SceneEnvironment {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "SceneEnvironment".into()
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        let atmosphere = generator.subschema_for::<Option<SceneAtmosphere>>();
        schemars::json_schema!({
            "type": "object",
            "properties": { "atmosphere": atmosphere },
            "additionalProperties": false,
        })
    }
}

#[cfg(feature = "schema")]
pub fn section_schemas(
    registry: &BrickRegistry,
    definitions: &mut serde_json::Map<String, Value>,
) -> serde_json::Map<String, Value> {
    let mut properties = serde_json::Map::new();
    for section in registry.sections() {
        let mut schema = serde_json::to_value(section.schema()).expect("schema_serializes");
        let Some(fields) = schema.as_object_mut() else {
            continue;
        };
        fields.remove("$schema");
        if let Some(Value::Object(nested)) = fields.remove("$defs") {
            for (name, definition) in nested {
                definitions.entry(name).or_insert(definition);
            }
        }
        let title = fields
            .remove("title")
            .and_then(|title| title.as_str().map(str::to_owned))
            .unwrap_or_else(|| section.name.to_owned());
        definitions.insert(title.clone(), schema);
        properties.insert(
            section.key.to_owned(),
            serde_json::json!({
                "anyOf": [{ "$ref": format!("#/$defs/{title}") }, { "type": "null" }],
            }),
        );
    }
    properties
}
