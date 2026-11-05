use bevy_ecs::prelude::World;
use engine_core::{Error, ErrorCode, SceneContext};
use engine_registry::SettingsEntry;
use engine_settings::Expected;
use serde_json::Value;

use super::TerrainClock;
use crate::{Game, GameHost};

#[derive(Clone, Copy, Debug)]
pub struct Setting {
    pub name: &'static str,
    pub values: &'static [&'static str],
}

impl Setting {
    pub fn accepted(&self) -> String {
        self.values.join("|")
    }

    pub fn default_value(&self) -> &'static str {
        self.values[0]
    }
}

pub const OVERLAY: &str = "set";

macro_rules! words {
    ([$($word:literal => $value:expr),+ $(,)?]) => {
        &[$($word),+]
    };
}

macro_rules! parse {
    ($value:ident, [$($word:literal => $set:expr),+ $(,)?]) => {
        match $value {
            $($word => Some($set),)+
            _ => None,
        }
    };
}

macro_rules! settings {
    ($(
        $field:ident: $ty:ty = $name:literal, $values:tt => $apply:ident;
    )*) => {
        #[derive(Clone, Debug, Default, PartialEq)]
        pub struct HostSettings {
            $(pub $field: Option<$ty>,)*
            pub overlay: Vec<(String, Value)>,
        }

        pub const SETTINGS: &[Setting] = &[$(Setting {
            name: $name,
            values: words!($values),
        }),*];

        impl HostSettings {
            pub fn set(&mut self, name: &str, value: &str) -> Result<bool, Error> {
                match name {
                    $($name => {
                        let Some(parsed) = parse!(value, $values) else {
                            let values: &[&str] = words!($values);
                            return Err(Error::new(ErrorCode::InvalidValue, "$")
                                .got(value)
                                .variants(values.iter().copied()));
                        };
                        self.$field = parsed;
                        Ok(true)
                    })*
                    OVERLAY => {
                        self.overlay.push(overlaid(value)?);
                        Ok(true)
                    }
                    _ => Ok(false),
                }
            }

            pub fn apply<G: Game>(&self, host: &mut GameHost<G>) -> Result<(), Error> {
                $(if let Some(value) = self.$field {
                    host.$apply(value);
                })*
                for (key, value) in &self.overlay {
                    host.set_setting(key, value.clone())
                        .map_err(|error| error.within(&format!("{OVERLAY}.{key}")))?;
                }
                Ok(())
            }
        }
    };
}

settings! {
    between: bool = "between",
        ["on" => Some(true), "off" => Some(false)] => set_between_frames;
    terrain_clock: TerrainClock = "terrain-clock",
        ["wall" => Some(TerrainClock::Wall), "frames" => Some(TerrainClock::Frames)] => set_terrain_clock;
}

impl HostSettings {
    pub fn takes(name: &str) -> bool {
        name == OVERLAY || SETTINGS.iter().any(|setting| setting.name == name)
    }
}

fn overlaid(entry: &str) -> Result<(String, Value), Error> {
    let Some((key, value)) = entry.split_once('=').filter(|(key, _)| !key.is_empty()) else {
        return Err(Error::new(ErrorCode::InvalidValue, "$").got(entry));
    };
    let value = engine_settings::parse_json::<Value>(value).map_err(|error| error.within("$"))?;
    Ok((key.to_owned(), value))
}

fn scoped<R>(world: &mut World, run: impl FnOnce(&mut World) -> R) -> R {
    match world.get_resource::<SceneContext>().cloned() {
        Some(context) => context.scope(|| run(world)),
        None => run(world),
    }
}

fn split(key: &str) -> (&str, Option<&str>) {
    match key.split_once('.') {
        Some((section, field)) => (section, Some(field)),
        None => (key, None),
    }
}

fn not_object(section: &str) -> Error {
    Error::new(ErrorCode::InvalidType, format!("$.{section}")).expected(Expected::Object)
}

fn known_field(entry: &SettingsEntry, key: &str, field: &str) -> Result<(), Error> {
    let schema = entry.schema();
    let fields = schema.get("properties").and_then(Value::as_object);
    if fields.is_some_and(|fields| fields.contains_key(field)) {
        return Ok(());
    }
    Err(Error::new(ErrorCode::UnknownField, format!("$.{key}"))
        .got(field)
        .variants(fields.into_iter().flatten().map(|(name, _)| name.as_str())))
}

fn set_agreeing(
    world: &mut World,
    entry: &SettingsEntry,
    value: &Value,
    before: &Value,
) -> Result<(), Error> {
    entry.set(world, value)?;
    let Err(refused) = engine_registry::check_sections(world) else {
        return Ok(());
    };
    entry.set(world, before)?;
    Err(refused)
}

pub fn has_setting(world: &World, key: &str) -> bool {
    let (section, _) = split(key);
    engine_registry::settings(world, section).is_ok()
}

pub fn apply_setting(world: &mut World, key: &str, value: &Value) -> Result<(), Error> {
    let (section, field) = split(key);
    let entry = engine_registry::settings(world, section)?;
    scoped(world, |world| {
        let before = entry.current(world);
        let Some(field) = field else {
            return set_agreeing(world, &entry, value, &before);
        };
        known_field(&entry, key, field)?;
        let mut current = before.clone();
        let Some(fields) = current.as_object_mut() else {
            return Err(not_object(section));
        };
        fields.insert(field.to_owned(), value.clone());
        set_agreeing(world, &entry, &current, &before)
    })
}

pub fn reset_setting(world: &mut World, key: &str) -> Result<(), Error> {
    let (section, field) = split(key);
    let entry = engine_registry::settings(world, section)?;
    let Some(field) = field else {
        entry.reset(world);
        return Ok(());
    };
    scoped(world, |world| {
        let defaults = entry.default_value();
        let mut current = entry.current(world);
        let Some(fields) = current.as_object_mut() else {
            return Err(not_object(section));
        };
        known_field(&entry, key, field)?;
        match defaults.get(field) {
            Some(default) => {
                fields.insert(field.to_owned(), default.clone());
            }
            None => {
                fields.remove(field);
            }
        }
        entry.set(world, &current)
    })
}

pub fn run_operation(world: &mut World, name: &str, request: Value) -> Result<Value, Error> {
    let operation = world
        .get_resource::<engine_registry::Registry>()
        .and_then(|registry| registry.operation(name))
        .copied()
        .ok_or_else(|| {
            Error::new(ErrorCode::UnknownVariant, "$.name")
                .expected(Expected::Present)
                .got(name)
        })?;
    operation
        .run(world, request)
        .map_err(|error| error.within("$.args"))
}
