use bevy_ecs::prelude::World;
use engine_core::{Error, ErrorCode};
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
    pipeline: bool = "pipeline",
        ["on" => Some(true), "off" => Some(false)] => set_tick_pipelining;
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

pub fn apply_setting(world: &mut World, key: &str, value: &Value) -> Result<(), Error> {
    engine_registry::settings(world, key)?.set(world, value)?;
    Ok(())
}
