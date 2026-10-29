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

macro_rules! words {
    ([$($word:literal => $value:expr),+ $(,)?]) => {
        &[$($word),+]
    };
    ((named $named:path)) => {
        <$named>::NAMES
    };
}

macro_rules! parse {
    ($value:ident, [$($word:literal => $set:expr),+ $(,)?]) => {
        match $value {
            $($word => Some($set),)+
            _ => None,
        }
    };
    ($value:ident, (named $named:path)) => {
        <$named>::named($value).map(Some)
    };
}

macro_rules! settings {
    ($(
        $field:ident: $ty:ty = $name:literal, $values:tt => $apply:ident;
    )*) => {
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
        pub struct HostSettings {
            $(pub $field: Option<$ty>,)*
        }

        pub const SETTINGS: &[Setting] = &[$(Setting {
            name: $name,
            values: words!($values),
        }),*];

        impl HostSettings {
            pub fn set(&mut self, name: &str, value: &str) -> Result<bool, engine_core::Error> {
                match name {
                    $($name => {
                        let Some(parsed) = parse!(value, $values) else {
                            let values: &[&str] = words!($values);
                            return Err(engine_core::Error::new(engine_core::ErrorCode::InvalidValue, "$")
                                .got(value)
                                .variants(values.iter().copied()));
                        };
                        self.$field = parsed;
                        Ok(true)
                    })*
                    _ => Ok(false),
                }
            }

            pub fn apply<G: Game>(&self, host: &mut GameHost<G>) {
                $(if let Some(value) = self.$field {
                    host.$apply(value);
                })*
            }
        }
    };
}

settings! {
    pipeline: bool = "pipeline",
        ["on" => Some(true), "off" => Some(false)] => set_tick_pipelining;
    taa: bool = "taa",
        ["scene" => None, "on" => Some(true), "off" => Some(false)] => set_temporal_antialiasing;
    occlusion: bool = "occlusion",
        ["scene" => None, "on" => Some(true), "off" => Some(false)] => set_occlusion_culling;
    subpixel: bool = "subpixel",
        ["scene" => None, "on" => Some(true), "off" => Some(false)] => set_subpixel_culling;
    between: bool = "between",
        ["on" => Some(true), "off" => Some(false)] => set_between_frames;
    terrain_clock: TerrainClock = "terrain-clock",
        ["wall" => Some(TerrainClock::Wall), "frames" => Some(TerrainClock::Frames)] => set_terrain_clock;
    debug_view: engine_core::DebugViewMode = "debug-view",
        (named engine_core::DebugViewMode) => set_debug_view;
    dynamic_resolution: bool = "dynamic-resolution",
        ["host" => None, "on" => Some(true), "off" => Some(false)] => set_dynamic_resolution;
}

impl HostSettings {
    pub fn takes(name: &str) -> bool {
        SETTINGS.iter().any(|setting| setting.name == name)
    }
}
