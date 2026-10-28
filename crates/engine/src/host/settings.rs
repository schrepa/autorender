use super::TerrainClock;
use crate::{Game, GameHost};

#[derive(Clone, Copy, Debug)]
pub struct Setting {
    pub name: &'static str,
    pub values: &'static [&'static str],
    pub what: &'static str,
}

impl Setting {
    pub fn described(&self) -> String {
        let mut values: Vec<String> = self.values.iter().map(|&value| value.into()).collect();
        if let Some(default) = values.first_mut() {
            default.push_str(" (default)");
        }
        let last = values.pop().unwrap_or_default();
        if values.is_empty() {
            format!("{last}: {}", self.what)
        } else {
            format!("{} or {last}: {}", values.join(", "), self.what)
        }
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
        $field:ident: $ty:ty = $name:literal, $what:literal, $values:tt => $apply:ident;
    )*) => {
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
        pub struct HostSettings {
            $(#[doc = $what] pub $field: Option<$ty>,)*
        }

        pub const SETTINGS: &[Setting] = &[$(Setting {
            name: $name,
            values: words!($values),
            what: $what,
        }),*];

        impl HostSettings {
            pub fn set(&mut self, name: &str, value: &str) -> Result<bool, String> {
                match name {
                    $($name => {
                        let Some(parsed) = parse!(value, $values) else {
                            let values: &[&str] = words!($values);
                            return Err(format!("{} is {}, not {value}", $name, values.join(" or ")));
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
    presentation: bool = "presentation",
        "how ordinary frames are presented: blended, composed and culled on the GPU, or exactly on the CPU",
        ["gpu" => Some(true), "cpu" => Some(false)] => set_gpu_presentation;
    pipeline: bool = "pipeline",
        "fixed ticks split across the frames around their due time, on GPU-presented frames",
        ["on" => Some(true), "off" => Some(false)] => set_tick_pipelining;
    taa: bool = "taa",
        "temporal antialiasing of GPU-presented frames, over the scene's setting",
        ["scene" => None, "on" => Some(true), "off" => Some(false)] => set_temporal_antialiasing;
    occlusion: bool = "occlusion",
        "occlusion culling of GPU-presented frames, over the scene's setting",
        ["scene" => None, "on" => Some(true), "off" => Some(false)] => set_occlusion_culling;
    subpixel: bool = "subpixel",
        "sub-pixel culling of GPU-presented frames, over the scene's setting",
        ["scene" => None, "on" => Some(true), "off" => Some(false)] => set_subpixel_culling;
    between: bool = "between",
        "terrain loading between frames, in the time they leave free, or within frames alone",
        ["on" => Some(true), "off" => Some(false)] => set_between_frames;
    terrain_clock: TerrainClock = "terrain-clock",
        "what terrain times its work by: the host's clock, or frames alone, each a slice of samples_per_frame with none between them",
        ["wall" => Some(TerrainClock::Wall), "frames" => Some(TerrainClock::Frames)] => set_terrain_clock;
    debug_view: engine_core::DebugViewMode = "debug-view",
        "the debug view frames show in place of the shaded image, over the scene's",
        (named engine_core::DebugViewMode) => set_debug_view;
    dynamic_resolution: bool = "dynamic-resolution",
        "dynamic resolution of display frames, over the host's choice: on for play in a window or a page, off offscreen and in measured runs",
        ["host" => None, "on" => Some(true), "off" => Some(false)] => set_dynamic_resolution;
}

impl HostSettings {
    pub fn takes(name: &str) -> bool {
        SETTINGS.iter().any(|setting| setting.name == name)
    }
}
