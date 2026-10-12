//! The engine switches every host takes, declared once ([`SETTINGS`]):
//! each one's name, which a page parameter and a native option share, the
//! values it takes, its default first, and what it does. Hosts read them
//! into [`HostSettings`] with [`HostSettings::set`], the one parser, and
//! apply them to the game they run; the page's parameters, the native
//! usage text and `bench/options.json` list them from the table.

use super::TerrainClock;
use crate::{Game, GameHost};

/// An engine switch hosts take.
#[derive(Clone, Copy, Debug)]
pub struct Setting {
    /// The page parameter's and the native option's name.
    pub name: &'static str,
    /// The values it takes, its default first.
    pub values: &'static [&'static str],
    /// What it does.
    pub what: &'static str,
}

impl Setting {
    /// Its values and what it does, as the page and the native host
    /// describe it: `gpu (default) or cpu: how...`.
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

/// Declares the settings: [`HostSettings`]' fields, [`SETTINGS`], the
/// parser and what applying them calls on a game's host.
macro_rules! settings {
    ($(
        $field:ident: $ty:ty = $name:literal, $what:literal,
        [$($word:literal => $value:expr),+ $(,)?] => $apply:ident;
    )*) => {
        /// The engine switches a host was given; one not given keeps what
        /// the game and its scene chose.
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
        pub struct HostSettings {
            $(#[doc = $what] pub $field: Option<$ty>,)*
        }

        /// Every engine switch hosts take.
        pub const SETTINGS: &[Setting] = &[$(Setting {
            name: $name,
            values: &[$($word),+],
            what: $what,
        }),*];

        impl HostSettings {
            /// Sets the switch `name` to `value`; whether `name` is one. A
            /// value it does not take fails, naming those it does.
            pub fn set(&mut self, name: &str, value: &str) -> Result<bool, String> {
                match name {
                    $($name => {
                        self.$field = match value {
                            $($word => $value,)+
                            _ => {
                                return Err(format!(
                                    "{} is {}, not {value}",
                                    $name,
                                    [$($word),+].join(" or ")
                                ));
                            }
                        };
                        Ok(true)
                    })*
                    _ => Ok(false),
                }
            }

            /// Applies the switches given to `host`, in this and later
            /// engines.
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
    terrain_clock: TerrainClock = "terrain-clock",
        "what terrain times its work by: the host's clock, or frames alone, each a slice of samples_per_frame with none between them",
        ["wall" => Some(TerrainClock::Wall), "frames" => Some(TerrainClock::Frames)] => set_terrain_clock;
}

impl HostSettings {
    /// Whether `name` is an engine switch.
    pub fn takes(name: &str) -> bool {
        SETTINGS.iter().any(|setting| setting.name == name)
    }
}
