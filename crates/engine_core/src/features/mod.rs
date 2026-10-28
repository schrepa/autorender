mod navigation;
mod terrain;
mod time_of_day;
mod world_streaming;

pub use navigation::*;
pub use terrain::*;
pub use time_of_day::*;
pub use world_streaming::*;

use bevy_app::App;
use bevy_ecs::prelude::{Component, Resource, World};
use serde::{Serialize, Serializer};

#[macro_export]
macro_rules! engine_features {
    ($then:ident) => {
        $then! {
            terrain: Terrain => TerrainPlugin, needs [];
            world_streaming: WorldStreaming => WorldStreamingPlugin, needs [];
            navigation: Navigation => NavigationPlugin, needs [Terrain];
            time_of_day: TimeOfDay => TimeOfDayPlugin, needs [];
            collision: Collision => CollisionPlugin, needs [];
            physics: Physics => PhysicsPlugin, needs [Collision];
        }
    };
}

macro_rules! core_features {
    ($($(#[$doc:meta])* $name:ident: $variant:ident => $plugin:ident, needs [$($need:ident),*];)*) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum Feature {
            $(
                $(#[$doc])*
                $variant,
            )*
        }

        impl Feature {
            pub const ALL: [Self; [$(stringify!($name)),*].len()] = [$(Self::$variant),*];

            pub const fn plugin(self) -> &'static str {
                match self {
                    $(Self::$variant => stringify!($plugin),)*
                }
            }

            pub const fn name(self) -> &'static str {
                match self {
                    $(Self::$variant => stringify!($name),)*
                }
            }

            pub const fn needs(self) -> &'static [Self] {
                match self {
                    $(Self::$variant => &[$(Self::$need),*],)*
                }
            }
        }
    };
}

engine_features!(core_features);

impl Serialize for Feature {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.name())
    }
}

impl Feature {
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|feature| feature.name() == name)
    }

    const fn bit(self) -> u32 {
        1 << self as u32
    }
}

pub fn install(app: &mut App, feature: Feature) {
    let world = app.world_mut();
    world.resource_mut::<Features>().insert(feature);
    crate::environment::init_sections(world, Some(feature));
}

pub(crate) fn check_needs(world: &World) {
    let Some(&features) = world.get_resource::<Features>() else {
        return;
    };
    for feature in features.iter() {
        for &need in feature.needs() {
            assert!(
                features.has(need),
                "{} needs {}: add {} too, or leave {} out",
                feature.plugin(),
                need.name(),
                need.plugin(),
                feature.name()
            );
        }
    }
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Features(u32);

impl Features {
    pub fn has(self, feature: Feature) -> bool {
        self.0 & feature.bit() != 0
    }

    pub fn insert(&mut self, feature: Feature) {
        self.0 |= feature.bit();
    }

    pub fn iter(self) -> impl Iterator<Item = Feature> {
        Feature::ALL
            .into_iter()
            .filter(move |&feature| self.has(feature))
    }
}

pub fn has_feature(world: &World, feature: Feature) -> bool {
    world
        .get_resource::<Features>()
        .is_some_and(|features| features.has(feature))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroundSample {
    pub height: f32,
    pub level: u32,
}

#[derive(Resource, Clone, Copy)]
pub struct Ground {
    pub sample: fn(&World, f32, f32) -> Option<GroundSample>,
    pub revision: fn(&World) -> u64,
}

impl Ground {
    pub fn sample_in(world: &World, x: f32, z: f32) -> Option<GroundSample> {
        let ground = world.get_resource::<Self>()?;
        (ground.sample)(world, x, z)
    }
}

#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HeightHeld(pub bool);

#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StreamingFocus;
