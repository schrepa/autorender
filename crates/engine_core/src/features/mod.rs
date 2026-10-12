//! The engine's features: the one list of them, what each plugin of the
//! engine's bundle adds, the scenes, environment sections and development
//! tools that name them, and the ground the features that stand things on
//! it read. A feature's folder here holds the feature: its settings,
//! components, systems and plugin.

mod animation;
mod deformation;
mod navigation;
mod ocean;
mod terrain;
mod time_of_day;
mod world_streaming;

pub use animation::*;
pub use deformation::*;
pub use navigation::*;
pub use ocean::*;
pub use terrain::*;
pub use time_of_day::*;
pub use world_streaming::*;

use bevy_app::App;
use bevy_ecs::prelude::{Component, Resource, World};
use serde::{Serialize, Serializer};

/// Every engine feature in the bundle's order: the one list of them. Each
/// entry is `name: Variant => Plugin, needs [Variant, ...];`, the feature's
/// name as scenes and development replies write it, its [`Feature`]
/// variant, the plugin adding it and the features whose plugins that plugin
/// needs. Code listing features passes a macro that receives this list
/// (`engine_features!(callback)`) instead of writing them out: [`Feature`]
/// here, and the facade's `EnginePlugins`, which adds every plugin in this
/// order and leaves out those `without` names.
#[macro_export]
macro_rules! engine_features {
    ($then:ident) => {
        $then! {
            /// Heightfield terrains streaming around the camera, with their
            /// bakes, placed objects, roads, bridges, inland water and
            /// waterfalls. It provides the [`Ground`](crate::Ground).
            terrain: Terrain => TerrainPlugin, needs [];
            /// Streamed entities spawning near the camera.
            world_streaming: WorldStreaming => WorldStreamingPlugin, needs [];
            /// Navigation agents walking over terrain.
            navigation: Navigation => NavigationPlugin, needs [Terrain];
            /// The sea and the entities riding it.
            ocean: Ocean => OceanPlugin, needs [];
            /// Clip players posing their targets, and skins.
            animation: Animation => AnimationPlugin, needs [];
            /// The time of day moving the sun.
            time_of_day: TimeOfDay => TimeOfDayPlugin, needs [];
            /// Tracks deformers press into the ground.
            deformation: Deformation => DeformationPlugin, needs [];
            /// Colliders, spatial queries and sensor events.
            collision: Collision => CollisionPlugin, needs [];
        }
    };
}

/// engine_core's part: [`Feature`] and what it says of each feature.
macro_rules! core_features {
    ($($(#[$doc:meta])* $name:ident: $variant:ident => $plugin:ident, needs [$($need:ident),*];)*) => {
        /// One feature a plugin adds to the engine. Scenes authoring a
        /// feature's components or sections, and development requests about
        /// it, need its plugin: without it they fail with `missing_plugin`.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum Feature {
            $(
                $(#[$doc])*
                #[doc = ""]
                #[doc = concat!("Its plugin is `", stringify!($plugin), "`.")]
                $variant,
            )*
        }

        impl Feature {
            /// Every feature, in the bundle's order.
            pub const ALL: [Self; [$(stringify!($name)),*].len()] = [$(Self::$variant),*];

            /// The plugin adding it.
            pub const fn plugin(self) -> &'static str {
                match self {
                    $(Self::$variant => stringify!($plugin),)*
                }
            }

            /// Its name, as scenes and development replies write it.
            pub const fn name(self) -> &'static str {
                match self {
                    $(Self::$variant => stringify!($name),)*
                }
            }

            /// The features whose plugins its plugin needs: an engine with
            /// it but without one of them does not start.
            pub const fn needs(self) -> &'static [Self] {
                match self {
                    $(Self::$variant => &[$(Self::$need),*],)*
                }
            }
        }
    };
}

engine_features!(core_features);

/// Features serialize as their names.
impl Serialize for Feature {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.name())
    }
}

impl Feature {
    /// The feature named `name`.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|feature| feature.name() == name)
    }

    const fn bit(self) -> u32 {
        1 << self as u32
    }
}

/// Registers `feature` with the engine as its plugin builds: the engine has
/// it from then on, and its environment sections that are on unless a scene
/// turns them off start at their defaults. Every feature's plugin calls it
/// first.
pub fn install(app: &mut App, feature: Feature) {
    let world = app.world_mut();
    world.resource_mut::<Features>().insert(feature);
    crate::environment::init_sections(world, Some(feature));
}

/// Panics unless every feature `world`'s engine has came with the features
/// it needs: what an engine checks before its first frame.
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

/// The features this engine's plugins added: each plugin registers its
/// own as it builds.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Features(u32);

impl Features {
    /// Whether `feature`'s plugin was added.
    pub fn has(self, feature: Feature) -> bool {
        self.0 & feature.bit() != 0
    }

    /// Registers `feature`: what [`install`] does as its plugin builds.
    pub fn insert(&mut self, feature: Feature) {
        self.0 |= feature.bit();
    }

    /// The features added, in [`Feature::ALL`] order.
    pub fn iter(self) -> impl Iterator<Item = Feature> {
        Feature::ALL
            .into_iter()
            .filter(move |&feature| self.has(feature))
    }
}

/// Whether `world`'s engine has `feature`.
pub fn has_feature(world: &World, feature: Feature) -> bool {
    world
        .get_resource::<Features>()
        .is_some_and(|features| features.has(feature))
}

/// The ground under a point, as [`Ground`] answers it: its height and the
/// level of detail answering, 0 the finest.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroundSample {
    pub height: f32,
    pub level: u32,
}

/// The ground what stands on it reads: grounded entities stand on it and
/// deformers press tracks into it, and games may read it too. The terrain
/// plugin provides the terrains'; a game with ground of its own installs a
/// `Ground` reading it instead. Without one nothing stands on the ground.
#[derive(Resource, Clone, Copy)]
pub struct Ground {
    /// The ground under world `(x, z)`; None where none is resident or a
    /// hole is cut.
    pub sample: fn(&World, f32, f32) -> Option<GroundSample>,
    /// Advances whenever the ground may have changed, so what stands on it
    /// stands again.
    pub revision: fn(&World) -> u64,
}

impl Ground {
    /// The ground under world `(x, z)` in `world`, if it has a ground.
    pub fn sample_in(world: &World, x: f32, z: f32) -> Option<GroundSample> {
        let ground = world.get_resource::<Self>()?;
        (ground.sample)(world, x, z)
    }
}

/// Holds an entity's height for something else, such as a walking
/// navigation agent keeping to its path's decks: grounding leaves it while
/// set.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HeightHeld(pub bool);

/// Keeps the world loaded around its entity as around the camera:
/// terrains keep their finest tiles resident there, so collision and
/// queries hold, and world streaming loads the cells near it.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StreamingFocus;
