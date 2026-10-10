//! The engine's features: what each plugin of the engine's bundle adds,
//! the scenes, environment sections and development tools that name them,
//! and the ground the features that stand things on it read.

use bevy_ecs::prelude::{Component, Resource, World};
use serde::Serialize;

/// One feature a plugin adds to the engine. Scenes authoring a feature's
/// components or sections, and development requests about it, need its
/// plugin: without it they fail with `missing_plugin`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Feature {
    /// Heightfield terrains streaming around the camera, with their bakes,
    /// placed objects, roads, bridges, inland water and waterfalls
    /// (`TerrainPlugin`). It provides the [`Ground`].
    Terrain,
    /// Streamed entities spawning near the camera (`WorldStreamingPlugin`).
    WorldStreaming,
    /// Navigation agents walking over terrain (`NavigationPlugin`); needs
    /// terrain.
    Navigation,
    /// The sea and the entities riding it (`OceanPlugin`).
    Ocean,
    /// Clip players posing their targets, and skins (`AnimationPlugin`).
    Animation,
    /// The time of day moving the sun (`TimeOfDayPlugin`).
    TimeOfDay,
    /// Tracks deformers press into the ground (`DeformationPlugin`).
    Deformation,
    /// Colliders, spatial queries and sensor events (`CollisionPlugin`).
    Collision,
}

impl Feature {
    /// Every feature, in the bundle's order.
    pub const ALL: [Self; 8] = [
        Self::Terrain,
        Self::WorldStreaming,
        Self::Navigation,
        Self::Ocean,
        Self::Animation,
        Self::TimeOfDay,
        Self::Deformation,
        Self::Collision,
    ];

    /// The plugin adding it.
    pub const fn plugin(self) -> &'static str {
        match self {
            Self::Terrain => "TerrainPlugin",
            Self::WorldStreaming => "WorldStreamingPlugin",
            Self::Navigation => "NavigationPlugin",
            Self::Ocean => "OceanPlugin",
            Self::Animation => "AnimationPlugin",
            Self::TimeOfDay => "TimeOfDayPlugin",
            Self::Deformation => "DeformationPlugin",
            Self::Collision => "CollisionPlugin",
        }
    }

    /// Its name, as scenes and development replies write it.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Terrain => "terrain",
            Self::WorldStreaming => "world_streaming",
            Self::Navigation => "navigation",
            Self::Ocean => "ocean",
            Self::Animation => "animation",
            Self::TimeOfDay => "time_of_day",
            Self::Deformation => "deformation",
            Self::Collision => "collision",
        }
    }

    /// The feature named `name`.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|feature| feature.name() == name)
    }

    const fn bit(self) -> u32 {
        1 << self as u32
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

    /// Registers `feature`: what its plugin does as it builds.
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
