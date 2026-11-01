#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

extern crate engine_ecs as bevy_ecs;

pub mod bloom;
pub mod dither;
pub mod exposure;
pub mod ground;
pub mod lighting;
pub mod lights;
pub mod meshes;
pub mod navigation;
pub mod particles;
pub mod physics;
pub mod probes;
pub mod reflections;
pub mod shadows;
pub mod sky;
mod smokes;
pub mod solids;
pub mod splines;
pub mod streaming;
pub mod sun;
pub mod terrain;
pub mod tonemap;
pub mod water;
pub mod wind;

use engine_place::PlaceDecl;

mod degrees {
    use serde::{Deserialize, Deserializer, Serializer};

    pub(crate) fn serialize<S: Serializer>(
        radians: &f32,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.serialize_f32(radians.to_degrees())
    }

    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f32, D::Error> {
        f32::deserialize(deserializer).map(f32::to_radians)
    }
}

fn is_zero(value: &f32) -> bool {
    *value == 0.0
}

pub const PLACES: &[PlaceDecl] = &[
    PlaceDecl::slot::<sun::SunPath>(),
    PlaceDecl::slot::<sun::Sun>(),
    PlaceDecl::slot::<sky::SkyLight>(),
    PlaceDecl::slot::<exposure::Exposure>(),
    PlaceDecl::slot::<tonemap::Display>(),
    PlaceDecl::slot::<lighting::Lighting>(),
    PlaceDecl::slot::<lights::Lights>(),
    PlaceDecl::component::<lights::LocalLight>("local_light"),
    PlaceDecl::slot::<shadows::ShadowMaps>(),
    PlaceDecl::slot::<shadows::CasterRoutes>(),
    PlaceDecl::slot::<shadows::ShadowCasters>(),
    PlaceDecl::component::<shadows::ShadowChanges>("shadow_changes"),
    PlaceDecl::slot::<probes::ProbeViews>(),
    PlaceDecl::slot::<probes::ProbeCaptures>(),
    PlaceDecl::slot::<probes::ProbeLight>(),
    PlaceDecl::component::<probes::ReflectionProbe>("reflection_probe"),
    PlaceDecl::slot::<bloom::Glow>(),
    PlaceDecl::slot::<reflections::Reflected>(),
    PlaceDecl::slot::<dither::DitherPlace>(),
    PlaceDecl::slot::<meshes::Instances>(),
    PlaceDecl::component::<meshes::Cube>("cube"),
    PlaceDecl::component::<meshes::Tint>("tint"),
    PlaceDecl::slot::<particles::Transparency>(),
    PlaceDecl::component::<particles::ParticleEmitter>("particle_emitter"),
    PlaceDecl::slot::<ground::Ground>(),
    PlaceDecl::slot::<ground::TerrainTiles>(),
    PlaceDecl::slot::<ground::GroundSurface>(),
    PlaceDecl::slot::<ground::Heightfield>(),
    PlaceDecl::slot::<ground::GroundTiles>(),
    PlaceDecl::slot::<ground::Horizons>(),
    PlaceDecl::slot::<ground::Bounce>(),
    PlaceDecl::slot::<ground::PlacedTiles>(),
    PlaceDecl::slot::<ground::Cover>(),
    PlaceDecl::list::<ground::FarCover>(),
    PlaceDecl::slot::<ground::Occupancy>(),
    PlaceDecl::slot::<ground::Foliage>(),
    PlaceDecl::component::<ground::HeightHeld>("height_held"),
    PlaceDecl::component::<ground::Grounded>("grounded"),
    PlaceDecl::component::<ground::GroundedAt>("grounded_at"),
    PlaceDecl::component::<navigation::NavAgent>("nav_agent"),
    PlaceDecl::component::<navigation::NavWalk>("nav_walk"),
    PlaceDecl::slot::<water::Waterline>(),
    PlaceDecl::slot::<water::WaterObstacles>(),
    PlaceDecl::slot::<water::WaterFlow>(),
    PlaceDecl::slot::<water::Hydrology>(),
    PlaceDecl::component::<water::WaterBody>("water_body"),
    PlaceDecl::slot::<wind::Wind>(),
    PlaceDecl::slot::<physics::Collision>(),
    PlaceDecl::component::<physics::Collider>("collider"),
    PlaceDecl::component::<physics::RigidBody>("rigid_body"),
    PlaceDecl::component::<solids::SolidSource>("solid_source"),
    PlaceDecl::component::<splines::SplineMesh>("spline_mesh"),
    PlaceDecl::component::<streaming::StreamingFocus>("streaming_focus"),
    PlaceDecl::list::<terrain::TerrainLayers>(),
    PlaceDecl::slot::<terrain::DrawnRivers>(),
];
