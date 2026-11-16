#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

extern crate engine_ecs as bevy_ecs;

pub mod ground;
pub mod paths;
pub mod terrain;
pub mod water;

use engine_place::PlaceDecl;

pub const PLACES: &[PlaceDecl] = &[
    PlaceDecl::slot::<ground::TerrainTiles>(),
    PlaceDecl::slot::<ground::GroundSurface>(),
    PlaceDecl::slot::<ground::Heightfield>(),
    PlaceDecl::slot::<ground::GroundBaking>(),
    PlaceDecl::slot::<ground::GroundTiles>(),
    PlaceDecl::component::<ground::GroundLayerAsks>("ground_layer_asks"),
    PlaceDecl::component::<ground::GroundLayersHeld>("ground_layers_held"),
    PlaceDecl::slot::<ground::Horizons>(),
    PlaceDecl::slot::<ground::Bounce>(),
    PlaceDecl::slot::<ground::PlacedTiles>(),
    PlaceDecl::slot::<ground::Cover>(),
    PlaceDecl::list::<ground::FarCover>(),
    PlaceDecl::slot::<ground::Foliage>(),
    PlaceDecl::slot::<water::Waterline>(),
    PlaceDecl::slot::<water::Hydrology>(),
    PlaceDecl::list::<paths::ScenePathSources>(),
    PlaceDecl::list::<terrain::TerrainLayers>(),
];
