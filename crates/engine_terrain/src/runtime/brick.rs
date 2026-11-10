use super::programs::{SurfaceEntry, SurfaceProgramGpu};
use super::publish::publish_terrains;
use super::solids::sync_solids;
use super::*;
use crate::jobs::expr_layer::EXPRESSION;
use crate::scene::{
    INVALID_STREAMED, TERRAIN, focus_kind, prefetch_kind, terrain_kind, terrain_paths,
};
use engine_core::{
    Brick, Bricks, Cook, CookList, GpuBits, GpuStruct, JobKind, SceneComponents, Schedule, Source,
};
use engine_heightfield::{INVALID_TERRAIN, MISSING_LAYER, UNKNOWN_MAP, UNREADABLE_MAP};
use engine_open_world_kit::ground::{GroundSurface, TerrainTiles};
use engine_open_world_kit::paths::{ScenePathSource, ScenePathSources};
use engine_open_world_kit::terrain::TerrainLayers;
use engine_open_world_kit::water::Hydrology;
use engine_places_world::streaming::{StreamedKey, StreamedKeys, Streams};
use engine_render::RenderBricks;

pub struct TerrainBrick;

const GENERATED: &[&str] = &[
    crate::gpu::WGSL,
    <SurfaceEntry as GpuStruct>::WGSL,
    <engine_heightfield::tile::Control as GpuBits>::WGSL,
];
const ERRORS: &[ErrorCode] = &[
    INVALID_TERRAIN,
    MISSING_LAYER,
    UNKNOWN_MAP,
    UNREADABLE_MAP,
    UNKNOWN_TERRAIN,
    INVALID_STREAMED,
];

impl Brick for TerrainBrick {
    const SOURCE: &'static Source = &crate::SOURCE;
    const JOBS: &'static [&'static JobKind] = &[&TILE_BUILD, &BAKE];

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .when(|settings: Res<TerrainSettings>| settings.enabled)
            .settings::<TerrainSettings>()
            .stats::<TerrainStats>()
            .errors(ERRORS)
            .resource::<Terrains>()
            .resource::<SurfacePrograms>()
            .resource::<TerrainBridges>()
            .resource::<TerrainRedo>()
            .fill::<TerrainTiles>(None)
            .fill::<Hydrology>(None)
            .fill::<GroundSurface>(Some(crate::shaders::terrain::faces::surface::MODULE))
            .join::<SceneComponents>(terrain_kind(), None)
            .join::<SceneComponents>(prefetch_kind(), None)
            .join::<SceneComponents>(focus_kind(), None)
            .join::<StreamedKeys>(
                StreamedKey {
                    key: TERRAIN,
                    streams: Streams::Never(INVALID_STREAMED),
                },
                None,
            )
            .join::<ScenePathSources>(
                ScenePathSource {
                    key: TERRAIN,
                    paths: terrain_paths,
                },
                None,
            )
            .join::<TerrainLayers>(&EXPRESSION, None)
            .join::<CookList>(Cook::new(super::cook::cook), None)
            .generated(GENERATED)
            .systems(
                Schedule::ViewSync,
                (
                    stream_terrains,
                    publish_terrains,
                    sync_solids,
                    bridges::sync,
                ),
            )
            .operation::<TerrainEditControl>()
            .operation::<TerrainUndoControl>()
            .operation::<TerrainRedoControl>()
            .operation::<TerrainRiverControl>()
            .observation::<TerrainHeightQuery>()
            .observation::<TerrainRouteQuery>()
            .observation::<TerrainInspect>()
            .observation::<TerrainLayerSweep>()
            .observation::<TerrainListing>()
            .observation::<TerrainSweepQuery>()
            .part::<SurfaceProgramGpu>();
    }
}
