use super::programs::{SurfaceEntry, SurfaceProgramGpu};
use super::publish::publish_terrains;
use super::solids::sync_solids;
use super::*;
use crate::jobs::GROUND_GRID;
use crate::jobs::expr_layer::EXPRESSION;
use crate::scene::{INVALID_STREAMED, TERRAIN, prefetch_kind, terrain_kind, terrain_paths};
use engine_core::{
    Brick, Bricks, Cook, CookList, GpuBits, GpuStruct, JobKind, SceneComponents, Schedule, Source,
};
use engine_heightfield::{INVALID_TERRAIN, MISSING_LAYER, UNKNOWN_MAP, UNREADABLE_MAP};
use engine_places_simulation::ground::{Ground, UNKNOWN_TERRAIN};
use engine_places_world::ground::{GroundSurface, TerrainTiles};
use engine_places_world::paths::{ScenePathSource, ScenePathSources};
use engine_places_world::streaming::{StreamedKey, StreamedKeys, Streams};
use engine_places_world::terrain::TerrainLayers;
use engine_places_world::water::Hydrology;
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
    const JOBS: &'static [&'static JobKind] = &[&TILE_BUILD, &BAKE, &GROUND_GRID];

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
            .fill::<Ground>(None)
            .fill::<TerrainTiles>(None)
            .fill::<Hydrology>(None)
            .fill::<GroundSurface>(Some(crate::shaders::terrain::faces::surface::MODULE))
            .join::<SceneComponents>(terrain_kind(), None)
            .join::<SceneComponents>(prefetch_kind(), None)
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
                (stream_terrains, publish_terrains, sync_solids, bridges::sync),
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
            .part::<SurfaceProgramGpu>();
    }
}
