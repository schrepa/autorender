use super::ground_face::publish_ground;
use super::programs::{SurfaceEntry, SurfaceProgramGpu, publish_programs};
use super::publish::{publish_hydrology, publish_tiles};
use super::solids::sync_solids;
use super::*;
use crate::expr_layer::EXPRESSION;
use crate::jobs::GROUND_GRID;
use crate::scene::{prefetch_kind, terrain_kind};
use engine_core::{Brick, Bricks, GpuStruct, JobKind, SceneComponents, Schedule, Source};
use engine_places::ground::{Ground, GroundSurface, TerrainTiles};
use engine_places::terrain::TerrainLayers;
use engine_places::water::Hydrology;
use engine_render::RenderBricks;

pub struct TerrainBrick;

const GENERATED: &[&str] = &[crate::gpu::WGSL, <SurfaceEntry as GpuStruct>::WGSL];

impl Brick for TerrainBrick {
    const SOURCE: &'static Source = &crate::SOURCE;
    const JOBS: &'static [&'static JobKind] = &[&TILE_BUILD, &BAKE, &GROUND_GRID];

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<TerrainSettings>()
            .stats::<TerrainStats>()
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
            .join::<TerrainLayers>(&EXPRESSION, None)
            .generated(GENERATED)
            .systems(
                Schedule::ViewSync,
                (
                    stream_terrains,
                    sync_solids,
                    publish_ground,
                    publish_tiles,
                    publish_hydrology,
                    publish_programs,
                ),
            )
            .systems(Schedule::WorldSync, bridges::sync)
            .systems(Schedule::Waiting, stream_terrains)
            .systems(Schedule::BetweenFrames, (work_between_frames, publish_ground))
            .operation::<TerrainEditControl>()
            .operation::<TerrainUndoControl>()
            .operation::<TerrainRedoControl>()
            .operation::<TerrainRiverControl>()
            .observation::<TerrainHeightQuery>()
            .observation::<TerrainRouteQuery>()
            .observation::<TerrainInspect>()
            .part::<SurfaceProgramGpu>();
    }
}
