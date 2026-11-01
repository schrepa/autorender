use super::ground_face::publish_ground;
use super::programs::{SurfaceProgramGpu, publish_programs};
use super::publish::{publish_hydrology, publish_tiles};
use super::solids::sync_solids;
use super::*;
use crate::jobs::GROUND_GRID;
use crate::scene::build_terrains;
use engine_core::{Brick, Bricks, ComponentKind, JobKind, SceneComponents, Schedule, Source};
use engine_places::ground::{Ground, GroundSurface, TerrainTiles};
use engine_places::terrain::TerrainLayers;
use engine_places::water::Hydrology;
use engine_render::RenderBricks;

pub struct TerrainBrick;

impl Brick for TerrainBrick {
    const SOURCE: &'static Source = &crate::SOURCE;
    const JOBS: &'static [&'static JobKind] = &[&TILE_BUILD, &BAKE, &GROUND_GRID];

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .when(|terrain: Res<TerrainSettings>| terrain.enabled)
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
            .join::<SceneComponents>(ComponentKind::of::<TerrainScene>("terrain"), None)
            .join::<SceneComponents>(ComponentKind::of::<TerrainPrefetch>("prefetch"), None)
            .join::<TerrainLayers>(crate::expr_layer::expression_kind(), None)
            .systems(Schedule::Presentation, bridges::sync)
            .systems(
                Schedule::ViewSync,
                (
                    build_terrains,
                    stream_terrains,
                    publish_ground,
                    publish_tiles,
                    publish_hydrology,
                    publish_programs,
                ),
            )
            .systems(Schedule::WorldSync, sync_solids)
            .systems(Schedule::BetweenFrames, work_between_frames)
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
