use super::programs::{SurfaceEntry, SurfaceProgramGpu};
use super::publish::publish_terrains;
use super::solids::sync_solids;
use super::*;
use crate::expr_layer::EXPRESSION;
use crate::jobs::GROUND_GRID;
use crate::scene::{prefetch_kind, terrain_kind};
use engine_core::{
    Brick, Bricks, Explain, ExplainAsk, ExplainRecord, GlobalTransforms, GpuStruct, JobKind,
    SceneComponents, Schedule, Source,
};
use engine_places::ground::{Ground, GroundSurface, GroundWork, TerrainTiles};
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
            .fill::<GroundWork>(None)
            .fill::<GroundSurface>(Some(crate::shaders::terrain::faces::surface::MODULE))
            .join::<SceneComponents>(terrain_kind(), None)
            .join::<SceneComponents>(prefetch_kind(), None)
            .join::<TerrainLayers>(&EXPRESSION, None)
            .generated(GENERATED)
            .systems(
                Schedule::ViewSync,
                (stream_terrains, publish_terrains, sync_solids),
            )
            .systems(Schedule::ViewSync, settle_view)
            .systems(Schedule::Stream, bridges::sync)
            .systems(
                Schedule::Waiting,
                (stream_terrains, publish_terrains, settle_waiting),
            )
            .systems(
                Schedule::BetweenFrames,
                (work_between_frames, publish_terrains),
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
            .explain::<Residency>()
            .part::<SurfaceProgramGpu>();
    }
}

struct Residency;

impl Explain for Residency {
    type Params = (Res<'static, Terrains>, Res<'static, GlobalTransforms>);

    fn explain(
        (terrains, globals): bevy_ecs::system::SystemParamItem<'_, '_, Self::Params>,
        ask: &ExplainAsk<'_>,
    ) -> Option<ExplainRecord> {
        let ExplainAsk::Entity { id, entity } = ask else {
            return None;
        };
        let position = globals.get(*entity)?.matrix()?.w_axis;
        let sample = terrains.sample(position.x, position.z);
        let decided = if sample.is_some() {
            "resident"
        } else {
            "not_resident"
        };
        Some(
            ExplainRecord::new(decided)
                .with("entity", id)
                .with("position_m", [position.x, position.y, position.z])
                .with("ground_level", sample.as_ref().map(|sample| sample.level))
                .with(
                    "ground_height_m",
                    sample.as_ref().map(|sample| sample.height),
                ),
        )
    }
}
