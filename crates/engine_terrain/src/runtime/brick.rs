use super::ground_face::{TerrainGround, check_live_water};
use super::solids::{TerrainDecks, TerrainFields, TerrainPlaced};
use super::water_body::TerrainWater;
use super::*;
use crate::jobs::{ShadingJob, SteadiedJob, TerrainWorker};
use crate::layer::{TerrainLayerKinds, TerrainLayersList};
use crate::scene::{PREFETCH, TERRAIN};
use engine_core::{
    Brick, Bricks, DetailConsumer, DetailConsumersList, EnvironmentSection, ExplainRecord,
    ExplainRequest, Fill, FixedSimulation, FixedSimulationSystems, GlobalTransforms, JobWorker,
    Join, TerrainDebug, ViewSync, ViewSyncSystems, WorldSync, WorldSyncSystems,
};
use engine_places::{Ground, GroundSlot, WaterBodiesList, WorldSolidsList};
use engine_scene::SceneComponentsList;

pub struct TerrainBrick;

impl Brick for TerrainBrick {
    fn register(bricks: &mut Bricks) {
        bricks
            .fill(Fill::<GroundSlot>::new().cpu(Ground::new("terrain", TerrainGround)))
            .settings::<TerrainSettings>()
            .join(Join::<DetailConsumersList>::new().cpu(DetailConsumer {
                name: "tiles",
                setting: "terrain.max_tiles",
                stat: "terrain.tile_cut",
            }))
            .join(Join::<DetailConsumersList>::new().cpu(DetailConsumer {
                name: "terrain_leaves",
                setting: "terrain.subdivision_leaves",
                stat: "terrain.leaf_cut",
            }))
            .records(&[
                TileJob::KIND.name(),
                BakeJob::KIND.name(),
                LightingJob::KIND.name(),
                BounceJob::KIND.name(),
                ShadingJob::KIND.name(),
                SteadiedJob::KIND.name(),
                SETTLED_KIND,
            ])
            .resource::<Terrains>()
            .join(Join::<WorldSolidsList>::new().cpu(("ground", Box::new(TerrainFields))))
            .join(Join::<WorldSolidsList>::new().cpu(("decks", Box::new(TerrainDecks))))
            .join(Join::<WorldSolidsList>::new().cpu(("placed", Box::new(TerrainPlaced))))
            .join(Join::<WaterBodiesList>::new().cpu(Box::new(TerrainWater)))
            .resource::<TerrainLayerKinds>();
        for &kind in crate::expr::KINDS {
            bricks.join(Join::<TerrainLayersList>::new().cpu(kind));
        }
        bricks
            .join(Join::<SceneComponentsList>::new().cpu(&TERRAIN))
            .join(Join::<SceneComponentsList>::new().cpu(&PREFETCH))
            .systems(ViewSync, stream_terrains.in_set(ViewSyncSystems::Stream))
            .systems(
                FixedSimulation,
                flow::step.in_set(FixedSimulationSystems::Ride),
            )
            .systems(
                WorldSync,
                (
                    bridges::sync,
                    waterfalls::sync,
                    white_water::sync,
                    water_sim::place_water_windows,
                )
                    .chain()
                    .in_set(WorldSyncSystems::Spawn),
            )
            .extract(extract_terrains)
            .operation::<TerrainControl>()
            .operation::<TerrainEditControl>()
            .operation::<TerrainUndoControl>()
            .operation::<TerrainRedoControl>()
            .operation::<TerrainRiverControl>()
            .operation::<TerrainHeightQuery>()
            .operation::<TerrainRouteQuery>()
            .inspector::<Terrains, TerrainsView>(|_, terrains| TerrainsView::of(terrains))
            .inspector::<TerrainDebug, TerrainDebugView>(|_, debug| TerrainDebugView::of(debug))
            .component_inspector::<TerrainPrefetch, TerrainPrefetch>(|_, _, prefetch| *prefetch)
            .explain(explain_residency);
        crate::smokes::register(bricks);
    }
}

fn explain_residency(world: &World, request: &ExplainRequest) -> Option<ExplainRecord> {
    let ExplainRequest::Entity(entity) = request else {
        return None;
    };
    let terrains = world.get_resource::<Terrains>()?;
    let position = world
        .get_resource::<GlobalTransforms>()?
        .get(entity.entity)?
        .matrix()?
        .w_axis;
    let sample = terrains.sample(position.x, position.z);
    let decided = if sample.is_some() {
        "resident"
    } else {
        "not_resident"
    };
    Some(
        ExplainRecord::new(decided)
            .with("entity", &entity.id)
            .with("position_m", [position.x, position.y, position.z])
            .with("ground_level", sample.as_ref().map(|sample| sample.level))
            .with(
                "ground_height_m",
                sample.as_ref().map(|sample| sample.height),
            ),
    )
}

pub(crate) fn join(world: &mut World) {
    if !world.contains_resource::<Terrains>() {
        return;
    }
    check_live_water(world);
    if let Some(mut jobs) = world.get_resource_mut::<Jobs>()
        && !jobs.works_here()
    {
        jobs.work_here(terrain_worker);
    }
}

fn terrain_worker() -> Box<dyn JobWorker> {
    Box::new(TerrainWorker::new())
}

impl JobWorker for TerrainWorker {
    fn share(&mut self, key: &str, bytes: Arc<[u8]>) -> Result<(), String> {
        TerrainWorker::share(self, key, bytes)
    }

    fn unshare(&mut self, key: &str) {
        TerrainWorker::unshare(self, key);
    }

    fn run(&mut self, kind: &str, input: &[u8]) -> Result<Vec<u8>, String> {
        TerrainWorker::run(self, kind, input)
    }
}

macro_rules! sections {
    ($($settings:ident: $key:literal),* $(,)?) => {
        $(
            impl EnvironmentSection for $settings {
                const KEY: &'static str = $key;
                const NAME: &'static str = stringify!($settings);
                const OPT_IN: bool = false;
                fn section_off() -> Self {
                    Self::off()
                }
                fn section_check(&self) -> Result<(), InvalidSetting> {
                    self.validate()
                }
                fn section_sanitized(self) -> Self {
                    self.sanitized()
                }
                #[cfg(feature = "schema")]
                fn schema() -> schemars::Schema {
                    schemars::schema_for!($settings)
                }
            }
        )*
    };
}

sections! {
    TerrainSettings: "terrain",
    GroundCover: "ground_cover",
    Occupancy: "occupancy",
    Water: "water",
}
