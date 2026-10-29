use super::*;
use crate::jobs::TerrainWorker;
use engine_core::{
    Brick, Bricks, EnvironmentSection, FixedSimulation, FixedSimulationSystems, Ground,
    GroundSample, JobWorker, TerrainDebug, ViewSync, ViewSyncSystems, WaterSources, WorldSync,
    WorldSyncSystems,
};

const WATER_SOURCE: &str = "terrain";

pub struct TerrainBrick;

impl Brick for TerrainBrick {
    fn register(bricks: &mut Bricks) {
        bricks
            .settings::<TerrainSettings>()
            .settings::<GroundCover>()
            .settings::<Wind>()
            .settings::<Occupancy>()
            .settings::<Water>()
            .resource::<Terrains>()
            .insert_resource(Ground {
                sample: ground_sample,
                revision: |world| {
                    world
                        .get_resource::<Terrains>()
                        .map_or(0, Terrains::revision)
                },
            })
            .systems(ViewSync, stream_terrains.in_set(ViewSyncSystems::Stream))
            .systems(FixedSimulation, flow::step.in_set(FixedSimulationSystems::Ride))
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
            .inspector::<Terrains, TerrainsView>(TerrainsView::of)
            .inspector::<TerrainDebug, TerrainDebugView>(TerrainDebugView::of);
    }
}

pub(crate) fn join(world: &mut World) {
    if !world.contains_resource::<Terrains>() {
        return;
    }
    if let Some(mut sources) = world.get_resource_mut::<WaterSources>()
        && !sources.names().any(|name| name == WATER_SOURCE)
    {
        sources.register(WATER_SOURCE, water_source::terrain_water);
    }
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

fn ground_sample(world: &World, x: f32, z: f32) -> Option<GroundSample> {
    let sample = world
        .get_resource::<Terrains>()?
        .sample(x, z)
        .filter(|sample| !sample.hole)?;
    Some(GroundSample {
        height: sample.height,
        level: sample.level,
    })
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
    Wind: "wind",
    Occupancy: "occupancy",
    Water: "water",
}
