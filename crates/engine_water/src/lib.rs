#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants,
    clippy::iter_over_hash_type,
    clippy::expect_used
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod debug;
pub mod flow;
mod flow_step;
mod gpu;
pub mod jobs;
mod rewater;
mod settings;
mod settled;
pub mod shallow_water;
mod sights;
mod state;
mod steady_water;
mod tile_water;
mod views;
mod water;
mod water_blocks;
mod water_falls;
mod water_kernels;
mod water_particles;
mod water_settle;
mod water_sim;
mod water_source;
mod waterfalls;
mod white_water;
mod windows;

use bevy_ecs::prelude::*;
use engine_core::{Brick, Bricks, Cut, JobKind, Schedule, Source, Spawner, stats};
use engine_places::water::{WaterFlow, Waterline};
use engine_render::{DebugModes, RenderBricks};

pub(crate) use engine_codec as codec;
pub(crate) use engine_heightfield::{Fall, Layers, Pyramid, Rivers, TileKey};
pub(crate) use engine_heightfield::tile::{Footprint, Solid, StampOp as Stamp, Tile, TileJob};

pub use flow::{Block, Pool, RiverFlow, WEIR};
pub use settings::Water;
pub use state::SettleFeedback;
pub use steady_water::*;
pub use tile_water::*;
pub use views::PoolView;
pub use windows::{LIVE_PIXELS, LONGEST_FRAME, WaterSourceCell, WaterWindow, live_steps};

use state::{FallSet, ParticleTallies, RiverBody, WaterState};
use waterfalls::TerrainFalls;
use white_water::TerrainWhiteWater;

stats! {
    #[section("water_stats")]
    pub struct WaterStats {
        #[pending] pub looks_left: u32,
        pub blocks: u32,
        pub dropped_blocks: u32,
        pub live_cut: u32,
        pub live_steps: u32,
        pub live_behind: u32,
        pub live_setup: u32,
        pub live_waiting: u32,
        pub touches: u32,
        pub dropped_touches: u32,
        pub particle_texels: u32,
        pub foam_points: u32,
        pub foam_reach: u32,
        pub foam_particles: u32,
        pub wave_particles: u32,
        pub particles_born: u32,
        pub dropped_particles: u32,
        pub cell_cut: Cut,
        pub foam_cut: Cut,
        pub leaf_cut: Cut,
        pub rewatered_tiles: u32,
        pub stale_water_tiles: u32,
        pub settled_records: u32,
        pub settling_records: u32,
        pub falls: u32,
        pub white_water: u32,
    }
}

pub struct WaterBrick;

impl Brick for WaterBrick {
    const SOURCE: &'static Source = &SOURCE;
    const JOBS: &'static [&'static JobKind] = &[&jobs::TILE_WATER, &jobs::STEADIED, &jobs::SETTLED];

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .when(|water: Res<Water>| water.enabled)
            .settings::<Water>()
            .stats::<WaterStats>()
            .resource::<WaterState>()
            .resource::<SettleFeedback>()
            .resource::<ParticleTallies>()
            .resource::<TerrainFalls>()
            .resource::<TerrainWhiteWater>()
            .fill::<WaterFlow>(None)
            .fill::<Waterline>(Some(shaders::water::faces::water::MODULE));
        for mode in debug::modes() {
            bricks.join::<DebugModes>(mode, Some(shaders::water::debug::MODULE));
        }
        bricks
            .systems(
                Schedule::ViewSync,
                (
                    state::rewater,
                    settled::request_settle,
                    settled::take_settled,
                    state::publish_waterline,
                    state::publish_bodies,
                    state::water_stats,
                ),
            )
            .systems(
                Schedule::WorldSync,
                (
                    waterfalls::sync,
                    white_water::sync,
                    windows::place_water_windows,
                ),
            )
            .systems(Schedule::FixedPostSimulation, flow_step::step)
            .off(despawn_water)
            .part::<water::WaterGpu>();
    }
}

fn despawn_water(
    owned: Query<Entity, Or<(With<RiverBody>, With<FallSet>)>>,
    mut falls: ResMut<TerrainFalls>,
    mut white: ResMut<TerrainWhiteWater>,
    mut state: ResMut<WaterState>,
    mut stats: ResMut<WaterStats>,
    mut spawner: Spawner<WaterBrick>,
) {
    for entity in &owned {
        spawner.despawn(entity);
    }
    for entity in falls.take().into_iter().chain(white.take()) {
        spawner.despawn(entity);
    }
    state.clear();
    *stats = WaterStats::default();
}
