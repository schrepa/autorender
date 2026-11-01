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
mod work;

use bevy_ecs::prelude::*;
use engine_core::{Brick, Bricks, Cut, JobKind, Jobs, Schedule, Source, Spawner, stats};
use engine_places::ground::Heightfield;
use engine_places::water::Waterline;
use engine_render::{DebugModes, DebugShown, RenderBricks};

pub(crate) use engine_codec as codec;
pub(crate) use engine_heightfield::tile::Footprint;
pub(crate) use engine_heightfield::{Layers, NARROWEST, Rivers, TileKey};

pub use flow::{Block, Pool, RiverFlow, WEIR};
pub use settings::Water;
pub use steady_water::*;
pub use tile_water::*;
pub use views::PoolView;
pub use water_settle::SettleFeedback;
pub use windows::{LIVE_PIXELS, LONGEST_FRAME, WaterRead, WaterSourceCell, WaterWindow, live_steps};

use state::{RiverBody, WaterState};
use water_particles::ParticleTallies;
use waterfalls::TerrainFalls;
use white_water::TerrainWhiteWater;

stats! {
    #[section("water_stats")]
    pub struct WaterStats {
        #[pending] pub looks_left: u32,
        #[pending] pub settles_left: u32,
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
        pub failed_jobs: u64,
    }
}

pub struct WaterBrick;

impl Brick for WaterBrick {
    const SOURCE: &'static Source = &SOURCE;
    const JOBS: &'static [&'static JobKind] = &[
        &jobs::TILE_WATER,
        &jobs::PARTS,
        &jobs::STEADIED,
        &jobs::SETTLED,
    ];

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
            .piece::<gpu::WaterPiece>()
            .fill::<Waterline>(Some(shaders::water::faces::water::MODULE))
            .observation::<views::WaterFlowQuery>();
        for mode in debug::modes() {
            bricks.join::<DebugModes>(mode, Some(shaders::water::modes::MODULE));
        }
        bricks
            .systems(
                Schedule::ViewSync,
                (
                    state::follow,
                    water_settle::hand_settles,
                    settled::request_settle,
                    rewater::rewater,
                    water_settle::plan_settles,
                    water_blocks::place_blocks,
                    water_sim::plan_sim,
                    water_particles::plan_particles,
                    state::publish_waterline,
                    state::publish_bodies,
                    publish_piece,
                    water_stats,
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
            .systems(Schedule::FixedSimulation, flow_step::step)
            .off(despawn_water)
            .generated(gpu::GENERATED)
            .part::<water::WaterGpu>();
    }
}

fn water_stats(
    water: Res<Water>,
    state: Res<WaterState>,
    tallies: Res<ParticleTallies>,
    heightfield: Res<Heightfield>,
    mut stats: ResMut<WaterStats>,
) {
    use gpu::{
        PARTICLE_FOAM_BORN, PARTICLE_FOAM_DROPPED, PARTICLE_FOAM_LIVE, PARTICLE_WAVES_BORN,
        PARTICLE_WAVES_DROPPED, PARTICLE_WAVES_LIVE,
    };
    let frame = water.enabled && !state.watered.is_empty();
    let (cell_cut, [live_cut, live_steps, live_behind, live_setup, live_waiting]) = if frame {
        state.live.cells()
    } else {
        (Cut::default(), [0; 5])
    };
    let [touches, dropped_touches] = if frame { state.live.touches() } else { [0; 2] };
    let [leaves, refused] = if frame {
        heightfield.water_leaves
    } else {
        [0; 2]
    };
    let ((texels, points, foam_cut, reach), counts) = if frame {
        (state.particles.stats(), tallies.counts)
    } else {
        ((0, 0, Cut::default(), 0), Default::default())
    };
    let tally = |word: u32| {
        if state.particles.grid.is_some() {
            counts.get(word as usize).copied().unwrap_or(0)
        } else {
            0
        }
    };
    let next = WaterStats {
        looks_left: state.looks_left.min(u64::from(u32::MAX)) as u32,
        settles_left: state.settles_left.min(u64::from(u32::MAX)) as u32,
        blocks: if frame { state.blocks.resident() } else { 0 },
        dropped_blocks: if frame { state.blocks.dropped } else { 0 },
        live_cut,
        live_steps,
        live_behind,
        live_setup,
        live_waiting,
        touches,
        dropped_touches,
        particle_texels: texels,
        foam_points: points,
        foam_reach: reach,
        foam_particles: tally(PARTICLE_FOAM_LIVE),
        wave_particles: tally(PARTICLE_WAVES_LIVE),
        particles_born: tally(PARTICLE_FOAM_BORN) + tally(PARTICLE_WAVES_BORN),
        dropped_particles: tally(PARTICLE_FOAM_DROPPED) + tally(PARTICLE_WAVES_DROPPED),
        cell_cut,
        foam_cut,
        leaf_cut: Cut::of(leaves.saturating_add(refused), leaves),
        rewatered_tiles: state.rewatered_tiles,
        stale_water_tiles: state.stale_water_tiles,
        settled_records: state.settled_records,
        settling_records: state.settling_records,
        falls: state
            .terrains
            .iter()
            .map(|terrain| terrain.falls.len() as u32)
            .sum(),
        white_water: stats.white_water,
        failed_jobs: state.failures.count,
    };
    stats.set_if_neq(next);
}

fn publish_piece(shown: Res<DebugShown>, mut piece: ResMut<gpu::WaterPiece>) {
    let mine = shown.owner == Some(std::any::TypeId::of::<WaterBrick>());
    piece.set_if_neq(gpu::WaterPiece {
        debug: [u32::from(mine), shown.local, 0, 0],
    });
}

fn despawn_water(
    owned: Query<Entity, With<RiverBody>>,
    (mut falls, mut white): (ResMut<TerrainFalls>, ResMut<TerrainWhiteWater>),
    mut state: ResMut<WaterState>,
    mut stats: ResMut<WaterStats>,
    mut line: ResMut<Waterline>,
    mut jobs: Jobs,
    mut spawner: Spawner<WaterBrick>,
) {
    for entity in &owned {
        spawner.despawn(entity);
    }
    for entity in falls.take().into_iter().chain(white.take()) {
        spawner.despawn(entity);
    }
    state.clear(&mut jobs);
    *stats = WaterStats::default();
    line.watered.clear();
    line.errors.clear();
}
