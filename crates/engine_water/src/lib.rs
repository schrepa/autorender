#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants,
    clippy::iter_over_hash_type,
    clippy::expect_used
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod debug;
mod drawn;
mod emitters;
mod eye;
pub mod flow;
mod flow_step;
mod gpu;
pub mod jobs;
mod rewater;
mod settings;
mod settled;
mod sights;
mod state;
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
use engine_core::{
    Attributes, Brick, Bricks, Cook, CookList, Cut, JobKind, KindDecl, KindSkip, Schedule, Source,
    SurfaceSlot, stats,
};
use engine_open_world_kit::ground::Heightfield;
use engine_open_world_kit::water::Waterline;
use engine_places_rendering::reflections::TraceDepths;
use engine_places_rendering::sky::EyeMedium;
use engine_render::{DebugModes, DebugShown, RenderBricks, SlotModule, SurfaceKind};

pub(crate) use engine_heightfield::TileKey;
pub(crate) use engine_heightfield::tile::Footprint;

pub use flow::{Block, FlowState, Pool, RiverFlow, WEIR};
pub use jobs::steady_water::*;
pub use jobs::tile_water::*;
pub use settings::Water;
pub use views::PoolView;
pub use water_settle::SettleFeedback;
pub use windows::{
    LIVE_PIXELS, LONGEST_FRAME, WaterRead, WaterSourceCell, WaterWindow, live_steps,
};

use state::WaterState;
use water_particles::ParticleTallies;
use water_sim::LiveSteps;

stats! {
    #[section("water_stats")]
    pub struct WaterStats {
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
        &jobs::DRAWN,
        &jobs::REWATER,
        &jobs::REACH,
    ];

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .when(|water: Res<Water>| water.enabled)
            .settings::<Water>()
            .stats::<WaterStats>()
            .errors(&[engine_heightfield::INVALID_TERRAIN])
            .resource::<WaterState>()
            .resource::<SettleFeedback>()
            .resource::<ParticleTallies>()
            .resource::<LiveSteps>()
            .piece::<gpu::WaterPiece>()
            .surface_kind(SurfaceKind { name: "water" })
            .join::<TraceDepths>((), Some(shaders::water::trace_depth::MODULE))
            .join::<CookList>(Cook::new(settled::cook), None)
            .material_slot(SlotModule {
                slot: SurfaceSlot::Normal.name(),
                name: "ripples",
                module: shaders::water::ripples::MODULE,
                maps: &[],
                values: &["ripples"],
                reach: 0.0,
            })
            .fill::<Waterline>(Some(shaders::water::faces::water::MODULE))
            .fill::<EyeMedium>(None)
            .observation::<views::WaterFlowQuery>()
            .draw_kind(KindDecl {
                kind: water::WATER_KIND,
                needs: Attributes::NONE,
                streams: &[],
                skips: &[
                    KindSkip {
                        pass: "shadow",
                        reason: "the water surface casts no shadows",
                    },
                    KindSkip {
                        pass: "probe",
                        reason: "the water surface stays out of probe captures",
                    },
                    KindSkip {
                        pass: "color",
                        reason: "the water surface draws in its own pass, after the scene it refracts",
                    },
                    KindSkip {
                        pass: "prepass",
                        reason: "the water surface refracts the scene behind it, so it fills no depth before it",
                    },
                ],
            })
            .pass::<water::WaterPass>()
            .draws::<water::WaterGpu>(water::WATER_KIND, water::WaterGpu::draw);
        for (index, mode) in debug::modes().enumerate() {
            let wgsl = (index == 0).then_some(shaders::water::modes::MODULE);
            bricks.join::<DebugModes>(mode, wgsl);
        }
        bricks
            .systems(
                Schedule::ViewSync,
                (
                    state::follow,
                    water_settle::hand_settles,
                    settled::request_settle,
                    rewater::rewater,
                    drawn::draw,
                    water_settle::plan_settles,
                    water_blocks::place_blocks,
                    water_sim::plan_sim,
                    water_particles::plan_particles,
                    state::publish_waterline,
                    state::publish_bodies,
                    eye::publish_medium,
                    publish_piece,
                    water_stats,
                ),
            )
            .systems(
                Schedule::WorldSync,
                (emitters::sync, windows::place_water_windows),
            )
            .systems(Schedule::FixedSimulation, flow_step::step)
            .generated(gpu::GENERATED)
            .part::<water::WaterGpu>();
    }
}

fn water_stats(
    state: Res<WaterState>,
    (tallies, steps): (Res<ParticleTallies>, Res<LiveSteps>),
    heightfield: Res<Heightfield>,
    mut stats: ResMut<WaterStats>,
) {
    use gpu::{
        PARTICLE_FOAM_BORN, PARTICLE_FOAM_DROPPED, PARTICLE_FOAM_LIVE, PARTICLE_WAVES_BORN,
        PARTICLE_WAVES_DROPPED, PARTICLE_WAVES_LIVE,
    };
    let left = u64::from(state.squares_left).saturating_add(state.settles_left);
    state
        .unsettled
        .left(u32::try_from(left).unwrap_or(u32::MAX));
    let frame = !state.watered.is_empty();
    let live = state.live.cells().filter(|_| frame);
    let [live_steps, live_behind] = if live.is_some() {
        [steps.steps, steps.behind]
    } else {
        [0; 2]
    };
    let (cell_cut, [live_cut, live_setup, live_waiting]) = live.unwrap_or_default();
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
        white_water: state
            .emitters
            .values()
            .map(emitters::Emitters::white_water)
            .sum(),
        failed_jobs: state.failures.count,
    };
    stats.set_if_neq(next);
}

fn publish_piece(shown: Res<DebugShown>, mut piece: ResMut<gpu::WaterPiece>) {
    let mine = shown.owner == Some(std::any::TypeId::of::<WaterBrick>());
    piece.set_if_neq(gpu::WaterPiece {
        debug: [u32::from(mine), debug::shown(shown.local), 0, 0],
    });
}
