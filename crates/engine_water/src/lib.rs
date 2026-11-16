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
mod look;
mod rewater;
mod settings;
mod sights;
mod state;
mod views;
mod water;
mod water_blocks;
mod water_detail;
mod water_falls;
mod water_source;
mod waterfalls;
mod waves;

use bevy_ecs::prelude::*;
use engine_core::{
    Attributes, Brick, Bricks, ColorPass, JobKind, KindDecl, KindSkip, PassRole, PassViews,
    Schedule, Source, SurfaceSlot, stats,
};
use engine_open_world_kit::ground::Heightfield;
use engine_open_world_kit::water::Waterline;
use engine_places_rendering::lighting::inputs;
use engine_places_rendering::reflections::TraceDepths;
use engine_places_rendering::sky::EyeMedium;
use engine_render::{DebugModes, DebugShown, PointInput, RenderBricks, SlotModule, SurfaceKind};

pub use flow::{Block, FlowState, Pool, RiverFlow, WEIR};
pub use jobs::tile_water::*;
pub use settings::{
    Water, WaterBoils, WaterBumps, WaterCaustics, WaterDebugView, WaterDriftLines, WaterFallSheets,
    WaterFallSpray, WaterFalls, WaterFlow, WaterFoam, WaterLace, WaterMist, WaterOptics,
    WaterRipples, WaterShore, WaterWaves,
};
pub use views::PoolView;

use state::WaterState;

stats! {
    #[section("water_stats")]
    pub struct WaterStats {
        pub blocks: u32,
        pub dropped_blocks: u32,
        pub leaves: u32,
        pub rewatered_tiles: u32,
        pub stale_water_tiles: u32,
        pub falls: u32,
        pub failed_jobs: u64,
    }
}

const SKIPS: &[KindSkip] = &[
    KindSkip::role(
        PassRole::Depth,
        PassViews::Owned,
        "the water surface casts no shadows",
    ),
    KindSkip::role(
        PassRole::Lit,
        PassViews::Owned,
        "the water surface stays out of probe captures",
    ),
    KindSkip::pass::<ColorPass>(
        "the water surface draws in its own pass, after the scene it refracts",
    ),
    KindSkip::role(
        PassRole::Depth,
        PassViews::Main,
        "the water surface refracts the scene behind it, so it fills no depth before it",
    ),
];

const SETS: [PointInput; 8] = [
    inputs::SMOOTHER_ACROSS,
    inputs::DIRECT_BLOCKED,
    inputs::ADDED_LIGHT,
    inputs::ROUGHNESS,
    inputs::MIRRORED_ROUGH,
    inputs::METALNESS,
    inputs::OCCLUSION,
    inputs::TRANSMISSION,
];

pub struct WaterBrick;

impl Brick for WaterBrick {
    const SOURCE: &'static Source = &SOURCE;
    const JOBS: &'static [&'static JobKind] = &[&jobs::TILE_WATER, &jobs::DRAWN, &jobs::REWATER];

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .switch::<Water>()
            .stats::<WaterStats>()
            .errors(&[engine_heightfield::INVALID_TERRAIN])
            .resource::<WaterState>()
            .resource::<waves::Waves>()
            .piece::<gpu::WaterPiece>()
            .surface_kind(SurfaceKind { name: "water" })
            .join::<TraceDepths>((), Some(shaders::water::trace_depth::MODULE))
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
                skips: SKIPS,
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
                    rewater::rewater,
                    drawn::draw,
                    water_blocks::place_blocks,
                    waves::follow,
                    state::publish_waterline,
                    state::publish_bodies,
                    eye::publish_medium,
                    publish_piece,
                    water_stats,
                ),
            )
            .systems(Schedule::WorldSync, emitters::sync)
            .systems(Schedule::FixedSimulation, flow_step::step)
            .generated(gpu::GENERATED)
            .part::<water::WaterGpu>();
        bricks.point_input(inputs::ROUGHNESS);
        for input in SETS {
            bricks.point_set(input.set());
        }
    }
}

fn water_stats(
    state: Res<WaterState>,
    heightfield: Res<Heightfield>,
    mut stats: ResMut<WaterStats>,
) {
    let frame = !state.watered.is_empty();
    let next = WaterStats {
        blocks: if frame { state.blocks.resident() } else { 0 },
        dropped_blocks: if frame { state.blocks.dropped } else { 0 },
        leaves: if frame { heightfield.water_leaves } else { 0 },
        rewatered_tiles: state.rewatered_tiles,
        stale_water_tiles: state.stale_water_tiles,
        falls: state
            .terrains
            .iter()
            .map(|terrain| terrain.falls.len() as u32)
            .sum(),
        failed_jobs: state.failures.count,
    };
    stats.set_if_neq(next);
}

fn publish_piece(
    shown: Res<DebugShown>,
    water: Res<Water>,
    mut piece: ResMut<gpu::WaterPiece>,
) {
    let mine = shown.owner == Some(std::any::TypeId::of::<WaterBrick>());
    let view = &water.debug_view;
    piece.set_if_neq(gpu::WaterPiece {
        debug: [u32::from(mine), debug::shown(shown.local), 0, 0],
        ranges: [view.deepest, view.fastest, view.froude_top, view.critical],
        breaking: [water.foam.breaking[0], water.foam.breaking[1], 0.0, 0.0],
    });
}
