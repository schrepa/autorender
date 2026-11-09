#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod debug;
mod gpu;
pub mod jobs;
mod settings;
mod shadings;
mod state;
mod terrain;
mod terrain_bounce;
mod terrain_horizons;
mod terrain_macro;
mod terrain_maps;
mod terrain_sky;
mod trace;
mod world_trace;

use bevy_ecs::prelude::*;
use engine_core::{Attributes, Brick, Bricks, Cook, CookList, JobKind, KindDecl, Schedule, Source};
use engine_places_rendering::lighting::{Ambient, SunBlockers};
use engine_places_rendering::reflections::{TraceDepths, TraceSource};
use engine_places_rendering::sky::OpenSky;
use engine_places_world::ground::{Bounce, GroundBaking, GroundTiles, Heightfield, Horizons};
use engine_render::{DebugModes, RenderBricks, SurfaceKind};

pub use gpu::HorizonsPiece;
pub use jobs::shading::TileShading;
pub use settings::{MAX_BOUNCE_LEVELS, TerrainShading, TerrainShadingStats};

use terrain::TerrainGpu;

pub struct TerrainShadingBrick;

impl Brick for TerrainShadingBrick {
    const SOURCE: &'static Source = &SOURCE;
    const JOBS: &'static [&'static JobKind] = &[
        &jobs::BOUNCE,
        &jobs::LIGHTING,
        &jobs::MACRO_COLORS,
        &jobs::SHADING,
    ];

    fn register(bricks: &mut Bricks<'_>) {
        use shaders::terrain_shading::{ambient, blockers, faces, modes, trace_depth};
        bricks
            .when(|shading: Res<TerrainShading>| shading.enabled)
            .settings::<TerrainShading>()
            .stats::<TerrainShadingStats>()
            .errors(&[engine_heightfield::INVALID_TERRAIN])
            .resource::<state::Bakes>()
            .resource::<state::Atlas>()
            .resource::<state::SurfaceTotals>()
            .resource::<state::PartStats>()
            .resource::<state::RecordsSettling>()
            .piece::<HorizonsPiece>()
            .generated(debug::GENERATED)
            .generated(gpu::GENERATED)
            .surface_kind(SurfaceKind { name: "ground" })
            .join::<TraceDepths>((), Some(trace_depth::MODULE))
            .join::<CookList>(Cook::new(shadings::cook), None)
            .fill::<Heightfield>(Some(faces::heightfield::MODULE))
            .fill::<GroundTiles>(Some(faces::ground_tiles::MODULE))
            .fill::<GroundBaking>(None)
            .fill::<TraceSource>(Some(faces::trace_source::MODULE))
            .within(horizons_on, |bricks| {
                bricks
                    .fill::<Horizons>(Some(faces::horizons::MODULE))
                    .fill::<OpenSky>(Some(faces::open_sky::MODULE))
                    .join::<SunBlockers>((), Some(blockers::MODULE))
                    .within(bounce_on, |bricks| {
                        bricks
                            .fill::<Bounce>(Some(faces::bounce::MODULE))
                            .join::<Ambient>((), Some(ambient::MODULE));
                    });
            });
        for (index, (name, _)) in debug::MODES.into_iter().enumerate() {
            let wgsl = (index == 0).then_some(modes::MODULE);
            bricks.join::<DebugModes>(debug::member(name), wgsl);
        }
        bricks
            .systems(
                Schedule::ViewSync,
                (
                    state::bake,
                    shadings::shade,
                    state::color,
                    state::place_tiles,
                    state::publish_piece,
                    state::publish_heightfield,
                    state::count,
                ),
            )
            .part::<TerrainGpu>()
            .part::<trace::TraceGpu>()
            .draw_kind(KindDecl {
                kind: terrain::TERRAIN,
                vertex: shaders::terrain_shading::draw::VS_TERRAIN.name(),
                needs: Attributes::NONE,
                casts_shadows: true,
                seen_by_probes: true,
            })
            .draws::<TerrainGpu>(terrain::TERRAIN, TerrainGpu::draw);
    }
}

fn horizons_on(shading: Res<TerrainShading>) -> bool {
    shading.sun_shadows
}

fn bounce_on(shading: Res<TerrainShading>) -> bool {
    shading.bounce
}
