#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod debug;
mod gpu;
mod holes;
pub mod jobs;
mod settings;
mod shadings;
mod state;
mod terrain;
mod terrain_bounce;
mod terrain_horizons;
mod terrain_macro;
mod terrain_maps;
mod terrain_rough;
mod terrain_sky;
mod terrain_sun;
mod trace;
mod world_trace;

use bevy_ecs::prelude::*;
use engine_core::{
    Attributes, BlendedPass, Brick, Bricks, ColorPass, Cook, CookList, JobKind, KindDecl, KindSkip,
    Schedule, Source,
};
use engine_open_world_kit::ground::{Bounce, GroundBaking, GroundTiles, Heightfield, Horizons};
use engine_places_rendering::lighting::{Ambient, SunBlockers, inputs};
use engine_places_rendering::reflections::{TraceDepths, TraceSource};
use engine_render::{DebugModes, PointInput, RenderBricks, SurfaceKind};

pub use gpu::HorizonsPiece;
pub use jobs::shading::TileShading;
pub use settings::{MAX_BOUNCE_LEVELS, TerrainShading, TerrainShadingStats};

use terrain::TerrainGpu;

const SKIPS: &[KindSkip] = &[
    KindSkip::pass::<ColorPass>(terrain::SHADED_ELSEWHERE),
    KindSkip::pass::<BlendedPass>(terrain::NEVER_BLENDS),
];

const SETS: [PointInput; 13] = [
    inputs::ADDED_INDIRECT,
    inputs::AMBIENT_BEND,
    inputs::IGNORE_FACE,
    inputs::TANGLED,
    inputs::RECEIVER_LIFT_FIRST,
    inputs::RECEIVER_LIFTS,
    inputs::ROUGHNESS,
    inputs::METALNESS,
    inputs::OCCLUSION,
    inputs::TRANSMISSION,
    inputs::MIRRORED,
    inputs::DIRECT_BLOCKED,
    inputs::DIRECT_SHARE,
];

pub struct TerrainShadingBrick;

impl Brick for TerrainShadingBrick {
    const SOURCE: &'static Source = &SOURCE;
    const JOBS: &'static [&'static JobKind] = &[
        &jobs::BOUNCE,
        &jobs::LIGHTING,
        &jobs::MACRO_COLORS,
        &jobs::ROUGH,
        &jobs::SHADING,
    ];

    fn register(bricks: &mut Bricks<'_>) {
        use shaders::terrain_shading::{ambient, blockers, faces, modes, trace_depth};
        bricks
            .switch::<TerrainShading>()
            .stats::<TerrainShadingStats>()
            .errors(&[engine_heightfield::INVALID_TERRAIN])
            .resource::<state::Bakes>()
            .resource::<state::Atlas>()
            .resource::<state::SurfaceTotals>()
            .resource::<state::PartStats>()
            .resource::<state::RecordsSettling>()
            .resource::<terrain_sun::SunPlan>()
            .resource::<holes::HoledGround>()
            .resource::<terrain_rough::RoughLights>()
            .piece::<HorizonsPiece>()
            .generated(debug::GENERATED)
            .generated(gpu::GENERATED)
            .surface_kind(SurfaceKind { name: "ground" })
            .join::<TraceDepths>((), Some(trace_depth::MODULE))
            .join::<CookList>(Cook::new(shadings::cook), None)
            .join::<CookList>(Cook::new(terrain_rough::cook), None)
            .fill::<Heightfield>(Some(faces::heightfield::MODULE))
            .fill::<GroundTiles>(Some(faces::ground_tiles::MODULE))
            .fill::<GroundBaking>(None)
            .fill::<TraceSource>(Some(faces::trace_source::MODULE))
            .within(horizons_on, |bricks| {
                bricks
                    .fill::<Horizons>(Some(faces::horizons::MODULE))
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
                    terrain_rough::roughen,
                    shadings::shade,
                    state::color,
                    state::lend_layers,
                    state::place_tiles,
                    state::publish_piece,
                    terrain_sun::plan,
                    state::publish_heightfield,
                    state::count,
                    holes::hold_material,
                ),
            )
            .part::<TerrainGpu>()
            .part::<trace::TraceGpu>()
            .draw_kind(KindDecl {
                kind: terrain::TERRAIN,
                needs: Attributes::NONE,
                streams: &[],
                skips: SKIPS,
            })
            .draws::<TerrainGpu>(terrain::TERRAIN, TerrainGpu::draw);
        bricks.material_slot(holes::SLOT_MODULE);
        for input in SETS {
            bricks.point_set(input.set());
        }
    }
}

fn horizons_on(shading: Res<TerrainShading>) -> bool {
    shading.sun_shadows
}

fn bounce_on(shading: Res<TerrainShading>) -> bool {
    shading.bounce
}
