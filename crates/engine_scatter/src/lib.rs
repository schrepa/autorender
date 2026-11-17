#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod far_cover;
mod gpu;
mod groups;
mod hidden;
mod jobs;
mod layouts;
mod pool;
mod settings;
mod snapping;
mod systems;

use engine_core::{
    Attributes, BASE_COLOR_MAP, Brick, Bricks, JobKind, KindDecl, Module, NORMAL_MAP, Schedule,
    Source, SurfaceSlot,
};
use engine_open_world_kit::ground::{FarCover, Foliage, PlacedTiles};
use engine_places::surface::{
    METALLIC_ROUGHNESS_MAP, METALNESS, OCCLUSION, OCCLUSION_MAP, OCCLUSION_STRENGTH, ROUGHNESS,
    TRANSMISSION,
};
use engine_places_rendering::lighting::inputs;
use engine_render::{PointInput, RenderBricks, SlotModule, SlotPrepare};

pub use settings::{
    CanopySettings, FoliageSettings, PlacedLook, PlacedPart, ScatterSettings, ScatterStats,
};

const SURFACE_MAPS: &[&str] = &[BASE_COLOR_MAP, NORMAL_MAP, METALLIC_ROUGHNESS_MAP];

const GROUNDED: [&str; 4] = [
    SurfaceSlot::BaseColor.name(),
    SurfaceSlot::Normal.name(),
    ROUGHNESS,
    METALNESS,
];

pub(crate) const WIND: &str = "wind";

const CARD_READS: [PointInput; 3] = [inputs::ROUGHNESS, inputs::METALNESS, inputs::OCCLUSION];

const CARD_SETS: [PointInput; 5] = [
    inputs::ROUGHNESS,
    inputs::METALNESS,
    inputs::TRANSMISSION,
    inputs::DIRECT_SOFTNESS,
    inputs::DIRECT_BLOCKED,
];

pub(crate) const PLANT_LOOK: [&str; 4] = [
    "light_through",
    "crown_softness",
    "hue_variation",
    "brightness_variation",
];

type Look = (
    &'static str,
    Module,
    &'static [&'static str],
    &'static [&'static str],
    &'static [&'static str],
    f32,
);

const LOOKS: [Look; 4] = [
    (
        "grounded",
        shaders::scatter::grounded::MODULE,
        &GROUNDED,
        SURFACE_MAPS,
        &[],
        0.0,
    ),
    (
        "foliage",
        shaders::scatter::leaves::MODULE,
        &[
            SurfaceSlot::BaseColor.name(),
            SurfaceSlot::Opacity.name(),
            SurfaceSlot::Cutoff.name(),
            SurfaceSlot::Normal.name(),
            ROUGHNESS,
            METALNESS,
            OCCLUSION,
            TRANSMISSION,
            SurfaceSlot::Kind.name(),
            inputs::DIRECT_SOFTNESS.name,
        ],
        SURFACE_MAPS,
        &[
            "crown_shade",
            PLANT_LOOK[0],
            PLANT_LOOK[1],
            PLANT_LOOK[2],
            PLANT_LOOK[3],
        ],
        0.0,
    ),
    (
        WIND,
        shaders::scatter::wind::MODULE,
        &[SurfaceSlot::Position.name()],
        &[],
        &[],
        layouts::WIND_REACH,
    ),
    (
        "world_mapped",
        shaders::scatter::world_mapped::MODULE,
        &[
            SurfaceSlot::BaseColor.name(),
            SurfaceSlot::Normal.name(),
            ROUGHNESS,
            METALNESS,
            OCCLUSION,
        ],
        &[
            BASE_COLOR_MAP,
            NORMAL_MAP,
            METALLIC_ROUGHNESS_MAP,
            OCCLUSION_MAP,
        ],
        &[
            "world_tiling",
            "top_red",
            "top_green",
            "top_blue",
            "top_coverage",
            "top_roughness",
            OCCLUSION_STRENGTH,
        ],
        0.0,
    ),
];

pub struct ScatterBrick;

impl Brick for ScatterBrick {
    const SOURCE: &'static Source = &SOURCE;
    const JOBS: &'static [&'static JobKind] = &[&jobs::CANOPY];

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<ScatterSettings>()
            .settings::<FoliageSettings>()
            .settings::<CanopySettings>()
            .stats::<ScatterStats>()
            .resource::<systems::ScatterState>()
            .resource::<systems::ScatterFeedback>()
            .resource::<systems::Resnapped>()
            .resource::<far_cover::CanopyMaps>()
            .piece::<layouts::CanopyPiece>()
            .generated(layouts::GENERATED)
            .fill::<Foliage>(Some(shaders::scatter::faces::foliage::MODULE))
            .fill::<PlacedTiles>(Some(shaders::scatter::faces::placed_tiles::MODULE))
            .join::<FarCover>((), Some(shaders::scatter::faces::far_cover::MODULE))
            .systems(
                Schedule::ViewSync,
                (
                    systems::publish_foliage,
                    systems::take_impostors,
                    systems::place_pool,
                    far_cover::bake_canopy,
                    systems::publish_canopy,
                    systems::publish_assets,
                    systems::plan_cull,
                    systems::publish_tiles,
                    systems::publish_changes,
                    systems::scatter_stats,
                ),
            )
            .draw_kind(KindDecl {
                kind: gpu::PLACED,
                needs: Attributes::NONE,
                streams: &[],
                skips: &[],
            })
            .draws::<gpu::ScatterGpu>(gpu::PLACED, gpu::ScatterGpu::draw)
            .slot_prepare(SlotPrepare {
                module: shaders::scatter::world_mapped::MODULE,
            });
        for (name, module, slots, maps, values, reach) in LOOKS {
            for &slot in slots {
                bricks.material_slot(SlotModule {
                    slot,
                    name,
                    module,
                    maps,
                    values,
                    reach,
                });
            }
        }
        for input in CARD_READS {
            bricks.point_input(input);
        }
        for input in CARD_SETS {
            bricks.point_set(input.set());
        }
    }
}
