#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod far_cover;
mod gpu;
mod groups;
mod layouts;
mod occupancy;
mod pool;
mod settings;
mod systems;

use engine_core::{
    Attributes, BASE_COLOR_MAP, Brick, Bricks, KindDecl, METALLIC_ROUGHNESS_MAP, Module,
    NORMAL_MAP, OCCLUSION_MAP, Schedule, Source, SurfaceSlot,
};
use engine_places_rendering::lighting::Ambient;
use engine_places_rendering::occlusion::OcclusionSource;
use engine_open_world_kit::ground::{FarCover, Foliage, PlacedTiles};
use engine_render::{RenderBricks, SlotModule};

pub use settings::{
    FoliageSettings, OccupancySettings, PlacedLod, PlacedLook, PlacedPart, ScatterSettings,
    ScatterStats,
};

const SURFACE_MAPS: &[&str] = &[BASE_COLOR_MAP, NORMAL_MAP, METALLIC_ROUGHNESS_MAP];

const GROUNDED: [SurfaceSlot; 4] = [
    SurfaceSlot::BaseColor,
    SurfaceSlot::Normal,
    SurfaceSlot::Roughness,
    SurfaceSlot::Metalness,
];

pub(crate) const WIND: &str = "wind";

type Look = (
    &'static str,
    Module,
    &'static [SurfaceSlot],
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
            SurfaceSlot::BaseColor,
            SurfaceSlot::Cutoff,
            SurfaceSlot::Normal,
            SurfaceSlot::Roughness,
            SurfaceSlot::Metalness,
            SurfaceSlot::Transmission,
            SurfaceSlot::Kind,
            SurfaceSlot::LightInputs,
        ],
        SURFACE_MAPS,
        &[],
        0.0,
    ),
    (
        WIND,
        shaders::scatter::wind::MODULE,
        &[SurfaceSlot::Position],
        &[],
        &[],
        layouts::WIND_REACH,
    ),
    (
        "world_mapped",
        shaders::scatter::world_mapped::MODULE,
        &[
            SurfaceSlot::BaseColor,
            SurfaceSlot::Normal,
            SurfaceSlot::Roughness,
            SurfaceSlot::Metalness,
            SurfaceSlot::Occlusion,
        ],
        &[BASE_COLOR_MAP, NORMAL_MAP, METALLIC_ROUGHNESS_MAP, OCCLUSION_MAP],
        &[
            "world_tiling",
            "top_red",
            "top_green",
            "top_blue",
            "top_coverage",
            "top_roughness",
        ],
        0.0,
    ),
];

pub struct ScatterBrick;

impl Brick for ScatterBrick {
    const SOURCE: &'static Source = &SOURCE;
    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<ScatterSettings>()
            .settings::<FoliageSettings>()
            .settings::<OccupancySettings>()
            .stats::<ScatterStats>()
            .resource::<systems::ScatterState>()
            .resource::<systems::ScatterFeedback>()
            .resource::<occupancy::OccupancyWindows>()
            .resource::<far_cover::FarCoverTable>()
            .piece::<layouts::CanopyPiece>()
            .generated(layouts::GENERATED)
            .fill::<Foliage>(Some(shaders::scatter::faces::foliage::MODULE))
            .fill::<PlacedTiles>(Some(shaders::scatter::faces::placed_tiles::MODULE))
            .fill::<OcclusionSource>(None)
            .join::<Ambient>((), Some(shaders::scatter::ambient::MODULE))
            .join::<FarCover>((), Some(shaders::scatter::faces::far_cover::MODULE))
            .systems(
                Schedule::ViewSync,
                (
                    systems::publish_foliage,
                    systems::take_impostors,
                    systems::place_windows,
                    systems::place_pool,
                    systems::publish_assets,
                    systems::plan_cull,
                    systems::mark_moved,
                    systems::publish_tiles,
                    systems::publish_changes,
                    systems::scatter_stats,
                ),
            )
            .draw_kind(KindDecl {
                kind: gpu::PLACED,
                vertex: shaders::scatter::draw::VS_OBJECT.name(),
                needs: Attributes::NONE,
                casts_shadows: true,
                seen_by_probes: true,
            })
            .draws::<gpu::ScatterGpu>(gpu::PLACED, gpu::ScatterGpu::draw);
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
    }
}
