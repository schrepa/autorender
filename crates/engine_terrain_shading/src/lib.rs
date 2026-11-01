include!(concat!(env!("OUT_DIR"), "/brick.rs"));

pub mod bounce;
mod debug;
mod gpu;
pub mod jobs;
pub mod lighting;
pub mod macro_colors;
mod settings;
mod shading;
mod state;
mod terrain;
mod terrain_bounce;
mod terrain_horizons;
mod terrain_macro;
mod terrain_maps;
mod terrain_sky;

use bevy_ecs::prelude::*;
use engine_core::{Brick, Bricks, JobKind, Schedule, Source};
use engine_places::ground::{Bounce, GroundTiles, Heightfield, Horizons};
use engine_places::shadows::{ShadowCasters, ShadowMaps};
use engine_places::probes::ProbeViews;
use engine_render::{DebugMode, DebugModes, Layer, MainView, RenderBricks};

pub use gpu::HorizonsPiece;
pub use settings::{MAX_BOUNCE_LEVELS, TerrainShading, TerrainShadingStats};
pub use shading::TileShading;

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
        use shaders::terrain_shading::faces;
        bricks
            .when(|shading: Res<TerrainShading>| shading.enabled)
            .settings::<TerrainShading>()
            .stats::<TerrainShadingStats>()
            .resource::<state::Bakes>()
            .resource::<state::Atlas>()
            .resource::<state::SurfaceTotals>()
            .resource::<state::PartStats>()
            .piece::<HorizonsPiece>()
            .fill::<Heightfield>(Some(faces::heightfield::MODULE))
            .fill::<GroundTiles>(Some(faces::ground_tiles::MODULE))
            .within(horizons_on, |bricks| {
                bricks
                    .fill::<Horizons>(Some(faces::horizons::MODULE))
                    .within(bounce_on, |bricks| {
                        bricks.fill::<Bounce>(Some(faces::bounce::MODULE));
                    });
            })
            .join::<DebugModes>(
                DebugMode {
                    name: "terrain",
                    decode: debug::terrain,
                },
                None,
            )
            .systems(
                Schedule::ViewSync,
                (
                    state::bake,
                    state::color,
                    state::place_tiles,
                    state::publish_changes,
                    state::publish_piece,
                    state::rest_surface,
                    state::count,
                ),
            )
            .off(state::forget)
            .part::<TerrainGpu>()
            .draws::<TerrainGpu, MainView>(Layer::Opaque, TerrainGpu::draw_main)
            .draws::<TerrainGpu, ShadowCasters>((), TerrainGpu::draw_casters)
            .draws::<TerrainGpu, ProbeViews>((), TerrainGpu::draw_probe);
    }
}

fn horizons_on(shading: Res<TerrainShading>, maps: Res<ShadowMaps>) -> bool {
    maps.on && shading.sun_shadows
}

fn bounce_on(shading: Res<TerrainShading>) -> bool {
    shading.bounce
}
