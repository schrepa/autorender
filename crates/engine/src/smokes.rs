mod hierarchy;
mod meshes;
mod scene_kit;
mod spline;
mod streamed_scene;
mod surfaces;
mod terrain_views;
mod texture;

use engine_core::CORE_ROW;
use engine_core::smoke::{SmokeEntry, SmokeResult, SmokeSession};

fn entry(
    name: &'static str,
    gpu: bool,
    run: fn(&mut dyn SmokeSession) -> SmokeResult,
) -> SmokeEntry {
    SmokeEntry {
        name,
        brick: CORE_ROW,
        gpu,
        run,
    }
}

pub fn core() -> Vec<SmokeEntry> {
    vec![
        entry("hierarchy", true, hierarchy::run),
        entry("spline", true, spline::run),
        entry("meshes", true, meshes::run),
        entry("surfaces", true, surfaces::run),
        entry("texture", true, texture::run),
        entry("streamed_scene", true, streamed_scene::run),
        entry("debug_view_terrain", true, terrain_views::run),
    ]
}
