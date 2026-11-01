use engine::Engine;
use engine_core::BrickList;
use engine_render::Installation;
use serde_json::json;

pub type List = fn(BrickList) -> BrickList;

pub fn engine(list: List) -> Result<Engine, String> {
    engine::engine(list).map_err(|error| error.to_string())
}

pub fn noop(engine: &mut Engine) -> Result<Installation, String> {
    engine_render::install_noop(engine.app.world_mut())
        .map_err(|problems| crate::report(&problems).err().unwrap_or_default())
}

pub fn check(engine: &mut Engine) -> Result<(), String> {
    let installed = noop(engine)?;
    println!(
        "{}",
        json!({
            "event": "installed",
            "parts": installed.parts.len(),
            "modules": installed.modules.len(),
            "layouts": installed.layouts,
            "frame_variants": installed.frame_variants,
        })
    );
    Ok(())
}
