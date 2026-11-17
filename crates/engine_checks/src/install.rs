use engine::Engine;
use engine_render::Installation;

pub fn noop(engine: &mut Engine) -> Result<Installation, String> {
    engine_render::install_noop(engine.app.world_mut())
        .map_err(|problems| crate::report(&problems).err().unwrap_or_default())
}
