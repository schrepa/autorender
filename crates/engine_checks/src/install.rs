use engine::{Engine, GpuRequest};
use engine_render::Installation;

pub fn noop(engine: &mut Engine, gpu: GpuRequest) -> Result<Installation, String> {
    engine_render::install_noop(engine.app.world_mut(), gpu)
        .map_err(|problems| crate::report(&problems).err().unwrap_or_default())
}
