use engine_core::smoke::{SmokeResult, SmokeSession};

pub use engine_render::place_smokes::frames;

pub fn lighting(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "lighting")
}

pub fn meshes(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "meshes")
}

pub fn sun(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "sun")
}

pub fn shadows(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "shadows")
}

pub fn lights(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "lights")
}

pub fn sky(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "sky")
}

pub fn antialiasing(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "antialiasing")
}

pub fn exposure(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "exposure")
}

pub fn tonemap(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "tonemap")
}

pub fn bloom(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "bloom")
}

pub fn dither(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "dither")
}

pub fn reflections(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "reflections")
}

pub fn probe_light(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "probe_light")
}

pub fn occupancy(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "occupancy")
}

pub fn placed(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "placed")
}

pub fn cover(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "cover")
}

pub fn horizons(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "horizons")
}

pub fn heightfield(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "heightfield")
}

pub fn ground_tiles(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "ground_tiles")
}

pub fn water(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "water")
}

pub fn ground_cover(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "ground_cover")
}

pub fn scatter(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "scatter")
}

pub fn terrain_shading(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "terrain_shading")
}

pub fn probes(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "probes")
}

pub fn particles(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "particles")
}

pub fn debug_view(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "debug_view")
}

pub fn post_effects(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "post_effects")
}
