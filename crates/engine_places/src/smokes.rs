use engine_place::{Runs, Smoke, SmokeFailure, SmokeSession, smoke_eq};

const fn frames(name: &'static str) -> Smoke {
    Smoke {
        name,
        runs: Runs::Gpu,
        run: drawn,
    }
}

fn drawn(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    smoke_eq!(session.frame(), true, "frame_drawn");
    Ok(())
}

pub const LIGHTING: Smoke = frames("slot_lighting");
pub const SUN: Smoke = frames("slot_sun");
pub const SHADOWS: Smoke = frames("slot_shadows");
pub const LIGHTS: Smoke = frames("slot_lights");
pub const SKY: Smoke = frames("slot_sky");
pub const EXPOSURE: Smoke = frames("slot_exposure");
pub const TONEMAP: Smoke = frames("slot_tonemap");
pub const BLOOM: Smoke = frames("slot_bloom");
pub const DITHER: Smoke = frames("slot_dither");
pub const REFLECTIONS: Smoke = frames("slot_reflections");
pub const PROBE_LIGHT: Smoke = frames("slot_probe_light");
pub const OCCUPANCY: Smoke = frames("slot_occupancy");
pub const PLACED_TILES: Smoke = frames("slot_placed_tiles");
pub const COVER: Smoke = frames("slot_cover");
pub const HORIZONS: Smoke = frames("slot_horizons");
pub const HEIGHTFIELD: Smoke = frames("slot_heightfield");
pub const GROUND_TILES: Smoke = frames("slot_ground_tiles");
pub const WATER: Smoke = frames("slot_water");
pub const INSTANCES: Smoke = frames("slot_instances");
pub const TRANSPARENCY: Smoke = frames("slot_transparency");
