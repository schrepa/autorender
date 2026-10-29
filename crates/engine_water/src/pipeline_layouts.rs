use engine_faces::heightfield::TERRAIN;
use engine_render::bindings::EMPTY;

use crate::bindings::{WATER, WATER_BEHIND, WATER_PARTICLES, WATER_SETUP, WATER_SIM};

engine_render::pipeline_layouts! {
    WATER_PIPELINES = [(frame without []), WATER, WATER_BEHIND];
    WATER_SIM_PIPELINES = [WATER_SIM];
    WATER_SETUP_PIPELINES = [EMPTY, TERRAIN, WATER_SETUP];
    WATER_PARTICLE_PIPELINES = [(frame without []), WATER, WATER_PARTICLES];
}
