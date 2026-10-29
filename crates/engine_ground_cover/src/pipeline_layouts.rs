use engine_render::bindings::EMPTY;

use crate::bindings::{COVER_DRAW, COVER_GROW};

engine_render::pipeline_layouts! {
    LIT_COVER_PIPELINES = [(frame without [refraction]), COVER_DRAW, EMPTY, (lights without [])];
    COVER_GROW_PIPELINES = [COVER_GROW];
}
