use engine_faces::heightfield::TERRAIN;
use engine_faces::shadows::SHADOW_VIEW;
use engine_render::bindings::EMPTY;

use crate::bindings::{
    BOUNCE, TERRAIN_CONTROL, TERRAIN_NORMALS, TERRAIN_RESOLVE, TERRAIN_SUBDIVISION,
    TERRAIN_SUBDIVISION_CAMERA,
};

engine_render::pipeline_layouts! {
    TERRAIN_PIPELINES = [(frame without [sky_tables, refraction]), TERRAIN];
    LIT_TERRAIN_PIPELINES = [
        (frame without [sky_tables, refraction]),
        TERRAIN,
        EMPTY,
        (lights without []),
    ];
    LIT_TERRAIN_RESOLVE_PIPELINES = [
        (frame without [sky_tables, refraction]),
        TERRAIN,
        TERRAIN_RESOLVE,
        (lights without []),
    ];
    TERRAIN_SUBDIVISION_PIPELINES = [TERRAIN_SUBDIVISION_CAMERA, TERRAIN, TERRAIN_SUBDIVISION];
    TERRAIN_CASTER_PIPELINES = [SHADOW_VIEW, TERRAIN];
    TERRAIN_NORMAL_PIPELINES = [TERRAIN_NORMALS];
    TERRAIN_CONTROL_PIPELINES = [TERRAIN_CONTROL];
    BOUNCE_PIPELINES = [BOUNCE];
}
