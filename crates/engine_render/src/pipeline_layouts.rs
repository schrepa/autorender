use crate::bindings::{
    EMPTY, GroupRef, IMPOSTOR_BAKE, IMPOSTOR_DRAW, MATERIAL, MIPMAPS, POST,
    PRESENTATION_COMPACT, PRESENTATION_COMPOSE, PRESENTATION_CULL, PRESENTATION_FADES,
    PRESENTATION_LEVEL, PRESENTATION_TESTS, PYRAMID_DOWN, PYRAMID_FIRST, RETAINED_POSES,
    TEMPORAL_POSES, TRANSPARENT_COMPOSITE,
};

pub const UNIFORM_BINDING_BYTES: u64 = 16_384;

#[derive(Clone, Copy, Debug)]
pub struct StageLimits {
    pub bind_groups: u32,
    pub uniform_buffers: u32,
    pub storage_buffers: u32,
    pub sampled_textures: u32,
    pub samplers: u32,
    pub storage_textures: u32,
    pub dynamic_uniform_buffers: u32,
    pub dynamic_storage_buffers: u32,
}

pub const REQUIRED_LIMITS: StageLimits = StageLimits {
    bind_groups: 4,
    uniform_buffers: 12,
    storage_buffers: 8,
    sampled_textures: 16,
    samplers: 16,
    storage_textures: 4,
    dynamic_uniform_buffers: 8,
    dynamic_storage_buffers: 4,
};

#[derive(Clone, Copy, Debug)]
pub struct PipelineLayout {
    pub label: &'static str,
    pub groups: &'static [GroupRef],
    pub limits: StageLimits,
}

#[doc(hidden)]
#[macro_export]
macro_rules! group_ref {
    (($group:ident without [$($set:ident),* $(,)?])) => {
        $crate::bindings::GroupRef::Shared {
            group: stringify!($group),
            without: &[$(stringify!($set)),*],
        }
    };
    ($table:ident) => {
        $crate::bindings::GroupRef::Own(&$table)
    };
}

#[macro_export]
macro_rules! pipeline_layouts {
    ($(
        $(#[$doc:meta])*
        $name:ident = [$($group:tt),+ $(,)?] $(needing { $($field:ident: $value:literal),+ $(,)? })?;
    )*) => {
        $(
            $(#[$doc])*
            pub const $name: $crate::pipeline_layouts::PipelineLayout =
                $crate::pipeline_layouts::PipelineLayout {
                    label: concat!("engine/", stringify!($name)),
                    groups: &[$($crate::group_ref!($group)),+],
                    limits: $crate::pipeline_layouts::StageLimits {
                        $($($field: $value,)+)?
                        ..$crate::pipeline_layouts::REQUIRED_LIMITS
                    },
                };
        )*

        pub const PIPELINE_LAYOUTS: &[(&str, &$crate::pipeline_layouts::PipelineLayout)] =
            &[$((stringify!($name), &$name)),*];
    };
}

crate::pipeline_layouts! {
    LIT_MESH_PIPELINES = [
        (frame without [refraction]),
        MATERIAL,
        RETAINED_POSES,
        (lights without []),
    ];
    LIT_TEMPORAL_MESH_PIPELINES = [
        (frame without [refraction]),
        MATERIAL,
        TEMPORAL_POSES,
        (lights without []),
    ];
    LIT_IMPOSTOR_PIPELINES = [
        (frame without [refraction]),
        IMPOSTOR_DRAW,
        RETAINED_POSES,
        (lights without []),
    ];
    LIT_TEMPORAL_IMPOSTOR_PIPELINES = [
        (frame without [refraction]),
        IMPOSTOR_DRAW,
        TEMPORAL_POSES,
        (lights without []),
    ];
    IMPOSTOR_BAKE_PIPELINES = [EMPTY, MATERIAL, IMPOSTOR_BAKE];
    POST_PIPELINES = [POST, (frame without []), (composite without [])];
    TRANSPARENT_COMPOSITE_PIPELINES = [TRANSPARENT_COMPOSITE, (frame without [])];
    MIPMAP_PIPELINES = [MIPMAPS];
    PYRAMID_FIRST_PIPELINES = [PYRAMID_FIRST];
    PYRAMID_DOWN_PIPELINES = [PYRAMID_DOWN];
    PRESENTATION_COMPOSE_PIPELINES = [PRESENTATION_COMPOSE, PRESENTATION_LEVEL];
    PRESENTATION_CULL_PIPELINES = [PRESENTATION_CULL];
    PRESENTATION_TEST_PIPELINES = [PRESENTATION_TESTS];
    PRESENTATION_COMPACT_PIPELINES = [PRESENTATION_COMPACT];
    PRESENTATION_FADE_PIPELINES = [PRESENTATION_FADES];
}

pub const LAYOUT_BUDGET: usize = 54;
