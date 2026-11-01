use crate::bindings::{
    EMPTY, IMPOSTOR_BAKE, MATERIAL, MIPMAPS, PRESENTATION_COMPACT, PRESENTATION_COMPOSE,
    PRESENTATION_CULL, PRESENTATION_FADES, PRESENTATION_LEVEL, PRESENTATION_TESTS, PYRAMID_DOWN,
    PYRAMID_FIRST, Table,
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
    pub groups: &'static [&'static Table],
    pub limits: StageLimits,
}

macro_rules! pipeline_layouts {
    ($(
        $name:ident = [$($group:ident),+ $(,)?] $(needing { $($field:ident: $value:literal),+ $(,)? })?;
    )*) => {
        $(
            pub const $name: PipelineLayout = PipelineLayout {
                label: concat!("engine/", stringify!($name)),
                groups: &[$(&$group),+],
                limits: StageLimits {
                    $($($field: $value,)+)?
                    ..REQUIRED_LIMITS
                },
            };
        )*

        pub const PIPELINE_LAYOUTS: &[(&str, &PipelineLayout)] = &[$((stringify!($name), &$name)),*];
    };
}

pipeline_layouts! {
    IMPOSTOR_BAKE_PIPELINES = [EMPTY, MATERIAL, IMPOSTOR_BAKE];
    MIPMAP_PIPELINES = [MIPMAPS];
    PYRAMID_FIRST_PIPELINES = [PYRAMID_FIRST];
    PYRAMID_DOWN_PIPELINES = [PYRAMID_DOWN];
    PRESENTATION_COMPOSE_PIPELINES = [PRESENTATION_COMPOSE, PRESENTATION_LEVEL];
    PRESENTATION_CULL_PIPELINES = [PRESENTATION_CULL];
    PRESENTATION_TEST_PIPELINES = [PRESENTATION_TESTS];
    PRESENTATION_COMPACT_PIPELINES = [PRESENTATION_COMPACT];
    PRESENTATION_FADE_PIPELINES = [PRESENTATION_FADES];
}
