use crate::bindings::{MIPMAPS, Table};

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

impl StageLimits {
    pub(crate) fn of(limits: &wgpu::Limits) -> Self {
        Self {
            bind_groups: limits.max_bind_groups,
            uniform_buffers: limits.max_uniform_buffers_per_shader_stage,
            storage_buffers: limits.max_storage_buffers_per_shader_stage,
            sampled_textures: limits.max_sampled_textures_per_shader_stage,
            samplers: limits.max_samplers_per_shader_stage,
            storage_textures: limits.max_storage_textures_per_shader_stage,
            dynamic_uniform_buffers: limits.max_dynamic_uniform_buffers_per_pipeline_layout,
            dynamic_storage_buffers: limits.max_dynamic_storage_buffers_per_pipeline_layout,
        }
    }
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
    MIPMAP_PIPELINES = [MIPMAPS];
}
