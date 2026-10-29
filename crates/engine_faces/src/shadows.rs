use engine_core::Pyramid;
use engine_render::bindings::Sample::Depth;
use engine_render::bindings::View::D2Array;
use engine_render::bindings::{Stages, Table, dynamic_uniform, read_of, texture};
use engine_render::gpu_data::{ColorData, PoseData};

const V: Stages = Stages::VERTEX;
const F: Stages = Stages::FRAGMENT;
const VF: Stages = Stages::VERTEX_FRAGMENT;

pub const SHADOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth16Unorm;

pub const GROUND_TEXELS: f32 = 2.0;

pub const ROUTES_BYTES: usize = engine_core::MAX_MOVING_TILES as usize
    * (size_of::<[[f32; 4]; 4]>() + size_of::<[f32; 4]>())
    + size_of::<[f32; 4]>();

pub const SHADOW_VIEW: Table = Table {
    label: "engine/shadow-view-layout",
    bindings: &[
        dynamic_uniform("view", 96, V),
        read_of("routes", ROUTES_BYTES, VF),
    ],
};

pub const SHADOW_POSES: Table = Table {
    label: "engine/shadow-pose-layout",
    bindings: &[
        read_of("slot_poses", size_of::<PoseData>(), V),
        read_of("slot_colors", size_of::<ColorData>(), V),
        texture("shadow_regions", Depth, D2Array, F),
    ],
};

#[derive(Clone, Copy, Debug)]
pub struct KeptRefresh {
    pub planes: [[f32; 4]; 6],
    pub texel: f32,
}

pub fn ground_level(pyramid: &Pyramid, texel: f32) -> u32 {
    let coarsest = pyramid.levels().saturating_sub(1);
    (0..coarsest)
        .find(|&level| pyramid.spacing(level) >= GROUND_TEXELS * texel)
        .unwrap_or(coarsest)
}

pub trait ShadowCastersFace {
    fn region_views(&self) -> &[(usize, KeptRefresh)];
}
