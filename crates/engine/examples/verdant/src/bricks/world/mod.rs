engine::brick!(world);

mod gpu;

use engine::prelude::*;
use engine_core::{GpuStruct, KindDecl, Pass, PassName, PassViews, Writes, gpu_struct};

use crate::names::{COLOR, COLOR_PASS, DEPTH, WORLD};

pub const MAX_BLADES: u32 = 1_000_000;
pub const MAX_PARTICLES: u32 = 50_000;
pub const WAKE_SIDE: u32 = 256;
pub const POINT_LIGHT_COUNT: u32 = 33;

gpu_struct! {
    pub struct WorldUniforms {
        pub time_wind: [f32; 4],
        pub camera_pos: [f32; 4],
        pub camera_fwd: [f32; 4],
        pub camera_right: [f32; 4],
        pub camera_up: [f32; 4],
        pub sun_dir: [f32; 4],
        pub sun_color: [f32; 4],
        pub sky_zenith: [f32; 4],
        pub sky_horizon: [f32; 4],
        pub haze_color: [f32; 4],
        pub ground_ambient: [f32; 4],
        pub cloud_lit: [f32; 4],
        pub cloud_shadow: [f32; 4],
        pub wake_body: [f32; 4],
        pub wake_cursor: [f32; 4],
        pub view_proj: [[f32; 4]; 4],
        pub inv_view_proj: [[f32; 4]; 4],
    }
}

gpu_struct! {
    pub struct MeshVertex {
        pub pos_mat: [f32; 4],
        pub norm_ao: [f32; 4],
        pub uv_extra: [f32; 4],
        pub tint: [f32; 4],
    }
}

gpu_struct! {
    pub struct GrassBlade {
        pub data: [f32; 4],
    }
}

gpu_struct! {
    pub struct Particle {
        pub pos_phase: [f32; 4],
        pub vel_kind: [f32; 4],
    }
}

gpu_struct! {
    pub struct PointLight {
        pub pos_radius: [f32; 4],
        pub color_intensity: [f32; 4],
    }
}

gpu_struct! {
    pub struct WakeCell {
        pub bend: [f32; 4],
    }
}

const GENERATED: &[&str] = &[
    <WorldUniforms as GpuStruct>::WGSL,
    <MeshVertex as GpuStruct>::WGSL,
    <GrassBlade as GpuStruct>::WGSL,
    <Particle as GpuStruct>::WGSL,
    <PointLight as GpuStruct>::WGSL,
    <WakeCell as GpuStruct>::WGSL,
];

pub struct ColorPass;

impl PassName for ColorPass {
    const NAME: &'static str = COLOR_PASS;
    const PASS: Pass = Pass {
        writes: Writes(&[COLOR, DEPTH]),
        views: PassViews::Every,
    };
}

pub struct WorldBrick;

impl Brick for WorldBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .pass::<ColorPass>()
            .draw_kind(KindDecl {
                kind: WORLD,
                passes: &[COLOR_PASS],
            })
            .generated(GENERATED)
            .draws::<gpu::WorldGpu>(WORLD, gpu::WorldGpu::draw_world);
    }
}
