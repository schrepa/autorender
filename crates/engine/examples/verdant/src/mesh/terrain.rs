use engine::math;
use glam::{Vec2, Vec3};

use crate::mesh::{MAT_TERRAIN, MAT_WATER, MeshBuilder, SurfaceStyle};
use crate::terrain::{WATER_LEVEL, path_clearance, terrain_height, terrain_normal};

fn grid_coord(u: f32, span: f32) -> f32 {
    let s = u * 2.0 - 1.0;
    let abs_s = s.abs();
    let curved = abs_s * 0.42 + abs_s * abs_s * abs_s * 0.58;
    s.signum() * curved * span
}

pub fn build_terrain(builder: &mut MeshBuilder) {
    let res: u32 = 192;
    let span = 350.0;
    let base = builder.vertices.len() as u32;
    for iz in 0..=res {
        let vz = iz as f32 / res as f32;
        let z = grid_coord(vz, span) - 38.0;
        for ix in 0..=res {
            let vx = ix as f32 / res as f32;
            let x = grid_coord(vx, span);
            let y = terrain_height(x, z);
            let normal = terrain_normal(x, z);
            let path = path_clearance(x, z);
            let dist = (x * x + z * z).sqrt();
            let ao = math::clamp(0.76 + 0.24 * normal.y, 0.4, 1.0);
            let style = SurfaceStyle {
                mat: MAT_TERRAIN,
                ao,
                moss: 0.0,
                tint: [0.22 + 0.08 * path, 0.34 - 0.04 * path, 0.14 + 0.05 * path, 0.0],
            };
            builder.push_vertex(
                Vec3::new(x, y, z),
                normal,
                Vec2::new(x * 0.1, z * 0.1),
                Vec2::new(path, math::clamp(dist / 240.0, 0.0, 1.0)),
                style,
            );
        }
    }
    let stride = res + 1;
    for iz in 0..res {
        for ix in 0..res {
            let a = base + iz * stride + ix;
            let b = a + 1;
            let c = a + stride + 1;
            let d = a + stride;
            builder.push_quad(a, d, c, b, false);
        }
    }

    let water_res: u32 = 24;
    let water_base = builder.vertices.len() as u32;
    let water_x_span = 360.0;
    let water_style = SurfaceStyle {
        mat: MAT_WATER,
        ao: 1.0,
        moss: 0.0,
        tint: [0.06, 0.14, 0.22, 0.0],
    };
    for iz in 0..=water_res {
        let fz = iz as f32 / water_res as f32;
        let z = -20.0 - fz * 260.0;
        for ix in 0..=water_res {
            let fx = ix as f32 / water_res as f32;
            let x = (fx * 2.0 - 1.0) * water_x_span;
            builder.push_vertex(
                Vec3::new(x, WATER_LEVEL, z),
                Vec3::Y,
                Vec2::new(x * 0.02, z * 0.02),
                Vec2::ZERO,
                water_style,
            );
        }
    }
    let w_stride = water_res + 1;
    for iz in 0..water_res {
        for ix in 0..water_res {
            let a = water_base + iz * w_stride + ix;
            let b = a + 1;
            let c = a + w_stride + 1;
            let d = a + w_stride;
            builder.push_quad(a, b, c, d, false);
        }
    }
}
