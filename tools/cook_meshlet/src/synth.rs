use engine_math::{atan2, clamp, cos, sin};
use engine_meshlet::SourceMesh;
use glam::{Vec2, Vec3};

use crate::gltf_load::compute_curvature_ao;

pub fn build_terrain_column_mesh() -> SourceMesh {
    let cols = 175_usize;
    let rows = 174_usize;
    let v_count = cols * rows;
    let mut positions = Vec::with_capacity(v_count);
    let mut uvs = Vec::with_capacity(v_count);
    let mut blend = Vec::with_capacity(v_count);
    let span = 32.0_f32;
    for rz in 0..rows {
        let vz = (rz as f32) / ((rows - 1) as f32);
        let z = (vz - 0.5) * span;
        for rx in 0..cols {
            let ux = (rx as f32) / ((cols - 1) as f32);
            let x = (ux - 0.5) * span;
            let r = (x * x + z * z).sqrt();
            let angle = atan2(z, x);
            let hills = 1.6 * sin(x * 0.24) * cos(z * 0.21)
                + 0.65 * sin(x * 0.63 + z * 0.47)
                + 0.22 * cos(x * 1.45 - z * 1.12);
            let pedestal = if r < 6.5 {
                let step = clamp((6.5 - r) * 0.85, 0.0, 2.5);
                let terraces = (step * 2.0).floor() * 0.35 + (step * 2.0).fract() * 0.12;
                let column = if r < 2.8 {
                    let flutes = 0.14 * cos(angle * 24.0);
                    clamp((2.8 - r) * 3.5, 0.0, 1.0) * (6.2 + flutes)
                } else {
                    0.0
                };
                terraces + column
            } else {
                0.0
            };
            let y = hills + pedestal;
            positions.push(Vec3::new(x, y, z));
            uvs.push(Vec2::new(ux, vz));
            blend.push(clamp(0.5 + 0.25 * sin(x * 0.35) * cos(z * 0.35), 0.2, 0.95));
        }
    }
    let tri_count = (cols - 1) * (rows - 1) * 2;
    let mut indices = Vec::with_capacity(tri_count * 3);
    for rz in 0..(rows - 1) {
        for rx in 0..(cols - 1) {
            let i00 = (rz * cols + rx) as u32;
            let i10 = (rz * cols + rx + 1) as u32;
            let i01 = ((rz + 1) * cols + rx) as u32;
            let i11 = ((rz + 1) * cols + rx + 1) as u32;
            indices.extend_from_slice(&[i00, i01, i10, i10, i01, i11]);
        }
    }
    let mut normals = vec![Vec3::ZERO; v_count];
    for &[a, b, c] in indices.as_chunks::<3>().0 {
        let (a, b, c) = (a as usize, b as usize, c as usize);
        let cross = (positions[b] - positions[a]).cross(positions[c] - positions[a]);
        normals[a] += cross;
        normals[b] += cross;
        normals[c] += cross;
    }
    for n in &mut normals {
        *n = if n.length_squared() > 1.0e-18 {
            n.normalize()
        } else {
            Vec3::Y
        };
    }
    let ao = compute_curvature_ao(&positions, &normals, &indices);
    SourceMesh {
        positions,
        normals,
        uvs,
        ao,
        blend,
        indices,
        material_ids: vec![0; tri_count],
    }
}

pub fn build_multimaterial_foliage_mesh() -> SourceMesh {
    let mut mesh = SourceMesh::default();
    let segs = 48_usize;
    let rings = 32_usize;
    let height = 10.0_f32;
    let base_v = mesh.positions.len() as u32;
    for ry in 0..=rings {
        let ty = (ry as f32) / (rings as f32);
        let y = ty * height;
        let radius = 1.2 - ty * 0.25;
        for sx in 0..=segs {
            let tx = (sx as f32) / (segs as f32);
            let theta = tx * core::f32::consts::TAU;
            let flute = 0.06 * cos(theta * 16.0);
            let r = radius + flute;
            let pos = Vec3::new(r * cos(theta), y, r * sin(theta));
            let n = Vec3::new(cos(theta), 0.0, sin(theta)).normalize();
            mesh.positions.push(pos);
            mesh.normals.push(n);
            mesh.uvs.push(Vec2::new(tx, ty));
            mesh.ao.push(0.85);
            mesh.blend.push(0.9);
        }
    }
    let stride = segs + 1;
    for ry in 0..rings {
        let mat = if (12..=15).contains(&ry) { 2_u8 } else { 0_u8 };
        for sx in 0..segs {
            let a = base_v + (ry * stride + sx) as u32;
            let b = base_v + (ry * stride + sx + 1) as u32;
            let c = base_v + ((ry + 1) * stride + sx) as u32;
            let d = base_v + ((ry + 1) * stride + sx + 1) as u32;
            mesh.indices.extend_from_slice(&[a, b, c, b, d, c]);
            mesh.material_ids.extend_from_slice(&[mat, mat]);
        }
    }
    for sprig in 0..420_u32 {
        let f = sprig as f32;
        let theta = f * 2.399_963;
        let elev = (f / 420.0) * 6.5 + 3.0;
        let rad = 2.0 + (1.0 - (elev - 3.0) / 6.5) * 1.8;
        let center = Vec3::new(rad * cos(theta), elev, rad * sin(theta));
        let fwd = Vec3::new(cos(theta), 0.35, sin(theta)).normalize();
        let side = Vec3::new(-sin(theta), 0.0, cos(theta)).normalize();
        let up = side.cross(fwd).normalize();
        let v0 = mesh.positions.len() as u32;
        let pts = [
            center - fwd * 0.25,
            center + side * 0.18 + up * 0.05,
            center + fwd * 0.32 + up * 0.08,
            center - side * 0.18 + up * 0.05,
            center + up * 0.12,
        ];
        for (k, &p) in pts.iter().enumerate() {
            mesh.positions.push(p);
            mesh.normals.push(up);
            mesh.uvs.push(Vec2::new((k as f32) * 0.25, 0.5));
            mesh.ao.push(0.75);
            mesh.blend.push(0.8);
        }
        mesh.indices.extend_from_slice(&[
            v0,
            v0 + 1,
            v0 + 4,
            v0 + 1,
            v0 + 2,
            v0 + 4,
            v0 + 2,
            v0 + 3,
            v0 + 4,
            v0 + 3,
            v0,
            v0 + 4,
        ]);
        mesh.material_ids.extend_from_slice(&[4, 4, 4, 4]);
    }
    mesh
}
