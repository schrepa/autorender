use engine::math;
use glam::{Vec2, Vec3};

use crate::mesh::{MAT_BARK, MAT_BLOSSOM, MAT_PINE, MAT_STONE, MeshBuilder, SurfaceStyle};
use crate::terrain::terrain_height;

const BARK_TINT: [f32; 4] = [0.14, 0.10, 0.09, 0.0];
const BOULDER_TINT: [f32; 4] = [0.22, 0.22, 0.20, 0.0];

fn sty(mat: f32, ao: f32, moss: f32, tint: [f32; 4]) -> SurfaceStyle {
    SurfaceStyle {
        mat,
        ao,
        moss,
        tint,
    }
}

fn hash_f(mut n: u32) -> f32 {
    n = (n ^ 61) ^ (n >> 16);
    n = n.wrapping_mul(9);
    n = n ^ (n >> 4);
    n = n.wrapping_mul(0x27d4_eb2d);
    n = n ^ (n >> 15);
    (n as f32) / (u32::MAX as f32)
}

pub fn build_flora_and_rocks(builder: &mut MeshBuilder) {
    let hero_pos = Vec3::new(-6.95, terrain_height(-6.95, -1.45) - 0.22, -1.45);
    build_mossy_boulder(builder, hero_pos + Vec3::new(-0.7, 0.25, 0.45), Vec3::new(1.95, 1.10, 1.70), 11);
    build_mossy_boulder(builder, hero_pos + Vec3::new(1.05, 0.05, -0.95), Vec3::new(1.40, 0.82, 1.30), 23);
    build_hero_sakura(builder, hero_pos, 1.04, 101);

    let comp_pos = Vec3::new(8.25, terrain_height(8.25, -5.8) - 0.22, -5.8);
    build_mossy_boulder(builder, comp_pos + Vec3::new(0.45, 0.10, 0.55), Vec3::new(1.65, 0.92, 1.45), 37);
    build_companion_sakura(builder, comp_pos, 209);

    let boulder_spots = [
        (Vec2::new(5.4, 5.8), Vec3::new(1.35, 0.72, 1.15), 41),
        (Vec2::new(-5.4, 8.0), Vec3::new(1.50, 0.78, 1.25), 53),
        (Vec2::new(7.2, 2.0), Vec3::new(1.55, 0.85, 1.35), 67),
        (Vec2::new(-9.2, 4.4), Vec3::new(1.95, 1.05, 1.65), 79),
        (Vec2::new(9.8, -2.2), Vec3::new(1.75, 0.92, 1.50), 89),
    ];
    for (xz, scale, seed) in boulder_spots {
        let p = Vec3::new(xz.x, terrain_height(xz.x, xz.y) - 0.16, xz.y);
        build_mossy_boulder(builder, p, scale, seed);
    }

    let pine_spots = [
        (Vec2::new(-15.5, -12.4), 1.22, 0.35, 301),
        (Vec2::new(15.2, -10.8), 1.14, -0.40, 401),
        (Vec2::new(-20.5, 3.6), 1.02, 0.55, 501),
        (Vec2::new(19.8, 2.2), 1.04, -0.32, 601),
        (Vec2::new(-10.6, -17.5), 0.98, 0.28, 701),
        (Vec2::new(9.6, -18.2), 1.06, -0.25, 809),
        (Vec2::new(-25.0, -5.5), 1.28, 0.42, 911),
    ];
    for (xz, scale, lean_x, seed) in pine_spots {
        let p = Vec3::new(xz.x, terrain_height(xz.x, xz.y) - 0.25, xz.y);
        build_windswept_pine(builder, p, scale, lean_x, seed);
    }
}

fn build_mossy_boulder(builder: &mut MeshBuilder, center: Vec3, scale: Vec3, seed: u32) {
    let lat: u32 = 10;
    let lon: u32 = 14;
    let base = builder.vertices.len() as u32;
    let phase = seed as f32 * 1.37;
    for i in 0..=lat {
        let v = i as f32 / lat as f32;
        let theta = v * core::f32::consts::PI;
        let (st, ct) = math::sin_cos(theta);
        for j in 0..=lon {
            let u = j as f32 / lon as f32;
            let phi = u * core::f32::consts::TAU;
            let (sp, cp) = math::sin_cos(phi);
            let dir = Vec3::new(st * cp, ct, st * sp);
            let bump = 1.0
                + 0.16 * math::sin(dir.x * 3.1 + phase) * math::cos(dir.z * 2.7 - phase)
                + 0.10 * math::sin(dir.y * 4.3 + dir.x * 2.1);
            let local = dir * scale * bump;
            let n = (dir / scale).normalize_or_zero();
            let ao = math::clamp(0.45 + 0.55 * (ct * 0.5 + 0.5), 0.35, 1.0);
            builder.push_vertex(
                center + local,
                n,
                Vec2::new(u * 2.0, v * 2.0),
                Vec2::new(0.0, 1.0),
                sty(MAT_STONE, ao, 1.0, BOULDER_TINT),
            );
        }
    }
    let stride = lon + 1;
    for i in 0..lat {
        for j in 0..lon {
            let a = base + i * stride + j;
            builder.push_quad(a, a + stride, a + stride + 1, a + 1, false);
        }
    }
}

fn build_hero_sakura(builder: &mut MeshBuilder, origin: Vec3, scale: f32, seed: u32) {
    let trunk = [
        (origin + Vec3::new(-0.25, -0.35, 0.10) * scale, 0.68 * scale),
        (origin + Vec3::new(-0.08, 0.75, 0.05) * scale, 0.50 * scale),
        (origin + Vec3::new(0.32, 1.85, 0.18) * scale, 0.42 * scale),
        (origin + Vec3::new(0.88, 2.95, 0.30) * scale, 0.36 * scale),
        (origin + Vec3::new(1.52, 3.95, 0.36) * scale, 0.30 * scale),
        (origin + Vec3::new(2.05, 4.75, 0.30) * scale, 0.22 * scale),
    ];
    builder.push_tube(&trunk, 14, sty(MAT_BARK, 0.75, 0.65, BARK_TINT), 1.2);

    for r in 0..5 {
        let ang = (r as f32 / 5.0) * core::f32::consts::TAU + 0.3;
        let (sa, ca) = math::sin_cos(ang);
        let root_pts = [
            (origin + Vec3::new(ca * 0.30, 0.45, sa * 0.30) * scale, 0.26 * scale),
            (origin + Vec3::new(ca * 0.95, 0.02, sa * 0.95) * scale, 0.16 * scale),
            (origin + Vec3::new(ca * 1.55, -0.35, sa * 1.55) * scale, 0.07 * scale),
        ];
        builder.push_tube(&root_pts, 8, sty(MAT_BARK, 0.70, 0.85, BARK_TINT), 0.5);
    }

    let boughs: [(Vec3, Vec3, Vec3, f32, f32); 6] = [
        (Vec3::new(1.35, 3.95, 0.28), Vec3::new(2.80, 5.15, 0.72), Vec3::new(4.10, 5.85, 0.95), 0.21, 2.15),
        (Vec3::new(1.75, 4.55, 0.26), Vec3::new(2.90, 6.20, -0.45), Vec3::new(4.35, 7.05, -1.05), 0.19, 2.35),
        (Vec3::new(1.25, 3.70, 0.26), Vec3::new(-0.55, 5.15, 0.80), Vec3::new(-2.45, 5.75, 1.20), 0.20, 2.35),
        (Vec3::new(1.55, 4.25, 0.16), Vec3::new(0.25, 6.00, -1.35), Vec3::new(-1.25, 6.75, -2.55), 0.18, 2.35),
        (Vec3::new(1.85, 4.75, 0.26), Vec3::new(1.55, 6.55, 0.35), Vec3::new(1.85, 7.80, 0.10), 0.18, 2.40),
        (Vec3::new(0.80, 2.95, 0.30), Vec3::new(1.75, 3.65, 1.75), Vec3::new(2.65, 3.95, 2.55), 0.16, 1.75),
    ];

    for (idx, &(p0, p1, p2, r0, cloud_r)) in boughs.iter().enumerate() {
        let mid_a = p0.lerp(p1, 0.55) + Vec3::new(0.0, 0.22, 0.0);
        let mid_b = p1.lerp(p2, 0.55) + Vec3::new(0.0, 0.16, 0.0);
        let pts = [
            (origin + p0 * scale, r0 * scale),
            (origin + mid_a * scale, r0 * 0.78 * scale),
            (origin + p1 * scale, r0 * 0.60 * scale),
            (origin + mid_b * scale, r0 * 0.40 * scale),
            (origin + p2 * scale, r0 * 0.18 * scale),
        ];
        builder.push_tube(&pts, 9, sty(MAT_BARK, 0.78, 0.25, BARK_TINT), 0.8);

        let cloud_mid = origin + p1.lerp(p2, 0.45) * scale + Vec3::new(0.0, 0.42 * scale, 0.0);
        let cloud_tip = origin + p2 * scale + Vec3::new(0.0, 0.28 * scale, 0.0);
        let s_base = seed + (idx as u32) * 977;
        push_blossom_cloud(builder, cloud_mid, cloud_r * 0.86 * scale, 210, s_base, false);
        push_blossom_cloud(builder, cloud_tip, cloud_r * scale, 260, s_base + 503, false);
    }
}

fn build_companion_sakura(builder: &mut MeshBuilder, origin: Vec3, seed: u32) {
    let trunk = [
        (origin + Vec3::new(0.0, -0.3, 0.0), 0.46),
        (origin + Vec3::new(-0.28, 1.35, 0.12), 0.34),
        (origin + Vec3::new(-0.72, 2.85, 0.24), 0.24),
        (origin + Vec3::new(-1.10, 4.05, 0.20), 0.15),
    ];
    builder.push_tube(&trunk, 10, sty(MAT_BARK, 0.75, 0.50, BARK_TINT), 0.9);
    let tips = [
        (Vec3::new(-0.70, 2.80, 0.22), Vec3::new(-2.85, 4.35, 1.15), 2.05),
        (Vec3::new(-0.95, 3.65, 0.20), Vec3::new(-1.85, 5.55, -0.95), 2.15),
        (Vec3::new(-0.85, 3.30, 0.15), Vec3::new(0.95, 4.85, 0.65), 1.95),
        (Vec3::new(-1.05, 4.00, 0.20), Vec3::new(-0.65, 5.95, 0.45), 2.10),
    ];
    for (idx, &(start, end, rad)) in tips.iter().enumerate() {
        let mid = start.lerp(end, 0.5) + Vec3::new(0.0, 0.35, 0.0);
        let pts = [(origin + start, 0.16), (origin + mid, 0.10), (origin + end, 0.04)];
        builder.push_tube(&pts, 7, sty(MAT_BARK, 0.78, 0.2, BARK_TINT), 0.4);
        push_blossom_cloud(builder, origin + end, rad, 180, seed + (idx as u32) * 733, false);
    }
}

fn build_windswept_pine(builder: &mut MeshBuilder, origin: Vec3, scale: f32, lean_x: f32, seed: u32) {
    let trunk = [
        (origin + Vec3::new(0.0, -0.4, 0.0) * scale, 0.40 * scale),
        (origin + Vec3::new(lean_x * 0.7, 1.8, 0.1) * scale, 0.30 * scale),
        (origin + Vec3::new(lean_x * 1.5, 3.8, -0.15) * scale, 0.22 * scale),
        (origin + Vec3::new(lean_x * 2.1, 5.6, -0.20) * scale, 0.12 * scale),
    ];
    builder.push_tube(&trunk, 9, sty(MAT_BARK, 0.74, 0.45, BARK_TINT), 0.6);

    let pads = [
        (Vec3::new(lean_x * 1.2 + 1.4, 3.3, 0.5), 1.75),
        (Vec3::new(lean_x * 1.4 - 1.5, 4.1, -0.4), 1.65),
        (Vec3::new(lean_x * 1.9 + 0.9, 5.1, -0.6), 1.85),
        (Vec3::new(lean_x * 2.2 - 0.3, 6.1, -0.1), 1.95),
    ];
    for (idx, &(off, rad)) in pads.iter().enumerate() {
        let limb_start = origin + Vec3::new(lean_x * 1.4, off.y - 0.45, 0.0) * scale;
        let limb_end = origin + off * scale;
        builder.push_tube(
            &[(limb_start, 0.11 * scale), (limb_end, 0.04 * scale)],
            6,
            sty(MAT_BARK, 0.74, 0.15, BARK_TINT),
            0.0,
        );
        push_blossom_cloud(builder, limb_end, rad * scale, 110, seed + (idx as u32) * 541, true);
    }
}

fn push_blossom_cloud(
    builder: &mut MeshBuilder,
    center: Vec3,
    radius: f32,
    count: u32,
    seed: u32,
    is_pine: bool,
) {
    let mat = if is_pine { MAT_PINE } else { MAT_BLOSSOM };
    for i in 0..count {
        let h0 = hash_f(seed.wrapping_add(i * 13 + 1));
        let h1 = hash_f(seed.wrapping_add(i * 13 + 2));
        let h2 = hash_f(seed.wrapping_add(i * 13 + 3));
        let h3 = hash_f(seed.wrapping_add(i * 13 + 4));
        let h4 = hash_f(seed.wrapping_add(i * 13 + 5));

        let phi = h0 * core::f32::consts::TAU;
        let cos_t = h1 * 2.0 - 1.0;
        let sin_t = math::max(0.0, 1.0 - cos_t * cos_t).sqrt();
        let radial = Vec3::new(
            sin_t * math::cos(phi),
            cos_t * if is_pine { 0.42 } else { 0.64 },
            sin_t * math::sin(phi),
        );
        let dist = radius * (0.28 + 0.72 * h2.powf(0.55));
        let pos = center + radial * dist;
        let macro_n = (radial + Vec3::new(0.0, 0.26, 0.0)).normalize_or_zero();
        let ao = math::clamp(
            0.22 + 0.78 * (dist / radius) * (0.45 + 0.55 * (cos_t * 0.5 + 0.5)),
            0.18,
            1.0,
        );

        let card_size = radius * (if is_pine { 0.25 } else { 0.21 } + 0.16 * h3);
        let yaw = h4 * core::f32::consts::TAU;
        let pitch = (h0 - 0.5) * 1.15;
        let (sy, cy) = math::sin_cos(yaw);
        let (sp, cp) = math::sin_cos(pitch);
        let right = Vec3::new(cy, 0.0, -sy) * card_size;
        let up = Vec3::new(sy * sp, cp, cy * sp) * card_size;

        let tint = if is_pine {
            [0.04 + 0.03 * h3, 0.13 + 0.09 * h4, 0.06 + 0.04 * h2, 0.22]
        } else {
            [0.98 + 0.02 * h3, 0.44 + 0.34 * h4, 0.56 + 0.28 * h2, 0.72]
        };
        let style = sty(mat, ao, 0.0, tint);
        let extra = Vec2::new(h0 * core::f32::consts::TAU, h3);
        let a = builder.push_vertex(pos - right - up, macro_n, Vec2::new(0.0, 0.0), extra, style);
        let b = builder.push_vertex(pos + right - up, macro_n, Vec2::new(1.0, 0.0), extra, style);
        let c = builder.push_vertex(pos + right + up, macro_n, Vec2::new(1.0, 1.0), extra, style);
        let d = builder.push_vertex(pos - right + up, macro_n, Vec2::new(0.0, 1.0), extra, style);
        builder.push_quad(a, b, c, d, true);
    }
}
