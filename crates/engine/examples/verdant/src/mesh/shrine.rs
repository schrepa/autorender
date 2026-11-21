use engine::math;
use glam::{Vec2, Vec3};

use crate::mesh::{
    MAT_GLOW, MAT_GOLD, MAT_ROPE, MAT_STONE, MAT_TIMBER, MAT_VERMILION, MeshBuilder, SurfaceStyle,
};
use crate::terrain::{SHRINE_TERRACE_Z, TORII_Z, lantern_positions, path_center_x, terrain_height};

const STONE_TINT: [f32; 4] = [0.28, 0.27, 0.25, 0.0];
const DARK_STONE: [f32; 4] = [0.16, 0.16, 0.15, 0.0];
const VERMILION_TINT: [f32; 4] = [0.74, 0.12, 0.06, 0.0];
const DEEP_TIMBER: [f32; 4] = [0.11, 0.09, 0.09, 0.0];
const ROOF_TILE: [f32; 4] = [0.11, 0.12, 0.16, 0.0];
const GOLD_TINT: [f32; 4] = [0.92, 0.72, 0.28, 0.0];
const ROPE_TINT: [f32; 4] = [0.68, 0.56, 0.34, 0.0];
const PAPER_TINT: [f32; 4] = [0.94, 0.91, 0.85, 0.18];
const LANTERN_GLOW: [f32; 4] = [1.0, 0.62, 0.18, 1.0];

fn sty(mat: f32, ao: f32, moss: f32, tint: [f32; 4]) -> SurfaceStyle {
    SurfaceStyle {
        mat,
        ao,
        moss,
        tint,
    }
}

pub fn build_shrine_complex(builder: &mut MeshBuilder) {
    build_terrace_and_steps(builder);
    build_flagstone_path(builder);
    build_torii_gate(builder);
    build_sanctuary_shrine(builder);
    for pos in lantern_positions() {
        build_stone_lantern(builder, pos);
    }
}

fn build_terrace_and_steps(builder: &mut MeshBuilder) {
    let top_y = 5.12;
    let center = Vec3::new(0.0, top_y - 0.98, SHRINE_TERRACE_Z - 0.8);
    builder.push_box(center, Vec3::new(4.85, 0.98, 3.85), 0.0, sty(MAT_STONE, 0.68, 0.82, STONE_TINT));

    for i in 0..9 {
        let t = i as f32 / 8.0;
        let x = -4.65 + t * 9.3;
        let jitter = 0.04 * math::sin(i as f32 * 2.7);
        builder.push_box(
            Vec3::new(x, top_y + 0.02 + jitter * 0.3, SHRINE_TERRACE_Z + 3.05),
            Vec3::new(0.54, 0.14, 0.36),
            jitter * 0.15,
            sty(MAT_STONE, 0.85, 0.75, STONE_TINT),
        );
    }

    let steps = 11;
    let z_start = 6.45;
    let z_end = -1.10;
    let y_start = 3.24;
    let y_end = top_y;
    for s in 0..steps {
        let t0 = s as f32 / steps as f32;
        let t1 = (s + 1) as f32 / steps as f32;
        let tm = (t0 + t1) * 0.5;
        let z = z_start + (z_end - z_start) * tm;
        let y_top = y_start + (y_end - y_start) * t1;
        let height = 0.68;
        let width = 1.66 + 0.14 * (1.0 - tm);
        builder.push_box(
            Vec3::new(0.0, y_top - height * 0.5, z),
            Vec3::new(width, height * 0.5, 0.44),
            0.0,
            sty(MAT_STONE, 0.76, 0.74, STONE_TINT),
        );
        for side in [-1.0_f32, 1.0_f32] {
            builder.push_box(
                Vec3::new(side * (width + 0.26), y_top - 0.14, z),
                Vec3::new(0.26, 0.44, 0.46),
                0.0,
                sty(MAT_STONE, 0.72, 0.95, DARK_STONE),
            );
        }
    }
    builder.push_box(
        Vec3::new(0.0, top_y - 0.36, -1.65),
        Vec3::new(2.25, 0.38, 0.85),
        0.0,
        sty(MAT_STONE, 0.78, 0.68, STONE_TINT),
    );
}

fn build_flagstone_path(builder: &mut MeshBuilder) {
    let count = 20;
    for i in 0..count {
        let t = i as f32 / (count - 1) as f32;
        let z = 7.05 + t * 14.5;
        let cx = path_center_x(z);
        let sway = 0.24 * math::sin(i as f32 * 2.13);
        let x = cx + sway;
        let y = terrain_height(x, z);
        let yaw = 0.22 * math::cos(i as f32 * 1.77);
        let sx = 0.72 + 0.20 * math::sin(i as f32 * 3.1).abs();
        let sz = 0.46 + 0.14 * math::cos(i as f32 * 2.4).abs();
        builder.push_box(
            Vec3::new(x, y - 0.01, z),
            Vec3::new(sx, 0.10, sz),
            yaw,
            sty(MAT_STONE, 0.80, 0.68, STONE_TINT),
        );
    }
}

fn build_torii_gate(builder: &mut MeshBuilder) {
    let z = TORII_Z;
    let base_y = 3.92;
    let half_span = 2.68;
    let height = 5.65;
    let lean = 0.16;

    for side in [-1.0_f32, 1.0_f32] {
        let px = side * half_span;
        let plinth_profile = [
            (0.46, -0.35),
            (0.45, 0.15),
            (0.39, 0.32),
            (0.37, 0.72),
            (0.31, 0.88),
            (0.28, 1.18),
            (0.25, 1.24),
        ];
        builder.push_lathe(
            Vec3::new(px, base_y, z),
            &plinth_profile,
            8,
            core::f32::consts::FRAC_PI_8,
            sty(MAT_STONE, 0.80, 0.95, DARK_STONE),
        );
        let pillar_pts = [
            (Vec3::new(px, base_y + 0.95, z), 0.255),
            (Vec3::new(px - side * lean * 0.5, base_y + height * 0.52, z), 0.232),
            (Vec3::new(px - side * lean, base_y + height, z), 0.208),
        ];
        builder.push_tube(&pillar_pts, 18, sty(MAT_VERMILION, 0.85, 0.12, VERMILION_TINT), 0.0);

        let top_x = px - side * lean;
        let daiwa = [(0.26, 0.0), (0.28, 0.06), (0.28, 0.18), (0.25, 0.24)];
        builder.push_lathe(
            Vec3::new(top_x, base_y + height - 0.24, z),
            &daiwa,
            16,
            0.0,
            sty(MAT_TIMBER, 0.82, 0.05, DEEP_TIMBER),
        );
    }

    let nuki_y = base_y + height - 1.15;
    builder.push_box(
        Vec3::new(0.0, nuki_y, z),
        Vec3::new(3.38, 0.18, 0.13),
        0.0,
        sty(MAT_VERMILION, 0.84, 0.08, VERMILION_TINT),
    );
    for side in [-1.0_f32, 1.0_f32] {
        builder.push_box(
            Vec3::new(side * (half_span - lean * 0.8), nuki_y + 0.04, z),
            Vec3::new(0.34, 0.26, 0.17),
            0.0,
            sty(MAT_TIMBER, 0.78, 0.05, DEEP_TIMBER),
        );
    }

    build_torii_lintel(builder, base_y + height + 0.08, z);
    build_torii_plaque_and_rope(builder, nuki_y, base_y + height - 0.08, z);
}

fn build_torii_lintel(builder: &mut MeshBuilder, y_center: f32, z: f32) {
    let segs: u32 = 30;
    let half_w = 4.25;
    let sides_profile = [
        Vec2::new(0.0, -0.22),
        Vec2::new(0.24, -0.10),
        Vec2::new(0.30, 0.12),
        Vec2::new(0.34, 0.30),
        Vec2::new(0.20, 0.42),
        Vec2::new(0.0, 0.46),
        Vec2::new(-0.20, 0.42),
        Vec2::new(-0.34, 0.30),
        Vec2::new(-0.30, 0.12),
        Vec2::new(-0.24, -0.10),
        Vec2::new(0.0, -0.22),
    ];
    let mut rings: Vec<Vec<u32>> = Vec::with_capacity((segs + 1) as usize);
    for i in 0..=segs {
        let u = (i as f32 / segs as f32) * 2.0 - 1.0;
        let abs_u = u.abs();
        let x = u * half_w;
        let lift = 0.72 * abs_u * abs_u * abs_u * abs_u + 0.14 * abs_u * abs_u;
        let scale = 1.0 + 0.18 * abs_u * abs_u;
        let slope = (2.88 * abs_u * abs_u * abs_u + 0.28 * abs_u) * u.signum() / half_w;
        let mut ring = Vec::with_capacity(11);
        for (p_idx, pt) in sides_profile.iter().enumerate() {
            let py = y_center + lift + pt.y * scale;
            let pz = z + pt.x * scale;
            let is_cap = (3..=7).contains(&p_idx);
            let mat = if is_cap { MAT_TIMBER } else { MAT_VERMILION };
            let tint = if is_cap { ROOF_TILE } else { VERMILION_TINT };
            let ny = if p_idx == 5 {
                1.0
            } else if p_idx == 0 || p_idx == 10 {
                -1.0
            } else if is_cap {
                0.65
            } else {
                -0.25
            };
            let nz = pt.x.signum() * 0.75;
            let normal = Vec3::new(-slope * 0.5, ny, nz).normalize_or_zero();
            let v = builder.push_vertex(
                Vec3::new(x, py, pz),
                normal,
                Vec2::new(u * 3.0, p_idx as f32 * 0.2),
                Vec2::new(0.0, if p_idx == 5 { 0.45 } else { 0.05 }),
                sty(mat, 0.85, 0.0, tint),
            );
            ring.push(v);
        }
        rings.push(ring);
    }
    for i in 0..segs as usize {
        for p in 0..10 {
            builder.push_quad(rings[i][p], rings[i][p + 1], rings[i + 1][p + 1], rings[i + 1][p], false);
        }
    }
    for side in [-1.0_f32, 1.0_f32] {
        builder.push_box(
            Vec3::new(side * (half_w + 0.03), y_center + 0.96, z),
            Vec3::new(0.06, 0.34, 0.36),
            0.0,
            sty(MAT_GOLD, 0.9, 0.0, GOLD_TINT),
        );
    }
}

fn build_torii_plaque_and_rope(builder: &mut MeshBuilder, nuki_y: f32, top_y: f32, z: f32) {
    let mid_y = (nuki_y + top_y) * 0.5;
    builder.push_box(
        Vec3::new(0.0, mid_y, z),
        Vec3::new(0.42, 0.52, 0.14),
        0.0,
        sty(MAT_TIMBER, 0.82, 0.05, DEEP_TIMBER),
    );
    builder.push_box(
        Vec3::new(0.0, mid_y, z + 0.13),
        Vec3::new(0.32, 0.40, 0.03),
        0.0,
        sty(MAT_TIMBER, 0.86, 0.0, DEEP_TIMBER),
    );
    for row in [-0.22_f32, 0.0, 0.22] {
        builder.push_box(
            Vec3::new(0.0, mid_y + row, z + 0.165),
            Vec3::new(0.18, 0.04, 0.015),
            0.0,
            sty(MAT_GOLD, 0.95, 0.0, GOLD_TINT),
        );
    }
    builder.push_box(
        Vec3::new(0.0, mid_y, z + 0.165),
        Vec3::new(0.04, 0.28, 0.015),
        0.0,
        sty(MAT_GOLD, 0.95, 0.0, GOLD_TINT),
    );

    let rope_segs = 20;
    let mut rope_pts = Vec::with_capacity(rope_segs + 1);
    for i in 0..=rope_segs {
        let t = i as f32 / rope_segs as f32;
        let u = t * 2.0 - 1.0;
        let x = u * 2.24;
        let sag = 0.42 * (1.0 - u * u);
        let radius = 0.065 + 0.075 * (1.0 - u * u).powf(1.2);
        rope_pts.push((Vec3::new(x, nuki_y - 0.14 - sag, z + 0.12), radius));
    }
    builder.push_tube(&rope_pts, 10, sty(MAT_ROPE, 0.85, 0.0, ROPE_TINT), 18.0);

    let paper = sty(MAT_ROPE, 0.95, 0.0, PAPER_TINT);
    for k in 0..5 {
        let u = -0.68 + k as f32 * 0.34;
        let x = u * 2.24;
        let sag = 0.42 * (1.0 - u * u);
        let top = Vec3::new(x, nuki_y - 0.22 - sag, z + 0.16);
        for seg in 0..3 {
            let sf = seg as f32;
            let zig = if seg % 2 == 0 { 0.05 } else { -0.05 };
            let p0 = top + Vec3::new(zig - 0.055, -sf * 0.14, 0.0);
            let p1 = top + Vec3::new(zig + 0.055, -sf * 0.14, 0.0);
            let p2 = top + Vec3::new(-zig + 0.055, -(sf + 1.0) * 0.14, 0.02);
            let p3 = top + Vec3::new(-zig - 0.055, -(sf + 1.0) * 0.14, 0.02);
            let a = builder.push_vertex(p0, Vec3::Z, Vec2::ZERO, Vec2::new(sf + 1.0, 0.0), paper);
            let b = builder.push_vertex(p1, Vec3::Z, Vec2::X, Vec2::new(sf + 1.0, 0.0), paper);
            let c = builder.push_vertex(p2, Vec3::Z, Vec2::ONE, Vec2::new(sf + 1.5, 0.0), paper);
            let d = builder.push_vertex(p3, Vec3::Z, Vec2::Y, Vec2::new(sf + 1.5, 0.0), paper);
            builder.push_quad(a, b, c, d, true);
        }
    }
}

fn build_sanctuary_shrine(builder: &mut MeshBuilder) {
    let base_y = 5.12;
    let z = SHRINE_TERRACE_Z - 1.25;

    builder.push_box(Vec3::new(0.0, base_y + 0.22, z), Vec3::new(2.75, 0.22, 2.20), 0.0, sty(MAT_TIMBER, 0.74, 0.10, DEEP_TIMBER));
    builder.push_box(Vec3::new(0.0, base_y + 1.55, z - 0.15), Vec3::new(2.05, 1.15, 1.50), 0.0, sty(MAT_TIMBER, 0.78, 0.05, [0.24, 0.17, 0.13, 0.0]));
    builder.push_box(Vec3::new(0.0, base_y + 1.48, z + 1.36), Vec3::new(0.86, 0.88, 0.05), 0.0, sty(MAT_GLOW, 1.0, 0.0, LANTERN_GLOW));

    for bar in [-0.42_f32, 0.0, 0.42] {
        builder.push_box(Vec3::new(bar, base_y + 1.48, z + 1.40), Vec3::new(0.03, 0.90, 0.03), 0.0, sty(MAT_TIMBER, 0.85, 0.0, DEEP_TIMBER));
    }
    for row in [0.95_f32, 1.48, 2.00] {
        builder.push_box(Vec3::new(0.0, base_y + row, z + 1.40), Vec3::new(0.90, 0.03, 0.03), 0.0, sty(MAT_TIMBER, 0.85, 0.0, DEEP_TIMBER));
    }

    for &cx in &[-2.42_f32, -1.15, 1.15, 2.42] {
        for &cz in &[-1.85_f32, 1.85] {
            let pts = [
                (Vec3::new(cx, base_y + 0.40, z + cz), 0.13),
                (Vec3::new(cx, base_y + 2.75, z + cz), 0.115),
            ];
            builder.push_tube(&pts, 12, sty(MAT_VERMILION, 0.85, 0.05, VERMILION_TINT), 0.0);
        }
    }
    for side in [-1.0_f32, 1.0_f32] {
        builder.push_box(Vec3::new(side * 2.55, base_y + 0.88, z + 0.15), Vec3::new(0.07, 0.06, 1.90), 0.0, sty(MAT_VERMILION, 0.85, 0.0, VERMILION_TINT));
        builder.push_lathe(
            Vec3::new(side * 2.55, base_y + 0.92, z + 2.02),
            &[(0.06, 0.0), (0.09, 0.08), (0.07, 0.18), (0.02, 0.26)],
            10,
            0.0,
            sty(MAT_GOLD, 0.92, 0.0, GOLD_TINT),
        );
    }

    build_curved_roof(builder, Vec3::new(0.0, base_y + 2.70, z));
}

fn build_curved_roof(builder: &mut MeshBuilder, base: Vec3) {
    let nx: u32 = 24;
    let nz: u32 = 16;
    let half_w = 3.85;
    let half_d = 3.25;
    let ridge_h = 2.15;

    let roof_point = |u: f32, v: f32| -> Vec3 {
        let abs_v = v.abs();
        let abs_u = u.abs();
        let x = u * half_w * (1.0 + 0.08 * abs_v * abs_v);
        let z = v * half_d * (1.0 + 0.06 * abs_u * abs_u);
        let slope = (1.0 - abs_v).powf(1.48) * ridge_h;
        let corner_lift = 0.56 * abs_u * abs_u * abs_u * abs_v * abs_v;
        let ridge_sag = 0.24 * abs_u * abs_u * (1.0 - abs_v);
        let tile_ripple = 0.028 * math::sin(u * 42.0).abs();
        Vec3::new(x, slope + corner_lift + ridge_sag + tile_ripple, z)
    };

    let idx_base = builder.vertices.len() as u32;
    for iz in 0..=nz {
        let v = (iz as f32 / nz as f32) * 2.0 - 1.0;
        for ix in 0..=nx {
            let u = (ix as f32 / nx as f32) * 2.0 - 1.0;
            let p = roof_point(u, v);
            let du = roof_point(u + 0.02, v) - roof_point(u - 0.02, v);
            let dv = roof_point(u, v + 0.02) - roof_point(u, v - 0.02);
            let n = dv.cross(du).normalize_or_zero();
            let ao = math::clamp(0.65 + 0.35 * (1.0 - v.abs()), 0.5, 1.0);
            builder.push_vertex(
                base + p,
                n,
                Vec2::new(u * 4.0, v * 4.0),
                Vec2::new(0.0, 0.38),
                sty(MAT_TIMBER, ao, 0.38, ROOF_TILE),
            );
        }
    }
    let stride = nx + 1;
    for iz in 0..nz {
        for ix in 0..nx {
            let a = idx_base + iz * stride + ix;
            builder.push_quad(a, a + stride, a + stride + 1, a + 1, false);
        }
    }

    let ridge_pts: Vec<(Vec3, f32)> = (0..=14)
        .map(|i| {
            let u = (i as f32 / 14.0) * 2.0 - 1.0;
            let y = ridge_h + 0.14 + 0.28 * u * u;
            (base + Vec3::new(u * (half_w * 0.96), y, 0.0), 0.16)
        })
        .collect();
    builder.push_tube(&ridge_pts, 10, sty(MAT_TIMBER, 0.85, 0.25, DEEP_TIMBER), 0.0);

    for side in [-1.0_f32, 1.0_f32] {
        builder.push_box(
            base + Vec3::new(side * (half_w * 0.98), ridge_h + 0.50, 0.0),
            Vec3::new(0.14, 0.30, 0.20),
            0.0,
            sty(MAT_GOLD, 0.9, 0.0, GOLD_TINT),
        );
    }
}

fn build_stone_lantern(builder: &mut MeshBuilder, origin: Vec3) {
    let base_profile = [
        (0.48, -0.25),
        (0.46, 0.12),
        (0.36, 0.24),
        (0.22, 0.34),
        (0.17, 0.72),
        (0.19, 1.02),
        (0.17, 1.32),
        (0.30, 1.44),
        (0.38, 1.54),
        (0.36, 1.62),
        (0.24, 1.64),
    ];
    builder.push_lathe(origin, &base_profile, 6, 0.0, sty(MAT_STONE, 0.78, 0.95, STONE_TINT));

    let firebox_y = origin.y + 1.64;
    let firebox_h = 0.46;
    for m in 0..6 {
        let angle = (m as f32 / 6.0) * core::f32::consts::TAU;
        let (sa, ca) = math::sin_cos(angle);
        let p = Vec3::new(origin.x + ca * 0.24, firebox_y + firebox_h * 0.5, origin.z + sa * 0.24);
        builder.push_box(p, Vec3::new(0.045, firebox_h * 0.5, 0.045), angle, sty(MAT_STONE, 0.78, 0.40, DARK_STONE));
    }

    let glow_profile = [(0.02, 0.04), (0.15, 0.12), (0.16, 0.34), (0.02, 0.42)];
    builder.push_lathe(Vec3::new(origin.x, firebox_y, origin.z), &glow_profile, 8, 0.0, sty(MAT_GLOW, 1.0, 0.0, LANTERN_GLOW));

    let roof_y = firebox_y + firebox_h;
    let roof_profile = [
        (0.28, 0.0),
        (0.56, 0.06),
        (0.52, 0.14),
        (0.32, 0.28),
        (0.14, 0.42),
        (0.16, 0.48),
        (0.11, 0.58),
        (0.01, 0.70),
    ];
    builder.push_lathe(Vec3::new(origin.x, roof_y, origin.z), &roof_profile, 6, 0.0, sty(MAT_STONE, 0.84, 1.0, STONE_TINT));
}
