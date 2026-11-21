use engine::math;
use glam::{Vec2, Vec3};

pub const SHRINE_TERRACE_Z: f32 = -4.6;
pub const TORII_Z: f32 = 2.6;
pub const WATER_LEVEL: f32 = -4.2;

pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = math::clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

pub fn path_center_x(z: f32) -> f32 {
    let fade = smoothstep(5.8, 16.5, z);
    let curve = -1.75 * math::sin((z - 5.8) * 0.135) - 0.42 * math::sin((z - 5.8) * 0.27);
    curve * fade
}

pub fn path_half_width(z: f32) -> f32 {
    if z < -0.6 {
        return 4.9;
    }
    if z < 6.8 {
        return 2.15;
    }
    1.18 + 0.16 * math::sin(z * 0.4)
}

pub fn path_clearance(x: f32, z: f32) -> f32 {
    if (-9.9..=0.0).contains(&z) {
        let dx = (x.abs() - 4.9) / 1.3;
        let dz = ((z + 4.9).abs() - 4.8) / 1.3;
        let d = math::max(dx, dz);
        return 1.0 - smoothstep(0.0, 1.0, d);
    }
    if (0.0..=23.5).contains(&z) {
        let cx = path_center_x(z);
        let half = path_half_width(z);
        let dist = (x - cx).abs();
        let end_fade = 1.0 - smoothstep(19.5, 23.5, z);
        return (1.0 - smoothstep(half * 0.72, half + 0.65, dist)) * end_fade;
    }
    0.0
}

fn crag_profile(x: f32, z: f32, phase: f32) -> f32 {
    let r1 = 1.0 - math::sin(x * 0.032 + phase + z * 0.008).abs();
    let r2 = 1.0 - math::sin(x * 0.074 - phase * 1.7 + z * 0.015).abs();
    let r3 = 1.0 - math::cos(x * 0.155 + phase * 2.3 - z * 0.022).abs();
    0.26 + 0.44 * r1 * r1 + 0.22 * r2 * r2 + 0.14 * r3 * r3
}

pub fn terrain_height(x: f32, z: f32) -> f32 {
    let hx = x * 0.044;
    let hz = (z + 1.8) * 0.040;
    let main_hill = 4.95 * math::exp(-(hx * hx + hz * hz));

    let roll_a = 0.62 * math::sin(x * 0.095 + 0.35) * math::cos(z * 0.085 - 0.25);
    let roll_b = 0.30 * math::sin(x * 0.19 - z * 0.16 + 0.8);
    let roll_c = 0.14 * math::cos(x * 0.37 + z * 0.31);
    let flank_left = 3.2 * math::exp(-((x + 26.0) * (x + 26.0) * 0.0022 + (z + 10.0) * (z + 10.0) * 0.0018));
    let flank_right = 2.8 * math::exp(-((x - 28.0) * (x - 28.0) * 0.0020 + (z + 14.0) * (z + 14.0) * 0.0016));

    let valley_t = smoothstep(-13.0, -72.0, z);
    let valley_drop = -11.2 * valley_t;

    let front_t = smoothstep(26.0, 88.0, z);
    let front_drop = -5.5 * front_t;

    let far_mtn_mask = smoothstep(-76.0, -142.0, z);
    let bay_saddle = 1.0 - 0.68 * math::exp(-((x - 8.0) * (x - 8.0) * 0.00024));
    let near_crags = 26.0
        * bay_saddle
        * math::exp(-((z + 152.0) * (z + 152.0) * 0.00048))
        * crag_profile(x, z, 0.7);
    let mid_crags = 42.0
        * (0.38 + 0.62 * bay_saddle)
        * math::exp(-((z + 225.0) * (z + 225.0) * 0.00026))
        * crag_profile(x * 0.85, z, 2.1);
    let far_alps = 58.0
        * (0.45 + 0.55 * bay_saddle)
        * math::exp(-((z + 300.0) * (z + 300.0) * 0.00018))
        * crag_profile(x * 0.72, z, 4.3);
    let left_summit = 42.0
        * math::exp(-((x + 118.0) * (x + 118.0) * 0.00036 + (z + 250.0) * (z + 250.0) * 0.00028))
        * (0.72 + 0.34 * crag_profile(x * 1.4, z, 1.3));
    let right_summit = 38.0
        * math::exp(-((x - 115.0) * (x - 115.0) * 0.00034 + (z + 235.0) * (z + 235.0) * 0.00026))
        * (0.70 + 0.36 * crag_profile(x * 1.35, z, 3.4));
    let side_peaks = 20.0
        * smoothstep(72.0, 175.0, x.abs())
        * crag_profile(z * 0.9, x, 0.5);

    let raw = main_hill
        + (roll_a + roll_b + roll_c) * (1.0 - 0.55 * valley_t)
        + flank_left
        + flank_right
        + valley_drop
        + front_drop
        + (near_crags + mid_crags + far_alps + left_summit + right_summit) * far_mtn_mask
        + side_peaks;

    let plateau_dx = x * 0.145;
    let plateau_dz = (z + 4.8) * 0.155;
    let plateau_w = math::exp(-(plateau_dx * plateau_dx + plateau_dz * plateau_dz));
    let terrace_blend = smoothstep(0.32, 0.78, plateau_w);
    let flat_top = raw * (1.0 - terrace_blend) + 4.82 * terrace_blend;

    let step_zone = smoothstep(-1.4, -0.3, z)
        * (1.0 - smoothstep(6.2, 7.6, z))
        * (1.0 - smoothstep(1.95, 3.15, x.abs()));
    let step_t = math::clamp((z - (-1.1)) / (6.45 - (-1.1)), 0.0, 1.0);
    let step_bed = 4.62 - step_t * 1.68;
    flat_top * (1.0 - step_zone * 0.96) + step_bed * (step_zone * 0.96)
}

pub fn terrain_normal(x: f32, z: f32) -> Vec3 {
    let eps = 0.22;
    let h_l = terrain_height(x - eps, z);
    let h_r = terrain_height(x + eps, z);
    let h_d = terrain_height(x, z - eps);
    let h_u = terrain_height(x, z + eps);
    Vec3::new(h_l - h_r, 2.0 * eps, h_d - h_u).normalize_or_zero()
}

pub fn lantern_positions() -> [Vec3; 4] {
    let spots = [
        Vec2::new(-3.45, 8.2),
        Vec2::new(3.85, 8.2),
        Vec2::new(-4.25, -0.25),
        Vec2::new(4.25, -0.25),
    ];
    spots.map(|p| Vec3::new(p.x, terrain_height(p.x, p.y), p.y))
}
