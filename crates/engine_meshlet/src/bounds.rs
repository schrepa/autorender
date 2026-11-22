use engine_math::{clamp, max};
use glam::Vec3;

pub fn aabb_of_points(points: &[Vec3]) -> (Vec3, Vec3) {
    if points.is_empty() {
        return (Vec3::ZERO, Vec3::ZERO);
    }
    let mut min_p = Vec3::splat(f32::INFINITY);
    let mut max_p = Vec3::splat(f32::NEG_INFINITY);
    for &p in points {
        min_p = min_p.min(p);
        max_p = max_p.max(p);
    }
    let extent = (max_p - min_p).max(Vec3::ZERO);
    (min_p, extent)
}

pub fn bounding_sphere_points(points: &[Vec3]) -> [f32; 4] {
    if points.is_empty() {
        return [0.0; 4];
    }
    if points.len() == 1 {
        let p = points[0];
        return [p.x, p.y, p.z, 0.0];
    }
    let mut ext = [0_usize; 6];
    for (idx, &p) in points.iter().enumerate() {
        if p.x < points[ext[0]].x {
            ext[0] = idx;
        }
        if p.x > points[ext[1]].x {
            ext[1] = idx;
        }
        if p.y < points[ext[2]].y {
            ext[2] = idx;
        }
        if p.y > points[ext[3]].y {
            ext[3] = idx;
        }
        if p.z < points[ext[4]].z {
            ext[4] = idx;
        }
        if p.z > points[ext[5]].z {
            ext[5] = idx;
        }
    }
    let mut best_a = points[ext[0]];
    let mut best_b = points[ext[1]];
    let mut best_d2 = -1.0_f32;
    for i in 0..6 {
        for j in (i + 1)..6 {
            let a = points[ext[i]];
            let b = points[ext[j]];
            let d2 = (b - a).length_squared();
            if d2 > best_d2 {
                best_d2 = d2;
                best_a = a;
                best_b = b;
            }
        }
    }
    let mut center = (best_a + best_b) * 0.5;
    let mut radius = best_d2.sqrt() * 0.5;
    for _ in 0..2 {
        for &p in points {
            let delta = p - center;
            let dist_sq = delta.length_squared();
            if dist_sq > radius * radius {
                let dist = dist_sq.sqrt();
                let new_r = (radius + dist) * 0.5;
                let shift = (new_r - radius) / dist;
                center += delta * shift;
                radius = new_r;
            }
        }
    }
    let mut exact_r = radius;
    for &p in points {
        let dist = (p - center).length();
        if dist > exact_r {
            exact_r = dist;
        }
    }
    let pad = exact_r * 1.0e-6;
    [center.x, center.y, center.z, exact_r + pad]
}

fn merge_two_spheres(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let ca = Vec3::new(a[0], a[1], a[2]);
    let cb = Vec3::new(b[0], b[1], b[2]);
    let ra = a[3];
    let rb = b[3];
    let delta = cb - ca;
    let dist = delta.length();
    if dist + rb <= ra {
        return a;
    }
    if dist + ra <= rb {
        return b;
    }
    let new_r = (dist + ra + rb) * 0.5;
    let center = if dist > 1.0e-12 {
        ca + delta * ((new_r - ra) / dist)
    } else {
        (ca + cb) * 0.5
    };
    [center.x, center.y, center.z, new_r]
}

fn extremal_sphere_pair(spheres: &[[f32; 4]]) -> (usize, usize) {
    if spheres.len() <= 24 {
        let mut best = (0_usize, 1_usize);
        let mut best_span = -1.0_f32;
        for i in 0..spheres.len() {
            let ca = Vec3::new(spheres[i][0], spheres[i][1], spheres[i][2]);
            for j in (i + 1)..spheres.len() {
                let cb = Vec3::new(spheres[j][0], spheres[j][1], spheres[j][2]);
                let span = (cb - ca).length() + spheres[i][3] + spheres[j][3];
                if span > best_span {
                    best_span = span;
                    best = (i, j);
                }
            }
        }
        return best;
    }
    let mut ext = [0_usize; 6];
    let mut min_v = [f32::INFINITY; 3];
    let mut max_v = [f32::NEG_INFINITY; 3];
    for (idx, s) in spheres.iter().enumerate() {
        for axis in 0..3 {
            if s[axis] - s[3] < min_v[axis] {
                min_v[axis] = s[axis] - s[3];
                ext[axis * 2] = idx;
            }
            if s[axis] + s[3] > max_v[axis] {
                max_v[axis] = s[axis] + s[3];
                ext[axis * 2 + 1] = idx;
            }
        }
    }
    let mut best = (ext[0], ext[1]);
    let mut best_span = -1.0_f32;
    for i in 0..6 {
        let a = ext[i];
        let ca = Vec3::new(spheres[a][0], spheres[a][1], spheres[a][2]);
        for &b in ext.iter().skip(i + 1) {
            let cb = Vec3::new(spheres[b][0], spheres[b][1], spheres[b][2]);
            let span = (cb - ca).length() + spheres[a][3] + spheres[b][3];
            if span > best_span {
                best_span = span;
                best = (a, b);
            }
        }
    }
    best
}

pub fn merge_spheres(spheres: &[[f32; 4]]) -> [f32; 4] {
    if spheres.is_empty() {
        return [0.0; 4];
    }
    if spheres.len() == 1 {
        return spheres[0];
    }
    let (a, b) = extremal_sphere_pair(spheres);
    let mut current = merge_two_spheres(spheres[a], spheres[b]);
    for &s in spheres {
        current = merge_two_spheres(current, s);
    }
    let center = Vec3::new(current[0], current[1], current[2]);
    let mut radius = current[3];
    for &s in spheres {
        let cs = Vec3::new(s[0], s[1], s[2]);
        let needed = (cs - center).length() + s[3];
        if needed > radius {
            radius = needed;
        }
    }
    let pad = radius * 1.0e-6;
    [center.x, center.y, center.z, radius + pad]
}

pub fn pack_normal_cone(axis: [i8; 3], cutoff: i8) -> u32 {
    u32::from(axis[0] as u8)
        | (u32::from(axis[1] as u8) << 8)
        | (u32::from(axis[2] as u8) << 16)
        | (u32::from(cutoff as u8) << 24)
}

pub fn compute_normal_cone(positions: &[Vec3], triangles: &[[u8; 3]]) -> u32 {
    let degenerate = pack_normal_cone([0, 0, 0], 127);
    if triangles.is_empty() {
        return degenerate;
    }
    let mut normals = Vec::with_capacity(triangles.len());
    let mut sum = Vec3::ZERO;
    for &[i0, i1, i2] in triangles {
        let p0 = positions[usize::from(i0)];
        let p1 = positions[usize::from(i1)];
        let p2 = positions[usize::from(i2)];
        let cross = (p1 - p0).cross(p2 - p0);
        let len_sq = cross.length_squared();
        if !len_sq.is_finite() || len_sq <= 1.0e-18 {
            return degenerate;
        }
        let n = cross / len_sq.sqrt();
        normals.push(n);
        sum += n;
    }
    let sum_len_sq = sum.length_squared();
    if !sum_len_sq.is_finite() || sum_len_sq <= 1.0e-12 {
        return degenerate;
    }
    let axis = sum / sum_len_sq.sqrt();
    let ax = (clamp(axis.x, -1.0, 1.0) * 127.0).round() as i8;
    let ay = (clamp(axis.y, -1.0, 1.0) * 127.0).round() as i8;
    let az = (clamp(axis.z, -1.0, 1.0) * 127.0).round() as i8;
    let q_axis = Vec3::new(f32::from(ax), f32::from(ay), f32::from(az));
    let q_len_sq = q_axis.length_squared();
    if q_len_sq <= 1.0e-4 {
        return degenerate;
    }
    let q_unit = q_axis / q_len_sq.sqrt();
    let mut min_dot = 1.0_f32;
    for &n in &normals {
        let d = n.dot(q_unit);
        if d < min_dot {
            min_dot = d;
        }
    }
    if min_dot <= 0.05 {
        return degenerate;
    }
    let sin_angle = max(0.0, 1.0 - min_dot * min_dot).sqrt();
    let raw_cutoff = (sin_angle * 127.0).ceil() as i32 + 1;
    if raw_cutoff >= 127 {
        return degenerate;
    }
    let cutoff = raw_cutoff.clamp(0, 126) as i8;
    pack_normal_cone([ax, ay, az], cutoff)
}

fn expand_bits_10(mut v: u32) -> u32 {
    v &= 0x3FF;
    v = (v | (v << 16)) & 0x0300_00FF;
    v = (v | (v << 8)) & 0x0300_F00F;
    v = (v | (v << 4)) & 0x030C_30C3;
    v = (v | (v << 2)) & 0x0924_9249;
    v
}

pub fn morton_3d(point: Vec3, bounds_min: Vec3, bounds_extent: Vec3) -> u32 {
    let quant = |value: f32, lo: f32, ext: f32| -> u32 {
        if ext <= 1.0e-12 {
            return 0;
        }
        let t = clamp((value - lo) / ext, 0.0, 1.0);
        (t * 1023.0).round() as u32
    };
    let x = expand_bits_10(quant(point.x, bounds_min.x, bounds_extent.x));
    let y = expand_bits_10(quant(point.y, bounds_min.y, bounds_extent.y));
    let z = expand_bits_10(quant(point.z, bounds_min.z, bounds_extent.z));
    x | (y << 1) | (z << 2)
}
