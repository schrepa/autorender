struct Pose {
    translation: vec3<f32>,
    rotation: vec4<f32>,
    scale: vec3<f32>,
}

fn pose_trs(pose: Pose) -> mat4x4<f32> {
    let x = pose.rotation.x;
    let y = pose.rotation.y;
    let z = pose.rotation.z;
    let w = pose.rotation.w;
    let x2 = x + x;
    let y2 = y + y;
    let z2 = z + z;
    let xx = x * x2;
    let xy = x * y2;
    let xz = x * z2;
    let yy = y * y2;
    let yz = y * z2;
    let zz = z * z2;
    let wx = w * x2;
    let wy = w * y2;
    let wz = w * z2;
    return mat4x4<f32>(
        vec4<f32>(1.0 - (yy + zz), xy + wz, xz - wy, 0.0) * pose.scale.x,
        vec4<f32>(xy - wz, 1.0 - (xx + zz), yz + wx, 0.0) * pose.scale.y,
        vec4<f32>(xz + wy, yz - wx, 1.0 - (xx + yy), 0.0) * pose.scale.z,
        vec4<f32>(pose.translation, 1.0),
    );
}

fn pose_blend_rotation(previous: vec4<f32>, current: vec4<f32>, alpha: f32) -> vec4<f32> {
    let p = normalize(previous);
    var c = normalize(current);
    let d = dot(p, c);
    if abs(d) >= LINEAR_ROTATION_DOT {
        if d < 0.0 {
            c = -c;
        }
        return normalize(p * (1.0 - alpha) + c * alpha);
    }
    var cosine = d;
    if cosine < 0.0 {
        c = -c;
        cosine = -cosine;
    }
    let theta = acos(cosine);
    let start = sin(theta * (1.0 - alpha));
    let end = sin(theta * alpha);
    return normalize((p * start + c * end) * (1.0 / sin(theta)));
}

fn drawn_local(previous: Pose, current: Pose, alpha: f32, blend: bool) -> mat4x4<f32> {
    if !blend || alpha >= 1.0 {
        return pose_trs(current);
    }
    if alpha == 0.0 {
        return pose_trs(previous);
    }
    let inverse = 1.0 - alpha;
    return pose_trs(Pose(
        previous.translation * inverse + current.translation * alpha,
        pose_blend_rotation(previous.rotation, current.rotation, alpha),
        previous.scale * inverse + current.scale * alpha,
    ));
}

fn pose_finite(value: vec4<f32>) -> bool {
    return all(abs(value) <= vec4<f32>(POSE_FLOAT_LIMIT));
}

fn pose_largest(value: vec3<f32>) -> f32 {
    return max(abs(value.x), max(abs(value.y), abs(value.z)));
}

fn pose_admitted(m: mat4x4<f32>) -> bool {
    if !(pose_finite(m[0]) && pose_finite(m[1]) && pose_finite(m[2]) && pose_finite(m[3])) {
        return false;
    }
    let x = m[0].xyz;
    let y = m[1].xyz;
    let z = m[2].xyz;
    let scale = max(pose_largest(x), max(pose_largest(y), pose_largest(z)));
    let determinant = dot(x, cross(y, z));
    return scale >= MIN_LINEAR_SCALE && scale <= COEFFICIENT_LIMIT
        && pose_largest(m[3].xyz) <= COEFFICIENT_LIMIT
        && abs(determinant) >= COMMON_DETERMINANT_MARGIN * ((scale * scale) * scale);
}
