fn pose_local(entry: PoseEntry, alpha: f32) -> mat4x4<f32> {
    return drawn_local(entry.previous[0], entry.previous[1], entry.previous[2],
        entry.current[0], entry.current[1], entry.current[2],
        pose_weight(alpha, entry.blend != 0u));
}

fn pose_base(entry: PoseEntry) -> mat4x4<f32> {
    let base = entry.base;
    return transpose(mat4x4<f32>(base[0], base[1], base[2], vec4<f32>(0.0, 0.0, 0.0, 1.0)));
}
