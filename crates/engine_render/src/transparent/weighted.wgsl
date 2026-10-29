struct TransparentTargets {
    @location(0) accumulated: vec4<f32>,
    @location(1) revealed: vec2<f32>,
}

fn weighted_particle(color: vec3<f32>, opacity: f32, distance: f32) -> TransparentTargets {
    let near = distance / 5.0;
    let far = distance / 200.0;
    let far3 = far * far * far;
    let weight = opacity * clamp(10.0 / (1.0e-5 + near * near + far3 * far3), 0.01, 3000.0);
    var targets: TransparentTargets;
    targets.accumulated = vec4<f32>(color * opacity, opacity) * weight;
    targets.revealed = vec2<f32>(opacity);
    return targets;
}
