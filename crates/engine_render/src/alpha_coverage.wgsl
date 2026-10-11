// Material-specific alpha-test factors. Derivatives are evaluated by the
// sampling caller before this opacity-dependent choice. Raw alpha is kept.
fn alpha_coverage_factor(mip: f32, opacity: f32) -> f32 {
    let legacy = 1.0 + 0.25 * max(mip, 0.0);
    let metadata = material_maps.alpha_coverage;
    if metadata.x == 0.0 || opacity != 1.0 || !(mip < metadata.y) {
        return legacy;
    }
    let last = u32(metadata.z) - 1u;
    let level = clamp(mip, 0.0, f32(last));
    let first = u32(floor(level));
    let next = min(first + 1u, last);
    let a = material_maps.alpha_factors[first >> 2u][first & 3u];
    let b = material_maps.alpha_factors[next >> 2u][next & 3u];
    return mix(a, b, level - f32(first));
}
