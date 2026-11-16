struct ViewDetail {
    eye: vec3<f32>,
    pixel_angle: f32,
    texel: f32,
    pixel_error: f32,
}

struct DetailLevel {
    level: u32,
    blend: f32,
}

fn across(view: ViewDetail, world: vec3<f32>) -> f32 {
    return length(world - view.eye) * view.pixel_angle + view.texel;
}

fn pixels(view: ViewDetail, size: f32, world: vec3<f32>) -> f32 {
    return size / across(view, world);
}

fn fits(view: ViewDetail, error: f32, world: vec3<f32>) -> bool {
    return error <= across(view, world) * view.pixel_error;
}

fn band(at: f32) -> f32 {
    return saturate(2.0 * at - 1.0);
}

fn handover(view: ViewDetail, error: f32, world: vec3<f32>) -> f32 {
    return band(log2(across(view, world) * view.pixel_error / max(error, DETAIL_FLOOR)) + 1.0);
}

fn detail_level(span: f32, base: f32, levels: u32) -> DetailLevel {
    let coarsest = f32(max(levels, 1u) - 1u);
    let ratio = max(span / max(base, DETAIL_FLOOR), DETAIL_FLOOR);
    let at = clamp(log2(ratio), 0.0, coarsest);
    var chosen: DetailLevel;
    chosen.level = u32(floor(at));
    chosen.blend = band(at - floor(at));
    return chosen;
}

fn seen_share(seen: f32) -> f32 {
    return saturate(1.0 + log2(max(seen, SEEN_FLOOR)));
}
