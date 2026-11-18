pub fn pack_surface([first, second]: [f32; 2], emission: f32) -> u32 {
    let unit = |value: f32, max: f32| {
        let value = if value.is_nan() { 0.0 } else { value };
        (engine_core::math::clamp(value / max, 0.0, 1.0) * 255.0).round() as u32
    };
    unit(first, 1.0) | unit(second, 1.0) << 8 | unit(emission, Material::MAX_EMISSION) << 16
}

pub fn pack_color(color: [f32; 4]) -> [u32; 2] {
    let unit = |value: f32, fallback: f32| {
        let value = if value.is_nan() { fallback } else { value };
        (engine_core::math::clamp(value, 0.0, 1.0) * 65535.0).round() as u32
    };
    [
        unit(color[0], 0.0) | unit(color[1], 0.0) << 16,
        unit(color[2], 0.0) | unit(color[3], 1.0) << 16,
    ]
}

