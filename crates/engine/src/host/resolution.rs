use crate::resolution::Resolution;

pub fn view_size(
    width: f64,
    height: f64,
    pixel_ratio: f64,
    settings: &Resolution,
) -> Option<(u32, u32)> {
    if !(width > 0.0 && height > 0.0 && width.is_finite() && height.is_finite()) {
        return None;
    }
    let ratio = settings.max_pixel_ratio.map_or(known(pixel_ratio), |most| {
        known(pixel_ratio).min(f64::from(most))
    });
    let mut scale = ratio * known(f64::from(settings.scale));
    if let Some(most) = settings.max_width {
        scale = scale.min(f64::from(most) / width);
    }
    if let Some(most) = settings.max_height {
        scale = scale.min(f64::from(most) / height);
    }
    let physical = |logical: f64| (logical * scale).round().max(1.0) as u32;
    Some((physical(width), physical(height)))
}

fn known(value: f64) -> f64 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        1.0
    }
}

pub fn render_size(
    (width, height): (u32, u32),
    pixel_ratio: f64,
    settings: &Resolution,
    scale: f64,
) -> Option<(u32, u32)> {
    if width == 0 || height == 0 {
        return None;
    }
    let ratio = known(pixel_ratio);
    let (wide, tall) = view_size(
        f64::from(width) / ratio,
        f64::from(height) / ratio,
        ratio,
        settings,
    )?;
    let scale = known(scale);
    let scaled = |policy: u32, display: u32| {
        ((f64::from(policy.min(display)) * scale).round() as u32).clamp(1, display)
    };
    Some((scaled(wide, width), scaled(tall, height)))
}
