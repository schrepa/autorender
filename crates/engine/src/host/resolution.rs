//! The size a view draws at, on every host: a resolution budget that keeps
//! fill cost independent of how large and dense the display is.

use super::policy::{MAX_PIXEL_RATIO, MAX_VIEW_HEIGHT, MAX_VIEW_WIDTH};

/// The physical size a view `width` × `height` logical pixels across draws
/// at, on a display of `pixel_ratio` physical pixels per logical one, at
/// the user's resolution `scale`: [`MAX_PIXEL_RATIO`] physical pixels per
/// logical one at most, times `scale`, and no more than [`MAX_VIEW_WIDTH`]
/// × [`MAX_VIEW_HEIGHT`], at the view's aspect. `None` for a view with no
/// area; an unknown ratio or scale counts as 1.
pub fn view_size(width: f64, height: f64, pixel_ratio: f64, scale: f64) -> Option<(u32, u32)> {
    if !(width > 0.0 && height > 0.0 && width.is_finite() && height.is_finite()) {
        return None;
    }
    let known = |value: f64| {
        if value.is_finite() && value > 0.0 {
            value
        } else {
            1.0
        }
    };
    let scale = (known(pixel_ratio).min(MAX_PIXEL_RATIO) * known(scale))
        .min(f64::from(MAX_VIEW_WIDTH) / width)
        .min(f64::from(MAX_VIEW_HEIGHT) / height);
    let physical = |logical: f64| (logical * scale).round().max(1.0) as u32;
    Some((physical(width), physical(height)))
}
