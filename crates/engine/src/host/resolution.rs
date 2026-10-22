//! The size a view draws at, on every host: a resolution budget that keeps
//! fill cost independent of how large and dense the display is, the user's
//! scale of it, and dynamic resolution's, which keeps frames within a GPU
//! time budget. Hosts hand the engine the display's whole size; frames draw
//! at this size inside it and upscale to it.

use engine_core::DynamicResolution;

use super::policy::{
    DYNAMIC_AIM, DYNAMIC_LOWER_FRAMES, DYNAMIC_OVER, DYNAMIC_PROBE_FRAMES, DYNAMIC_RAISE_FRAMES,
    DYNAMIC_STEP, DYNAMIC_UNDER, MAX_PIXEL_RATIO, MAX_VIEW_HEIGHT, MAX_VIEW_WIDTH,
};

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
    let scale = (known(pixel_ratio).min(MAX_PIXEL_RATIO) * known(scale))
        .min(f64::from(MAX_VIEW_WIDTH) / width)
        .min(f64::from(MAX_VIEW_HEIGHT) / height);
    let physical = |logical: f64| (logical * scale).round().max(1.0) as u32;
    Some((physical(width), physical(height)))
}

/// `value`, where it is a finite positive number; 1 otherwise.
fn known(value: f64) -> f64 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        1.0
    }
}

/// The size display frames draw at on a display `width` × `height` physical
/// pixels of `pixel_ratio` physical pixels per logical one: the view's size
/// by the resolution policy ([`view_size`]) at the user's resolution
/// `scale`, within the display, times dynamic resolution's `dynamic` scale;
/// a pixel at least each way. `None` for a display with no area.
pub fn render_size(
    (width, height): (u32, u32),
    pixel_ratio: f64,
    scale: f64,
    dynamic: f64,
) -> Option<(u32, u32)> {
    if width == 0 || height == 0 {
        return None;
    }
    let ratio = known(pixel_ratio);
    let (wide, tall) = view_size(
        f64::from(width) / ratio,
        f64::from(height) / ratio,
        ratio,
        scale,
    )?;
    let dynamic = known(dynamic);
    let scaled = |policy: u32, display: u32| {
        ((f64::from(policy.min(display)) * dynamic).round() as u32).clamp(1, display)
    };
    Some((scaled(wide, width), scaled(tall, height)))
}

/// Dynamic resolution's scale of the size frames draw at: lowered at once
/// toward what keeps frames within their budget after [`DYNAMIC_LOWER_FRAMES`]
/// frames in a row over it, raised a [`DYNAMIC_STEP`] after frames in a row
/// under it, held between; on the step's multiples within the settings'
/// bounds.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DynamicScale {
    scale: f64,
    over: u32,
    under: u32,
}

impl Default for DynamicScale {
    fn default() -> Self {
        Self {
            scale: 1.0,
            over: 0,
            under: 0,
        }
    }
}

impl DynamicScale {
    /// The scale frames draw at under `settings`: 1 while they are off.
    pub(crate) fn scale(&self, settings: &DynamicResolution) -> f64 {
        if settings.enabled {
            self.scale
                .clamp(f64::from(settings.min_scale), f64::from(settings.max_scale))
        } else {
            1.0
        }
    }

    /// Starts over at the most scale with no frames counted: after a pause,
    /// or with dynamic resolution off.
    pub(crate) fn reset(&mut self) {
        *self = Self::default();
    }

    /// Counts a frame that took `ms` of its `budget` milliseconds under
    /// `settings`: its GPU work where the device `timed` it, else the time
    /// since the frame before, which cannot tell how far under the budget a
    /// frame waiting for the display ran, so the scale rises only after
    /// [`DYNAMIC_PROBE_FRAMES`] frames within it.
    pub(crate) fn frame(
        &mut self,
        settings: &DynamicResolution,
        ms: f64,
        budget: f64,
        timed: bool,
    ) {
        let (low, high) = (f64::from(settings.min_scale), f64::from(settings.max_scale));
        let scale = self.scale.clamp(low, high);
        let (under, frames) = if timed {
            (DYNAMIC_UNDER, DYNAMIC_RAISE_FRAMES)
        } else {
            (DYNAMIC_OVER, DYNAMIC_PROBE_FRAMES)
        };
        if ms > budget * DYNAMIC_OVER {
            self.over += 1;
            self.under = 0;
            if self.over >= DYNAMIC_LOWER_FRAMES {
                // A frame's GPU work goes with its pixels, the scale's square.
                let aimed = scale * (budget * DYNAMIC_AIM / ms).sqrt();
                let stepped =
                    ((aimed / DYNAMIC_STEP).floor() * DYNAMIC_STEP).min(scale - DYNAMIC_STEP);
                self.scale = stepped.clamp(low, high);
                self.over = 0;
            }
        } else if ms < budget * under {
            self.under += 1;
            self.over = 0;
            if self.under >= frames {
                self.scale = (scale + DYNAMIC_STEP).min(high);
                self.under = 0;
            }
        } else {
            self.over = 0;
            self.under = 0;
        }
    }
}
