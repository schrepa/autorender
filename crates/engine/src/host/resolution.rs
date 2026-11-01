use crate::resolution::DynamicResolution;

use super::policy::{
    DYNAMIC_AIM, DYNAMIC_LOWER_FRAMES, DYNAMIC_OVER, DYNAMIC_PROBE_FRAMES, DYNAMIC_RAISE_FRAMES,
    DYNAMIC_STEP, DYNAMIC_UNDER, MAX_PIXEL_RATIO, MAX_VIEW_HEIGHT, MAX_VIEW_WIDTH,
};

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
    pub(crate) fn scale(&self, settings: &DynamicResolution) -> f64 {
        if settings.enabled {
            self.scale
                .clamp(f64::from(settings.min_scale), f64::from(settings.max_scale))
        } else {
            1.0
        }
    }

    pub(crate) fn reset(&mut self) {
        *self = Self::default();
    }

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
