use glam::{UVec2, Vec2};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ViewportClass {
    Compact,
    Medium,
    #[default]
    Wide,
}

impl ViewportClass {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Compact => "compact",
            Self::Medium => "medium",
            Self::Wide => "wide",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameFacts {
    pub display_size: UVec2,
    pub pixel_ratio: f32,
    pub safe_insets_pixels: [f32; 4],
}

impl Default for FrameFacts {
    fn default() -> Self {
        Self {
            display_size: UVec2::ZERO,
            pixel_ratio: 1.0,
            safe_insets_pixels: [0.0; 4],
        }
    }
}

impl FrameFacts {
    pub fn effective_pixel_ratio(self) -> f32 {
        if self.pixel_ratio.is_finite() && self.pixel_ratio > 0.0 {
            self.pixel_ratio
        } else {
            1.0
        }
    }

    pub fn logical_size(self) -> Vec2 {
        let ratio = self.effective_pixel_ratio();
        Vec2::new(
            self.display_size.x as f32 / ratio,
            self.display_size.y as f32 / ratio,
        )
    }

    pub fn portrait(self) -> bool {
        self.display_size.y > self.display_size.x
    }

    pub fn viewport_class(self) -> ViewportClass {
        let logical = self.logical_size();
        let short = if logical.x < logical.y {
            logical.x
        } else {
            logical.y
        };
        if short < 600.0 {
            ViewportClass::Compact
        } else if short < 900.0 {
            ViewportClass::Medium
        } else {
            ViewportClass::Wide
        }
    }

    pub fn safe_rect_pixels(self) -> [f32; 4] {
        let width = self.display_size.x as f32;
        let height = self.display_size.y as f32;
        let bound = |value: f32, limit: f32| {
            let positive = if value.is_finite() && value > 0.0 {
                value
            } else {
                0.0
            };
            if positive < limit { positive } else { limit }
        };
        let top = bound(self.safe_insets_pixels[0], height * 0.45);
        let right = bound(self.safe_insets_pixels[1], width * 0.45);
        let bottom = bound(self.safe_insets_pixels[2], height * 0.45);
        let left = bound(self.safe_insets_pixels[3], width * 0.45);
        let safe_w = width - left - right;
        let safe_h = height - top - bottom;
        [
            left,
            top,
            if safe_w > 0.0 { safe_w } else { 0.0 },
            if safe_h > 0.0 { safe_h } else { 0.0 },
        ]
    }
}
