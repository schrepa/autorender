use glam::UVec2;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameFacts {
    pub display_size: UVec2,
    pub pixel_ratio: f32,
}

impl Default for FrameFacts {
    fn default() -> Self {
        Self {
            display_size: UVec2::ZERO,
            pixel_ratio: 1.0,
        }
    }
}
