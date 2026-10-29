#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sun {
    pub direction: [f32; 4],
    pub color: [f32; 4],
}

impl Sun {
    pub const NONE: Self = Self {
        direction: [0.0, 1.0, 0.0, 0.0],
        color: [0.0, 0.0, 0.0, 1.0],
    };
}

pub trait SunFace {
    fn sun(&self) -> Sun;
}
