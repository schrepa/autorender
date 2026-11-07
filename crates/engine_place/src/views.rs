use bevy_ecs::component::Component;
use bevy_ecs::entity::Entity;
use serde::{Deserialize, Serialize};

use crate::{ApplyAt, Ask, Error, ErrorCode, FrameId, Layers, NodeId, TextureHandle};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Target {
    #[default]
    Surface,
    Texture(TextureHandle),
}

impl Target {
    pub const fn surface(self) -> bool {
        matches!(self, Self::Surface)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Rect {
    pub const FULL: Self = Self {
        x: 0.0,
        y: 0.0,
        width: 1.0,
        height: 1.0,
    };

    pub fn check(self, path: &str) -> Result<(), Error> {
        let within = |value: f32| value.is_finite() && (0.0..=1.0).contains(&value);
        let fields = [
            ("x", self.x),
            ("y", self.y),
            ("width", self.width),
            ("height", self.height),
        ];
        for (name, value) in fields {
            if !within(value) {
                return Err(Error::new(ErrorCode::OutOfRange, format!("{path}.{name}"))
                    .range(0.0, 1.0)
                    .got(value));
            }
        }
        if self.width <= 0.0 || self.x + self.width > 1.0 {
            return Err(Error::new(ErrorCode::OutOfRange, format!("{path}.width"))
                .range(0.0, f64::from(1.0 - self.x))
                .got(self.width));
        }
        if self.height <= 0.0 || self.y + self.height > 1.0 {
            return Err(Error::new(ErrorCode::OutOfRange, format!("{path}.height"))
                .range(0.0, f64::from(1.0 - self.y))
                .got(self.height));
        }
        Ok(())
    }

    pub fn pixels(self, (width, height): (u32, u32)) -> [u32; 4] {
        let at = |share: f32, whole: u32| (share * whole as f32).round() as u32;
        let x = at(self.x, width).min(width.saturating_sub(1));
        let y = at(self.y, height).min(height.saturating_sub(1));
        let right = at(self.x + self.width, width).clamp(x + 1, width.max(x + 1));
        let bottom = at(self.y + self.height, height).clamp(y + 1, height.max(y + 1));
        [x, y, right - x, bottom - y]
    }
}

impl Default for Rect {
    fn default() -> Self {
        Self::FULL
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum Clear {
    #[default]
    All,
    DepthOnly,
    Nothing,
}

impl Clear {
    pub const fn color(self) -> bool {
        matches!(self, Self::All)
    }

    pub const fn depth(self) -> bool {
        matches!(self, Self::All | Self::DepthOnly)
    }
}

#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct View {
    pub camera: NodeId,
    pub target: Target,
    pub viewport: Rect,
    pub layers: Layers,
    pub clear: Clear,
    pub order: i32,
}

impl View {
    pub const fn new(camera: NodeId) -> Self {
        Self {
            camera,
            target: Target::Surface,
            viewport: Rect::FULL,
            layers: Layers::ALL,
            clear: Clear::All,
            order: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ViewHistory {
    #[default]
    Fresh,
    Valid {
        since: FrameId,
    },
}

impl ViewHistory {
    pub const fn valid(self) -> bool {
        matches!(self, Self::Valid { .. })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ViewCut {
    pub view: Entity,
}

impl Ask for ViewCut {
    const NAME: &'static str = "view_cut";
    const AT: ApplyAt = ApplyAt::FrameStart;
}
