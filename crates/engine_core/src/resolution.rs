//! The size display frames draw at: the user's resolution scale of the size
//! the hosts' resolution policy gives the view, within the display, and
//! dynamic resolution, which scales that size so frames keep within a GPU
//! time budget. Frames draw at it from the top-left corner of targets made
//! at the display's whole size, which the frame's last pass upscales to:
//! temporal antialiasing's resolve, or the screen effects' composite.
//! Material textures sample as much sharper as the frame draws smaller.
//! Debug views and exact frames draw at the display's size.

use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use crate::environment::InvalidSetting;
use engine_settings::{ExtraRules, settings};

settings! {
    /// The user's resolution scale: display frames draw at this share of
    /// the size the resolution policy gives the view on each axis, within
    /// the display's own size, and upscale to the display. 1 by default.
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneResolution")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Resolution {
        /// The share of the resolution policy's size display frames draw at
        /// on each axis: below 1 draws fewer pixels, above 1 more where the
        /// display holds them.
        pub scale: f32 = 1.0 => range(0.25, 2.0),
    }
}

impl Resolution {
    /// What a world without these settings reads as: the policy's size.
    pub fn off() -> Self {
        Self::default()
    }
}

settings! {
    extra_rules;
    /// Dynamic resolution settings: whether display frames scale the size
    /// they draw at, between the least and the most scale, so their GPU
    /// time keeps within a budget. It goes by two timestamps a frame where
    /// the device times its work, else by the time between frames; it lowers
    /// the scale at once while frames run over the budget and raises it a
    /// step at a time while they run under, in steps that keep temporal
    /// antialiasing's history steady. Off unless a scene names it; hosts
    /// turn it on for play in a window or a page, and keep it off for
    /// offscreen captures and measured runs.
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneDynamicResolution")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct DynamicResolution {
        /// Whether display frames scale the size they draw at to keep within
        /// the budget.
        pub enabled: bool = true => any,
        /// The least scale frames draw at on each axis, of the size the
        /// resolution policy and the user's scale give: at most `max_scale`.
        pub min_scale: f32 = 0.5 => range(0.25, 1.0),
        /// The most scale frames draw at on each axis.
        pub max_scale: f32 = 1.0 => range(0.25, 1.0),
        /// Milliseconds of GPU time each frame keeps within; omitted, the
        /// display's refresh interval where the host knows it, else a
        /// sixtieth of a second.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub budget_ms: Option<f32> = None => range(1.0, 1000.0),
    }
}

impl ExtraRules for DynamicResolution {
    /// The least scale is at most the most.
    fn check_extra(&self) -> Result<(), InvalidSetting> {
        if self.min_scale > self.max_scale {
            return Err(InvalidSetting::new(
                "min_scale",
                "must be at most max_scale",
            ));
        }
        Ok(())
    }

    /// A least scale above the most is the most.
    fn sanitize_extra(self) -> Self {
        Self {
            min_scale: crate::math::min(self.min_scale, self.max_scale),
            ..self
        }
    }
}

impl DynamicResolution {
    /// What a world without these settings reads as: no dynamic resolution.
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}
