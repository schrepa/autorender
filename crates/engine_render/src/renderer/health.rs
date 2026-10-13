//! The renderer's health: what the device's callbacks report, which each
//! frame reads first. Losing the device, or an error that names no one
//! lazily created feature by its objects' labels, fails the renderer for
//! good; an error naming one disables that feature alone, as its creation
//! failing validation does, and frames render without it.

use super::*;
use crate::render_features::RenderFeatures;

/// What the device's callbacks report, shared with them: a rare message or
/// set of features, never ECS data.
#[derive(Default)]
pub(super) struct Reports {
    /// Why the renderer failed for good.
    fatal: Option<String>,
    /// The features uncaptured errors named since a frame last read them,
    /// as a set.
    named: u64,
}

impl Reports {
    /// Records the device's loss: fatal.
    pub(super) fn lost(&mut self, reason: wgpu::DeviceLostReason, message: &str) {
        self.fatal.get_or_insert_with(|| {
            format!("GPU device lost ({reason:?}): {message}. Reload to recreate the engine.")
        });
    }

    /// Records an uncaptured `error`: the one lazily created feature its
    /// message names, or else the renderer's failure.
    pub(super) fn uncaptured(&mut self, error: &wgpu::Error) {
        let message = error.to_string();
        // Browsers report a frame's command failing, then its submission
        // failing because it did: the first says why, and decides.
        if message.contains("[Invalid CommandBuffer") {
            return;
        }
        match RenderFeatures::named_by(&message) {
            0 => {
                self.fatal.get_or_insert_with(|| {
                    format!("WebGPU rendering failed: {message}. Reload to recreate the engine.")
                });
            }
            named => self.named |= named,
        }
    }
}

impl Renderer {
    /// Reads what the device reported since the last frame: the renderer's
    /// failure, or the features errors named, which it disables with those
    /// whose creation failed validation.
    pub(super) fn check_health(&mut self) -> Result<(), RenderError> {
        let mut named = 0;
        if let Ok(mut reports) = self.health.lock() {
            if let Some(message) = &reports.fatal {
                return Err(RenderError(message.clone()));
            }
            named = std::mem::take(&mut reports.named);
        }
        if self.features.disable(named) != 0 && self.features.temporal.disabled() {
            self.forget_temporal();
        }
        Ok(())
    }

    /// The renderer's features disabled for good, by name.
    pub fn disabled_features(&self) -> engine_core::Names {
        engine_core::Names {
            bits: self.features.disabled(),
            names: RenderFeatures::NAMES,
        }
    }

    /// Injects a fault into the lazily created feature named `feature`, to
    /// check that the renderer disables it alone: its next creation makes
    /// an invalid buffer of its own, which its first frame uses.
    ///
    /// # Errors
    ///
    /// When no lazily created feature has that name.
    #[cfg(feature = "dev")]
    pub fn inject_fault(&mut self, feature: &str) -> Result<(), RenderError> {
        if self.features.inject(feature) {
            return Ok(());
        }
        Err(RenderError(format!(
            "No lazily created feature is named {feature:?}; they are {}.",
            RenderFeatures::lazy_names().collect::<Vec<_>>().join(", ")
        )))
    }
}
