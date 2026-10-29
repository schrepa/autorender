use super::*;
use crate::render_features::RenderFeatures;

#[derive(Default)]
pub(super) struct Reports {
    fatal: Option<Error>,
    named: u64,
}

impl Reports {
    pub(super) fn lost(&mut self, reason: wgpu::DeviceLostReason, message: &str) {
        self.fatal.get_or_insert_with(|| {
            failed("$.device")
                .expected(Expected::Live)
                .got((format!("{reason:?}"), message))
        });
    }

    pub(super) fn uncaptured(&mut self, error: &wgpu::Error) {
        let message = error.to_string();
        if message.contains("[Invalid CommandBuffer") {
            return;
        }
        match RenderFeatures::named_by(&message) {
            0 => {
                self.fatal
                    .get_or_insert_with(|| failed("$.gpu").got(message));
            }
            named => self.named |= named,
        }
    }
}

impl Renderer {
    pub(super) fn check_health(&mut self) -> Result<(), Error> {
        let mut named = 0;
        if let Ok(mut reports) = self.health.lock() {
            if let Some(fatal) = &reports.fatal {
                return Err(fatal.clone());
            }
            named = std::mem::take(&mut reports.named);
        }
        if self.features.disable(named) != 0 && self.features.temporal.disabled() {
            self.forget_temporal();
        }
        Ok(())
    }

    pub fn disabled_features(&self) -> engine_core::Names {
        engine_core::Names {
            bits: self.features.disabled(),
            names: RenderFeatures::NAMES,
        }
    }

    #[cfg(feature = "dev")]
    pub fn lose_device(&mut self) {
        self.device.destroy();
        crate::platform::poll(&self.device);
    }

    #[cfg(feature = "dev")]
    pub fn inject_fault(&mut self, feature: &str) -> Result<(), Error> {
        if self.features.inject(feature) {
            return Ok(());
        }
        Err(Error::new(ErrorCode::UnknownVariant, "$.feature")
            .got(feature)
            .variants(RenderFeatures::lazy_names()))
    }
}
