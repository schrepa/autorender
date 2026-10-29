use super::*;
use crate::temporal::TemporalPass;
use std::task::{Context, Poll, Waker};

#[derive(Default)]
pub(crate) struct Reports {
    fatal: Option<Error>,
    messages: Vec<String>,
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
        self.messages.push(message);
    }
}

impl Renderer {
    pub(super) fn check_health(&mut self) -> Result<(), Error> {
        let installed = self.installing.as_mut().map(|validation| {
            validation
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()))
        });
        match installed {
            Some(Poll::Ready(None)) => self.installing = None,
            Some(Poll::Ready(Some(error))) => {
                self.installing = None;
                return Err(failed_with("$.pipelines", error));
            }
            Some(Poll::Pending) | None => {}
        }
        let mut named = 0;
        if let Ok(mut reports) = self.core.health.lock() {
            for message in std::mem::take(&mut reports.messages) {
                match self.parts.named_by(&message) {
                    0 => {
                        reports
                            .fatal
                            .get_or_insert_with(|| failed("$.gpu").got(message));
                    }
                    bits => named |= bits,
                }
            }
            if let Some(fatal) = &reports.fatal {
                return Err(fatal.clone());
            }
        }
        if self.parts.disable(named) != 0
            && self
                .parts
                .get::<TemporalPass>()
                .is_some_and(|temporal| temporal.disabled())
        {
            self.forget_temporal();
        }
        Ok(())
    }

    pub fn disabled_features(&self) -> engine_core::Names {
        engine_core::Names {
            bits: self.parts.disabled(),
            names: self.parts.names(),
        }
    }

    #[cfg(feature = "dev")]
    pub fn lose_device(&mut self) {
        self.core.device.destroy();
        crate::platform::poll(&self.core.device);
    }

    #[cfg(feature = "dev")]
    pub fn inject_fault(&mut self, feature: &str) -> Result<(), Error> {
        if self.parts.inject(feature) {
            return Ok(());
        }
        Err(
            Error::new(engine_core::ErrorCode::UnknownVariant, "$.feature")
                .got(feature)
                .variants(self.parts.lazy_names()),
        )
    }
}
