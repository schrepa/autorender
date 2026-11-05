use super::*;
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
                .got_whole((format!("{reason:?}"), message))
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

fn named_by(parts: &[crate::install::PartInfo], message: &str) -> Option<usize> {
    let mut named = None;
    for (index, part) in parts.iter().enumerate() {
        if part.labels.iter().any(|label| message.contains(label)) {
            if named.is_some() {
                return None;
            }
            named = Some(index);
        }
    }
    named
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
        let linked = self.core.pipeline_counts.take_errors();
        if let Ok(mut reports) = self.core.health.lock() {
            if let Some(error) = linked.into_iter().next() {
                reports.fatal.get_or_insert(error);
            }
            for message in std::mem::take(&mut reports.messages) {
                let named = self
                    .run
                    .as_ref()
                    .and_then(|run| named_by(&run.machine.declared.parts, &message));
                match (named, self.run.as_mut()) {
                    (Some(part), Some(run)) => {
                        if !run.machine.declared.parts[part].lazy {
                            continue;
                        }
                        let saved = run.part;
                        run.part = part as u32;
                        let error = failed("$.gpu").got_whole(&message);
                        run.fault(error);
                        run.part = saved;
                    }
                    (Some(_) | None, _) => {
                        reports
                            .fatal
                            .get_or_insert_with(|| failed("$.gpu").got_whole(message));
                    }
                }
            }
            if let Some(fatal) = &reports.fatal {
                return Err(fatal.clone());
            }
        }
        Ok(())
    }

    pub fn faulted_parts(&self) -> Vec<&'static str> {
        let Some(run) = self.run.as_ref() else {
            return Vec::new();
        };
        run.faulted
            .iter()
            .zip(&run.machine.declared.parts)
            .filter(|(fault, _)| fault.is_some())
            .map(|(_, part)| part.name)
            .collect()
    }

    #[cfg(feature = "dev")]
    pub fn lose_device(&mut self) {
        self.core.device.destroy();
        crate::platform::poll(&self.core.device);
    }

    #[cfg(feature = "dev")]
    pub fn inject_fault(&mut self, feature: &str) -> Result<(), Error> {
        let Some(run) = self.run.as_mut() else {
            return Err(
                Error::new(engine_core::ErrorCode::UnknownVariant, "$.feature").got(feature),
            );
        };
        let parts = &run.machine.declared.parts;
        let Some(index) = parts
            .iter()
            .position(|known| known.lazy && known.name == feature)
        else {
            let names: Vec<&str> = parts
                .iter()
                .filter(|known| known.lazy)
                .map(|known| known.name)
                .collect();
            return Err(
                Error::new(engine_core::ErrorCode::UnknownVariant, "$.feature")
                    .got(feature)
                    .variants(names),
            );
        };
        let saved = run.part;
        run.part = index as u32;
        run.fault(failed("$.gpu").got("injected"));
        run.part = saved;
        Ok(())
    }
}
