use super::*;
use bevy_ecs::world::World;
use std::task::{Context, Poll, Waker};

#[derive(Clone, Debug, Default)]
enum Health {
    #[default]
    Live,
    Failed(Error),
    Lost(Error),
    Faulted(Error),
}

enum Change {
    Failed(Error),
    Lost(Error),
    BricksFaulted,
}

impl Health {
    fn after(&mut self, change: Change) {
        *self = match (std::mem::take(self), change) {
            (Self::Live, Change::Failed(error)) => Self::Failed(error),
            (Self::Live | Self::Failed(_), Change::Lost(error)) => Self::Lost(error),
            (Self::Lost(error), Change::BricksFaulted) => Self::Faulted(error),
            (state @ (Self::Failed(_) | Self::Lost(_) | Self::Faulted(_)), Change::Failed(_))
            | (state @ (Self::Lost(_) | Self::Faulted(_)), Change::Lost(_))
            | (
                state @ (Self::Live | Self::Failed(_) | Self::Faulted(_)),
                Change::BricksFaulted,
            ) => state,
        };
    }

    fn error(&self) -> Option<&Error> {
        match self {
            Self::Live => None,
            Self::Failed(error) | Self::Lost(error) | Self::Faulted(error) => Some(error),
        }
    }
}

#[derive(Default)]
pub(crate) struct Reports {
    health: Health,
    messages: Vec<String>,
}

impl Reports {
    pub(super) fn lost(&mut self, reason: wgpu::DeviceLostReason, message: &str) {
        self.health.after(Change::Lost(
            failed("$.device")
                .expected(Expected::Live)
                .got_whole((format!("{reason:?}"), message)),
        ));
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
                reports.health.after(Change::Failed(error));
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
                        let error = failed("$.gpu").got_whole(message);
                        reports.health.after(Change::Failed(error));
                    }
                }
            }
            if let Some(fatal) = reports.health.error() {
                return Err(fatal.clone());
            }
        }
        Ok(())
    }

    pub(super) fn fault_bricks(&mut self, world: &mut World) {
        let lost = self
            .core
            .health
            .lock()
            .ok()
            .and_then(|reports| match &reports.health {
                Health::Lost(error) => Some(error.clone()),
                Health::Live | Health::Failed(_) | Health::Faulted(_) => None,
            });
        let Some(error) = lost else {
            return;
        };
        let states = world.get_resource::<engine_registry::BrickStates>();
        let mut bricks: Vec<engine_registry::BrickId> = self
            .run
            .iter()
            .flat_map(|run| run.gates.iter().flatten())
            .filter_map(|gate| states.and_then(|states| states.brick(*gate)))
            .collect();
        bricks.sort_unstable();
        bricks.dedup();
        for brick in bricks {
            engine_registry::fault(world, brick, error.clone());
        }
        if let Ok(mut reports) = self.core.health.lock() {
            reports.health.after(Change::BricksFaulted);
        }
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
