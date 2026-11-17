use super::*;
use bevy_ecs::world::World;
use std::task::{Context, Poll, Waker};

#[derive(Clone, Debug, Default)]
enum Health {
    #[default]
    Live,
    Failed(Error),
    Lost(Error),
    Faulted,
    Recovering,
    Recovered,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceHealth {
    Live,
    Failed,
    Lost,
    Faulted,
    Recovering,
    Recovered,
}

enum Change {
    Failed(Error),
    Lost(Error),
    BricksFaulted,
    Recovering,
    Recovered,
    Settled,
}

impl Health {
    fn after(&mut self, change: Change) {
        *self = match (std::mem::take(self), change) {
            (Self::Live | Self::Recovering | Self::Recovered, Change::Failed(error)) => {
                Self::Failed(error)
            }
            (
                Self::Live | Self::Failed(_) | Self::Recovering | Self::Recovered,
                Change::Lost(error),
            ) => Self::Lost(error),
            (Self::Lost(_), Change::BricksFaulted) => Self::Faulted,
            (Self::Faulted, Change::Recovering) => Self::Recovering,
            (Self::Recovering, Change::Recovered) => Self::Recovered,
            (Self::Recovered, Change::Settled) => Self::Live,
            (state @ (Self::Failed(_) | Self::Lost(_) | Self::Faulted), Change::Failed(_))
            | (state @ (Self::Lost(_) | Self::Faulted), Change::Lost(_))
            | (
                state @ (Self::Live
                | Self::Failed(_)
                | Self::Faulted
                | Self::Recovering
                | Self::Recovered),
                Change::BricksFaulted,
            )
            | (
                state @ (Self::Live
                | Self::Failed(_)
                | Self::Lost(_)
                | Self::Recovering
                | Self::Recovered),
                Change::Recovering,
            )
            | (
                state @ (Self::Live
                | Self::Failed(_)
                | Self::Lost(_)
                | Self::Faulted
                | Self::Recovered),
                Change::Recovered,
            )
            | (
                state @ (Self::Live
                | Self::Failed(_)
                | Self::Lost(_)
                | Self::Faulted
                | Self::Recovering),
                Change::Settled,
            ) => state,
        };
    }

    fn fatal(&self) -> Option<&Error> {
        match self {
            Self::Failed(error) => Some(error),
            Self::Live | Self::Lost(_) | Self::Faulted | Self::Recovering | Self::Recovered => None,
        }
    }

    fn shown(&self) -> DeviceHealth {
        match self {
            Self::Live => DeviceHealth::Live,
            Self::Failed(_) => DeviceHealth::Failed,
            Self::Lost(_) => DeviceHealth::Lost,
            Self::Faulted => DeviceHealth::Faulted,
            Self::Recovering => DeviceHealth::Recovering,
            Self::Recovered => DeviceHealth::Recovered,
        }
    }
}

#[derive(Default)]
pub(crate) struct Reports {
    health: Health,
    messages: Vec<String>,
    notices: Vec<Error>,
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

fn refusal(machine: &crate::machine::Machine, (pass, program): (&str, u64), text: &str) -> Error {
    let label = (machine.pipelines.iter())
        .find(|built| built.program_key == program)
        .and_then(|built| match &built.decl.program {
            crate::install::Program::Render(desc) => Some(desc.vertex.name()),
            crate::install::Program::Compute(_) => Some(built.stages.first()?.entry.name()),
        });
    let path = match label {
        Some(label) => format!("$.pipelines.{pass}.{label}"),
        None => format!("$.pipelines.{pass}"),
    };
    failed(&path).got_whole(text)
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
        if let Ok(mut reports) = self.core.health.lock() {
            let mut refused = Vec::new();
            if let Some(run) = self.run.as_mut() {
                for (pass, program, text) in std::mem::take(&mut run.pipelines.refused) {
                    let notice = refusal(&run.machine, (pass, program), &text);
                    reports.notices.push(notice);
                    refused.push(text);
                }
            }
            reports.messages.extend(refused);
            for message in std::mem::take(&mut reports.messages) {
                let error = failed("$.gpu").got_whole(message);
                reports.health.after(Change::Failed(error));
            }
            if let Some(fatal) = reports.health.fatal() {
                return Err(fatal.clone());
            }
        }
        Ok(())
    }

    pub fn take_notices(&self) -> Vec<Error> {
        self.core
            .health
            .lock()
            .map(|mut reports| std::mem::take(&mut reports.notices))
            .unwrap_or_default()
    }

    pub fn device_health(&self) -> DeviceHealth {
        self.core
            .health
            .lock()
            .map_or(DeviceHealth::Failed, |reports| reports.health.shown())
    }

    pub(super) fn usable(&self) -> bool {
        matches!(
            self.device_health(),
            DeviceHealth::Live | DeviceHealth::Recovered
        )
    }

    fn change(&self, change: Change) {
        if let Ok(mut reports) = self.core.health.lock() {
            reports.health.after(change);
        }
    }

    pub(super) fn recover(&mut self, world: &mut World) {
        let state = self
            .core
            .health
            .lock()
            .ok()
            .map(|reports| reports.health.clone());
        match state {
            Some(Health::Faulted) => {
                self.recovering = Some(Box::pin(super::startup::open(
                    self.core.display.instance(),
                    self.core.display.power_preference(),
                )));
                self.change(Change::Recovering);
            }
            Some(Health::Recovering) => {
                let Some(opening) = self.recovering.as_mut() else {
                    return;
                };
                let Poll::Ready(opened) = opening
                    .as_mut()
                    .poll(&mut Context::from_waker(Waker::noop()))
                else {
                    return;
                };
                self.recovering = None;
                let adopted = opened.and_then(|opened| self.adopt_device(opened));
                let installed = adopted.and_then(|()| {
                    self.install(world).map_err(|problems| {
                        problems
                            .into_iter()
                            .next()
                            .unwrap_or_else(|| failed("$.device"))
                    })
                });
                match installed {
                    Ok(()) => {
                        for brick in std::mem::take(&mut self.lost_bricks) {
                            engine_registry::ask_on(world, brick);
                        }
                        self.change(Change::Recovered);
                    }
                    Err(error) => self.change(Change::Failed(error)),
                }
            }
            Some(Health::Recovered) => self.change(Change::Settled),
            Some(Health::Live | Health::Failed(_) | Health::Lost(_)) | None => {}
        }
    }

    pub(super) fn fault_bricks(&mut self, world: &mut World) {
        let lost = self
            .core
            .health
            .lock()
            .ok()
            .and_then(|reports| match &reports.health {
                Health::Lost(error) => Some(error.clone()),
                Health::Live
                | Health::Failed(_)
                | Health::Faulted
                | Health::Recovering
                | Health::Recovered => None,
            });
        let Some(error) = lost else {
            return;
        };
        let states = world.get_resource::<engine_registry::BrickStates>();
        let mut bricks: Vec<engine_registry::BrickId> = self
            .run
            .iter()
            .flat_map(|run| run.installs.gates.iter().flatten())
            .filter_map(|gate| states.and_then(|states| states.brick(*gate)))
            .collect();
        bricks.sort_unstable();
        bricks.dedup();
        for &brick in &bricks {
            engine_registry::fault(world, brick, error.clone());
        }
        self.lost_bricks = bricks;
        self.change(Change::BricksFaulted);
    }
}
