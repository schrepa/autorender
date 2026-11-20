mod input;
mod orders;

use std::any::Any;
use std::collections::VecDeque;

use bevy_ecs::world::World;
use engine_core::{Error, ErrorCode, Expected};
use engine_dev::{Applied, Request};
use engine_registry::{Kind, NameEntry, ReachMut, register};
use engine_render::Renderer;
use serde_json::{Value, json};

use crate::frame_stats::FrameStats;
use crate::host::DeviceGatherer;
use crate::{Game, GameHost};
use orders::{Order, Running};

#[derive(Default)]
pub(crate) struct DevHost {
    agent: DeviceGatherer,
    tickets: u32,
    lines: VecDeque<(u32, String)>,
    running: Option<Running>,
    stepping: bool,
    held: bool,
    out: DevTake,
}

#[derive(Default)]
pub struct DevTake {
    pub replies: Vec<(u32, String)>,
    pub pictures: Vec<Shot>,
    pub reload: bool,
}

pub struct Shot {
    pub file: String,
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

impl DevHost {
    pub(crate) fn stepping(&self) -> bool {
        self.stepping
    }

    pub(crate) fn drop_edges(&mut self) {
        self.agent.drop_edges();
    }

    pub(crate) fn take_input(&mut self) -> engine_core::Devices {
        self.agent.take()
    }
}

pub(crate) fn install(world: &mut World) {
    engine_dev::install(world);
    engine_render::gpu_names(world);
    register(
        world,
        NameEntry::read("frame", Kind::State, |reach, _| {
            if let Some(renderer) = reach.part::<Renderer>() {
                renderer.time_frame();
            }
            Ok(serde_json::to_value(reach.world.resource::<FrameStats>())
                .expect("frame_stats_serialize"))
        }),
    );
    register(world, input::entry());
    orders::install(world);
}

type Listing = fn(&str) -> Result<Vec<(String, u64)>, String>;

pub fn kept_entry(listing: Listing) -> NameEntry {
    NameEntry::read("kept", Kind::State, move |reach, _| {
        let place = reach
            .world
            .get_resource::<engine_core::Kept>()
            .and_then(engine_core::Kept::place)
            .ok_or_else(|| {
                Error::new(ErrorCode::UnsupportedFeature, "$.kept").got("this host keeps nothing")
            })?;
        let listed =
            listing(place).map_err(|error| Error::new(ErrorCode::SourceIo, "$.kept").got(error))?;
        Ok(json!({
            "place": place,
            "kept": listed
                .into_iter()
                .map(|(name, bytes)| json!({ "name": name, "bytes": bytes }))
                .collect::<Vec<_>>(),
        }))
    })
}

const UNCHANGED_WRITES: &str = "unchanged_writes";
const UNCHANGED_WRITES_CHANGES: &str =
    "hashes every byte written to the GPU on the CPU, so CPU render time grows with bytes written";

pub fn instruments() -> &'static [&'static str] {
    static NAMES: std::sync::LazyLock<Vec<&'static str>> = std::sync::LazyLock::new(|| {
        std::iter::once(UNCHANGED_WRITES)
            .chain(engine_render::GPU_INSTRUMENTS)
            .collect()
    });
    &NAMES
}

pub fn instrument(
    reach: &mut ReachMut<'_>,
    name: &str,
    on: bool,
) -> Result<&'static str, &'static str> {
    if name == UNCHANGED_WRITES {
        return engine_render::unchanged_writes(reach.world, on)
            .map(|()| UNCHANGED_WRITES_CHANGES)
            .map_err(|_| "unchanged_writes_refused");
    }
    match reach.part::<Renderer>() {
        Some(renderer) => renderer.gpu_instrument(name, on),
        None => Err("no_renderer"),
    }
}

impl<G: Game> GameHost<G> {
    pub fn dev_push(&mut self, line: &str) -> u32 {
        let development = &mut self.development;
        development.tickets = development.tickets.wrapping_add(1);
        development
            .lines
            .push_back((development.tickets, line.to_owned()));
        development.tickets
    }

    pub fn dev_take(&mut self) -> DevTake {
        std::mem::take(&mut self.development.out)
    }

    pub fn dev_reloaded(&mut self, source: &str) {
        let Some(running) = self.development.running.take() else {
            return;
        };
        let Order::Reload(reload) = &running.order else {
            self.development.running = Some(running);
            return;
        };
        let outcome = crate::game_host::reload_checked(
            self.scene.instance(),
            source,
            (&reload.expected_revision, &reload.revision),
        )
        .and_then(|document| self.arrive(document));
        self.development.running = Some(running);
        if outcome.is_err() {
            self.dev_reload_ended(outcome);
        }
    }

    pub fn dev_reload_ended(&mut self, outcome: crate::Result<()>) {
        let Some(running) = self
            .development
            .running
            .take_if(|running| matches!(running.order, Order::Reload(_)))
        else {
            return;
        };
        let refused = self
            .app
            .engine()
            .app
            .world()
            .resource::<engine_core::Assets>()
            .groups()
            .into_iter()
            .find_map(|group| match group.stage {
                engine_core::GroupStage::Failed(error) => Some(error),
                engine_core::GroupStage::Arriving
                | engine_core::GroupStage::Compiling
                | engine_core::GroupStage::Ready => None,
            });
        let outcome = outcome.and_then(|()| refused.map_or_else(|| Ok(json!({"reloaded": true})), Err));
        self.answer(&running, outcome);
    }

    pub(crate) fn dev_apply(&mut self) {
        while self.development.running.is_none() {
            let Some((ticket, line)) = self.development.lines.pop_front() else {
                return;
            };
            let request = match Request::parse(&line) {
                Ok(request) => request,
                Err(refused) => {
                    self.reply(ticket, refused.id.as_ref(), Err(refused.error));
                    continue;
                }
            };
            let applied = self.reach(|reach| engine_dev::apply(reach, &request));
            match applied {
                Ok(Applied::Value(value)) => self.reply(ticket, request.id.as_ref(), Ok(value)),
                Err(error) => self.reply(ticket, request.id.as_ref(), Err(error)),
                Ok(Applied::Loop(order)) => match Order::of(order) {
                    Some(order) => {
                        self.development.running = Some(Running {
                            ticket,
                            id: request.id,
                            order,
                        });
                    }
                    None => {
                        let refused = Error::new(ErrorCode::UnsupportedCommand, "$.name")
                            .expected(Expected::Supported)
                            .got(&request.name);
                        self.reply(ticket, request.id.as_ref(), Err(refused));
                    }
                },
            }
        }
    }

    fn reach<R>(&mut self, run: impl FnOnce(&mut ReachMut<'_>) -> R) -> R {
        let mut parts: Vec<&mut dyn Any> =
            vec![&mut self.development.agent, &mut *self.app.renderer];
        if let Some(scene) = self.scene.instance_mut() {
            parts.push(scene);
        }
        run(&mut ReachMut {
            world: self.app.engine.app.world_mut(),
            parts: &mut parts,
        })
    }

    fn reply(&mut self, ticket: u32, id: Option<&Value>, outcome: Result<Value, Error>) {
        let line = engine_dev::reply(id, self.app.engine().app.world(), outcome);
        self.development.out.replies.push((ticket, line));
    }

    fn answer(&mut self, running: &Running, outcome: Result<Value, Error>) {
        self.reply(running.ticket, running.id.as_ref(), outcome);
    }

    fn held(&self) -> bool {
        self.app
            .engine()
            .app
            .world()
            .resource::<engine_core::Held>()
            .paused
    }

    fn hold(&mut self) {
        self.app
            .engine_mut()
            .app
            .world_mut()
            .resource_mut::<engine_core::Held>()
            .paused = true;
        self.development.held = true;
    }
}
