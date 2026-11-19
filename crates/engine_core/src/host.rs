use bevy_ecs::prelude::Resource;
use serde::Serialize;

use crate::clock::Interval;

#[derive(Resource, Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
#[repr(u32)]
pub enum CursorMode {
    Drag = 0,
    #[default]
    Free = 1,
    Captured = 2,
}

impl CursorMode {
    pub const ALL: [(Self, &'static str); 3] = [
        (Self::Drag, "drag"),
        (Self::Free, "free"),
        (Self::Captured, "captured"),
    ];
}

#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Host {
    pub focused: bool,
    pub hidden: bool,
    pub showing: bool,
}

impl Host {
    pub(crate) const fn shown(self) -> bool {
        self.showing && !self.hidden
    }
}

#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Present {
    pub vsync: bool,
    pub latency: u32,
}

#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Show {
    WhenReady,
    Now,
}

impl crate::Engine {
    pub fn report_host(&mut self, host: Host) {
        if !host.shown() {
            self.interval = Interval::Fresh;
        }
        let world = self.app.world_mut();
        if world.get_resource::<Host>() != Some(&host) {
            world.insert_resource(host);
        }
    }

    pub(crate) fn shown(&self) -> bool {
        self.app
            .world()
            .get_resource::<Host>()
            .is_some_and(|host| host.shown())
    }
}
