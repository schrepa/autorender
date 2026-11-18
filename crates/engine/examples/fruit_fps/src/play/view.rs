use engine::prelude::{Component, Entity, Resource, settings};
use glam::Vec3;

#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct ViewOffsets {
    pub recoil_pitch_radians: f32,
    pub recoil_yaw_radians: f32,
    pub punch_pitch_radians: f32,
    pub punch_yaw_radians: f32,
    pub zoom: f32,
}

impl ViewOffsets {
    pub const NONE: ViewOffsets = ViewOffsets {
        recoil_pitch_radians: 0.0,
        recoil_yaw_radians: 0.0,
        punch_pitch_radians: 0.0,
        punch_yaw_radians: 0.0,
        zoom: 1.0,
    };

    pub fn pitch_radians(&self) -> f32 {
        self.recoil_pitch_radians + self.punch_pitch_radians
    }

    pub fn yaw_radians(&self) -> f32 {
        self.recoil_yaw_radians + self.punch_yaw_radians
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Watch {
    #[default]
    Own,
    Corpse,
    Teammate,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CorpseView {
    pub eye: Vec3,
    pub yaw_radians: f32,
    pub tick: u64,
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct Watched {
    pub slot: Option<u32>,
    pub entity: Option<Entity>,
    pub watch: Watch,
    pub corpse: Option<CorpseView>,
}

settings! {
    #[section("controls")]
    #[serde(deny_unknown_fields)]
    pub struct Controls {
        pub radians_per_pixel: f32 = 0.0022 => range(0.00001, 0.05),
        pub invert_pitch: bool = false => any,
        pub fov_degrees: f32 = 62.0 => range(40.0, 120.0),
        pub render_scale: f32 = 1.0 => range(0.25, 1.0),
    }
}
