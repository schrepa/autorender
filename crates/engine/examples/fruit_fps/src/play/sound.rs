use engine::prelude::{Resource, settings};
use glam::Vec3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Anchor {
    Listener,
    Point(Vec3),
    Actor(u32),
    Barrier,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Cue {
    pub event: String,
    pub anchor: Anchor,
    pub source: Option<u32>,
    pub gain: f32,
    pub hold_ticks: Option<u32>,
    pub tag: Option<u64>,
}

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct SoundCues {
    pub cues: Vec<Cue>,
}

impl SoundCues {
    pub fn push(&mut self, event: impl Into<String>, anchor: Anchor, source: Option<u32>) {
        self.cues.push(Cue {
            event: event.into(),
            anchor,
            source,
            gain: 1.0,
            hold_ticks: None,
            tag: None,
        });
    }
}

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct SoundLoad {
    pub ready: bool,
    pub asked: u32,
    pub loaded: u32,
    pub failed: u32,
    pub refusal: Option<String>,
}

settings! {
    #[section("volume")]
    #[serde(deny_unknown_fields)]
    pub struct Volume {
        pub master: f32 = 1.0 => range(0.0, 1.0),
        pub effects: f32 = 1.0 => range(0.0, 1.0),
        pub interface: f32 = 1.0 => range(0.0, 1.0),
        pub footsteps: f32 = 1.0 => range(0.0, 1.0),
    }
}
