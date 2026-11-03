use std::sync::Arc;

use bevy_ecs::prelude::{Component, Resource};
use engine_place::Slot;
use glam::Vec3;
use serde::{Deserialize, Serialize};

const MAX_ORDERS: usize = 1024;
const RADIUS: f32 = 30.0;

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum Bus {
    Master,
    Music,
    #[default]
    Effects,
    Ambient,
    Voice,
}

impl Bus {
    pub const ALL: [Self; 5] = [
        Self::Master,
        Self::Music,
        Self::Effects,
        Self::Ambient,
        Self::Voice,
    ];

    pub const fn index(self) -> usize {
        self as usize
    }
}

#[derive(Component, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[schemars(rename = "SceneSound")]
#[serde(default, deny_unknown_fields)]
pub struct SoundSource {
    pub file: String,
    pub bus: Bus,
    pub volume: f32,
    pub pitch: f32,
    pub looping: bool,
    pub spatial: bool,
    pub radius: f32,
    pub autoplay: bool,
    pub stream: bool,
    #[serde(skip)]
    pub bytes: Option<Arc<[u8]>>,
}

impl Default for SoundSource {
    fn default() -> Self {
        Self {
            file: String::new(),
            bus: Bus::default(),
            volume: 1.0,
            pitch: 1.0,
            looping: false,
            spatial: false,
            radius: RADIUS,
            autoplay: true,
            stream: false,
            bytes: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SoundHandle(u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VoiceHandle(u32);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OneShot {
    pub bus: Bus,
    pub volume: f32,
    pub pitch: f32,
    pub at: Option<Vec3>,
    pub radius: f32,
}

impl Default for OneShot {
    fn default() -> Self {
        Self {
            bus: Bus::default(),
            volume: 1.0,
            pitch: 1.0,
            at: None,
            radius: RADIUS,
        }
    }
}

#[doc(hidden)]
pub enum SoundOrder {
    Load {
        sound: SoundHandle,
        bytes: Arc<[u8]>,
    },
    Play {
        voice: VoiceHandle,
        sound: SoundHandle,
        shot: OneShot,
    },
    PlayBytes {
        voice: VoiceHandle,
        bytes: Arc<[u8]>,
        shot: OneShot,
    },
    Stop(VoiceHandle),
    Volume(VoiceHandle, f32),
}

#[derive(Resource, Default)]
pub struct Sounds {
    next_sound: u32,
    next_voice: u32,
    orders: Vec<SoundOrder>,
    cut: u32,
}

impl Slot for Sounds {
    const NAME: &'static str = "sounds";
}

impl Sounds {
    pub fn load(&mut self, bytes: impl Into<Arc<[u8]>>) -> SoundHandle {
        self.next_sound += 1;
        let sound = SoundHandle(self.next_sound);
        self.order(SoundOrder::Load {
            sound,
            bytes: bytes.into(),
        });
        sound
    }

    pub fn play(&mut self, sound: SoundHandle, shot: OneShot) -> VoiceHandle {
        let voice = self.voice();
        self.order(SoundOrder::Play { voice, sound, shot });
        voice
    }

    pub fn play_bytes(&mut self, bytes: impl Into<Arc<[u8]>>, shot: OneShot) -> VoiceHandle {
        let voice = self.voice();
        self.order(SoundOrder::PlayBytes {
            voice,
            bytes: bytes.into(),
            shot,
        });
        voice
    }

    pub fn stop(&mut self, voice: VoiceHandle) {
        self.order(SoundOrder::Stop(voice));
    }

    pub fn set_volume(&mut self, voice: VoiceHandle, volume: f32) {
        self.order(SoundOrder::Volume(voice, volume));
    }

    #[doc(hidden)]
    pub fn is_idle(&self) -> bool {
        self.orders.is_empty() && self.cut == 0
    }

    #[doc(hidden)]
    pub fn take_orders(&mut self) -> Vec<SoundOrder> {
        std::mem::take(&mut self.orders)
    }

    #[doc(hidden)]
    pub fn take_cut(&mut self) -> u32 {
        std::mem::take(&mut self.cut)
    }

    fn voice(&mut self) -> VoiceHandle {
        self.next_voice += 1;
        VoiceHandle(self.next_voice)
    }

    fn order(&mut self, order: SoundOrder) {
        if self.orders.len() < MAX_ORDERS {
            self.orders.push(order);
        } else {
            self.cut = self.cut.saturating_add(1);
        }
    }
}
