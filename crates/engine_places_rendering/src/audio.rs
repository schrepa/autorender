use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use bevy_ecs::prelude::{Component, Res, Resource};
use bevy_ecs::system::SystemParamItem;
use engine_place::{Call, Error, Lands, Operation, Slot};
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

#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(transparent)]
pub struct SoundHandle(u32);

#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(transparent)]
pub struct VoiceHandle(u32);

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct OneShot {
    pub bus: Bus,
    pub volume: f32,
    pub pitch: f32,
    #[schemars(with = "Option<[f32; 3]>")]
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

#[derive(Default)]
struct Queue {
    next_sound: u32,
    next_voice: u32,
    orders: Vec<SoundOrder>,
    cut: u32,
}

#[derive(Resource, Default)]
pub struct Sounds {
    queue: Mutex<Queue>,
}

impl Slot for Sounds {
    const NAME: &'static str = "sounds";
    const CALLS: &'static [Call] = &[
        Call::of::<PlaySound>(),
        Call::of::<StopVoice>(),
        Call::of::<VoiceVolume>(),
    ];
}

#[derive(Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PlaySound {
    pub sound: SoundHandle,
    #[serde(default)]
    pub shot: OneShot,
}

#[derive(Serialize, schemars::JsonSchema)]
pub struct Voiced {
    pub voice: Option<VoiceHandle>,
}

impl Operation for PlaySound {
    const NAME: &'static str = "sound_play";
    const LANDS: Lands = Lands::NextFrame;
    type Params = Res<'static, Sounds>;
    type Request = Self;
    type Reply = Voiced;

    fn run(sounds: SystemParamItem<'_, '_, Self::Params>, request: Self) -> Result<Voiced, Error> {
        Ok(Voiced {
            voice: sounds.play(request.sound, request.shot),
        })
    }
}

#[derive(Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StopVoice {
    pub voice: VoiceHandle,
}

#[derive(Serialize, schemars::JsonSchema)]
pub struct Ordered {
    pub ordered: bool,
}

impl Operation for StopVoice {
    const NAME: &'static str = "sound_stop";
    const LANDS: Lands = Lands::NextFrame;
    type Params = Res<'static, Sounds>;
    type Request = Self;
    type Reply = Ordered;

    fn run(sounds: SystemParamItem<'_, '_, Self::Params>, request: Self) -> Result<Ordered, Error> {
        Ok(Ordered {
            ordered: sounds.stop(request.voice),
        })
    }
}

#[derive(Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct VoiceVolume {
    pub voice: VoiceHandle,
    pub volume: f32,
}

impl Operation for VoiceVolume {
    const NAME: &'static str = "sound_volume";
    const LANDS: Lands = Lands::NextFrame;
    type Params = Res<'static, Sounds>;
    type Request = Self;
    type Reply = Ordered;

    fn run(sounds: SystemParamItem<'_, '_, Self::Params>, request: Self) -> Result<Ordered, Error> {
        Ok(Ordered {
            ordered: sounds.set_volume(request.voice, request.volume),
        })
    }
}

impl Sounds {
    pub fn load(&self, bytes: impl Into<Arc<[u8]>>) -> Option<SoundHandle> {
        let mut queue = self.lock();
        let sound = SoundHandle(queue.next_sound + 1);
        queue
            .order(SoundOrder::Load {
                sound,
                bytes: bytes.into(),
            })
            .then(|| {
                queue.next_sound += 1;
                sound
            })
    }

    pub fn play(&self, sound: SoundHandle, shot: OneShot) -> Option<VoiceHandle> {
        let mut queue = self.lock();
        let voice = VoiceHandle(queue.next_voice + 1);
        queue
            .order(SoundOrder::Play { voice, sound, shot })
            .then(|| queue.took(voice))
    }

    pub fn play_bytes(&self, bytes: impl Into<Arc<[u8]>>, shot: OneShot) -> Option<VoiceHandle> {
        let mut queue = self.lock();
        let voice = VoiceHandle(queue.next_voice + 1);
        queue
            .order(SoundOrder::PlayBytes {
                voice,
                bytes: bytes.into(),
                shot,
            })
            .then(|| queue.took(voice))
    }

    pub fn stop(&self, voice: VoiceHandle) -> bool {
        self.lock().order(SoundOrder::Stop(voice))
    }

    pub fn set_volume(&self, voice: VoiceHandle, volume: f32) -> bool {
        self.lock().order(SoundOrder::Volume(voice, volume))
    }

    #[doc(hidden)]
    pub fn take_orders(&self) -> Vec<SoundOrder> {
        std::mem::take(&mut self.lock().orders)
    }

    #[doc(hidden)]
    pub fn take_cut(&self) -> u32 {
        std::mem::take(&mut self.lock().cut)
    }

    fn lock(&self) -> MutexGuard<'_, Queue> {
        self.queue.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Queue {
    fn took(&mut self, voice: VoiceHandle) -> VoiceHandle {
        self.next_voice = voice.0;
        voice
    }

    fn order(&mut self, order: SoundOrder) -> bool {
        let room = self.orders.len() < MAX_ORDERS;
        if room {
            self.orders.push(order);
        } else {
            self.cut = self.cut.saturating_add(1);
        }
        room
    }
}
