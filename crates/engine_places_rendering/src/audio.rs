use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use bevy_ecs::prelude::{Component, Entity, Query, Res, Resource};
use bevy_ecs::system::SystemParamItem;
use engine_place::{Call, Error, ErrorCode, Lands, Named, Operation, Slot};
use glam::Vec3;
use serde::{Deserialize, Serialize};

const MAX_ORDERS: usize = 1024;
const RADIUS: f32 = 30.0;

pub const SOUND_LOADING: ErrorCode = ErrorCode::brick("sound_loading");

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
    Play {
        voice: VoiceHandle,
        entity: Entity,
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

#[doc(hidden)]
#[derive(Clone)]
pub enum SoundStanding {
    Loading,
    Ready,
    Refused(Error),
}

#[derive(Default)]
enum Hearing {
    #[default]
    Unheard,
    Heard {
        device: Result<(), Error>,
        live: usize,
        limit: usize,
    },
}

#[derive(Default)]
struct Queue {
    next_voice: u32,
    orders: Vec<SoundOrder>,
    cut: u32,
    hearing: Hearing,
    standing: BTreeMap<Entity, SoundStanding>,
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
    pub entity: String,
    #[serde(default)]
    pub shot: OneShot,
}

#[derive(Serialize, schemars::JsonSchema)]
pub struct Voiced {
    pub voice: VoiceHandle,
}

impl Operation for PlaySound {
    const NAME: &'static str = "sound_play";
    const LANDS: Lands = Lands::NextFrame;
    type Params = (
        Res<'static, Sounds>,
        Named<'static, 'static>,
        Query<'static, 'static, &'static SoundSource>,
    );
    type Request = Self;
    type Reply = Voiced;

    fn run(
        (sounds, named, sources): SystemParamItem<'_, '_, Self::Params>,
        request: Self,
    ) -> Result<Voiced, Error> {
        let entity = named.entity(&request.entity, "$.entity")?;
        let source = sources.get(entity).map_err(|_| {
            Error::new(ErrorCode::MissingComponent, "$.entity").got(&request.entity)
        })?;
        let voice = sounds.play((entity, source), request.shot)?;
        Ok(Voiced { voice })
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
    pub fn play(
        &self,
        (entity, source): (Entity, &SoundSource),
        shot: OneShot,
    ) -> Result<VoiceHandle, Error> {
        let mut queue = self.lock();
        queue.playable(entity, source)?;
        let voice = VoiceHandle(queue.next_voice + 1);
        if !queue.order(SoundOrder::Play {
            voice,
            entity,
            shot,
        }) {
            return Err(Error::new(ErrorCode::QueueFull, "$").at_most(MAX_ORDERS as u64));
        }
        Ok(queue.took(voice))
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

    #[doc(hidden)]
    pub fn hear(&self, device: Result<(), Error>, live: usize, limit: usize) {
        self.lock().hearing = Hearing::Heard {
            device,
            live,
            limit,
        };
    }

    #[doc(hidden)]
    pub fn stand(&self, entity: Entity, standing: SoundStanding) {
        self.lock().standing.insert(entity, standing);
    }

    #[doc(hidden)]
    pub fn forget(&self, entity: Entity) {
        self.lock().standing.remove(&entity);
    }

    fn lock(&self) -> MutexGuard<'_, Queue> {
        self.queue.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Queue {
    fn playable(&self, entity: Entity, source: &SoundSource) -> Result<(), Error> {
        let Hearing::Heard {
            device,
            live,
            limit,
        } = &self.hearing
        else {
            return Err(Error::new(ErrorCode::PlaceNotFilled, "$").got(Sounds::NAME));
        };
        device.clone()?;
        let queued = self
            .orders
            .iter()
            .filter(|order| matches!(order, SoundOrder::Play { .. } | SoundOrder::PlayBytes { .. }))
            .count();
        if live + queued >= *limit {
            return Err(Error::new(ErrorCode::QueueFull, "$").at_most(*limit as u64));
        }
        match self.standing.get(&entity) {
            Some(SoundStanding::Refused(error)) => Err(error.clone()),
            Some(SoundStanding::Ready) => Ok(()),
            Some(SoundStanding::Loading) | None if source.bytes.is_some() => Ok(()),
            Some(SoundStanding::Loading) | None => {
                Err(Error::new(SOUND_LOADING, "$.entity").got(&source.file))
            }
        }
    }

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
