use std::collections::BTreeMap;
use std::io::Cursor;
use std::sync::Arc;

use engine_core::math::clamp;
use engine_places::audio::Bus;
use glam::{Mat3, Quat, Vec3};
use kira::listener::ListenerHandle;
use kira::sound::static_sound::{StaticSoundData, StaticSoundHandle};
use kira::sound::streaming::{StreamingSoundData, StreamingSoundHandle};
use kira::sound::{FromFileError, PlaybackState, SoundData};
use kira::track::{SpatialTrackBuilder, SpatialTrackHandle, TrackBuilder, TrackHandle};
use kira::{AudioManager, AudioManagerSettings, Decibels, DefaultBackend, Easing, Tween};

use crate::mixer::Start;
use crate::settings::MOST_VOICES;

const SILENT: f32 = 0.001;
const DECIBELS_PER_DECADE: f32 = 20.0;
const ORIGIN: [f32; 3] = [0.0; 3];
const FACING: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

fn decibels(amplitude: f32) -> Decibels {
    if amplitude <= SILENT {
        Decibels::SILENCE
    } else {
        Decibels(DECIBELS_PER_DECADE * amplitude.log10())
    }
}

enum Sound {
    Held(StaticSoundData),
    Streamed(Arc<[u8]>),
    Failed(String),
}

enum Playing {
    Held(StaticSoundHandle),
    Streamed(StreamingSoundHandle<FromFileError>),
}

impl Playing {
    fn state(&self) -> PlaybackState {
        match self {
            Self::Held(handle) => handle.state(),
            Self::Streamed(handle) => handle.state(),
        }
    }

    fn stop(&mut self) {
        match self {
            Self::Held(handle) => handle.stop(Tween::default()),
            Self::Streamed(handle) => handle.stop(Tween::default()),
        }
    }

    fn set(&mut self, volume: f32, pitch: f32) {
        let (volume, pitch) = (decibels(volume), f64::from(pitch));
        match self {
            Self::Held(handle) => {
                handle.set_volume(volume, Tween::default());
                handle.set_playback_rate(pitch, Tween::default());
            }
            Self::Streamed(handle) => {
                handle.set_volume(volume, Tween::default());
                handle.set_playback_rate(pitch, Tween::default());
            }
        }
    }
}

struct Voice {
    playing: Playing,
    track: Option<SpatialTrackHandle>,
    radius: f32,
}

struct Device {
    manager: AudioManager<DefaultBackend>,
    listener: ListenerHandle,
    tracks: [TrackHandle; Bus::ALL.len()],
}

impl Device {
    fn spatial(&mut self, bus: Bus) -> Result<SpatialTrackHandle, String> {
        self.tracks[bus.index()]
            .add_spatial_sub_track(self.listener.id(), ORIGIN, spatial_track())
            .map_err(|failure| failure.to_string())
    }
}

fn spatial_track() -> SpatialTrackBuilder {
    SpatialTrackBuilder::new()
        .volume(Decibels::SILENCE)
        .attenuation_function(None::<Easing>)
        .spatialization_strength(1.0_f32)
        .persist_until_sounds_finish(true)
        .sound_capacity(1)
        .sub_track_capacity(1)
}

fn bus_track(manager: &mut AudioManager<DefaultBackend>) -> Result<TrackHandle, String> {
    let room = MOST_VOICES as usize;
    let builder = TrackBuilder::new()
        .sub_track_capacity(room)
        .sound_capacity(room);
    manager
        .add_sub_track(builder)
        .map_err(|failure| failure.to_string())
}

fn held(data: &StaticSoundData, play: &Start) -> StaticSoundData {
    let data = data
        .volume(decibels(play.volume))
        .playback_rate(f64::from(play.pitch));
    if play.looping {
        data.loop_region(0.0..)
    } else {
        data
    }
}

fn streamed(bytes: &Arc<[u8]>, play: &Start) -> Result<StreamingSoundData<FromFileError>, String> {
    let data = StreamingSoundData::from_cursor(Cursor::new(Arc::clone(bytes)))
        .map_err(|failure| failure.to_string())?
        .volume(decibels(play.volume))
        .playback_rate(f64::from(play.pitch));
    Ok(if play.looping {
        data.loop_region(0.0..)
    } else {
        data
    })
}

fn on_track<D: SoundData>(
    bus: &mut TrackHandle,
    track: Option<&mut SpatialTrackHandle>,
    data: D,
) -> Result<D::Handle, String> {
    let played = match track {
        Some(track) => track.play(data),
        None => bus.play(data),
    };
    played.map_err(|failure| failure.to_string())
}

#[derive(Default)]
pub(crate) struct Backend {
    device: Option<Device>,
    sounds: BTreeMap<u32, Sound>,
    voices: BTreeMap<u32, Voice>,
}

impl Backend {
    pub(crate) fn open(&mut self) -> Result<(), String> {
        let mut manager = AudioManager::<DefaultBackend>::new(AudioManagerSettings::default())
            .map_err(|failure| failure.to_string())?;
        let listener = manager
            .add_listener(ORIGIN, FACING)
            .map_err(|failure| failure.to_string())?;
        let tracks = [
            bus_track(&mut manager)?,
            bus_track(&mut manager)?,
            bus_track(&mut manager)?,
            bus_track(&mut manager)?,
            bus_track(&mut manager)?,
        ];
        self.device = Some(Device {
            manager,
            listener,
            tracks,
        });
        Ok(())
    }

    pub(crate) fn opened(&self) -> bool {
        self.device.is_some()
    }

    pub(crate) fn close(&mut self) {
        self.voices.clear();
        self.device = None;
    }

    pub(crate) fn decode(&mut self, sound: u32, bytes: Arc<[u8]>, stream: bool, _may_open: bool) {
        let decoded = if stream {
            StreamingSoundData::from_cursor(Cursor::new(Arc::clone(&bytes)))
                .map(|_| Sound::Streamed(bytes))
        } else {
            StaticSoundData::from_cursor(Cursor::new(bytes)).map(Sound::Held)
        };
        let entry = decoded.unwrap_or_else(|failure| Sound::Failed(failure.to_string()));
        self.sounds.insert(sound, entry);
    }

    pub(crate) fn forget(&mut self, sound: u32) {
        self.sounds.remove(&sound);
    }

    pub(crate) fn arrival(&self, sound: u32) -> Option<Result<(), String>> {
        Some(match self.sounds.get(&sound) {
            Some(Sound::Held(_) | Sound::Streamed(_)) => Ok(()),
            Some(Sound::Failed(failure)) => Err(failure.clone()),
            None => Err("sound_missing".to_owned()),
        })
    }

    pub(crate) fn start(&mut self, voice: u32, sound: u32, play: &Start) -> Result<(), String> {
        let device = self.device.as_mut().ok_or("no_device")?;
        let sound = self.sounds.get(&sound).ok_or("sound_missing")?;
        let mut track = play.radius.map(|_| device.spatial(play.bus)).transpose()?;
        let bus = &mut device.tracks[play.bus.index()];
        let playing = match sound {
            Sound::Held(data) => Playing::Held(on_track(bus, track.as_mut(), held(data, play))?),
            Sound::Streamed(bytes) => {
                Playing::Streamed(on_track(bus, track.as_mut(), streamed(bytes, play)?)?)
            }
            Sound::Failed(failure) => return Err(failure.clone()),
        };
        let radius = play.radius.unwrap_or_default();
        self.voices.insert(
            voice,
            Voice {
                playing,
                track,
                radius,
            },
        );
        Ok(())
    }

    pub(crate) fn stop(&mut self, voice: u32) {
        if let Some(mut voice) = self.voices.remove(&voice) {
            voice.playing.stop();
        }
    }

    pub(crate) fn finished(&self, voice: u32) -> bool {
        self.voices
            .get(&voice)
            .is_none_or(|voice| voice.playing.state() == PlaybackState::Stopped)
    }

    pub(crate) fn set_voice(&mut self, voice: u32, volume: f32, pitch: f32) {
        if let Some(voice) = self.voices.get_mut(&voice) {
            voice.playing.set(volume, pitch);
        }
    }

    pub(crate) fn place(&mut self, voice: u32, at: Option<Vec3>, eye: Vec3) {
        let Some(voice) = self.voices.get_mut(&voice) else {
            return;
        };
        let radius = voice.radius;
        let Some(track) = voice.track.as_mut() else {
            return;
        };
        let fade = at.map_or(0.0, |at| clamp(1.0 - at.distance(eye) / radius, 0.0, 1.0));
        if let Some(at) = at {
            track.set_position(at.to_array(), Tween::default());
        }
        track.set_volume(decibels(fade), Tween::default());
    }

    pub(crate) fn listen(&mut self, eye: Vec3, forward: Vec3, up: Vec3) {
        let Some(device) = self.device.as_mut() else {
            return;
        };
        let right = forward.cross(up).normalize_or_zero();
        let above = right.cross(forward);
        let turn = Quat::from_mat3(&Mat3::from_cols(right, above, -forward));
        device
            .listener
            .set_position(eye.to_array(), Tween::default());
        device
            .listener
            .set_orientation(turn.to_array(), Tween::default());
    }

    pub(crate) fn set_bus(&mut self, bus: Bus, volume: f32, paused: bool) {
        let Some(device) = self.device.as_mut() else {
            return;
        };
        let track = &mut device.tracks[bus.index()];
        match bus {
            Bus::Master => device
                .manager
                .main_track()
                .set_volume(decibels(volume), Tween::default()),
            Bus::Music | Bus::Effects | Bus::Ambient | Bus::Voice => {
                track.set_volume(decibels(volume), Tween::default());
            }
        }
        if paused {
            track.pause(Tween::default());
        } else {
            track.resume(Tween::default());
        }
    }

    pub(crate) fn tick(&mut self) {}
}
