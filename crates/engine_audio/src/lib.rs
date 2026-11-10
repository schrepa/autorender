#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

#[cfg(target_arch = "wasm32")]
mod browser;
mod mixer;
#[cfg(not(target_arch = "wasm32"))]
mod native;
mod scene;
mod settings;
mod views;

#[cfg(target_arch = "wasm32")]
use browser as backend;
#[cfg(not(target_arch = "wasm32"))]
use native as backend;

pub use mixer::{SOUND_PART, SoundEnded, SoundEndedRow, SoundEnds};
pub use settings::{Audio, AudioStats};

use engine_core::{Brick, Bricks, ComponentKind, SceneComponents, Schedule, Source};
use engine_places_rendering::audio::{SOUND_LOADING, SoundSource, Sounds};

pub struct AudioBrick;

impl Brick for AudioBrick {
    const SOURCE: &'static Source = &SOURCE;
    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .switch::<Audio>()
            .stats::<AudioStats>()
            .errors(&[mixer::AUDIO_FAILED, SOUND_LOADING])
            .resource::<mixer::Mixer>()
            .resource::<mixer::SoundEnds>()
            .event::<mixer::SoundEnded, mixer::SoundEnds>()
            .fill::<Sounds>(None)
            .join::<SceneComponents>(
                ComponentKind::of::<SoundSource>("sound")
                    .check(scene::check_sound)
                    .files(scene::sound_files)
                    .prepare(scene::prepare_sound)
                    .schema(scene::sound_schema),
                None,
            )
            .systems(Schedule::Presentation, mixer::mix)
            .on_remove::<SoundSource, _>(mixer::release)
            .operation::<views::AudioDevice>()
            .observation::<views::AudioVoices>();
    }
}
