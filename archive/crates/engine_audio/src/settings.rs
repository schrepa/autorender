use engine_core::{settings, stats};
use engine_places_rendering::audio::Bus;

pub(crate) const MOST_VOICES: u32 = 512;

settings! {
    #[section("audio")]
    #[switch(on)]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneAudio")]
    pub struct Audio {
        pub master: f32 = 1.0 => range(0.0, 1.0),
        pub music: f32 = 1.0 => range(0.0, 1.0),
        pub effects: f32 = 1.0 => range(0.0, 1.0),
        pub ambient: f32 = 1.0 => range(0.0, 1.0),
        pub voice: f32 = 1.0 => range(0.0, 1.0),
        pub max_voices: u32 = 64 => range(1, MOST_VOICES),
    }
}

impl Audio {
    pub(crate) fn volume_of(&self, bus: Bus) -> f32 {
        match bus {
            Bus::Master => self.master,
            Bus::Music => self.music,
            Bus::Effects => self.effects,
            Bus::Ambient => self.ambient,
            Bus::Voice => self.voice,
        }
    }
}

stats! {
    #[section("audio_stats")]
    pub struct AudioStats {
        pub device: bool,
        pub sounds_loaded: u32,
        pub voices: u32,
        pub voices_cut: u32,
        pub decode_failures: u32,
        pub latest_failure: Option<String>,
    }
}
