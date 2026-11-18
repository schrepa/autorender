use std::collections::BTreeMap;

use serde::Deserialize;

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Curve {
    pub flat: bool,
    pub near_m: f32,
    pub far_m: f32,
    pub floor: f32,
    pub power: f32,
    pub occluded: f32,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SoundFile {
    pub file: String,
    pub seconds: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Group {
    Effects,
    Interface,
    Footsteps,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SoundEntry {
    pub files: Vec<SoundFile>,
    pub loops: bool,
    pub gain: f32,
    pub jitter: f32,
    pub priority: u32,
    pub curve: String,
    pub own_gain: f32,
    pub group: Group,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct DeviceTiming {
    pub plant_repeat_seconds: f32,
    pub defuse_repeat_seconds: f32,
    pub beep_slow_seconds: f32,
    pub beep_fast_seconds: f32,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Table {
    pub folder: String,
    pub extension: String,
    pub seed: u64,
    pub voices: usize,
    pub master_gain: f32,
    pub latency_seconds: Option<f32>,
    pub glide_seconds: f32,
    pub steal_fade_seconds: f32,
    pub hold_fade_seconds: f32,
    pub audible_gain: f32,
    pub tune_step: f32,
    pub occlusion_every_ticks: u32,
    pub occlusion_pullback_m: f32,
    pub pan_width: f32,
    pub quiet_step_gain: f32,
    pub countdown_seconds: u32,
    pub device: DeviceTiming,
    pub curves: BTreeMap<String, Curve>,
    pub sounds: BTreeMap<String, SoundEntry>,
    pub events: BTreeMap<String, String>,
}

impl Table {
    pub fn parse(bytes: &[u8]) -> Result<Table, String> {
        let table: Table = serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
        if table.voices == 0 {
            return Err("voices: 0".to_owned());
        }
        if table.occlusion_every_ticks == 0 {
            return Err("occlusion_every_ticks: 0".to_owned());
        }
        for (event, sound) in &table.events {
            if !table.sounds.contains_key(sound) {
                return Err(format!("events.{event}: no sound {sound}"));
            }
        }
        for (name, sound) in &table.sounds {
            if sound.files.is_empty() {
                return Err(format!("sounds.{name}: no files"));
            }
            if !table.curves.contains_key(&sound.curve) {
                return Err(format!("sounds.{name}: no curve {}", sound.curve));
            }
        }
        for (name, curve) in &table.curves {
            if !curve.flat && curve.far_m <= curve.near_m {
                return Err(format!("curves.{name}: far_m must be above near_m"));
            }
        }
        Ok(table)
    }

    pub fn path(&self, file: &str) -> String {
        format!("{}{file}{}", self.folder, self.extension)
    }
}
