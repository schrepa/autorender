use std::collections::BTreeMap;
use std::sync::Arc;

use engine::{Error, ErrorCode, Expected};
use engine_core::SoundId;
use serde::{Deserialize, Deserializer};

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(from = "String")]
pub(super) struct Named {
    pub name: String,
    pub at: usize,
}

impl From<String> for Named {
    fn from(name: String) -> Self {
        Self { name, at: 0 }
    }
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Curve {
    pub flat: bool,
    pub near_m: f32,
    #[serde(default)]
    pub far_m: f32,
    #[serde(default)]
    pub far_from: Option<String>,
    pub floor: f32,
    pub power: f32,
    pub occluded: f32,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SoundFile {
    pub file: String,
    pub seconds: f32,
    #[serde(skip)]
    pub sound: Option<SoundId>,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SoundEntry {
    pub files: Vec<SoundFile>,
    pub loops: bool,
    pub gain: f32,
    pub jitter: f32,
    pub priority: u32,
    pub curve: Named,
    pub own_gain: Option<f32>,
    pub group: Named,
    #[serde(default)]
    pub gain_jitter_db: f32,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Clash {
    pub winner: Named,
    pub loser: Named,
    pub seconds: f32,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Table {
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
    pub clashes: Vec<Clash>,
    #[serde(deserialize_with = "listed")]
    pub curves: Vec<(String, Curve)>,
    #[serde(deserialize_with = "listed")]
    pub sounds: Vec<(String, SoundEntry)>,
    pub events: BTreeMap<String, Named>,
    #[serde(skip)]
    pub groups: Vec<String>,
}

fn listed<'de, D: Deserializer<'de>, T: Deserialize<'de>>(
    deserializer: D,
) -> Result<Vec<(String, T)>, D::Error> {
    BTreeMap::<String, T>::deserialize(deserializer).map(|listed| listed.into_iter().collect())
}

fn position<T>(listed: &[(String, T)], name: &str) -> Option<usize> {
    listed
        .binary_search_by(|(listed, _)| listed.as_str().cmp(name))
        .ok()
}

impl Table {
    pub fn read(bytes: Arc<[u8]>, at: &str) -> Result<Table, Error> {
        let text = std::str::from_utf8(&bytes).map_err(|_| {
            Error::new(ErrorCode::InvalidRecord, at.to_owned()).expected(Expected::Utf8)
        })?;
        let mut table: Table =
            engine::bricks::parse_json(text).map_err(|error| error.within(at))?;
        table.resolve().map_err(|problem| {
            Error::new(ErrorCode::InvalidValue, at.to_owned()).got(problem)
        })?;
        Ok(table)
    }

    fn resolve(&mut self) -> Result<(), String> {
        if self.voices == 0 {
            return Err("voices: 0".to_owned());
        }
        if self.occlusion_every_ticks == 0 {
            return Err("occlusion_every_ticks must be above 0".to_owned());
        }
        for (event, sound) in &mut self.events {
            sound.at = position(&self.sounds, &sound.name)
                .ok_or_else(|| format!("events.{event}: no sound {}", sound.name))?;
        }
        for clash in &mut self.clashes {
            for named in [&mut clash.winner, &mut clash.loser] {
                named.at = position(&self.sounds, &named.name)
                    .ok_or_else(|| format!("clashes: no sound {}", named.name))?;
            }
        }
        for (name, sound) in &mut self.sounds {
            if sound.files.is_empty() {
                return Err(format!("sounds.{name}: no files"));
            }
            sound.curve.at = position(&self.curves, &sound.curve.name)
                .ok_or_else(|| format!("sounds.{name}: no curve {}", sound.curve.name))?;
            sound.group.at = match self.groups.iter().position(|group| *group == sound.group.name) {
                Some(at) => at,
                None => {
                    self.groups.push(sound.group.name.clone());
                    self.groups.len() - 1
                }
            };
        }
        for (name, curve) in &self.curves {
            if !curve.flat && curve.far_from.is_none() && curve.far_m <= curve.near_m {
                return Err(format!("curves.{name}: far_m must be above near_m"));
            }
        }
        Ok(())
    }

    pub fn reach(&mut self, reach: &BTreeMap<String, f32>) -> Result<(), String> {
        for (name, curve) in &mut self.curves {
            let Some(from) = &curve.far_from else {
                continue;
            };
            let far = *reach
                .get(from)
                .ok_or_else(|| format!("curves.{name}.far_from: nothing reaches {from}"))?;
            if far <= curve.near_m {
                return Err(format!("curves.{name}: {from} = {far} is not above near_m"));
            }
            curve.far_m = far;
        }
        Ok(())
    }
}
