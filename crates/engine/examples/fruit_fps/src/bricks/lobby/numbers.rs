use std::collections::BTreeMap;

use serde::Deserialize;

use crate::play::lobby::LobbyCamera;

#[derive(Deserialize)]
struct LobbyFile {
    lobby: LobbyNumbers,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Place {
    pub(super) name: String,
    pub(super) at: [f32; 3],
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Pick {
    pub(super) clip: String,
    #[serde(default)]
    pub(super) window: Option<[f32; 2]>,
    pub(super) weight: f32,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Clips {
    pub(super) idle: String,
    pub(super) pick: Pick,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PickLayer {
    pub(super) from: String,
    pub(super) fade_seconds: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Timings {
    pub(super) idle_fade_seconds: f32,
    pub(super) warm_seconds: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Checks {
    pub(super) floor_probe_m: f32,
    pub(super) sight_height_m: f32,
    pub(super) frame_aspect: f32,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Member {
    pub(super) character: String,
    pub(super) at: [f32; 3],
    pub(super) turn_degrees: f32,
    pub(super) speed: f32,
    pub(super) lead_seconds: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Spot {
    pub(super) at: [f32; 3],
    pub(super) turn_degrees: f32,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Title {
    pub(super) camera: LobbyCamera,
    pub(super) group: Vec<Member>,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Select {
    pub(super) camera: LobbyCamera,
    pub(super) front: Spot,
    pub(super) back: Vec<Spot>,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct LobbyNumbers {
    pub(super) place: Place,
    pub(super) clips: BTreeMap<String, Clips>,
    pub(super) pick: PickLayer,
    pub(super) timings: Timings,
    pub(super) checks: Checks,
    pub(super) title: Title,
    pub(super) select: Select,
}

fn at_least(value: f32, least: f32, path: &str) -> Result<(), String> {
    if value.is_finite() && value >= least {
        Ok(())
    } else {
        Err(format!("numbers.json {path}: {value} is below {least}"))
    }
}

fn camera_checked(camera: &LobbyCamera, path: &str) -> Result<(), String> {
    let fov = camera.fov_degrees;
    if !(fov.is_finite() && fov > 1.0 && fov < 170.0) {
        return Err(format!(
            "numbers.json {path}.fov_degrees: {fov} is outside 1..170"
        ));
    }
    if camera.position == camera.target {
        return Err(format!(
            "numbers.json {path}: the camera stands on its target"
        ));
    }
    at_least(
        camera.drift.period_seconds,
        0.0,
        &format!("{path}.drift.period_seconds"),
    )
}

fn pick_checked(pick: &Pick, path: &str) -> Result<(), String> {
    at_least(pick.weight, 0.0, &format!("{path}.weight"))?;
    if pick.weight > 1.0 {
        return Err(format!(
            "numbers.json {path}.weight: {} is above 1",
            pick.weight
        ));
    }
    if let Some([start, end]) = pick.window {
        at_least(start, 0.0, &format!("{path}.window[0]"))?;
        at_least(end, start, &format!("{path}.window[1]"))?;
    }
    Ok(())
}

fn checked(lobby: LobbyNumbers) -> Result<LobbyNumbers, String> {
    let timings = lobby.timings;
    for (value, key) in [
        (timings.idle_fade_seconds, "timings.idle_fade_seconds"),
        (timings.warm_seconds, "timings.warm_seconds"),
        (lobby.pick.fade_seconds, "pick.fade_seconds"),
    ] {
        at_least(value, 0.0, &format!("$.lobby.{key}"))?;
    }
    for (character, clips) in &lobby.clips {
        pick_checked(&clips.pick, &format!("$.lobby.clips.{character}.pick"))?;
    }
    at_least(
        lobby.checks.floor_probe_m,
        0.0,
        "$.lobby.checks.floor_probe_m",
    )?;
    at_least(
        lobby.checks.frame_aspect,
        0.1,
        "$.lobby.checks.frame_aspect",
    )?;
    camera_checked(&lobby.title.camera, "$.lobby.title.camera")?;
    camera_checked(&lobby.select.camera, "$.lobby.select.camera")?;
    for (at, member) in lobby.title.group.iter().enumerate() {
        at_least(
            member.speed,
            0.01,
            &format!("$.lobby.title.group[{at}].speed"),
        )?;
        at_least(
            member.lead_seconds,
            0.0,
            &format!("$.lobby.title.group[{at}].lead_seconds"),
        )?;
    }
    Ok(lobby)
}

pub(super) fn read(text: &str) -> Result<LobbyNumbers, String> {
    let file: LobbyFile =
        serde_json::from_str(text).map_err(|error| format!("numbers.json lobby: {error}"))?;
    checked(file.lobby)
}
