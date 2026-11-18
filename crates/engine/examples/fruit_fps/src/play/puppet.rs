use std::collections::BTreeMap;
use std::sync::Arc;

use engine::prelude::{Component, Resource};
use engine_core::{Error, ErrorCode, SceneCx};
use glam::{Mat3, Quat, Vec3};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::play::events::arms::Weapon;
use crate::play::figures::{Figure, HandTarget};
use crate::weapons::{Loadout, Slot};

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BaseClip {
    pub clip: String,
    pub fade_seconds: f32,
    #[serde(default)]
    pub looping: Option<bool>,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LookTable {
    pub hips: String,
    pub limit_radians: f32,
    pub catch_up_radians_per_second: f32,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Casts {
    pub caster: OneShot,
    pub throw: OneShot,
    pub place: OneShot,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReloadHand {
    pub to: [f32; 3],
    pub share: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Aiming {
    pub seconds: f32,
    pub lower: f32,
    pub hand: [f32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DroppedLook {
    pub lift_m: f32,
    pub roll_radians: f32,
    pub turn_radians_per_id: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Switch {
    pub dip_m: f32,
    pub dip_radians: f32,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OneShot {
    pub clip: String,
    pub from: String,
    pub feather: u32,
    pub weight: f32,
    pub speed: f32,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Hits {
    pub head: String,
    pub front: String,
    pub back: String,
    pub side: String,
    pub low: String,
    pub from: String,
    pub feather: u32,
    pub weight: f32,
    pub speed: f32,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Death {
    pub from_front: String,
    pub from_back: String,
    pub from_side: String,
    pub speed: f32,
    pub fade_seconds: f32,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CrouchClips {
    pub from: f32,
    pub idle: BaseClip,
    pub walk: BaseClip,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CrouchTable {
    pub drop_m: f32,
    pub use_clips: bool,
    pub clips: CrouchClips,
}

impl CrouchTable {
    pub fn chosen(&self) -> Option<&CrouchClips> {
        self.use_clips.then_some(&self.clips)
    }
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plant {
    pub start: BaseClip,
    pub hold: BaseClip,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PinFeet {
    pub contact_m: f32,
    pub clips: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rate {
    pub min: f32,
    pub max: f32,
    pub stride_floor_mps: f32,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AimTable {
    pub chain: Vec<String>,
    pub share: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Axes {
    pub fingers: [f32; 3],
    pub palm: [f32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HandAxes {
    pub right: Axes,
    pub left: Axes,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Grip {
    pub fingers: [f32; 3],
    pub palm: [f32; 3],
    pub pole: [f32; 3],
    pub wrist_m: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HoldPose {
    pub pivot: [f32; 3],
    pub grip: [f32; 3],
    pub pitch_share: f32,
    pub right: Grip,
    #[serde(default)]
    pub left: Option<Grip>,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sway {
    pub metres_per_radian: f32,
    pub max_m: f32,
    pub follow_per_second: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bob {
    pub stride_m: f32,
    pub side_m: f32,
    pub up_m: f32,
    pub full_mps: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Kick {
    pub back_m: f32,
    pub up_radians: f32,
    pub return_per_second: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ViewPlace {
    pub rest: [f32; 3],
    pub scoped: [f32; 3],
    pub lowered: [f32; 3],
    pub lower_running: bool,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ViewTable {
    pub scale: f32,
    pub scope_seconds: f32,
    pub lower_seconds: f32,
    pub sway: Sway,
    pub bob: Bob,
    pub kick: Kick,
    pub fire: String,
    pub reload: String,
    #[serde(default)]
    pub equip: Option<String>,
    pub weapons: BTreeMap<String, ViewPlace>,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PuppetTable {
    pub aim: AimTable,
    pub crouch: CrouchTable,
    pub still_below_mps: f32,
    pub rate: Rate,
    pub idle: BaseClip,
    pub walk: BaseClip,
    pub run: BaseClip,
    pub turn: BaseClip,
    pub look: LookTable,
    pub rise: BaseClip,
    pub fall: BaseClip,
    pub hits: Hits,
    pub casts: Casts,
    pub melee: OneShot,
    pub victory: OneShot,
    pub death: Death,
    pub plant: Plant,
    pub defuse: BaseClip,
    pub land: OneShot,
    pub reload_hand: ReloadHand,
    pub switch: Switch,
    pub aiming: Aiming,
    pub dropped: DroppedLook,
    pub pin_feet: PinFeet,
    pub hold_blend_seconds: f32,
    pub hand_axes: HandAxes,
    pub holds: BTreeMap<String, HoldPose>,
    pub weapons: BTreeMap<String, String>,
    pub view: ViewTable,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WeaponFiles {
    pub figure: Figure,
    pub markers: String,
    pub mesh: String,
    pub material: String,
    pub held_mesh: String,
    pub held_material: String,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Markers {
    pub grip_r: Option<Vec3>,
    pub grip_l: Option<Vec3>,
    pub muzzle: Option<Vec3>,
    pub sight: Option<Vec3>,
    pub eject: Option<Vec3>,
    pub mag_socket: Option<Vec3>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WeaponLook {
    pub files: WeaponFiles,
    pub markers: Markers,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PuppetBook {
    pub table: PuppetTable,
    pub weapons: BTreeMap<String, WeaponLook>,
}

#[derive(Deserialize)]
struct MarkerEntry {
    node: String,
    position: [f32; 3],
}

#[derive(Deserialize)]
struct MarkerFile {
    markers: Vec<MarkerEntry>,
}

#[derive(Component, Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PuppetFile {
    pub file: String,
    pub weapons: BTreeMap<String, WeaponFiles>,
    #[serde(skip)]
    book: Option<Arc<PuppetBook>>,
}

fn text_of(cx: &SceneCx<'_>, uri: &str, at: &str) -> Result<Option<String>, Error> {
    let Some(bytes) = cx.buffer(uri, at)? else {
        return Ok(None);
    };
    String::from_utf8(bytes.to_vec())
        .map(Some)
        .map_err(|_| Error::new(ErrorCode::InvalidRecord, at.to_owned()).got(uri))
}

fn markers_of(text: &str, at: &str) -> Result<Markers, Error> {
    let file: MarkerFile = engine::bricks::parse_value_at(engine::bricks::parse_json(text)?, at)?;
    let find = |name: &str| {
        file.markers
            .iter()
            .find(|entry| entry.node == name)
            .map(|entry| Vec3::from_array(entry.position))
    };
    Ok(Markers {
        grip_r: find("grip_r"),
        grip_l: find("grip_l"),
        muzzle: find("muzzle"),
        sight: find("sight"),
        eject: find("eject"),
        mag_socket: find("mag_socket"),
    })
}

impl PuppetFile {
    pub fn files(&self) -> Vec<String> {
        let mut files = vec![self.file.clone()];
        for weapon in self.weapons.values() {
            files.extend(weapon.figure.files());
            files.push(weapon.markers.clone());
        }
        files
    }

    pub fn book(&self) -> Option<&Arc<PuppetBook>> {
        self.book.as_ref()
    }

    pub fn prepared(mut self, cx: &SceneCx<'_>, path: &str) -> Result<Self, Error> {
        let at = format!("{path}.file");
        let Some(text) = text_of(cx, &self.file, &at)? else {
            return Ok(self);
        };
        let whole = engine::bricks::parse_json(&text).map_err(|error| error.within(&at))?;
        let table: PuppetTable =
            engine::bricks::parse_value_at(whole, "$").map_err(|error| error.within(&at))?;
        let mut weapons = BTreeMap::new();
        for (key, files) in std::mem::take(&mut self.weapons) {
            let figure_at = format!("{path}.weapons.{key}.figure");
            let figure = files.figure.clone().present(cx, &figure_at)?;
            let markers_at = format!("{path}.weapons.{key}.markers");
            let markers = match text_of(cx, &files.markers, &markers_at)? {
                Some(text) => markers_of(&text, &markers_at)?,
                None => Markers::default(),
            };
            let files = WeaponFiles { figure, ..files };
            self.weapons.insert(key.clone(), files.clone());
            weapons.insert(key, WeaponLook { files, markers });
        }
        self.book = Some(Arc::new(PuppetBook { table, weapons }));
        Ok(self)
    }
}

impl Axes {
    pub fn frame(&self) -> Mat3 {
        frame(Vec3::from_array(self.fingers), Vec3::from_array(self.palm))
    }
}

impl Grip {
    pub fn frame(&self) -> Mat3 {
        frame(Vec3::from_array(self.fingers), Vec3::from_array(self.palm))
    }

    pub fn hand(&self, axes: &Axes) -> Quat {
        Quat::from_mat3(&(self.frame() * axes.frame().transpose())).normalize()
    }

    pub fn wrist(&self, grip: Vec3) -> Vec3 {
        grip - self.frame().y_axis * self.wrist_m
    }
}

fn frame(fingers: Vec3, palm: Vec3) -> Mat3 {
    let y = fingers.normalize_or(Vec3::Y);
    let x = (palm - y * palm.dot(y)).normalize_or(y.any_orthonormal_vector());
    Mat3::from_cols(x, y, x.cross(y))
}

pub fn weapon_key(loadout: &Loadout) -> &'static str {
    match (loadout.active, loadout.active_gun()) {
        (Slot::Primary | Slot::Sidearm, Some(gun)) => Weapon::Gun(gun.kind()).key(),
        (Slot::Melee, _) | (Slot::Primary | Slot::Sidearm, None) => Weapon::Melee.key(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActorMuzzle {
    pub slot: u32,
    pub weapon: &'static str,
    pub position: Vec3,
    pub direction: Vec3,
}

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct Muzzles {
    pub tick: u64,
    pub actors: Vec<ActorMuzzle>,
}

impl Muzzles {
    pub fn of(&self, slot: u32) -> Option<&ActorMuzzle> {
        self.actors.iter().find(|muzzle| muzzle.slot == slot)
    }
}

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct ViewModel {
    pub tick: u64,
    pub slot: Option<u32>,
    pub weapon: Option<&'static str>,
    pub clip: String,
    pub muzzle: Option<Vec3>,
    pub direction: Vec3,
    pub sight: Vec3,
    pub hands: [Option<HandTarget>; 2],
    pub scale: f32,
}
