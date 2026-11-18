use std::collections::BTreeMap;
use std::sync::Arc;

use engine::prelude::{Component, Entity, Resource};
use engine_core::{Error, ErrorCode, SceneCx};
use glam::{Mat3, Quat, Vec3};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::play::events::arms::{SwingEvent, Weapon};
use crate::play::figures::{ChestFrame, Figure, Form, HandTarget, Offset};
use crate::play::light::MapFilter;
use crate::rules::AbilitySlot;
use crate::weapons::{Loadout, Slot};

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BaseClip {
    pub clip: String,
    pub fade_seconds: f32,
    #[serde(default)]
    pub looping: Option<bool>,
    #[serde(default)]
    pub whole: bool,
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
pub struct CastClip {
    pub clip: String,
    pub event_seconds: f32,
    #[serde(default)]
    pub still_from: Option<String>,
    #[serde(default)]
    pub whole: bool,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Casts {
    pub clips: BTreeMap<String, CastClip>,
    pub suffix: BTreeMap<String, String>,
    pub from: String,
    pub feather: u32,
    pub weight: f32,
    pub fade_seconds: f32,
}

impl Casts {
    pub fn named(&self, cast: &CastClip, weapon: &str) -> Option<String> {
        Some(format!("{}{}", cast.clip, self.suffix.get(weapon)?))
    }

    pub fn clips(&self) -> Vec<String> {
        self.clips
            .values()
            .flat_map(|cast| {
                self.suffix
                    .values()
                    .map(move |suffix| format!("{}{suffix}", cast.clip))
            })
            .collect()
    }
}

pub fn swing_clip<'a>(
    (light, heavy): (&'a [String], &'a String),
    swing: &SwingEvent,
) -> Option<&'a String> {
    if swing.heavy {
        Some(heavy)
    } else {
        light.get(usize::from(swing.step) % light.len().max(1))
    }
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThrowClips {
    pub ready: BTreeMap<String, String>,
    pub hold: String,
    pub release: BTreeMap<String, String>,
    pub roll: BTreeMap<String, String>,
    pub rolled: Vec<String>,
    pub release_leave_seconds: f32,
    pub roll_leave_seconds: f32,
    pub hand: ThrowHand,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThrowHand {
    pub joint: String,
    pub along_m: f32,
    pub out_m: f32,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UpperTable {
    pub from: String,
    pub feather: u32,
    pub weight: f32,
    pub fade_seconds: f32,
    pub played_from: String,
    pub played: Vec<String>,
    pub fire: BTreeMap<String, String>,
    pub reload: BTreeMap<String, String>,
    pub throw: ThrowClips,
}

impl UpperTable {
    pub fn clips(&self) -> impl Iterator<Item = &str> {
        let throw = &self.throw;
        self.fire
            .values()
            .chain(self.reload.values())
            .chain(throw.ready.values())
            .chain(throw.release.values())
            .chain(throw.roll.values())
            .chain([&throw.hold])
            .map(String::as_str)
    }
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeleeClips {
    pub light: Vec<String>,
    pub heavy: String,
    pub from: String,
    pub feather: u32,
    pub weight: f32,
    pub speed: f32,
    pub fade_seconds: f32,
}

impl MeleeClips {
    pub fn clips(&self) -> impl Iterator<Item = &str> {
        self.light.iter().chain([&self.heavy]).map(String::as_str)
    }
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DrawClips {
    pub clips: BTreeMap<String, String>,
    pub from: String,
    pub feather: u32,
    pub weight: f32,
    pub fade_seconds: f32,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceClips {
    pub kneel: BaseClip,
    pub plant: BaseClip,
    pub defuse: BaseClip,
    pub stand: BaseClip,
}

impl DeviceClips {
    pub fn clips(&self) -> impl Iterator<Item = &str> {
        [&self.kneel, &self.plant, &self.defuse, &self.stand]
            .into_iter()
            .map(|base| base.clip.as_str())
    }
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
    #[serde(default)]
    pub window: Option<[f32; 2]>,
    #[serde(default)]
    pub whole: bool,
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
    pub headshot: String,
    pub speed: f32,
    pub fade_seconds: f32,
    #[serde(default)]
    pub whole: bool,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CrouchClips {
    pub from: f32,
    pub idle: BaseClip,
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
pub struct PinFeet {
    pub contact_m: f32,
    pub clips: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Band {
    pub min: f32,
    pub max: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Heading {
    Forward,
    Left,
    Right,
    Back,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Gait {
    pub clip: String,
    pub planted_mps: f32,
    pub direction: Heading,
    pub from_mps: f32,
    pub to_mps: f32,
    pub crouch: bool,
    pub dropped: bool,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Locomotion {
    pub gaits: Vec<Gait>,
    pub speed_fade_mps: f32,
    pub fade_seconds: f32,
    pub hip_turn_limit_radians: f32,
    pub rate: Band,
    pub stride: Band,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ViewTest {
    pub frustum: bool,
    pub map: bool,
    pub mesh_margin_m: f32,
    pub map_pad_m: f32,
    pub aspect_max: f32,
    pub item_radius_m: f32,
    pub hold_frames: u64,
    pub cut_m: f32,
    pub see_through: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FormsTable {
    pub full_to_m: f32,
    pub mid_from_m: f32,
    pub mid_back_m: f32,
    pub far_from_m: f32,
    pub reference_fov_degrees: f32,
}

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct Seen {
    pub frame: u64,
    pub shown: Vec<(Entity, Form, f32)>,
    pub judged: Vec<Entity>,
    pub frustum: u32,
    pub map: u32,
    pub held: u32,
}

impl Seen {
    pub fn judges(&self, entity: Entity) -> bool {
        self.judged.binary_search(&entity).is_ok()
    }

    pub fn shows(&self, entity: Entity) -> bool {
        self.form(entity).is_some()
    }

    pub fn form(&self, entity: Entity) -> Option<Form> {
        self.at(entity).map(|at| self.shown[at].1)
    }

    pub fn distance(&self, entity: Entity) -> Option<f32> {
        self.at(entity).map(|at| self.shown[at].2)
    }

    fn at(&self, entity: Entity) -> Option<usize> {
        self.shown
            .binary_search_by_key(&entity, |(known, ..)| *known)
            .ok()
    }
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EyeFit {
    pub reference: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeadCheck {
    pub slice_m: f32,
    pub solid_share: f32,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AimTable {
    pub chain: Vec<String>,
    pub share: f32,
    #[serde(default)]
    pub chest: Option<ChestFrame>,
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
    #[serde(default)]
    pub hide_from_share: Option<f32>,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ViewMelee {
    pub light: Vec<String>,
    pub heavy: String,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ViewInspect {
    pub clip: String,
    pub cancel_fade_seconds: f32,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ViewDevice {
    pub plant: String,
    pub defuse: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Throw {
    pub seconds: f32,
    pub lift_m: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Shade {
    pub indirect_floor: f32,
    pub maps: ShadeMaps,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ShadeMaps {
    pub base_color: MapFilter,
    pub normal: MapFilter,
    pub orm: MapFilter,
}

impl ShadeMaps {
    pub fn checked(&self, path: &str) -> Result<(), Error> {
        [
            (self.base_color, "base_color"),
            (self.normal, "normal"),
            (self.orm, "orm"),
        ]
        .into_iter()
        .try_for_each(|(filter, key)| filter.checked(&format!("{path}.{key}")))
        .map_err(|problem| Error::new(ErrorCode::InvalidRecord, path).got(problem))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PropsLook {
    pub spin_radians_per_second: f32,
    pub trim: PropTrim,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PropTrim {
    pub ally: [f32; 3],
    pub enemy: [f32; 3],
    pub strength: f32,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InHandProp {
    pub key: String,
    pub leave_seconds: f32,
    pub size_m: f32,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HandCast {
    pub clip: String,
    pub speed: f32,
    pub fade_seconds: f32,
    #[serde(default)]
    pub prop: Option<InHandProp>,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HandsTable {
    pub rest: [f32; 3],
    pub socket: String,
    pub ready: String,
    pub hold: String,
    pub abilities: BTreeMap<String, HandCast>,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ViewTable {
    pub hands: HandsTable,
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
    pub melee: ViewMelee,
    pub inspect: ViewInspect,
    pub device: ViewDevice,
    pub weapons: BTreeMap<String, ViewPlace>,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PuppetTable {
    pub aim: AimTable,
    pub head_check: HeadCheck,
    #[serde(default)]
    pub eye_fit: Option<EyeFit>,
    pub view_test: ViewTest,
    pub forms: FormsTable,
    pub crouch: CrouchTable,
    pub still_below_mps: f32,
    pub locomotion: Locomotion,
    pub idle: BaseClip,
    pub turn: BaseClip,
    pub look: LookTable,
    pub rise: BaseClip,
    pub fall: BaseClip,
    pub hits: Hits,
    pub casts: Casts,
    pub upper: UpperTable,
    pub melee: MeleeClips,
    pub draw: DrawClips,
    pub victory: OneShot,
    pub death: Death,
    pub device: DeviceClips,
    pub land: OneShot,
    pub switch: Switch,
    pub aiming: Aiming,
    pub dropped: DroppedLook,
    pub throw: Throw,
    pub props: PropsLook,
    pub shade: Shade,
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
    pub light: Option<Vec3>,
    pub tip: Option<Vec3>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WeaponLook {
    pub files: WeaponFiles,
    pub markers: Markers,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PlantedFiles {
    pub mesh: String,
    pub material: String,
    pub markers: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlantedLook {
    pub files: PlantedFiles,
    pub markers: Markers,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct HandsFiles {
    pub figure: Figure,
    pub mesh: String,
    pub material: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PuppetBook {
    pub table: PuppetTable,
    pub weapons: BTreeMap<String, WeaponLook>,
    pub planted: Option<PlantedLook>,
    pub hands: Option<HandsFiles>,
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
    #[serde(default)]
    pub planted: Option<PlantedFiles>,
    #[serde(default)]
    pub hands: Option<HandsFiles>,
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
        light: find("light"),
        tip: find("tip"),
    })
}

impl PuppetFile {
    pub fn files(&self) -> Vec<String> {
        let mut files = vec![self.file.clone()];
        for weapon in self.weapons.values() {
            files.extend(weapon.figure.files());
            files.push(weapon.markers.clone());
        }
        files.extend(self.planted.iter().map(|planted| planted.markers.clone()));
        files.extend(self.hands.iter().flat_map(|hands| hands.figure.files()));
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
        table
            .shade
            .maps
            .checked("$.shade.maps")
            .map_err(|error| error.within(&at))?;
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
        let planted = match self.planted.clone() {
            Some(files) => {
                let at = format!("{path}.planted.markers");
                let markers = match text_of(cx, &files.markers, &at)? {
                    Some(text) => markers_of(&text, &at)?,
                    None => Markers::default(),
                };
                Some(PlantedLook { files, markers })
            }
            None => None,
        };
        let hands = match self.hands.clone() {
            Some(files) => {
                let figure = files.figure.present(cx, &format!("{path}.hands.figure"))?;
                Some(HandsFiles { figure, ..files })
            }
            None => None,
        };
        self.hands.clone_from(&hands);
        self.book = Some(Arc::new(PuppetBook {
            table,
            weapons,
            planted,
            hands,
        }));
        Ok(self)
    }
}

impl Axes {
    pub fn frame(&self) -> Mat3 {
        frame(Vec3::from_array(self.fingers), Vec3::from_array(self.palm))
    }

    pub fn seat(&self, hand: &ThrowHand) -> Offset {
        let palm = Vec3::from_array(self.palm).normalize_or(Vec3::Y);
        let fingers = Vec3::from_array(self.fingers);
        let along =
            (fingers - palm * fingers.dot(palm)).normalize_or(palm.any_orthonormal_vector());
        let turn = Quat::from_mat3(&Mat3::from_cols(palm.cross(along), palm, along)).normalize();
        Offset {
            translation: (along * hand.along_m + palm * hand.out_m).to_array(),
            rotation: turn.to_array(),
            scale: 1.0,
        }
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

pub const DEVICE: &str = "device";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Task {
    Planting,
    Defusing,
}

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct Tasks {
    pub tick: u64,
    pub actors: Vec<(u32, Task)>,
}

impl Tasks {
    pub fn of(&self, slot: u32) -> Option<Task> {
        self.actors
            .iter()
            .find(|(actor, _)| *actor == slot)
            .map(|(_, task)| *task)
    }
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct DeviceLamp {
    pub position: Option<Vec3>,
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

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct Holding {
    pub held: Option<InHand>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InHand {
    pub holder: Entity,
    pub socket: String,
    pub seat: Offset,
    pub prop: String,
    pub size_m: f32,
    pub owner: u32,
    pub team: usize,
    pub ability: usize,
    pub cast: Option<u64>,
    pub release: Option<u64>,
}

impl InHand {
    pub fn holds(&self, tick: u64) -> bool {
        self.release.is_none_or(|release| tick < release)
    }

    pub fn hides(&self, (owner, ability, born): (u32, usize, u64)) -> bool {
        self.owner == owner && self.ability == ability && self.cast.is_some_and(|cast| born >= cast)
    }
}

pub fn leave_tick(cast: u64, seconds: f32, delta: f32) -> u64 {
    let ticks = if delta > 0.0 {
        (seconds / delta).ceil() as u64
    } else {
        0
    };
    cast + ticks
}

pub fn ability_index(slot: AbilitySlot) -> usize {
    match slot {
        AbilitySlot::C => 0,
        AbilitySlot::Q => 1,
        AbilitySlot::E => 2,
        AbilitySlot::X => 3,
    }
}
