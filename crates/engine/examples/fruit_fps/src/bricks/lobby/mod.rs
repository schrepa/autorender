engine::brick!(lobby);

mod numbers;

use std::collections::BTreeMap;
use std::f32::consts::{PI, TAU};
use std::sync::Arc;

use engine::math;
use engine::prelude::*;
use engine::{Devices, MouseButton};
use engine_core::{Assets, Stats};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::play::figures::biped;
use crate::play::flow::{Flow, Stage};
use crate::play::lobby::{LobbyCamera, LobbyView};
use crate::play::numbers::Numbers;
use crate::play::puppet::Puppet;
use crate::play::skinned::{
    BaseLayer, ClipNo, Figure, FigureClock, FigureFit, FigureJoints, FigureLimbs, FigurePlay,
    JointNo, UpperLayer,
};
use crate::play::world::Arena;

use numbers::{Clips, LobbyNumbers};

const CAMERA_BLEND_SECONDS: f64 = 0.58;
const STAGE_SWAP_SECONDS: f64 = 0.56;
const CAROUSEL_STEP_SECONDS: f64 = 0.54;
const CAROUSEL_FAR_SECONDS: f64 = 0.68;
const CAROUSEL_OVERSHOOT: f32 = 1.05;
const CAROUSEL_DEPTH_M: f32 = 2.05;
const CAROUSEL_WIDTH_M: f32 = 1.05;
const CAROUSEL_LIFT_M: f32 = 0.035;
const UNSPIN_SECONDS: f64 = 0.48;
const DRAG_START_PX: f32 = 4.0;
const DRAG_RAD_PER_PX: f32 = 0.011;
const DRAG_DAMP_RATE: f32 = 7.5;

#[derive(Component, Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LobbyFigure {
    pub character: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct LobbyFigureStats {
    pub character: String,
    pub position: [f32; 3],
    pub facing_degrees: f32,
    pub clip: Option<String>,
    pub seconds: f32,
    pub pick: Option<String>,
    pub pick_seconds: Option<f32>,
    pub pick_weight: f32,
    pub shown: bool,
    pub in_sight: Option<bool>,
    pub feet_frame: Option<[f32; 2]>,
    pub head_frame_y: Option<f32>,
}

stats! {
    #[section("lobby")]
    pub struct LobbyStats {
        pub stage: String,
        pub camera: String,
        pub place: String,
        pub camera_position: Option<[f32; 3]>,
        pub figures: Vec<LobbyFigureStats>,
        pub refusal: Option<String>,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shot {
    Title,
    Select,
}

impl Shot {
    fn of(stage: Stage) -> Option<Shot> {
        match stage {
            Stage::Title => Some(Shot::Title),
            Stage::Select => Some(Shot::Select),
            Stage::Loading | Stage::Playing | Stage::Paused | Stage::MatchEnd => None,
        }
    }

    fn key(self) -> &'static str {
        match self {
            Shot::Title => "title",
            Shot::Select => "select",
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Cue {
    cue: u32,
    warm_until: f64,
    picking: bool,
    done_at: Option<f64>,
}

#[derive(Clone, Copy, Debug, Default)]
struct CarouselRing {
    from_angle: f32,
    to_angle: f32,
    started: f64,
    duration: f64,
    primed: bool,
}

impl CarouselRing {
    fn sample(&self, now: f64) -> (f32, f32, f32) {
        if !self.primed || self.duration <= 0.0 {
            return (self.to_angle, 0.0, 1.0);
        }
        let raw = math::clamp(((now - self.started) / self.duration) as f32, 0.0, 1.0);
        if raw >= 1.0 {
            return (self.to_angle, 0.0, 1.0);
        }
        let delta = self.to_angle - self.from_angle;
        let u = ease_out_back(raw, CAROUSEL_OVERSHOOT);
        let du = ease_out_back_deriv(raw, CAROUSEL_OVERSHOOT) / self.duration as f32;
        (self.from_angle + delta * u, delta * du, raw)
    }

    fn snap(&mut self, target: f32) {
        self.from_angle = target;
        self.to_angle = target;
        self.started = 0.0;
        self.duration = 0.0;
        self.primed = true;
    }

    fn turn_to(&mut self, target: f32, now: f64) -> f32 {
        if !self.primed {
            self.snap(target);
            return 0.0;
        }
        let (cur, _, _) = self.sample(now);
        let delta = shortest_delta(cur, target);
        if delta.abs() < 1e-4 {
            return 0.0;
        }
        self.from_angle = cur;
        self.to_angle = cur + delta;
        self.started = now;
        self.duration = if delta.abs() > 1.8 {
            CAROUSEL_FAR_SECONDS
        } else {
            CAROUSEL_STEP_SECONDS
        };
        delta
    }
}

#[derive(Clone, Copy, Debug)]
struct FigureMotion {
    last_pos: Vec3,
    last_yaw: f32,
    last_scale: f32,
    base_scale: Vec3,
    blend_from: Option<(Vec3, f32, f32)>,
    blend_started: f64,
    blend_duration: f64,
}

impl FigureMotion {
    fn settled(pos: Vec3, yaw: f32, scale: f32, base_scale: Vec3) -> Self {
        Self {
            last_pos: pos,
            last_yaw: yaw,
            last_scale: scale,
            base_scale,
            blend_from: None,
            blend_started: 0.0,
            blend_duration: 0.0,
        }
    }

    fn begin_blend(&mut self, now: f64, duration: f64) {
        self.blend_from = Some((self.last_pos, self.last_yaw, self.last_scale));
        self.blend_started = now;
        self.blend_duration = duration;
    }

    fn apply(&mut self, now: f64, target: (Vec3, f32, f32)) -> (Vec3, f32, f32) {
        let (to_pos, to_yaw, to_scale) = target;
        let out = if let Some((from_pos, from_yaw, from_scale)) = self.blend_from
            && self.blend_duration > 0.0
        {
            let raw = math::clamp(((now - self.blend_started) / self.blend_duration) as f32, 0.0, 1.0);
            if raw >= 1.0 {
                self.blend_from = None;
                (to_pos, to_yaw, to_scale)
            } else {
                let u = ease_out_quint(raw);
                let pos = from_pos.lerp(to_pos, u);
                let yaw = from_yaw + shortest_delta(from_yaw, to_yaw) * u;
                let scale = from_scale + (to_scale - from_scale) * u;
                (pos, yaw, scale)
            }
        } else {
            (to_pos, to_yaw, to_scale)
        };
        self.last_pos = out.0;
        self.last_yaw = out.1;
        self.last_scale = out.2;
        out
    }
}

fn shortest_delta(from: f32, to: f32) -> f32 {
    (to - from + PI).rem_euclid(TAU) - PI
}

fn ease_out_quint(t: f32) -> f32 {
    let k = 1.0 - math::clamp(t, 0.0, 1.0);
    1.0 - k * k * k * k * k
}

fn ease_out_back(t: f32, overshoot: f32) -> f32 {
    let x = math::clamp(t, 0.0, 1.0) - 1.0;
    1.0 + (overshoot + 1.0) * x * x * x + overshoot * x * x
}

fn ease_out_back_deriv(t: f32, overshoot: f32) -> f32 {
    let x = math::clamp(t, 0.0, 1.0) - 1.0;
    3.0 * (overshoot + 1.0) * x * x + 2.0 * overshoot * x
}

type Read = Result<Arc<LobbyNumbers>, String>;

#[derive(Resource, Default)]
struct LobbyBook {
    numbers: Option<(u64, Read, Vec<String>)>,
    shot: Option<Shot>,
    chosen: String,
    hero: Option<Entity>,
    ring: CarouselRing,
    user_yaw: f32,
    user_yaw_vel: f32,
    unspin: Option<(Entity, f32, f64)>,
    drag_px: f32,
    drag_x: Option<f32>,
    last_frame: f64,
    cues: BTreeMap<Entity, Cue>,
    motions: BTreeMap<Entity, FigureMotion>,
    bound: BTreeMap<Entity, Bound>,
}

struct Bound {
    lobby: Arc<LobbyNumbers>,
    picked: Picked,
    missing: Vec<String>,
}

#[derive(Clone, Copy)]
struct Picked {
    idle: Option<ClipNo>,
    pick: Option<ClipNo>,
    from: Option<JointNo>,
}

fn bound(
    book: &mut LobbyBook,
    (entity, figure, assets): (Entity, &Ref<Figure>, &Assets),
    (lobby, clips, character): (&Arc<LobbyNumbers>, &Clips, &str),
    problems: &mut Vec<String>,
) -> Picked {
    let fresh = (book.bound.get(&entity)).is_some_and(|held| {
        Arc::ptr_eq(&held.lobby, lobby) && held.picked.from.is_some() && !figure.is_changed()
    });
    if !fresh {
        let mut missing = Vec::new();
        clips_missing(clips, (character, figure), &mut missing);
        let skeleton = assets.get(figure.skeleton).ok();
        let picked = Picked {
            idle: figure.clip(&clips.idle),
            pick: figure.clip(&clips.pick.clip),
            from: skeleton.and_then(|skeleton| skeleton.find(&lobby.pick.from)).map(JointNo),
        };
        let made = Bound {
            lobby: Arc::clone(lobby),
            picked,
            missing,
        };
        book.bound.insert(entity, made);
    }
    let held = &book.bound[&entity];
    problems.extend(held.missing.iter().cloned());
    held.picked
}

pub struct LobbyBrick;

impl Brick for LobbyBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .join::<SceneComponents>(
                ComponentKind::of::<LobbyFigure>("lobby_figure")
                    .schema(engine_core::schema_of::<LobbyFigure>),
            )
            .moves::<LobbyFigure>(Pace::Frame)
            .resource::<LobbyView>()
            .resource::<LobbyBook>()
            .dev(|dev| {
                dev.stats::<LobbyStats>();
            })
            .systems(Schedule::Presentation, stage);
    }
}

fn roster_of(whole: &serde_json::Value) -> Vec<String> {
    whole
        .get("characters")
        .and_then(serde_json::Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.get("key").and_then(serde_json::Value::as_str))
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn numbers_of(book: &mut LobbyBook, numbers: &Numbers) -> Option<(Read, Vec<String>)> {
    let game = numbers.get()?;
    let fresh = book
        .numbers
        .as_ref()
        .is_none_or(|(revision, ..)| *revision != numbers.revision());
    if fresh {
        let read = numbers::read(&game.whole).map(Arc::new);
        let roster = roster_of(&game.whole);
        book.numbers = Some((numbers.revision(), read, roster));
    }
    book.numbers
        .as_ref()
        .map(|(_, read, roster)| (read.clone(), roster.clone()))
}

struct Wanted {
    at: Vec3,
    turn_degrees: f32,
    scale: f32,
    yaw_offset: f32,
    speed: f32,
    lead_seconds: f32,
    path: String,
}

type Listed = [(Entity, String)];

fn entity_of(listed: &Listed, character: &str) -> Option<Entity> {
    listed
        .iter()
        .find(|(_, known)| known == character)
        .map(|(entity, _)| *entity)
}

fn carousel_order<'a>(
    lobby: &'a LobbyNumbers,
    roster: &'a [String],
    listed: &'a Listed,
) -> Vec<&'a str> {
    let mut order: Vec<&str> = roster.iter().map(String::as_str).collect();
    for member in &lobby.title.group {
        if !order.contains(&member.character.as_str()) {
            order.push(member.character.as_str());
        }
    }
    for (_, character) in listed {
        if !order.contains(&character.as_str()) {
            order.push(character.as_str());
        }
    }
    order
}

fn ring_target_angle(order: &[&str], chosen: &str) -> f32 {
    let count = order.len().max(1);
    let slot = order.iter().position(|known| *known == chosen).unwrap_or(0);
    -(slot as f32) * (TAU / count as f32)
}

fn title_plan(
    lobby: &LobbyNumbers,
    listed: &Listed,
    problems: &mut Vec<String>,
) -> Vec<(Entity, Wanted)> {
    let anchor = Vec3::from_array(lobby.place.at);
    let mut planned = Vec::new();
    for (at, member) in lobby.title.group.iter().enumerate() {
        let path = format!("$.lobby.title.group[{at}]");
        let Some(entity) = entity_of(listed, &member.character) else {
            problems.push(format!(
                "{path}.character {:?}: no scene entity carries lobby_figure {{character {:?}}}",
                member.character, member.character
            ));
            continue;
        };
        planned.push((
            entity,
            Wanted {
                at: anchor + Vec3::from_array(member.at),
                turn_degrees: member.turn_degrees,
                scale: 1.0,
                yaw_offset: 0.0,
                speed: member.speed,
                lead_seconds: member.lead_seconds,
                path,
            },
        ));
    }
    planned
}

fn select_plan(
    lobby: &LobbyNumbers,
    (listed, order, chosen): (&Listed, &[&str], &str),
    (ring_angle, ring_vel, ring_raw): (f32, f32, f32),
    problems: &mut Vec<String>,
) -> Vec<(Entity, Wanted)> {
    let anchor = Vec3::from_array(lobby.place.at);
    let front = anchor + Vec3::from_array(lobby.select.front.at);
    let cam_pos = anchor + Vec3::from_array(lobby.select.camera.position);
    let axis = (front - cam_pos)
        .with_y(0.0)
        .try_normalize()
        .unwrap_or(Vec3::NEG_Z);
    let lat = Vec3::new(-axis.z, 0.0, axis.x);
    let center = front + axis * CAROUSEL_DEPTH_M;
    let group = &lobby.title.group;
    let speed = |character: &str| {
        group
            .iter()
            .find(|member| member.character == character)
            .map_or(1.0, |member| member.speed)
    };
    if entity_of(listed, chosen).is_none() {
        problems.push(format!(
            "flow.character {chosen:?}: no scene entity carries lobby_figure {{character {chosen:?}}}"
        ));
    }
    let count = order.len().max(1);
    let step = TAU / count as f32;
    let motion_bank = math::clamp(ring_vel * 0.065, -0.26, 0.26);
    let travel_arc = math::sin(math::clamp(ring_raw, 0.0, 1.0) * PI);
    let arrive_pop = if ring_raw < 1.0 {
        math::sin(math::clamp(ring_raw, 0.0, 1.0) * PI) * 0.035
    } else {
        0.0
    };
    let mut planned = Vec::new();
    for (slot, character) in order.iter().enumerate() {
        let Some(entity) = entity_of(listed, character) else {
            continue;
        };
        let theta = slot as f32 * step + ring_angle;
        let (sin, cos) = math::sin_cos(theta);
        let is_chosen = *character == chosen;
        let lift = if is_chosen { CAROUSEL_LIFT_M * travel_arc } else { 0.0 };
        let at = center - axis * (CAROUSEL_DEPTH_M * cos) + lat * (CAROUSEL_WIDTH_M * sin)
            + Vec3::Y * lift;
        let frontness = (cos + 1.0) * 0.5;
        let scale = 0.88 + 0.12 * frontness + if is_chosen { arrive_pop } else { 0.0 };
        let path = if is_chosen {
            "$.lobby.select.front".to_owned()
        } else {
            format!("$.lobby.select.carousel[{slot}]")
        };
        planned.push((
            entity,
            Wanted {
                at,
                turn_degrees: lobby.select.front.turn_degrees,
                scale,
                yaw_offset: motion_bank,
                speed: speed(character),
                lead_seconds: 0.0,
                path,
            },
        ));
    }
    planned
}

fn update_turntable(
    book: &mut LobbyBook,
    devices: &Devices,
    shot: Shot,
    settings_open: bool,
    now: f64,
) {
    let dt = if book.last_frame > 0.0 {
        math::clamp((now - book.last_frame) as f32, 0.0, 0.1)
    } else {
        0.0
    };
    book.last_frame = now;
    if shot != Shot::Select || settings_open {
        book.drag_px = 0.0;
        book.drag_x = None;
        book.user_yaw_vel = 0.0;
        if shot != Shot::Select {
            book.user_yaw = 0.0;
        }
        return;
    }
    if devices.mouse.buttons.held(MouseButton::Left) {
        let dx = if let Some(pos) = devices.mouse.position_pixels {
            let delta = book.drag_x.map_or(0.0, |prev| pos.x - prev);
            book.drag_x = Some(pos.x);
            delta
        } else {
            book.drag_x = None;
            devices.mouse.motion_pixels.x
        };
        book.drag_px += dx.abs();
        if book.drag_px >= DRAG_START_PX && dx != 0.0 {
            let delta = dx * DRAG_RAD_PER_PX;
            book.user_yaw += delta;
            if dt > 0.001 {
                book.user_yaw_vel = delta / dt;
            }
        }
    } else {
        book.drag_px = 0.0;
        book.drag_x = None;
        if book.user_yaw_vel.abs() > 0.001 {
            book.user_yaw += book.user_yaw_vel * dt;
            book.user_yaw_vel *= (-DRAG_DAMP_RATE * dt).exp();
        } else {
            book.user_yaw_vel = 0.0;
        }
    }
}

fn cue_up(
    book: &mut LobbyBook,
    (shot, chosen, hero): (Option<Shot>, &str, Option<Entity>),
    (lobby, now, real_now): (&LobbyNumbers, f64, f64),
    (listed, order): (&Listed, &[&str]),
) -> (bool, bool, bool, f32) {
    let prev_shot = book.shot;
    let prev_hero = book.hero;
    let entering = prev_shot != shot;
    let from_hidden = prev_shot.is_none() && shot.is_some();
    book.shot = shot;
    let changed = book.chosen != chosen;
    let target_angle = ring_target_angle(order, chosen);
    let mut ring_delta = 0.0;
    if entering && shot == Some(Shot::Select) {
        book.ring.snap(target_angle);
        book.user_yaw = 0.0;
        book.user_yaw_vel = 0.0;
        book.unspin = None;
    } else if changed {
        if shot == Some(Shot::Select) {
            ring_delta = book.ring.turn_to(target_angle, real_now);
        }
        if let Some(old_hero) = prev_hero
            && book.user_yaw.abs() > 1e-3
        {
            book.unspin = Some((old_hero, shortest_delta(0.0, book.user_yaw), real_now));
        }
        book.user_yaw = 0.0;
        book.user_yaw_vel = 0.0;
    } else if entering {
        book.user_yaw = 0.0;
        book.user_yaw_vel = 0.0;
        book.unspin = None;
    }
    if changed {
        chosen.clone_into(&mut book.chosen);
    }
    book.hero = hero;
    book.cues
        .retain(|entity, _| listed.iter().any(|(known, _)| known == entity));
    book.motions
        .retain(|entity, _| listed.iter().any(|(known, _)| known == entity));
    book.bound
        .retain(|entity, _| listed.iter().any(|(known, _)| known == entity));
    let timings = lobby.timings;
    for (entity, _) in listed {
        let cue = book.cues.entry(*entity).or_default();
        if from_hidden {
            cue.warm_until = now + f64::from(timings.warm_seconds);
        }
        if shot == Some(Shot::Select) && (entering || changed) && hero == Some(*entity) {
            cue.cue += 1;
            cue.picking = true;
            cue.done_at = None;
        }
        if hero != Some(*entity) {
            cue.picking = false;
        }
    }
    if entering && !from_hidden {
        for motion in book.motions.values_mut() {
            motion.begin_blend(real_now, STAGE_SWAP_SECONDS);
        }
    }
    (entering, from_hidden, changed, ring_delta)
}

fn picked(
    lobby: &LobbyNumbers,
    (clips, bound, clock): (&Clips, Picked, Option<&FigureClock>),
    cue: &mut Cue,
    now: f64,
) -> Option<UpperLayer> {
    if !cue.picking {
        return None;
    }
    let ended = clock.is_some_and(|clock| clock.upper_cue == Some(cue.cue) && clock.upper_done);
    if ended && cue.done_at.is_none() {
        cue.done_at = Some(now);
    }
    let fade = f64::from(lobby.pick.fade_seconds);
    let share = match cue.done_at {
        Some(at) if fade > 0.0 => 1.0 - (now - at) / fade,
        Some(_) => 0.0,
        None => 1.0,
    };
    if share <= 0.0 {
        cue.picking = false;
        return None;
    }
    Some(UpperLayer {
        clip: bound.pick?,
        weight: clips.pick.weight * share as f32,
        looping: false,
        from: bound.from?,
        feather: 0,
        speed: 1.0,
        cue: cue.cue,
        window: clips.pick.window,
        fit_seconds: None,
        additive: false,
        played_from: None,
        fade_seconds: 0.0,
    })
}

fn played(
    lobby: &LobbyNumbers,
    (wanted, clips, bound, clock): (&Wanted, &Clips, Picked, Option<&FigureClock>),
    cue: &mut Cue,
    now: f64,
) -> (Option<FigurePlay>, bool) {
    let timings = lobby.timings;
    let warming = now < cue.warm_until;
    let speed = if warming && timings.warm_seconds > 0.0 {
        wanted.speed + wanted.lead_seconds / timings.warm_seconds
    } else {
        wanted.speed
    };
    let play = bound.idle.map(|idle| FigurePlay {
        base: BaseLayer {
            clip: idle,
            speed,
            looping: true,
            fade_seconds: timings.idle_fade_seconds,
            cue: 0,
            fit_seconds: None,
            blend: Vec::new(),
            stride: None,
            synced: false,
        },
        upper: picked(lobby, (clips, bound, clock), cue, now),
        aim: None,
        hold: None,
        crouch: None,
        pin_feet: None,
    });
    (play, !warming)
}

fn framed(shot: (Vec3, Vec3, f32), aspect: f32, point: Vec3) -> Option<[f32; 2]> {
    let (position, target, fov_degrees) = shot;
    let forward = (target - position).try_normalize()?;
    let right = forward.cross(Vec3::Y).try_normalize()?;
    let up = right.cross(forward);
    let toward = point - position;
    let depth = toward.dot(forward);
    if depth <= 0.0 {
        return None;
    }
    let tan = (fov_degrees.to_radians() * 0.5).tan();
    Some([
        0.5 + 0.5 * toward.dot(right) / (depth * tan * aspect),
        0.5 - 0.5 * toward.dot(up) / (depth * tan),
    ])
}

fn floored(
    arena: Option<&Arena>,
    (lobby, wanted): (&LobbyNumbers, &Wanted),
    problems: &mut Vec<String>,
) -> Vec3 {
    let Some(arena) = arena.filter(|arena| arena.ready()) else {
        return wanted.at;
    };
    let probe = lobby.checks.floor_probe_m;
    let lift = wanted.at.y;
    let sample = wanted.at.with_y(0.0);
    match arena.ground_under(sample + Vec3::Y * probe, probe * 2.0) {
        Some(hit) => sample.with_y(hit.point.y + lift),
        None => {
            problems.push(format!(
                "{}.at: no floor within {probe} m above or below {:?}",
                wanted.path,
                wanted.at.to_array()
            ));
            wanted.at
        }
    }
}

fn clips_missing(clips: &Clips, (character, figure): (&str, &Figure), problems: &mut Vec<String>) {
    for (clip, key) in [(&clips.idle, "idle"), (&clips.pick.clip, "pick.clip")] {
        if !figure.clips.contains_key(clip) {
            problems.push(format!(
                "$.lobby.clips.{character}.{key} {clip:?}: the lobby figure {character:?} has no such clip in figure.clips"
            ));
        }
    }
}

type Figures<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static LobbyFigure,
        Ref<'static, Figure>,
        &'static Node,
        Option<&'static mut FigurePlay>,
        (Option<&'static FigureJoints>, Option<&'static FigureClock>),
        (Option<&'static FigureLimbs>, Option<&'static FigureFit>),
    ),
>;

type Reads<'w> = (
    (Res<'w, Numbers>, Res<'w, Puppet>, Res<'w, Assets>),
    (Res<'w, Flow>, Res<'w, Devices>),
    (Res<'w, FrameTime>, Res<'w, FixedTime>),
    Option<Res<'w, Arena>>,
);

type Writes<'w, 's> = (
    ResMut<'w, LobbyBook>,
    ResMut<'w, LobbyView>,
    Stats<'w, LobbyStats>,
    (Moves<'w, 's, LobbyBrick>, Spawner<'w, 's, LobbyBrick>),
);

struct Scene<'a> {
    lobby: &'a LobbyNumbers,
    camera: LobbyCamera,
    drifted: (Vec3, Vec3, f32),
    arena: Option<&'a Arena>,
    fit: Option<FigureFit>,
    now: f64,
    real_now: f64,
    shot: Shot,
    hero: Option<Entity>,
    user_yaw: f32,
    unspin: Option<(Entity, f32, f64)>,
}

fn hide(
    spawner: &mut Spawner<LobbyBrick>,
    (entity, node, play, joints): (Entity, &Node, bool, bool),
) {
    if node.visible {
        spawner.insert(
            entity,
            Node {
                visible: false,
                ..*node
            },
        );
    }
    if play {
        spawner.remove::<FigurePlay>(entity);
    }
    if joints {
        spawner.remove::<FigureJoints>(entity);
    }
}

fn clear_lobby(
    book: &mut LobbyBook,
    view: &mut ResMut<'_, LobbyView>,
    figures: &Figures<'_, '_>,
    spawner: &mut Spawner<LobbyBrick>,
) {
    book.shot = None;
    book.hero = None;
    book.ring = CarouselRing::default();
    book.motions.clear();
    book.user_yaw = 0.0;
    book.user_yaw_vel = 0.0;
    book.unspin = None;
    book.drag_px = 0.0;
    book.drag_x = None;
    view.set_if_neq(LobbyView::default());
    for (entity, _, _, node, play, (joints, _), _) in figures {
        hide(spawner, (entity, node, play.is_some(), joints.is_some()));
    }
}

fn place_figures<'a>(
    scene: &'a Scene<'_>,
    (planned, lobby, assets): (&[(Entity, Wanted)], &'a Arc<LobbyNumbers>, &Assets),
    (book, figures): (&mut LobbyBook, &'a mut Figures<'_, '_>),
    (moves, spawner, problems): (
        &mut Moves<LobbyBrick>,
        &mut Spawner<LobbyBrick>,
        &mut Vec<String>,
    ),
) -> Vec<(&'a str, Option<Shown<'a>>)> {
    let mut rows = Vec::new();
    for (entity, lobby_figure, figure, node, play, (joints, clock), given) in figures {
        let character = lobby_figure.character.as_str();
        let clips = lobby.clips.get(character);
        if clips.is_none() {
            problems.push(format!("$.lobby.clips: no row for lobby figure {character:?}"));
        }
        let found = planned.iter().find(|(known, _)| *known == entity);
        let (Some((_, wanted)), Some(clips)) = (found, clips) else {
            book.motions.remove(&entity);
            hide(spawner, (entity, node, play.is_some(), joints.is_some()));
            rows.push((character, None));
            continue;
        };
        let picked = bound(book, (entity, &figure, assets), (lobby, clips, character), problems);
        let cue = book.cues.entry(entity).or_default();
        let shown = placed(
            scene,
            (wanted, clips, picked),
            (cue, &mut book.motions),
            (entity, node, play, (joints, clock), given),
            (moves, spawner, problems),
        );
        rows.push((character, Some(shown)));
    }
    rows
}

fn stage(
    ((numbers, puppet, assets), (flow, devices), (frame, fixed), arena): Reads,
    mut figures: Figures,
    (mut book, mut view, mut stats, (mut moves, mut spawner)): Writes,
) {
    let book = &mut *book;
    let mut problems = Vec::new();
    let read = numbers_of(book, &numbers);
    let (lobby, roster) = match &read {
        Some((Ok(lobby), roster)) => (Some(Arc::clone(lobby)), roster.as_slice()),
        Some((Err(problem), _)) => {
            problems.push(problem.clone());
            (None, &[][..])
        }
        None => (None, &[][..]),
    };
    let shot = lobby.as_ref().and(Shot::of(flow.stage));
    let mut listed: Vec<(Entity, String)> = figures
        .iter()
        .map(|(entity, lobby_figure, ..)| (entity, lobby_figure.character.clone()))
        .collect();
    listed.sort_unstable_by_key(|(entity, _)| *entity);
    let (Some(lobby), Some(shot)) = (lobby, shot) else {
        clear_lobby(book, &mut view, &figures, &mut spawner);
        report(&mut stats, (flow.stage, None, None), Vec::new, problems);
        return;
    };
    let order = carousel_order(&lobby, roster, &listed);
    let hero = (shot == Shot::Select)
        .then(|| entity_of(&listed, &flow.character))
        .flatten();
    let (now, real_now) = (fixed.elapsed_seconds, frame.real_elapsed);
    let (entering, from_hidden, changed, ring_delta) = cue_up(
        book,
        (Some(shot), &flow.character, hero),
        (&lobby, now, real_now),
        (&listed, &order),
    );
    update_turntable(book, &devices, shot, flow.settings_open, real_now);
    let ring_sample = book.ring.sample(real_now);
    let planned = match shot {
        Shot::Title => title_plan(&lobby, &listed, &mut problems),
        Shot::Select => select_plan(&lobby, (&listed, &order, &flow.character), ring_sample, &mut problems),
    };
    let camera = match shot {
        Shot::Title => lobby.title.camera,
        Shot::Select => lobby.select.camera,
    }
    .moved(Vec3::from_array(lobby.place.at));
    if entering || view.camera != Some(camera) {
        let duration = if from_hidden { 0.0 } else { CAMERA_BLEND_SECONDS };
        view.transition_to(camera, real_now, duration);
    } else if shot == Shot::Select && changed {
        view.nudge(real_now, if ring_delta >= 0.0 { 1.0 } else { -1.0 });
    }
    let drifted = view.shot(real_now).unwrap_or((
        Vec3::from_array(camera.position),
        Vec3::from_array(camera.target),
        camera.fov_degrees,
    ));
    let scene = Scene {
        lobby: &lobby,
        camera,
        drifted,
        arena: arena.as_deref(),
        fit: puppet.eye_fit(&numbers),
        now,
        real_now,
        shot,
        hero,
        user_yaw: book.user_yaw,
        unspin: book.unspin,
    };
    let rows = place_figures(
        &scene,
        (&planned, &lobby, &assets),
        (book, &mut figures),
        (&mut moves, &mut spawner, &mut problems),
    );
    report(
        &mut stats,
        (flow.stage, Some(shot), Some((lobby.place.name.as_str(), drifted.0))),
        || rows.into_iter().map(|(c, s)| row(&scene, c, s)).collect(),
        problems,
    );
}

struct Shown<'a> {
    feet: Vec3,
    yaw: f32,
    clips: &'a Clips,
    weight: Option<f32>,
    visible: bool,
    joints: Option<&'a FigureJoints>,
    clock: Option<&'a FigureClock>,
}

fn row(scene: &Scene<'_>, character: &str, shown: Option<Shown<'_>>) -> LobbyFigureStats {
    let Some(shown) = shown else {
        return LobbyFigureStats {
            character: character.to_owned(),
            ..LobbyFigureStats::default()
        };
    };
    let (checks, joints, clock) = (scene.lobby.checks, shown.joints, shown.clock);
    LobbyFigureStats {
        character: character.to_owned(),
        position: shown.feet.to_array(),
        facing_degrees: shown.yaw.to_degrees(),
        clip: Some(shown.clips.idle.clone()),
        seconds: clock.map_or(0.0, |clock| clock.base_seconds),
        pick_seconds: shown.weight.and(clock).map(|clock| clock.upper_seconds),
        pick_weight: shown.weight.unwrap_or(0.0),
        pick: shown.weight.map(|_| shown.clips.pick.clip.clone()),
        shown: shown.visible,
        in_sight: scene
            .arena
            .filter(|arena| arena.ready())
            .map(|arena| {
                arena.sees(scene.drifted.0, shown.feet + Vec3::Y * checks.sight_height_m)
            }),
        feet_frame: framed(scene.drifted, checks.frame_aspect, shown.feet),
        head_frame_y: joints
            .and_then(|joints| joints.top)
            .and_then(|top| framed(scene.drifted, checks.frame_aspect, top))
            .map(|[_, y]| y),
    }
}

fn extra_yaw(scene: &Scene<'_>, entity: Entity) -> f32 {
    if scene.shot == Shot::Select && scene.hero == Some(entity) {
        return scene.user_yaw;
    }
    if let Some((who, yaw, started)) = scene.unspin
        && who == entity
    {
        let t = math::clamp(((scene.real_now - started) / UNSPIN_SECONDS) as f32, 0.0, 1.0);
        if t < 1.0 {
            return yaw * (1.0 - ease_out_quint(t));
        }
    }
    0.0
}

type Standing<'r, 'a> = (
    Entity,
    &'r Node,
    Option<Mut<'r, FigurePlay>>,
    (Option<&'a FigureJoints>, Option<&'a FigureClock>),
    (Option<&'r FigureLimbs>, Option<&'r FigureFit>),
);

fn placed<'a>(
    scene: &Scene<'_>,
    (wanted, clips, picked): (&Wanted, &'a Clips, Picked),
    (cue, motions): (&mut Cue, &mut BTreeMap<Entity, FigureMotion>),
    (entity, node, play, (joints, clock), (limbs, fit)): Standing<'_, 'a>,
    (moves, spawner, problems): (
        &mut Moves<LobbyBrick>,
        &mut Spawner<LobbyBrick>,
        &mut Vec<String>,
    ),
) -> Shown<'a> {
    let lobby = scene.lobby;
    let feet = floored(scene.arena, (lobby, wanted), problems);
    let toward = Vec3::from_array(scene.camera.position) - feet;
    let base_yaw = math::atan2(toward.x, toward.z)
        + wanted.turn_degrees.to_radians()
        + wanted.yaw_offset
        + extra_yaw(scene, entity);
    let mut drawn_feet = feet;
    let mut drawn_yaw = base_yaw;
    if let Some(mut transform) = moves.get_mut(entity) {
        let motion = motions
            .entry(entity)
            .or_insert_with(|| FigureMotion::settled(feet, base_yaw, wanted.scale, transform.scale));
        let (pos, yaw, scale) = motion.apply(scene.real_now, (feet, base_yaw, wanted.scale));
        drawn_feet = pos;
        drawn_yaw = yaw;
        let next = Transform {
            translation: pos,
            rotation: Quat::from_rotation_y(yaw),
            scale: motion.base_scale * scale,
        };
        transform.set_if_neq(next);
    }
    let (wanted_play, shown) = played(lobby, (wanted, clips, picked, clock), cue, scene.now);
    let weight = (wanted_play.as_ref())
        .and_then(|play| play.upper)
        .map(|upper| upper.weight);
    match (play, wanted_play) {
        (Some(mut play), Some(wanted_play)) => {
            play.set_if_neq(wanted_play);
        }
        (None, Some(wanted_play)) => spawner.insert(entity, wanted_play),
        (_, None) => {}
    }
    if joints.is_none() {
        spawner.insert(entity, FigureJoints::default());
    }
    if limbs.is_none() {
        spawner.insert(entity, biped());
    }
    if fit != scene.fit.as_ref() {
        match &scene.fit {
            Some(fit) => spawner.insert(entity, fit.clone()),
            None => spawner.remove::<FigureFit>(entity),
        }
    }
    if node.visible != shown {
        spawner.insert(
            entity,
            Node {
                visible: shown,
                ..*node
            },
        );
    }
    Shown {
        feet: drawn_feet,
        yaw: drawn_yaw,
        clips,
        weight,
        visible: shown,
        joints,
        clock,
    }
}

fn report(
    stats: &mut Stats<LobbyStats>,
    (stage, shot, place): (Stage, Option<Shot>, Option<(&str, Vec3)>),
    figures: impl FnOnce() -> Vec<LobbyFigureStats>,
    mut problems: Vec<String>,
) {
    problems.dedup();
    stats.show(|| LobbyStats {
        stage: stage.key().to_owned(),
        camera: shot.map_or("player", Shot::key).to_owned(),
        place: place.map_or_else(String::new, |(name, _)| name.to_owned()),
        camera_position: place.map(|(_, at)| at.to_array()),
        figures: figures(),
        refusal: (!problems.is_empty()).then(|| problems.join("; ")),
    });
}
