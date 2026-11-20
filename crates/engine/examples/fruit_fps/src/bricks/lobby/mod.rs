engine::brick!(lobby);

mod numbers;

use std::collections::BTreeMap;
use std::sync::Arc;

use engine::math;
use engine::prelude::*;
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

type Read = Result<Arc<LobbyNumbers>, String>;

#[derive(Resource, Default)]
struct LobbyBook {
    numbers: Option<(u64, Read)>,
    shot: Option<Shot>,
    chosen: String,
    cues: BTreeMap<Entity, Cue>,
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

fn numbers_of(book: &mut LobbyBook, numbers: &Numbers) -> Option<Read> {
    let game = numbers.get()?;
    let fresh = book
        .numbers
        .as_ref()
        .is_none_or(|(revision, _)| *revision != numbers.revision());
    if fresh {
        let read = numbers::read(&game.whole).map(Arc::new);
        book.numbers = Some((numbers.revision(), read));
    }
    book.numbers.as_ref().map(|(_, read)| read.clone())
}

struct Wanted {
    at: Vec3,
    turn_degrees: f32,
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
    (listed, chosen): (&Listed, &str),
    problems: &mut Vec<String>,
) -> Vec<(Entity, Wanted)> {
    let anchor = Vec3::from_array(lobby.place.at);
    let group = &lobby.title.group;
    let speed = |character: &str| {
        group
            .iter()
            .find(|member| member.character == character)
            .map_or(1.0, |member| member.speed)
    };
    let mut order: Vec<&str> = group
        .iter()
        .map(|member| member.character.as_str())
        .collect();
    for (_, character) in listed {
        if !order.contains(&character.as_str()) {
            order.push(character.as_str());
        }
    }
    let mut planned = Vec::new();
    match entity_of(listed, chosen) {
        Some(entity) => planned.push((
            entity,
            Wanted {
                at: anchor + Vec3::from_array(lobby.select.front.at),
                turn_degrees: lobby.select.front.turn_degrees,
                speed: speed(chosen),
                lead_seconds: 0.0,
                path: "$.lobby.select.front".to_owned(),
            },
        )),
        None => problems.push(format!(
            "flow.character {chosen:?}: no scene entity carries lobby_figure {{character {chosen:?}}}"
        )),
    }
    let others = order.into_iter().filter(|character| *character != chosen);
    for ((at, spot), character) in lobby.select.back.iter().enumerate().zip(others) {
        if let Some(entity) = entity_of(listed, character) {
            planned.push((
                entity,
                Wanted {
                    at: anchor + Vec3::from_array(spot.at),
                    turn_degrees: spot.turn_degrees,
                    speed: speed(character),
                    lead_seconds: 0.0,
                    path: format!("$.lobby.select.back[{at}]"),
                },
            ));
        }
    }
    planned
}

fn cue_up(
    book: &mut LobbyBook,
    (shot, chosen, hero): (Option<Shot>, &str, Option<Entity>),
    (lobby, now): (&LobbyNumbers, f64),
    listed: &Listed,
) {
    let entering = book.shot != shot;
    let from_hidden = book.shot.is_none() && shot.is_some();
    book.shot = shot;
    let changed = book.chosen != chosen;
    if changed {
        chosen.clone_into(&mut book.chosen);
    }
    book.cues
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
    match arena.ground_under(wanted.at + Vec3::Y * probe, probe * 2.0) {
        Some(hit) => wanted.at.with_y(hit.point.y),
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
    Res<'w, Flow>,
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

fn stage(
    ((numbers, puppet, assets), flow, (frame, fixed), arena): Reads,
    mut figures: Figures,
    (mut book, mut view, mut stats, (mut moves, mut spawner)): Writes,
) {
    let book = &mut *book;
    let mut problems = Vec::new();
    let read = numbers_of(book, &numbers);
    let lobby = match &read {
        Some(Ok(lobby)) => Some(Arc::clone(lobby)),
        Some(Err(problem)) => {
            problems.push(problem.clone());
            None
        }
        None => None,
    };
    let shot = lobby.as_ref().and(Shot::of(flow.stage));
    let mut listed: Vec<(Entity, String)> = figures
        .iter()
        .map(|(entity, lobby_figure, ..)| (entity, lobby_figure.character.clone()))
        .collect();
    listed.sort_unstable_by_key(|(entity, _)| *entity);
    let (Some(lobby), Some(shot)) = (lobby, shot) else {
        book.shot = None;
        view.set_if_neq(LobbyView::default());
        for (entity, _, _, node, play, (joints, _), _) in &figures {
            hide(
                &mut spawner,
                (entity, node, play.is_some(), joints.is_some()),
            );
        }
        report(&mut stats, (flow.stage, None, None), Vec::new, problems);
        return;
    };
    let planned = match shot {
        Shot::Title => title_plan(&lobby, &listed, &mut problems),
        Shot::Select => select_plan(&lobby, (&listed, &flow.character), &mut problems),
    };
    let hero = (shot == Shot::Select)
        .then(|| entity_of(&listed, &flow.character))
        .flatten();
    let now = fixed.elapsed_seconds;
    let entering = book.shot != Some(shot);
    cue_up(
        book,
        (Some(shot), &flow.character, hero),
        (&lobby, now),
        &listed,
    );
    let camera = match shot {
        Shot::Title => lobby.title.camera,
        Shot::Select => lobby.select.camera,
    }
    .moved(Vec3::from_array(lobby.place.at));
    if entering || view.camera != Some(camera) {
        view.set_if_neq(LobbyView {
            camera: Some(camera),
            since: frame.real_elapsed,
        });
    }
    let drifted = view.shot(frame.real_elapsed).unwrap_or((
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
    };
    let mut rows = Vec::new();
    for (entity, lobby_figure, figure, node, play, (joints, clock), given) in &mut figures {
        let character = lobby_figure.character.as_str();
        let clips = lobby.clips.get(character);
        if clips.is_none() {
            problems.push(format!(
                "$.lobby.clips: no row for lobby figure {character:?}"
            ));
        }
        let found = planned.iter().find(|(known, _)| *known == entity);
        let (Some((_, wanted)), Some(clips)) = (found, clips) else {
            hide(
                &mut spawner,
                (entity, node, play.is_some(), joints.is_some()),
            );
            rows.push((character, None));
            continue;
        };
        let picks = (&lobby, clips, character);
        let picked = bound(book, (entity, &figure, &assets), picks, &mut problems);
        let cue = book.cues.entry(entity).or_default();
        let shown = placed(
            &scene,
            (wanted, clips, picked),
            cue,
            (entity, node, play, (joints, clock), given),
            (&mut moves, &mut spawner, &mut problems),
        );
        rows.push((character, Some(shown)));
    }
    report(
        &mut stats,
        (
            flow.stage,
            Some(shot),
            Some((lobby.place.name.as_str(), drifted.0)),
        ),
        || {
            (rows.into_iter())
                .map(|(character, shown)| row(&scene, character, shown))
                .collect()
        },
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
    cue: &mut Cue,
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
    let yaw = math::atan2(toward.x, toward.z) + wanted.turn_degrees.to_radians();
    if let Some(mut transform) = moves.get_mut(entity) {
        let next = Transform {
            translation: feet,
            rotation: Quat::from_rotation_y(yaw),
            scale: transform.scale,
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
        feet,
        yaw,
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
