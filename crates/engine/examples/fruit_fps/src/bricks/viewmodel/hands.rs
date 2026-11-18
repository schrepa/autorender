use std::collections::BTreeMap;

use engine::prelude::*;
use engine_core::math::max;
use engine_core::{Parent, SceneContext};

use super::place::{Held, camera_lit, camera_of, moving, shown_now};
use super::{FirstPersonHands, HandsStats, ViewModelBrick, ViewModelStats};
use crate::play::actors::{Actor, Alive};
use crate::play::camera::TestCamera;
use crate::play::events::powers::PowerEvents;
use crate::play::figures::{BaseLayer, FIGURE, Figure, FigureJoints, FigurePlay, Offset};
use crate::play::flow::Flow;
use crate::play::powers::Powers;
use crate::play::puppet::{
    HandCast, HandsTable, Holding, InHand, InHandProp, PuppetBook, PuppetFile, ability_index,
    leave_tick,
};
use crate::play::view::Watched;
use crate::rules::AbilitySlot;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Ready,
    Hold,
    Cast,
}

impl Phase {
    fn key(self) -> &'static str {
        match self {
            Phase::Ready => "ready",
            Phase::Hold => "hold",
            Phase::Cast => "cast",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Playing {
    ability: String,
    slot: usize,
    phase: Phase,
    clip: String,
    speed: f32,
    fade_seconds: f32,
    looping: bool,
    cue: u32,
    cast: Option<u64>,
    release: Option<u64>,
    prop: Option<InHandProp>,
}

#[derive(Resource, Default)]
pub(super) struct Hands {
    entity: Option<Entity>,
    playing: Option<Playing>,
    cue: u32,
    missing: Option<String>,
    refusal: Option<&'static str>,
    pub(super) shown: bool,
    pub(super) returned: bool,
}

struct Now<'a> {
    table: &'a HandsTable,
    clips: &'a BTreeMap<String, String>,
    tick: u64,
    delta: f32,
}

type Named<'a> = Option<(&'a str, AbilitySlot)>;

fn begun(hands: &mut Hands, now: &Now<'_>, playing: Playing) -> Option<Playing> {
    if !now.clips.contains_key(&playing.clip) {
        hands.missing = Some(playing.clip);
        return None;
    }
    hands.cue = hands.cue.wrapping_add(1);
    Some(Playing {
        cue: hands.cue,
        ..playing
    })
}

fn readied(
    hands: &mut Hands,
    now: &Now<'_>,
    (name, row): (&str, &HandCast),
    slot: AbilitySlot,
) -> Option<Playing> {
    let playing = Playing {
        ability: name.to_owned(),
        slot: ability_index(slot),
        phase: Phase::Ready,
        clip: now.table.ready.clone(),
        speed: 1.0,
        fade_seconds: 0.0,
        looping: false,
        cue: 0,
        cast: None,
        release: None,
        prop: row.prop.clone(),
    };
    begun(hands, now, playing)
}

fn cast(
    hands: &mut Hands,
    now: &Now<'_>,
    (name, row): (&str, &HandCast),
    slot: AbilitySlot,
) -> Option<Playing> {
    let release = row.prop.as_ref().map(|prop| {
        let seconds = prop.leave_seconds / max(row.speed, 1.0e-3);
        leave_tick(now.tick, seconds, now.delta)
    });
    let playing = Playing {
        ability: name.to_owned(),
        slot: ability_index(slot),
        phase: Phase::Cast,
        clip: row.clip.clone(),
        speed: row.speed,
        fade_seconds: row.fade_seconds,
        looping: false,
        cue: 0,
        cast: Some(now.tick),
        release,
        prop: row.prop.clone(),
    };
    begun(hands, now, playing)
}

fn stepped(
    hands: &mut Hands,
    now: &Now<'_>,
    (held, casted): (Named<'_>, Named<'_>),
    done: bool,
) -> Option<Playing> {
    let current = hands.playing.take();
    if let Some(((name, row), slot)) = casted.and_then(|(name, slot)| {
        now.table
            .abilities
            .get_key_value(name)
            .map(|found| (found, slot))
    }) {
        return cast(hands, now, (name.as_str(), row), slot);
    }
    match (current, held) {
        (Some(playing), _) if playing.phase == Phase::Cast => (!done).then_some(playing),
        (Some(playing), Some((name, _))) if playing.ability == name => {
            if playing.phase == Phase::Ready && done {
                let hold = Playing {
                    phase: Phase::Hold,
                    clip: now.table.hold.clone(),
                    looping: true,
                    ..playing
                };
                begun(hands, now, hold)
            } else {
                Some(playing)
            }
        }
        (Some(_) | None, Some((name, slot))) => now
            .table
            .abilities
            .get_key_value(name)
            .and_then(|(name, row)| readied(hands, now, (name.as_str(), row), slot)),
        (Some(_) | None, None) => None,
    }
}

fn named(powers: Option<&Powers>, slot: u32, ability: AbilitySlot) -> Option<&str> {
    let actor = powers?.actor(slot)?;
    Some(actor.slots[ability_index(ability)].name.as_str())
}

fn in_hand(
    playing: &Playing,
    (hands, (owner, team)): (Entity, (u32, usize)),
    socket: &str,
) -> Option<InHand> {
    let prop = playing.prop.as_ref()?;
    Some(InHand {
        holder: hands,
        socket: socket.to_owned(),
        seat: Offset::STILL,
        prop: prop.key.clone(),
        size_m: prop.size_m,
        owner,
        team,
        ability: playing.slot,
        cast: playing.cast,
        release: playing.release,
    })
}

type Decide<'w, 's> = (
    Res<'w, FixedTime>,
    (Option<Res<'w, Powers>>, Option<Res<'w, PowerEvents>>),
    Query<'w, 's, &'static PuppetFile>,
    (
        Option<Res<'w, Watched>>,
        Option<Res<'w, Flow>>,
        Option<Res<'w, TestCamera>>,
    ),
    Query<'w, 's, (&'static Actor, Has<Alive>)>,
    Query<'w, 's, &'static FigureJoints, With<FirstPersonHands>>,
);

fn viewer(
    (watched, flow, test): (Option<&Watched>, Option<&Flow>, Option<&TestCamera>),
    actors: &Query<(&Actor, Has<Alive>)>,
) -> Option<(u32, usize)> {
    let entity = watched?.entity?;
    let (actor, alive) = actors.get(entity).ok()?;
    (alive && shown_now(flow, test)).then_some((actor.slot, actor.team.index()))
}

pub(super) fn decide(
    (fixed, (powers, events), files, (watched, flow, test), actors, joints): Decide,
    (mut hands, mut holding): (ResMut<Hands>, ResMut<Holding>),
) {
    let was = hands.shown;
    hands.missing = None;
    hands.refusal = None;
    let book = files.iter().find_map(PuppetFile::book);
    let viewed = viewer(
        (watched.as_deref(), flow.as_deref(), test.as_deref()),
        &actors,
    );
    let files = book.and_then(|book| book.hands.as_ref().map(|files| (book, files)));
    let next = match (viewed, files) {
        (Some((slot, _)), Some((book, files))) => {
            let now = Now {
                table: &book.table.view.hands,
                clips: &files.figure.clips,
                tick: fixed.tick,
                delta: fixed.delta_seconds,
            };
            let powers = powers.as_deref();
            let held = powers
                .and_then(|powers| powers.held(slot))
                .map(|held| (held.name.as_str(), held.ability));
            let casted = events
                .iter()
                .flat_map(|events| &events.casts)
                .rfind(|cast| cast.slot == slot)
                .and_then(|cast| Some((named(powers, slot, cast.ability)?, cast.ability)));
            let done = hands.playing.as_ref().is_some_and(|playing| {
                hands
                    .entity
                    .and_then(|entity| joints.get(entity).ok())
                    .is_some_and(|joints| joints.base_cue == playing.cue && joints.base_done)
            });
            stepped(&mut hands, &now, (held, casted), done)
        }
        (Some(_), None) => {
            hands.refusal = Some("no `hands` in the scene's `puppet` component");
            None
        }
        (None, _) => None,
    };
    hands.playing = next;
    hands.shown = hands.playing.is_some();
    hands.returned = was && !hands.shown;
    let socket = book.map_or("", |book| book.table.view.hands.socket.as_str());
    let held = hands
        .playing
        .as_ref()
        .zip(hands.entity.zip(viewed))
        .and_then(|(playing, holder)| in_hand(playing, holder, socket));
    holding.set_if_neq(Holding { held });
}

fn wanted(playing: &Playing) -> FigurePlay {
    FigurePlay {
        base: BaseLayer {
            clip: playing.clip.clone(),
            speed: playing.speed,
            looping: playing.looping,
            fade_seconds: playing.fade_seconds,
            cue: playing.cue,
            fit_seconds: None,
            blend: Vec::new(),
            stride: None,
            synced: false,
        },
        upper: None,
        aim: None,
        hold: None,
        crouch: None,
        pin_feet: None,
    }
}

type Parts<'w, 's> = Query<
    'w,
    's,
    (
        &'static Node,
        Option<&'static Parent>,
        &'static mut FigurePlay,
        &'static FigureJoints,
    ),
    With<FirstPersonHands>,
>;

type Draw<'w, 's> = (
    (Res<'w, FixedTime>, Res<'w, Held>, Res<'w, Holding>),
    Option<Res<'w, SceneContext>>,
    Query<'w, 's, &'static PuppetFile>,
);

fn made(book: &PuppetBook, scene: &SceneContext) -> Result<(Drawn, Figure, Entity), &'static str> {
    let files = book
        .hands
        .as_ref()
        .ok_or("no `hands` in the scene's `puppet` component")?;
    let assets = &scene.assets;
    let mesh = assets.meshes.get(&files.mesh).copied();
    let material = assets.materials.get(&files.material).copied();
    let (Some(mesh), Some(material)) = (mesh, material) else {
        return Err("hands mesh or material missing from the scene");
    };
    let camera = camera_of(scene).ok_or("no scene entity `camera`")?;
    Ok((
        camera_lit(Drawn::new(FIGURE, mesh, material)),
        files.figure.clone(),
        camera,
    ))
}

pub(super) fn draw(
    ((fixed, gun, holding), scene, files): Draw,
    mut parts: Parts,
    (mut hands, mut stats, mut moves, mut spawner): (
        ResMut<Hands>,
        ResMut<ViewModelStats>,
        Moves<ViewModelBrick>,
        Spawner<ViewModelBrick>,
    ),
) {
    let book = files.iter().find_map(PuppetFile::book);
    let scale = book.map_or(1.0, |book| book.table.view.scale);
    let placed = Transform {
        translation: gun.offset() * scale,
        rotation: Quat::IDENTITY,
        scale: Vec3::splat(scale),
    };
    let shown = hands.shown;
    let existing = hands.entity.and_then(|entity| parts.get_mut(entity).ok());
    let mut seconds = 0.0;
    let mut item = None;
    let current = hands.playing.clone();
    match (existing, current.as_ref()) {
        (Some((node, parent, mut play, joints)), playing) => {
            let entity = hands.entity;
            if node.visible != shown
                && let Some(entity) = entity
            {
                spawner.insert(entity, moving(shown));
            }
            if let Some(playing) = playing {
                play.set_if_neq(wanted(playing));
                seconds = if joints.base_cue == playing.cue {
                    joints.base_seconds
                } else {
                    0.0
                };
                item = joints
                    .joints
                    .first()
                    .filter(|joint| joint.found)
                    .map(|joint| joint.position.to_array());
            }
            if let Some(entity) = entity {
                let camera = scene.as_deref().and_then(camera_of);
                if camera.is_some() && parent.map(Parent::parent) != camera {
                    spawner.set_parent(entity, camera);
                }
                if let Some(mut transform) = moves.get_mut(entity) {
                    transform.set_if_neq(placed);
                }
            }
        }
        (None, Some(playing)) => {
            let made = match (book, scene.as_deref()) {
                (Some(book), Some(scene)) => made(book, scene),
                (None, _) | (_, None) => Err("no prepared `puppet` component or scene"),
            };
            match made {
                Ok((drawn, figure, camera)) => {
                    let entity = spawner.spawn((
                        placed,
                        moving(shown),
                        drawn,
                        figure,
                        wanted(playing),
                        FigureJoints {
                            names: vec![book.map_or_else(String::new, |book| {
                                book.table.view.hands.socket.clone()
                            })],
                            ..FigureJoints::default()
                        },
                        FirstPersonHands,
                    ));
                    spawner.set_parent(entity, Some(camera));
                    hands.entity = Some(entity);
                }
                Err(why) => hands.refusal = Some(why),
            }
        }
        (None, None) => {}
    }
    let playing = hands.playing.as_ref();
    let prop = holding
        .held
        .as_ref()
        .filter(|held| held.holds(fixed.tick))
        .map(|held| held.prop.clone());
    let report = HandsStats {
        shown,
        ability: playing.map(|playing| playing.ability.clone()),
        phase: playing.map(|playing| playing.phase.key().to_owned()),
        clip: playing.map(|playing| playing.clip.clone()),
        seconds,
        prop,
        release_tick: playing.and_then(|playing| playing.release),
        item_r: item,
        missing: hands.missing.clone(),
        refusal: hands.refusal.map(str::to_owned),
    };
    if stats.hands != report {
        stats.hands = report;
    }
}
