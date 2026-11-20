use std::fmt::Write;

use engine::math;
use engine::prelude::*;
use engine::{FrameTime, Projection, Views};
use engine_core::DevSetting;

use crate::play::actors::{Actor, Body, Health, Human, Motion, Wish};
use crate::play::arms::Arms;
use crate::play::events::arms::{ArmsEvents, Target};
use crate::play::events::powers::PowerEvents;
use crate::play::events::round::RoundEvents;
use crate::play::numbers::Numbers;
use crate::play::round::{DeviceSpot, Round};
use crate::play::view::{Watch, Watched};
use crate::rules::{
    AbilitySlot, Cause, Event, Phase, Player, RoundEndReason, Team, View, WeaponId,
};
use crate::weapons::Stance;

use super::hud::{DamageMark, FeedLine, HudFacts, MapItem, ResultFacts, Shown};
use super::live::{self, character_key, character_name, named, set, tone, worded};
use super::{DamageEdge, Loaded, ScreenLoad, ScreenSetup, ScreenSource, Words};

const WALL_SECONDS: f64 = 1.0;
const DEVICE_ICON: &str = "device";
const DEMO_CHANNEL: f32 = 0.6;
const DEMO_SCORE: [&str; 2] = ["5", "3"];
const DEMO_POOR_NEED: [u32; 4] = [0, 1150, 2450, 4250];

#[derive(Resource, Default)]
pub(super) struct Recent {
    pub(super) ultimate: Option<(f64, u32)>,
    power_tick: Option<u64>,
    feed: Vec<(f64, usize, FeedLine)>,
    walls: Vec<(f64, usize)>,
    went_live: Option<f64>,
    point: Option<(f64, String)>,
    dealt: Vec<f32>,
    round_result: Option<ResultFacts>,
    match_result: Option<ResultFacts>,
    armor_break: Option<f64>,
    kill: Option<(f64, u32, String)>,
    damage: Vec<(f64, Vec3)>,
    round_tick: Option<u64>,
    arms_tick: Option<u64>,
    demo_shown: bool,
    healths: Vec<Option<f32>>,
    order: Vec<(u32, WeaponId)>,
}

impl Recent {
    pub(super) fn shows_demo(&self) -> bool {
        self.demo_shown
    }
}

fn cause_icon(line: &mut FeedLine, view: &View<'_>, killer: Option<usize>, cause: Cause) {
    let catalog = view.catalog;
    line.weapon.clear();
    line.ability.clear();
    line.world = false;
    match cause {
        Cause::Weapon(id) => {
            if let Some(entry) = catalog.weapon(id) {
                named(&mut line.weapon, "weapon_", &entry.name);
            }
        }
        Cause::Ability(slot) => {
            let index = slot.index();
            let name = killer
                .and_then(|killer| view.players.get(killer))
                .and_then(|player| catalog.characters.get(player.character))
                .map_or("", |entry| entry.ability_names[index].as_str());
            named(&mut line.ability, "ability_", name);
        }
        Cause::World => line.world = true,
    }
}
fn feed_line(
    view: &View<'_>,
    (killer, victim): (Option<usize>, usize),
    (headshot, cause): (bool, Cause),
    human: Option<usize>,
) -> FeedLine {
    let catalog = view.catalog;
    let mut line = FeedLine {
        headshot,
        mine: killer.is_some() && killer == human,
        ..FeedLine::default()
    };
    if let Some(player) = killer.and_then(|killer| view.players.get(killer)) {
        set(&mut line.killer, character_name(catalog, player));
        named(
            &mut line.killer_char,
            "char_",
            character_key(catalog, player),
        );
        line.killer_color = tone(view.side_of(player.team));
    }
    if let Some(player) = view.players.get(victim) {
        set(&mut line.victim, character_name(catalog, player));
        named(
            &mut line.victim_char,
            "char_",
            character_key(catalog, player),
        );
        line.victim_color = tone(view.side_of(player.team));
    }
    cause_icon(&mut line, view, killer, cause);
    line
}

fn kill_part(
    recent: &mut Recent,
    view: &View<'_>,
    (killer, victim, human): (Option<usize>, usize, Option<usize>),
    now: f64,
) {
    let (Some(killer), Some(human)) = (killer, human) else {
        return;
    };
    let (Some(me), Some(fallen)) = (view.players.get(killer), view.players.get(victim)) else {
        return;
    };
    if killer != human || fallen.team == me.team {
        return;
    }
    let mut icon = String::new();
    named(&mut icon, "char_", character_key(view.catalog, fallen));
    icon.push('_');
    icon.push_str(tone(view.side_of(fallen.team)).key());
    recent.kill = Some((now, me.round_kills, icon));
}

fn assist_part(
    line: &mut FeedLine,
    view: &View<'_>,
    (killer, victim, assistants): (Option<usize>, usize, &[usize]),
    human: Option<usize>,
) {
    let victim_team = view.players.get(victim).map(|player| player.team);
    let mut helped: Vec<usize> = Vec::new();
    for &slot in assistants {
        let enemy = view
            .players
            .get(slot)
            .is_some_and(|player| Some(player.team) != victim_team);
        if enemy && Some(slot) != killer && !helped.contains(&slot) {
            helped.push(slot);
        }
    }
    line.assisted = human.is_some_and(|human| helped.contains(&human));
    line.assist_more = helped.len().saturating_sub(1) as u32;
    let Some(player) = helped.first().and_then(|slot| view.players.get(*slot)) else {
        return;
    };
    named(
        &mut line.assist_char,
        "char_",
        character_key(view.catalog, player),
    );
    line.assist_color = tone(view.side_of(player.team));
}

fn point_reason<'a>(events: &[Event], human: usize, words: &'a Words) -> Option<&'a str> {
    let gained = events
        .iter()
        .any(|event| matches!(event, Event::UltimatePoint { player, .. } if *player == human));
    if !gained {
        return None;
    }
    let mut reason = words.point_orb.as_str();
    for event in events {
        if matches!(event, Event::KillFeed { killer: Some(killer), .. } if *killer == human) {
            reason = &words.point_kill;
        } else if matches!(event, Event::KillFeed { victim, .. } if *victim == human) {
            reason = &words.point_death;
        } else if matches!(event, Event::PlantCompleted { player, .. } if *player == human) {
            reason = &words.point_plant;
        } else if matches!(event, Event::DefuseCompleted { player } if *player == human) {
            reason = &words.point_defuse;
        }
    }
    Some(reason)
}

fn most(scores: impl Iterator<Item = (usize, u32)>) -> Option<usize> {
    let mut best: Option<(usize, u32)> = None;
    for (slot, score) in scores {
        if best.is_none_or(|(_, held)| score > held) {
            best = Some((slot, score));
        }
    }
    best.map(|(slot, _)| slot)
}

fn mvp_into(
    result: &mut ResultFacts,
    view: &View<'_>,
    slot: Option<usize>,
    (words, dealt): (&Words, &[f32]),
) {
    let catalog = view.catalog;
    let Some((slot, player)) = slot.and_then(|slot| Some((slot, view.players.get(slot)?))) else {
        return;
    };
    let mut kills = String::new();
    let mut deaths = String::new();
    let mut assists = String::new();
    let mut adr = String::new();
    let _ = write!(kills, "{}", player.kills);
    let _ = write!(deaths, "{}", player.deaths);
    let _ = write!(assists, "{}", player.assists);
    let _ = write!(adr, "{}", live::adr(view, dealt, slot));
    let name = character_name(catalog, player);
    let counts = [
        ("name", name),
        ("kills", kills.as_str()),
        ("deaths", deaths.as_str()),
        ("assists", assists.as_str()),
        ("adr", adr.as_str()),
    ];
    worded(&mut result.mvp, &words.mvp_match, &counts);
    worded(&mut result.mvp_stats, &words.mvp_stats, &counts);
    set(&mut result.mvp_name, name);
    let role = catalog
        .characters
        .get(player.character)
        .map_or("", |entry| entry.role.as_str());
    worded(
        &mut result.mvp_char,
        &words.mvp_char,
        &[("character", name), ("role", role)],
    );
    named(
        &mut result.mvp_icon,
        "char_",
        character_key(catalog, player),
    );
    result.mvp_portrait.clone_from(&result.mvp_icon);
    result.mvp_color = tone(view.side_of(player.team));
}
fn round_result(
    view: &View<'_>,
    (winner, reason): (Team, RoundEndReason),
    (human_team, device): (Option<Team>, DeviceSpot),
    words: &Words,
    kills: &[u32],
) -> ResultFacts {
    let catalog = view.catalog;
    let mut result = ResultFacts {
        color: tone(view.side_of(winner)),
        ..ResultFacts::default()
    };
    let won = human_team == Some(winner);
    set(
        &mut result.title,
        if won {
            &words.round_won
        } else {
            &words.round_lost
        },
    );
    let site = match device {
        DeviceSpot::Planted { site, .. }
        | DeviceSpot::Exploded { site, .. }
        | DeviceSpot::Defused { site, .. } => catalog.sites.get(site).map_or("", String::as_str),
        DeviceSpot::Nowhere | DeviceSpot::Carried { .. } | DeviceSpot::Dropped { .. } => "",
    };
    let template = match reason {
        RoundEndReason::Elimination => &words.elimination,
        RoundEndReason::Exploded => &words.exploded,
        RoundEndReason::Defused => &words.defused,
        RoundEndReason::Time => &words.time_out,
    };
    worded(
        &mut result.reason,
        template,
        &[("device", &catalog.device_name), ("site", site)],
    );
    let best = most(
        kills
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, kills)| *kills > 0),
    );
    if let Some(slot) = best {
        let mut round_kills = String::new();
        let _ = write!(round_kills, "{}", kills[slot]);
        if let Some(player) = view.players.get(slot) {
            worded(
                &mut result.mvp,
                &words.mvp_round,
                &[
                    ("name", character_name(catalog, player)),
                    ("kills", &round_kills),
                ],
            );
            named(
                &mut result.mvp_icon,
                "char_",
                character_key(catalog, player),
            );
            result.mvp_color = tone(view.side_of(player.team));
        }
    }
    result
}

fn match_result(
    view: &View<'_>,
    (winner, human_team): (Team, Option<Team>),
    words: &Words,
    dealt: &[f32],
) -> ResultFacts {
    let mut result = ResultFacts {
        color: tone(view.side_of(winner)),
        ..ResultFacts::default()
    };
    let us = human_team.unwrap_or(winner);
    let them = Team::BOTH
        .into_iter()
        .find(|team| *team != us)
        .unwrap_or(us);
    let mut ours = String::new();
    let mut theirs = String::new();
    let _ = write!(ours, "{}", view.score[us.index()]);
    let _ = write!(theirs, "{}", view.score[them.index()]);
    worded(
        &mut result.title,
        if us == winner {
            &words.victory
        } else {
            &words.defeat
        },
        &[("us", &ours), ("them", &theirs)],
    );
    let best = most(view.players.iter().map(|player| player.kills).enumerate());
    mvp_into(&mut result, view, best, (words, dealt));
    result
}

type Sources<'w, 's> = (
    Option<Res<'w, Round>>,
    (Option<Res<'w, RoundEvents>>, Option<Res<'w, PowerEvents>>),
    Option<Res<'w, ArmsEvents>>,
    (Query<'w, 's, &'static ScreenSource>, Res<'w, ScreenLoad>),
    Res<'w, FrameTime>,
);

fn powers_part(recent: &mut Recent, events: &PowerEvents, now: f64) {
    if recent.power_tick == Some(events.tick) {
        return;
    }
    recent.power_tick = Some(events.tick);
    if let Some(cast) = events
        .casts
        .iter()
        .rev()
        .find(|cast| cast.ability == AbilitySlot::X)
    {
        recent.ultimate = Some((now, cast.slot));
    }
}

pub(super) fn collect(
    (round, (events, powers), arms, (sources, load), time): Sources,
    humans: Query<&Actor, With<Human>>,
    bodies: Query<(&Actor, &Body)>,
    mut recent: ResMut<Recent>,
) {
    let Some(source) = sources.iter().next() else {
        return;
    };
    let (Some(round), Some(view)) = (round.as_deref(), round.as_deref().and_then(Round::view))
    else {
        return;
    };
    let now = time.elapsed;
    let recent = &mut *recent;
    if let Some(powers) = powers.as_deref() {
        powers_part(recent, powers, now);
    }
    let human = humans.iter().next().map(|actor| actor.slot as usize);
    let human_team = human
        .and_then(|slot| view.players.get(slot))
        .map(|player| player.team);
    recent.dealt.resize(view.players.len(), 0.0);
    if let Some(events) = events.filter(|events| recent.round_tick != Some(events.tick)) {
        recent.round_tick = Some(events.tick);
        for event in &events.events {
            if let Event::KillFeed {
                killer,
                victim,
                headshot,
                cause,
                assistants,
            } = event
            {
                let mut line = feed_line(&view, (*killer, *victim), (*headshot, *cause), human);
                assist_part(&mut line, &view, (*killer, *victim, assistants), human);
                line.wall = recent.walls.iter().any(|(_, slot)| slot == victim);
                recent.feed.push((now, *victim, line));
                kill_part(recent, &view, (*killer, *victim, human), now);
            } else if let Event::Phase {
                phase: Phase::Buy,
                round,
                ..
            } = event
            {
                if *round <= 1 {
                    recent.dealt.fill(0.0);
                }
                recent.round_result = None;
                recent.match_result = None;
            } else if let Event::Phase {
                phase: Phase::Live, ..
            } = event
            {
                recent.went_live = Some(now);
            } else if let Event::RoundEnd { winner, reason, .. } = event {
                let result = round_result(
                    &view,
                    (*winner, *reason),
                    (human_team, round.device),
                    &source.words,
                    &view
                        .players
                        .iter()
                        .map(|player| player.round_kills)
                        .collect::<Vec<_>>(),
                );
                recent.round_result = Some(result);
            } else if let Event::MatchEnd { winner, .. } = event {
                recent.match_result = Some(match_result(
                    &view,
                    (*winner, human_team),
                    &source.words,
                    &recent.dealt,
                ));
            }
        }
        if let Some(reason) =
            human.and_then(|human| point_reason(&events.events, human, &source.words))
        {
            recent.point = Some((now, reason.to_owned()));
        }
    }
    let lines = source.feed.lines;
    if recent.feed.len() > lines {
        let extra = recent.feed.len() - lines;
        recent.feed.drain(..extra);
    }
    if let Some(arms) = arms.filter(|arms| recent.arms_tick != Some(arms.tick)) {
        let edge = load.loaded().map(|_| &source.marks.damage);
        arms_part(recent, &arms, (&bodies, human), (edge, now));
    }
}

fn arms_part(
    recent: &mut Recent,
    arms: &ArmsEvents,
    (bodies, human): (&Query<(&Actor, &Body)>, Option<usize>),
    (edge, now): (Option<&DamageEdge>, f64),
) {
    recent.arms_tick = Some(arms.tick);
    recent.walls.retain(|(born, _)| now - born < WALL_SECONDS);
    for hit in arms.hits.iter().filter(|hit| hit.killed) {
        let walled = arms.shots.iter().any(|shot| {
            shot.shooter == hit.shooter
                && shot.pierced.is_some()
                && matches!(shot.target, Target::Actor { slot, .. } if slot == hit.victim)
        });
        if walled {
            let victim = hit.victim as usize;
            recent.walls.push((now, victim));
            for (born, slot, line) in &mut recent.feed {
                if *slot == victim && now - *born < WALL_SECONDS {
                    line.wall = true;
                }
            }
        }
    }
    for hit in &arms.hits {
        if let Some(dealt) = recent.dealt.get_mut(hit.shooter as usize) {
            *dealt += hit.health_lost + hit.armor_lost;
        }
        if Some(hit.shooter as usize) == human && hit.armor_broke {
            recent.armor_break = Some(now);
        }
        if Some(hit.victim as usize) == human
            && let Some((_, body)) = bodies.iter().find(|(actor, _)| actor.slot == hit.shooter)
        {
            let me = bodies
                .iter()
                .find(|(actor, _)| actor.slot == hit.victim)
                .map(|(_, mine)| mine.feet);
            let half = edge.map_or(0.0, |edge| edge.length_deg.to_radians() * 0.5);
            if let Some(me) = me {
                let new = bearing(body.feet, me);
                recent
                    .damage
                    .retain(|(_, from)| turned(bearing(*from, me) - new).abs() > half);
            }
            recent.damage.push((now, body.feet));
        }
    }
    let keep = f64::from(edge.map_or(0.0, |edge| edge.fade_seconds));
    recent.damage.retain(|(born, _)| now - born < keep);
}

fn bearing(from: Vec3, at: Vec3) -> f32 {
    let toward = from - at;
    math::atan2(-toward.x, -toward.z)
}

fn turned(angle: f32) -> f32 {
    let turn = std::f32::consts::TAU;
    angle - (angle / turn).round() * turn
}

fn stance(motion: Motion) -> Stance {
    match motion {
        Motion::Standing => Stance::Stand,
        Motion::Walking => Stance::Walk,
        Motion::Running => Stance::Run,
        Motion::Crouching => Stance::Crouch,
        Motion::Air => Stance::Air,
    }
}

fn spread_px(views: &Views, spread_deg: f32) -> f32 {
    let Some(main) = views.main() else {
        return 0.0;
    };
    let fov = match main.lens.projection {
        Projection::Perspective {
            vertical_fov_radians,
        } => vertical_fov_radians,
        Projection::Orthographic { .. } | Projection::Fitted { .. } => return 0.0,
    };
    let tan = |angle: f32| {
        let (sin, cos) = math::sin_cos(angle);
        sin / cos
    };
    let half = tan(fov * 0.5);
    if half <= 0.0 {
        return 0.0;
    }
    (tan(spread_deg.to_radians()) / half * views.frame().display_size.y as f32 * 0.5).round()
}

type HumanRow = (
    &'static Actor,
    Option<&'static Health>,
    Option<&'static Arms>,
    Option<&'static Body>,
    Option<&'static Wish>,
);

fn weapon_part(
    facts: &mut HudFacts,
    (arms, body): (Option<&Arms>, Option<&Body>),
    numbers: &Numbers,
    views: &Views,
    share: f32,
) {
    let weapon = &mut facts.weapon;
    let (Some(arms), Some(numbers)) = (arms, numbers.get()) else {
        weapon.mag = None;
        weapon.reserve = None;
        weapon.mag_size = None;
        return;
    };
    let motion = body.map_or(Stance::Stand, |body| stance(body.motion));
    let shown = arms.loadout.view(&numbers.weapons, motion);
    set(&mut weapon.name, shown.name);
    if arms.device_out {
        set(&mut weapon.icon, DEVICE_ICON);
    } else {
        named(&mut weapon.icon, "weapon_", shown.name);
    }
    weapon.down = arms.weapon_down;
    weapon.mag = shown.magazine;
    weapon.reserve = shown.reserve;
    weapon.reload = shown.reload_progress;
    weapon.mag_size = arms
        .loadout
        .active_gun()
        .map(|gun| numbers.weapons.gun(gun.kind()).magazine);
    weapon.low = match (weapon.mag, weapon.mag_size) {
        (Some(mag), Some(size)) => mag as f32 <= size as f32 * share,
        (Some(_) | None, Some(_) | None) => false,
    };
    facts.spread_px = spread_px(views, shown.spread_deg);
}

fn marks_part(
    facts: &mut HudFacts,
    recent: &Recent,
    (wish, body): (Option<&Wish>, Option<&Body>),
    (source, now): (&ScreenSource, f64),
) {
    let marks = &source.marks;
    facts.armor_break = recent.armor_break.and_then(|born| {
        let age = (now - born) as f32;
        (age < marks.armor.seconds).then_some(age)
    });
    let kill = &mut facts.kill;
    kill.active = false;
    if let Some((born, count, victim)) = &recent.kill
        && now - born < f64::from(source.low.kill_seconds)
    {
        kill.active = true;
        kill.count = *count;
        set(&mut kill.victim_icon, victim);
        let words = &source.words.kills;
        let at = (*count as usize).clamp(1, words.len()) - 1;
        set(&mut kill.label, &words[at]);
    }
    facts.damage.clear();
    let fade = marks.damage.fade_seconds;
    let (Some(wish), Some(body)) = (wish, body) else {
        return;
    };
    for (born, from) in &recent.damage {
        let age = (now - born) as f32;
        if age >= fade {
            continue;
        }
        facts.damage.push(DamageMark {
            angle_radians: turned(bearing(*from, body.feet) - wish.yaw_radians),
            age,
        });
    }
}

fn moments_part(facts: &mut HudFacts, recent: &Recent, lows: &super::Lows, now: f64) {
    facts.round.go = !facts.round.buying
        && recent
            .went_live
            .is_some_and(|born| now - born < f64::from(lows.go_seconds));
    let point = &mut facts.point;
    point.active = false;
    if let Some((born, reason)) = &recent.point
        && now - born < f64::from(lows.point_seconds)
    {
        point.active = true;
        set(&mut point.reason, reason);
    }
}

pub(super) fn spot(feet: Vec3) -> [f32; 2] {
    [feet.x, -feet.z]
}

fn compass(yaw_radians: f32) -> f32 {
    -yaw_radians.to_degrees()
}

type Placed<'a> = (&'a Actor, Option<&'a Health>, Option<&'a Body>);

fn minimap_part(
    facts: &mut HudFacts,
    (view, round): (&View<'_>, &Round),
    (slot, wish, body): (usize, Option<&Wish>, Option<&Body>),
    actors: &Query<Placed<'_>>,
) {
    let team = view.players.get(slot).map(|player| player.team);
    let tone = team.map(|team| live::tone(view.side_of(team)).key());
    let minimap = &mut facts.minimap;
    minimap.me.clear();
    if let (Some(wish), Some(body)) = (wish, body) {
        minimap.me.push(MapItem {
            pos: spot(body.feet),
            yaw_degrees: compass(wish.yaw_radians),
            ..MapItem::default()
        });
    }
    minimap.players.clear();
    let mut carrier = None;
    for (actor, _, placed) in actors {
        let Some(placed) = placed else {
            continue;
        };
        let at = actor.slot as usize;
        if matches!(round.device, DeviceSpot::Carried { slot: held, .. } if held == at) {
            carrier = Some((at, placed.feet));
        }
        let mate = view
            .players
            .get(at)
            .filter(|player| Some(player.team) == team && player.alive);
        if at != slot && mate.is_some() {
            minimap.players.push(MapItem {
                pos: spot(placed.feet),
                yaw_degrees: compass(actor.yaw_radians),
                color: tone,
                ..MapItem::default()
            });
        }
    }
    minimap.device.clear();
    minimap.planted.clear();
    if let DeviceSpot::Planted { at, .. } = round.device {
        minimap.planted.push(MapItem {
            pos: spot(at),
            ..MapItem::default()
        });
        return;
    }
    let seen = round.device.at().or_else(|| {
        carrier
            .filter(|(at, _)| view.players.get(*at).map(|player| player.team) == team)
            .map(|(_, feet)| feet)
    });
    if let Some(at) = seen {
        minimap.device.push(MapItem {
            pos: spot(at),
            ..MapItem::default()
        });
    }
}

fn recent_part(facts: &mut HudFacts, recent: &Recent, source: &ScreenSource, now: f64) {
    let keep = f64::from(source.feed.seconds);
    let fresh = recent
        .feed
        .iter()
        .rev()
        .filter(|(born, _, _)| now - born < keep);
    let count = fresh.clone().count();
    facts.feed.resize_with(count, FeedLine::default);
    for (shown, (_, _, line)) in facts.feed.iter_mut().zip(fresh) {
        shown.clone_from(line);
    }
    let result = match facts.phase {
        Some(Phase::Over) => recent.match_result.as_ref(),
        Some(Phase::Buy | Phase::Live | Phase::Planted | Phase::RoundEnd) | None => {
            recent.round_result.as_ref()
        }
    };
    match result {
        Some(result) => facts.result.clone_from(result),
        None => {
            let shown = &mut facts.result;
            for text in [
                &mut shown.title,
                &mut shown.reason,
                &mut shown.mvp,
                &mut shown.mvp_icon,
            ] {
                text.clear();
            }
        }
    }
}

type Reads<'w, 's> = (
    Option<Res<'w, Round>>,
    Option<Res<'w, Numbers>>,
    Res<'w, Views>,
    Res<'w, FrameTime>,
    (DevSetting<'w, ScreenSetup>, Option<Res<'w, Watched>>),
);

fn watched_slot(watched: Option<&Watched>) -> Option<u32> {
    watched
        .filter(|watched| watched.watch == Watch::Teammate)
        .and_then(|watched| watched.slot)
}

fn vitals_part(
    facts: &mut HudFacts,
    view: &View<'_>,
    (player, health): (&Player, Option<&Health>),
    lows: &super::Lows,
) {
    let me = &mut facts.player;
    me.health = health.map_or(0.0, |health| health.health);
    me.armor = health.map_or(0.0, |health| health.armour);
    me.health_low = player.alive && me.health <= lows.health;
    let entry = view.catalog.characters.get(player.character);
    set(
        &mut me.char_name,
        entry.map_or("", |entry| entry.name.as_str()),
    );
    set(&mut me.role, entry.map_or("", |entry| entry.role.as_str()));
    named(
        &mut me.icon,
        "char_",
        entry.map_or("", |entry| entry.key.as_str()),
    );
    let tier = player.armor.map(|armor| armor.tier);
    set(
        &mut me.armor_icon,
        match tier {
            Some(crate::rules::ArmorTier::Light) => "armor_light",
            Some(crate::rules::ArmorTier::Heavy) | None => "armor_heavy",
        },
    );
}

fn poor_part(facts: &mut HudFacts, poor: bool, loaded: &Loaded) {
    let shown = facts.buy.weapons.iter_mut().zip(&loaded.demo.buy.weapons);
    for (index, (card, base)) in shown.enumerate() {
        let short = poor && !base.owned;
        card.afford = base.afford && !short;
        card.need = if short {
            DEMO_POOR_NEED[index % DEMO_POOR_NEED.len()]
        } else {
            base.need
        };
        card.ask = short && index + 1 < DEMO_POOR_NEED.len();
        card.asked = short && index + 1 == DEMO_POOR_NEED.len();
    }
}

fn demo_part(
    facts: &mut HudFacts,
    setup: &ScreenSetup,
    (source, loaded): (&ScreenSource, &Loaded),
) {
    facts.blind = setup.demo_blind;
    facts.spectate.active = setup.demo_spectate;
    let orb = &mut facts.orb;
    orb.channel = setup.demo_channel;
    orb.hint = !setup.demo_channel;
    orb.progress = if setup.demo_channel {
        DEMO_CHANNEL
    } else {
        0.0
    };
    let words = &source.words;
    if setup.show == Some(Shown::MatchEnd) {
        worded(
            &mut facts.result.title,
            &words.victory,
            &[("us", DEMO_SCORE[0]), ("them", DEMO_SCORE[1])],
        );
    } else {
        set(&mut facts.result.title, &words.round_won);
    }
    poor_part(facts, setup.demo_poor, loaded);
    facts.scope_ease = setup.demo_scope;
    facts.scope = u32::from(setup.demo_scope > 0.0);
    facts.armor_break = setup.demo_armor.then_some(0.1);
    let kill = &mut facts.kill;
    kill.active = setup.demo_kills > 0;
    kill.count = setup.demo_kills;
    let words = &source.words.kills;
    let at = (setup.demo_kills as usize).clamp(1, words.len().max(1)) - 1;
    set(&mut kill.label, words.get(at).map_or("", String::as_str));
}

pub(super) fn fill(
    (round, numbers, views, time, (setup, watched)): Reads,
    humans: Query<HumanRow, With<Human>>,
    rows: Query<HumanRow>,
    actors: Query<Placed>,
    (sources, load): (Query<&ScreenSource>, Res<ScreenLoad>),
    mut recent: ResMut<Recent>,
    mut facts: ResMut<HudFacts>,
) {
    let Some(source) = sources.iter().next() else {
        return;
    };
    let Some(loaded) = load.loaded() else {
        return;
    };
    let view = round.as_deref().and_then(Round::view);
    let human = humans.iter().next();
    let setup = setup.get();
    let demo = setup.is_some_and(|setup| setup.demo);
    let (Some(view), Some((actor, health, arms, body, wish)), false) = (view, human, demo) else {
        if !recent.demo_shown {
            facts.clone_from(&loaded.demo);
            recent.demo_shown = true;
        }
        if let Some(setup) = setup {
            demo_part(&mut facts, setup, (source, loaded));
        }
        return;
    };
    recent.demo_shown = false;
    let slot = actor.slot as usize;
    let Some(player) = view.players.get(slot) else {
        return;
    };
    let recent = &mut *recent;
    recent.healths.clear();
    recent.healths.resize(view.players.len(), None);
    for (actor, health, _) in &actors {
        if let (Some(held), Some(health)) = (recent.healths.get_mut(actor.slot as usize), health) {
            *held = Some(health.health);
        }
    }
    let facts = &mut *facts;
    let (lows, words) = (&source.low, &source.words);
    facts.live = true;
    live::round_part(facts, &view, player.team, (lows, words));
    live::plates_part(facts, &view, player.team, &recent.healths);
    let seen = watched_slot(watched.as_deref()).and_then(|watched| {
        let row = rows.iter().find(|row| row.0.slot == watched)?;
        Some((view.players.get(watched as usize)?, row))
    });
    let (seen_player, (_, seen_health, seen_arms, seen_body, _)) =
        seen.unwrap_or((player, (actor, health, arms, body, wish)));
    live::abilities_part(facts, &view, (seen_player, player));
    live::action_part(facts, &view, slot, words);
    live::device_part(facts, &view, slot);
    let weapons = numbers
        .as_deref()
        .and_then(Numbers::get)
        .map(|numbers| &numbers.weapons);
    if facts.phase == Some(Phase::Buy) || facts.screen == Some(Shown::Buy) {
        live::buy_part(
            facts,
            &view,
            (slot, player),
            (lows, words),
            (weapons, &mut recent.order),
        );
    }
    vitals_part(facts, &view, (seen_player, seen_health), lows);
    facts.player.credits = player.credits;
    if let Some(numbers) = numbers.as_deref() {
        weapon_part(
            facts,
            (seen_arms, seen_body),
            numbers,
            &views,
            lows.magazine_share,
        );
    }
    if let Some(round) = round.as_deref() {
        minimap_part(facts, (&view, round), (slot, wish, body), &actors);
    }
    marks_part(facts, recent, (wish, body), (source, time.elapsed));
    moments_part(facts, recent, lows, time.elapsed);
    let round = &mut facts.round;
    round.ready = view.ready(slot);
    round.ready_left = if round.ready { round.time } else { 0.0 };
    round.ready_count = round.ready_left.ceil() as u32;
    recent_part(facts, recent, source, time.elapsed);
}

pub(super) fn fill_board(
    round: Option<Res<Round>>,
    humans: Query<&Actor, With<Human>>,
    (sources, recent): (Query<&ScreenSource>, Res<Recent>),
    mut facts: ResMut<HudFacts>,
) {
    if recent.demo_shown || !matches!(facts.screen, Some(Shown::Scoreboard | Shown::MatchEnd)) {
        return;
    }
    let (Some(source), Some(view), Some(actor)) = (
        sources.iter().next(),
        round.as_deref().and_then(Round::view),
        humans.iter().next(),
    ) else {
        return;
    };
    let slot = actor.slot as usize;
    let Some(player) = view.players.get(slot) else {
        return;
    };
    live::board_part(
        &mut facts,
        &view,
        (slot, player.team, &recent.dealt),
        &source.words,
    );
}
