use std::fmt::Write;

use engine::math;
use engine::prelude::*;
use engine::{FrameTime, Projection, Views};

use crate::play::actors::{Actor, Body, Health, Human, Motion, Wish};
use crate::play::arms::Arms;
use crate::play::events::arms::ArmsEvents;
use crate::play::events::round::RoundEvents;
use crate::play::numbers::Numbers;
use crate::play::round::{DeviceSpot, Round};
use crate::play::screen::{DamageMark, FeedLine, HitMark, HudFacts, MapItem, ResultFacts};
use crate::rules::{AbilitySlot, Cause, Event, Phase, RoundEndReason, Team, View, WeaponId};
use crate::weapons::Stance;

use super::live::{self, character_key, character_name, named, set, tone, worded};
use super::{ScreenSetup, ScreenSource, Words};

#[derive(Resource, Default)]
pub(super) struct Recent {
    feed: Vec<(f64, FeedLine)>,
    round_kills: Vec<u32>,
    dealt: Vec<f32>,
    round_result: Option<ResultFacts>,
    match_result: Option<ResultFacts>,
    hit: Option<(f64, HitMark)>,
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
            let index = match slot {
                AbilitySlot::C => 0,
                AbilitySlot::Q => 1,
                AbilitySlot::E => 2,
                AbilitySlot::X => 3,
            };
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
        line.killer_color = tone(view.side_of(player.team));
    }
    if let Some(player) = view.players.get(victim) {
        set(&mut line.victim, character_name(catalog, player));
        line.victim_color = tone(view.side_of(player.team));
    }
    cause_icon(&mut line, view, killer, cause);
    line
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
    let won = human_team == Some(winner);
    set(
        &mut result.title,
        if won { &words.victory } else { &words.defeat },
    );
    let best = most(view.players.iter().map(|player| player.kills).enumerate());
    mvp_into(&mut result, view, best, (words, dealt));
    result
}

type Sources<'w, 's> = (
    Option<Res<'w, Round>>,
    Option<Res<'w, RoundEvents>>,
    Option<Res<'w, ArmsEvents>>,
    Query<'w, 's, &'static ScreenSource>,
    Res<'w, FrameTime>,
);

pub(super) fn collect(
    (round, events, arms, sources, time): Sources,
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
    let human = humans.iter().next().map(|actor| actor.slot as usize);
    let human_team = human
        .and_then(|slot| view.players.get(slot))
        .map(|player| player.team);
    recent.round_kills.resize(view.players.len(), 0);
    recent.dealt.resize(view.players.len(), 0.0);
    if let Some(events) = events.filter(|events| recent.round_tick != Some(events.tick)) {
        recent.round_tick = Some(events.tick);
        for event in &events.events {
            if let Event::KillFeed {
                killer,
                victim,
                headshot,
                cause,
                ..
            } = event
            {
                recent.feed.push((
                    now,
                    feed_line(&view, (*killer, *victim), (*headshot, *cause), human),
                ));
                if let Some(kills) = killer.and_then(|killer| recent.round_kills.get_mut(killer)) {
                    *kills += 1;
                }
            } else if let Event::Phase {
                phase: Phase::Buy,
                round,
                ..
            } = event
            {
                if *round <= 1 {
                    recent.dealt.fill(0.0);
                }
                recent.round_kills.fill(0);
                recent.round_result = None;
                recent.match_result = None;
            } else if let Event::RoundEnd { winner, reason, .. } = event {
                let result = round_result(
                    &view,
                    (*winner, *reason),
                    (human_team, round.device),
                    &source.words,
                    &recent.round_kills,
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
    }
    let lines = source.feed.lines;
    if recent.feed.len() > lines {
        let extra = recent.feed.len() - lines;
        recent.feed.drain(..extra);
    }
    let Some(arms) = arms.filter(|arms| recent.arms_tick != Some(arms.tick)) else {
        return;
    };
    recent.arms_tick = Some(arms.tick);
    for hit in &arms.hits {
        if let Some(dealt) = recent.dealt.get_mut(hit.shooter as usize) {
            *dealt += hit.health_lost + hit.armor_lost;
        }
        if Some(hit.shooter as usize) == human {
            recent.hit = Some((
                now,
                HitMark {
                    age: 0.0,
                    headshot: hit.headshot,
                    killed: hit.killed,
                },
            ));
        }
        if Some(hit.victim as usize) == human
            && let Some((_, body)) = bodies.iter().find(|(actor, _)| actor.slot == hit.shooter)
        {
            recent.damage.push((now, body.feet));
        }
    }
    let keep = f64::from(source.marks.damage.seconds);
    recent.damage.retain(|(born, _)| now - born < keep);
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
    facts.hit = recent.hit.and_then(|(born, hit)| {
        let age = (now - born) as f32;
        (age < marks.hit.seconds).then_some(HitMark { age, ..hit })
    });
    facts.damage.clear();
    let (Some(wish), Some(body)) = (wish, body) else {
        return;
    };
    for (born, from) in &recent.damage {
        let age = (now - born) as f32;
        if age >= marks.damage.seconds {
            continue;
        }
        let toward = *from - body.feet;
        let bearing = math::atan2(-toward.x, -toward.z);
        let mut angle = bearing - wish.yaw_radians;
        let turn = std::f32::consts::TAU;
        angle -= (angle / turn).round() * turn;
        facts.damage.push(DamageMark {
            angle_radians: angle,
            age,
        });
    }
}

fn spot(feet: Vec3) -> [f32; 2] {
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
    let tone = team.map(|team| live::tone(view.side_of(team)));
    let minimap = &mut facts.minimap;
    minimap.me.clear();
    if let (Some(wish), Some(body)) = (wish, body) {
        minimap.me.push(MapItem {
            pos: spot(body.feet),
            yaw_degrees: compass(wish.yaw_radians),
            color: None,
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
            });
        }
    }
    minimap.device.clear();
    let seen = round.device.at().or_else(|| {
        carrier
            .filter(|(at, _)| view.players.get(*at).map(|player| player.team) == team)
            .map(|(_, feet)| feet)
    });
    if let Some(at) = seen {
        minimap.device.push(MapItem {
            pos: spot(at),
            yaw_degrees: 0.0,
            color: None,
        });
    }
}

fn recent_part(facts: &mut HudFacts, recent: &Recent, source: &ScreenSource, now: f64) {
    let keep = f64::from(source.feed.seconds);
    let fresh = recent
        .feed
        .iter()
        .rev()
        .filter(|(born, _)| now - born < keep);
    let count = fresh.clone().count();
    facts.feed.resize_with(count, FeedLine::default);
    for (shown, (_, line)) in facts.feed.iter_mut().zip(fresh) {
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
    Res<'w, ScreenSetup>,
);

pub(super) fn fill(
    (round, numbers, views, time, setup): Reads,
    humans: Query<HumanRow, With<Human>>,
    actors: Query<Placed>,
    sources: Query<&ScreenSource>,
    mut recent: ResMut<Recent>,
    mut facts: ResMut<HudFacts>,
) {
    let Some(source) = sources.iter().next() else {
        return;
    };
    let Some(loaded) = source.loaded() else {
        return;
    };
    let view = round.as_deref().and_then(Round::view);
    let human = humans.iter().next();
    let (Some(view), Some((actor, health, arms, body, wish)), false) = (view, human, setup.demo)
    else {
        if !recent.demo_shown {
            facts.clone_from(&loaded.demo);
            recent.demo_shown = true;
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
    live::abilities_part(facts, &view, player);
    live::action_part(facts, &view, slot, words);
    live::device_part(facts, &view, slot);
    let weapons = numbers
        .as_deref()
        .and_then(Numbers::get)
        .map(|numbers| &numbers.weapons);
    live::buy_part(
        facts,
        &view,
        (slot, player),
        (lows, words),
        (weapons, &mut recent.order),
    );
    live::board_part(facts, &view, (slot, player.team, &recent.dealt), words);
    let me = &mut facts.player;
    me.health = health.map_or(0.0, |health| health.health);
    me.armor = health.map_or(0.0, |health| health.armour);
    me.health_low = player.alive && me.health <= lows.health;
    me.credits = player.credits;
    let entry = view.catalog.characters.get(player.character);
    set(
        &mut me.char_name,
        entry.map_or("", |entry| entry.name.as_str()),
    );
    set(&mut me.role, entry.map_or("", |entry| entry.role.as_str()));
    let tier = player.armor.map(|armor| armor.tier);
    set(
        &mut me.armor_icon,
        match tier {
            Some(crate::rules::ArmorTier::Light) => "armor_light",
            Some(crate::rules::ArmorTier::Heavy) | None => "armor_heavy",
        },
    );
    if let Some(numbers) = numbers.as_deref() {
        weapon_part(facts, (arms, body), numbers, &views, lows.magazine_share);
    }
    if let Some(round) = round.as_deref() {
        minimap_part(facts, (&view, round), (slot, wish, body), &actors);
    }
    marks_part(facts, recent, (wish, body), (source, time.elapsed));
    recent_part(facts, recent, source, time.elapsed);
}
