use engine::prelude::*;
use engine::{FixedTime, FrameTime};

use crate::abilities::{AbilityId, DrawKind, DrawShape, Drawable, OrbState, StatusKind};
use crate::play::actors::{Actor, Body, Human};
use crate::play::bots::{BotMinds, TeamSight};
use crate::play::powers::Powers;
use crate::play::round::Round;
use crate::play::screen::{HudFacts, MapItem, StatusFacts};
use crate::play::view::{Watch, Watched};
use crate::rules::{Catalog, Team, View};

use super::ScreenSource;
use super::facts::{Recent, spot};
use super::live::{character_key, character_name, named, set, tone};

const SLOWED: &str = "slowed";
const REVEALED: &str = "revealed";
const SIGHT: &str = "sight";
const VULNERABLE: &str = "vulnerable";
const BOOSTED: &str = "boosted";
const CHAINED: &str = "chained";
const LOST_RADIUS_M: f32 = 1.6;

type Reads<'w, 's> = (
    Option<Res<'w, Round>>,
    Option<Res<'w, Powers>>,
    Option<Res<'w, Watched>>,
    Res<'w, Recent>,
    (Res<'w, FixedTime>, Res<'w, FrameTime>),
    Query<'w, 's, &'static ScreenSource>,
);

fn ability_icon(out: &mut String, catalog: &Catalog, ability: AbilityId) {
    match catalog
        .characters
        .get(ability.character)
        .and_then(|entry| entry.ability_names.get(ability.slot.index()))
    {
        Some(name) => named(out, "ability_", name),
        None => out.clear(),
    }
}

const KINDS: [(StatusKind, &str); 6] = [
    (StatusKind::Chained, CHAINED),
    (StatusKind::Slowed, SLOWED),
    (StatusKind::SightCut, SIGHT),
    (StatusKind::Vulnerable, VULNERABLE),
    (StatusKind::Revealed, REVEALED),
    (StatusKind::Boosted, BOOSTED),
];

fn statuses_part(
    facts: &mut HudFacts,
    (powers, source): (&Powers, &ScreenSource),
    (slot, team): (u32, Team),
    step: f32,
) {
    let enemy = team.other().index();
    let words = &source.words;
    let held = powers.statuses(slot);
    let mut count = 0;
    for (wanted, kind) in KINDS {
        let longest = held
            .iter()
            .filter(|status| status.kind == wanted)
            .filter(|status| status.kind != StatusKind::Revealed || status.to_team == Some(enemy))
            .max_by_key(|status| status.ticks_left);
        let Some(status) = longest else {
            continue;
        };
        if facts.statuses.len() == count {
            facts.statuses.push(StatusFacts::default());
        }
        let shown = &mut facts.statuses[count];
        set(&mut shown.kind, kind);
        let label = match kind {
            SLOWED => &words.status_slowed,
            SIGHT => &words.status_sight,
            VULNERABLE => &words.status_vulnerable,
            REVEALED => &words.status_revealed,
            CHAINED => &words.status_chained,
            _ => &words.status_boosted,
        };
        set(&mut shown.label, label);
        shown.left = status.ticks_left as f32 * step;
        count += 1;
    }
    facts.statuses.truncate(count);
}

fn drawn_at(drawable: &Drawable) -> (Vec3, f32) {
    match drawable.shape {
        DrawShape::Sphere { centre, radius } => (centre, radius),
        DrawShape::Box { centre, .. } => (centre, 0.0),
        DrawShape::Segment { a, b } => ((a + b) * 0.5, 0.0),
    }
}

fn utility_part(facts: &mut HudFacts, powers: &Powers, (catalog, team): (&Catalog, Team)) {
    let minimap = &mut facts.minimap;
    minimap.smokes.clear();
    minimap.utility.clear();
    for drawable in powers.drawables() {
        let (at, radius) = drawn_at(&drawable);
        match drawable.kind {
            DrawKind::Smoke => minimap.smokes.push(MapItem {
                pos: spot(at),
                radius_m: radius,
                ..MapItem::default()
            }),
            DrawKind::Trap
            | DrawKind::Wire
            | DrawKind::Sensor
            | DrawKind::Wall
            | DrawKind::Curtain
            | DrawKind::Zone
                if drawable.team == team.index() =>
            {
                let mut item = MapItem {
                    pos: spot(at),
                    ..MapItem::default()
                };
                ability_icon(&mut item.icon, catalog, drawable.ability);
                minimap.utility.push(item);
            }
            DrawKind::Projectile
            | DrawKind::Zone
            | DrawKind::Curtain
            | DrawKind::Wall
            | DrawKind::Trap
            | DrawKind::Wire
            | DrawKind::Sensor
            | DrawKind::StrikeMark => {}
        }
    }
}

fn enemies_part(
    facts: &mut HudFacts,
    (sight, view, team): (&TeamSight, &View<'_>, Team),
    bodies: &Query<(&Actor, &Body)>,
    (tick, step, keep): (u64, f32, f32),
) {
    let side = tone(view.side_of(team.other()));
    let index = team.index();
    let alive = |slot: u32| {
        view.players
            .get(slot as usize)
            .filter(|player| player.team != team && player.alive)
    };
    let minimap = &mut facts.minimap;
    minimap.enemies.clear();
    minimap.lost.clear();
    let mut shown: Vec<u32> = Vec::new();
    let seen = sight.seen.get(index).map_or(&[][..], Vec::as_slice);
    for &(enemy, _) in seen {
        if shown.contains(&enemy) {
            continue;
        }
        let (Some(player), Some((_, body))) = (
            alive(enemy),
            bodies.iter().find(|(actor, _)| actor.slot == enemy),
        ) else {
            continue;
        };
        let mut item = MapItem {
            pos: spot(body.feet),
            color: Some(side),
            ..MapItem::default()
        };
        named(&mut item.icon, "char_", character_key(view.catalog, player));
        item.icon.push('_');
        item.icon.push_str(side.key());
        shown.push(enemy);
        minimap.enemies.push(item);
    }
    let last = sight.last.get(index).map_or(&[][..], Vec::as_slice);
    for &(enemy, at_tick, feet) in last {
        let age = tick.saturating_sub(at_tick) as f32 * step;
        if shown.contains(&enemy) || alive(enemy).is_none() || keep <= 0.0 || age >= keep {
            continue;
        }
        minimap.lost.push(MapItem {
            pos: spot(feet),
            color: Some(side),
            fade: age / keep,
            radius_m: LOST_RADIUS_M,
            ..MapItem::default()
        });
    }
}

fn ultimate_part(
    facts: &mut HudFacts,
    (cast, view, team): (Option<(f64, u32)>, &View<'_>, Team),
    (now, keep): (f64, f64),
) {
    let banner = &mut facts.ult;
    let caster = cast
        .filter(|(born, _)| now - born < keep)
        .and_then(|(_, slot)| view.players.get(slot as usize));
    banner.active = caster.is_some();
    let Some(player) = caster else {
        return;
    };
    let catalog = view.catalog;
    let side = view.side_of(player.team);
    banner.ally = player.team == team;
    banner.color = tone(side);
    set(&mut banner.name, character_name(catalog, player));
    named(&mut banner.icon, "char_", character_key(catalog, player));
    banner.icon.push('_');
    banner.icon.push_str(banner.color.key());
    let ultimate = catalog
        .characters
        .get(player.character)
        .map_or("", |entry| entry.ability_names[3].as_str());
    set(&mut banner.ultimate, ultimate);
}

fn orbs_part(facts: &mut HudFacts, powers: &Powers, (slot, tick): (u32, u64)) {
    let orbs = &mut facts.minimap.orbs;
    orbs.clear();
    for orb in powers.orbs() {
        if orb.state == OrbState::Waiting {
            orbs.push(MapItem {
                pos: spot(orb.at),
                ..MapItem::default()
            });
        }
    }
    let progress = powers.orb_progress(slot, tick);
    let shown = &mut facts.orb;
    shown.channel = progress.is_some();
    shown.progress = progress.map_or(0.0, |(_, share)| share);
    shown.hint = progress.is_none() && powers.orb_in_reach(slot).is_some();
}

pub(super) fn fill_tactics(
    (round, powers, watched, recent, (fixed, frame), sources): Reads,
    humans: Query<&Actor, With<Human>>,
    bodies: Query<(&Actor, &Body)>,
    minds: Option<Res<BotMinds>>,
    mut facts: ResMut<HudFacts>,
) {
    if recent.shows_demo() {
        return;
    }
    let (Some(view), Some(powers), Some(actor), Some(source)) = (
        round.as_deref().and_then(Round::view),
        powers.as_deref(),
        humans.iter().next(),
        sources.iter().next(),
    ) else {
        return;
    };
    let Some(team) = view
        .players
        .get(actor.slot as usize)
        .map(|player| player.team)
    else {
        return;
    };
    let facts = &mut *facts;
    let viewer = watched
        .as_deref()
        .filter(|watched| watched.watch == Watch::Teammate)
        .and_then(|watched| watched.slot)
        .unwrap_or(actor.slot);
    let tick = fixed.tick;
    facts.blind = powers.blindness_at(viewer, tick as f64 + f64::from(frame.alpha));
    orbs_part(facts, powers, (actor.slot, tick));
    statuses_part(facts, (powers, source), (viewer, team), fixed.delta_seconds);
    utility_part(facts, powers, (view.catalog, team));
    ultimate_part(
        facts,
        (recent.ultimate, &view, team),
        (frame.elapsed, f64::from(source.low.ult_seconds)),
    );
    let sight = minds.as_deref().map(|minds| &minds.sight);
    if let Some(sight) = sight {
        enemies_part(
            facts,
            (sight, &view, team),
            &bodies,
            (tick, fixed.delta_seconds, source.low.lost_seconds),
        );
    } else {
        facts.minimap.enemies.clear();
        facts.minimap.lost.clear();
    }
}
