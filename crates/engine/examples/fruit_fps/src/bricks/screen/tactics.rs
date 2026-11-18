use engine::prelude::*;
use engine::{FixedTime, FrameTime};

use crate::abilities::{AbilityId, DrawKind, DrawShape, Drawable, Modifiers, OrbState};
use crate::play::actors::{Actor, Body, Human};
use crate::play::powers::Powers;
use crate::play::round::Round;
use crate::play::screen::{HudFacts, MapItem, StatusFacts};
use crate::play::view::{Watch, Watched};
use crate::rules::{Catalog, Team, View};

use super::ScreenSource;
use super::facts::{Hold, Mark, Recent, compass, spot};
use super::live::{named, set, tone};

const SLOWED: &str = "slowed";
const REVEALED: &str = "revealed";
const SIGHT: &str = "sight";
const VULNERABLE: &str = "vulnerable";
const BOOSTED: &str = "boosted";

type Reads<'w, 's> = (
    Option<Res<'w, Round>>,
    Option<Res<'w, Powers>>,
    Option<Res<'w, Watched>>,
    Res<'w, Recent>,
    (Res<'w, FixedTime>, Res<'w, FrameTime>),
    Query<'w, 's, &'static ScreenSource>,
);

#[derive(Default)]
pub(super) struct Lost {
    seen: Vec<(u32, MapItem, f64)>,
}

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

fn status_kind(hold: Hold, enemy: usize) -> Option<&'static str> {
    match hold {
        Hold::Reveal(team) if team == enemy => Some(REVEALED),
        Hold::Reveal(_) => None,
        Hold::Buff(_, modifiers) => buff_kind(modifiers),
    }
}

fn buff_kind(modifiers: Modifiers) -> Option<&'static str> {
    if modifiers.sight_m.is_some() {
        Some(SIGHT)
    } else if modifiers.speed < 1.0 {
        Some(SLOWED)
    } else if modifiers.damage_taken > 1.0 {
        Some(VULNERABLE)
    } else if modifiers.fire_rate > 1.0 || modifiers.reload < 1.0 {
        Some(BOOSTED)
    } else {
        None
    }
}

fn statuses_part(
    facts: &mut HudFacts,
    (marks, source): (&[Mark], &ScreenSource),
    (view, slot, team): (&View<'_>, u32, Team),
    (tick, step): (u64, f32),
) {
    let enemy = team.other().index();
    let words = &source.words;
    let mut count = 0;
    for kind in [SLOWED, SIGHT, VULNERABLE, REVEALED, BOOSTED] {
        let longest = marks
            .iter()
            .filter(|mark| mark.actor == slot && status_kind(mark.hold, enemy) == Some(kind))
            .max_by_key(|mark| mark.until);
        let Some(mark) = longest else {
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
            _ => &words.status_boosted,
        };
        set(&mut shown.label, label);
        ability_icon(&mut shown.icon, view.catalog, mark.ability);
        shown.left = mark.until.saturating_sub(tick) as f32 * step;
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
            _ => {}
        }
    }
}

fn enemies_part(
    facts: &mut HudFacts,
    (powers, view, team): (&Powers, &View<'_>, Team),
    bodies: &Query<(&Actor, &Body)>,
    (lost, now, keep): (&mut Lost, f64, f64),
) {
    let color = Some(tone(view.side_of(team.other())));
    for (actor, body) in bodies {
        let foe = view
            .players
            .get(actor.slot as usize)
            .filter(|player| player.team != team && player.alive);
        if foe.is_none() || !powers.revealed(actor.slot, team) {
            continue;
        }
        let item = MapItem {
            pos: spot(body.feet),
            yaw_degrees: compass(actor.yaw_radians),
            color,
            ..MapItem::default()
        };
        lost.seen.retain(|(slot, _, _)| *slot != actor.slot);
        lost.seen.push((actor.slot, item, now));
    }
    lost.seen.retain(|(slot, _, born)| {
        now - born < keep
            && view
                .players
                .get(*slot as usize)
                .is_some_and(|player| player.alive)
    });
    let enemies = &mut facts.minimap.enemies;
    enemies.clear();
    for (_, item, born) in &lost.seen {
        let mut shown = item.clone();
        shown.fade = ((now - born) / keep) as f32;
        enemies.push(shown);
    }
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
    mut lost: Local<Lost>,
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
    statuses_part(
        facts,
        (&recent.marks, source),
        (&view, actor.slot, team),
        (tick, fixed.delta_seconds),
    );
    utility_part(facts, powers, (view.catalog, team));
    let keep = f64::from(source.low.lost_seconds);
    enemies_part(
        facts,
        (powers, &view, team),
        &bodies,
        (&mut lost, frame.elapsed, keep),
    );
}
