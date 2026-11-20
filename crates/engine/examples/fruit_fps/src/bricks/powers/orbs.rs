use engine_core::Assets;
use glam::Vec3;

use super::{Book, OrbChannelStat, OrbEventStat, OrbStat, PowerStats, RECENT_ORBS};
use crate::abilities::{ActorId, OrbEvent, OrbNumbers, OrbState, Orbs, Stop, Taker};
use crate::play::layout::Layout;
use crate::play::paths::Paths;
use crate::play::powers::Powers;
use crate::play::round::Round;
use crate::play::world::Arena;
use crate::rules::{Phase, PointReason};

const FLOOR_SEARCH_FROM_M: f32 = 3.0;
const FLOOR_SEARCH_DEPTH_M: f32 = 10.0;

fn spots(numbers: &OrbNumbers, layout: &Layout, arena: &Arena) -> Result<Vec<(String, Vec3)>, String> {
    numbers
        .spots
        .iter()
        .map(|name| {
            let callout = (layout.map.callouts)
                .iter()
                .find(|callout| &callout.name == name)
                .ok_or_else(|| format!("orb spot `{name}` is not a callout of the map"))?;
            let above = Vec3::new(callout.pos[0], FLOOR_SEARCH_FROM_M, -callout.pos[1]);
            let floor = arena
                .ray(above, Vec3::NEG_Y, FLOOR_SEARCH_DEPTH_M)
                .ok_or_else(|| format!("orb spot `{name}` has no floor under it"))?;
            Ok((name.clone(), above + Vec3::NEG_Y * floor.distance))
        })
        .collect()
}

pub(super) fn placed(
    book: &mut Book,
    powers: &mut Powers,
    (paths, (assets_changed, assets)): (Option<&Paths>, (bool, &Assets)),
    arena: &Arena,
) {
    if powers.orbs.rules.is_some() {
        return;
    }
    let (Some(numbers), Some(layout)) = (book.orbs.as_ref(), paths.and_then(Paths::layout)) else {
        return;
    };
    let tried = (book.revision, arena.revision(), layout);
    if book.orb_tried == Some(tried) && !assets_changed {
        return;
    }
    book.orb_tried = Some(tried);
    let found = (assets.get(layout))
        .map_err(|error| format!("map callouts: {error}"))
        .and_then(|layout| spots(numbers, layout, arena));
    match found {
        Ok(found) => {
            powers.orbs.place(numbers.rules, found);
            book.orb_problem = None;
        }
        Err(problem) => book.orb_problem = Some(problem),
    }
}

pub(super) fn step(
    orbs: &mut Orbs,
    round: Option<&mut Round>,
    takers: &[Taker],
    (restarted, tick): (bool, u64),
) -> Vec<OrbEvent> {
    let mut events = if restarted {
        orbs.restore()
    } else {
        Vec::new()
    };
    let Some(round) = round else {
        return events;
    };
    match round.phase() {
        Some(Phase::Buy | Phase::Live) => events.extend(orbs.tick(takers, tick)),
        Some(Phase::Planted | Phase::RoundEnd | Phase::Over) => events.extend(orbs.vanish()),
        None => {}
    }
    if let Some(game) = round.game.as_mut() {
        for event in &events {
            if let OrbEvent::Taken {
                actor: ActorId(slot),
                ..
            } = event
            {
                game.grant_ultimate_point(*slot as usize, PointReason::Orb);
            }
        }
    }
    events
}

pub(super) fn remember(book: &mut Book, events: &[OrbEvent], tick: u64) {
    for event in events {
        let (orb, slot, change) = match *event {
            OrbEvent::Started { orb, actor } => (orb, Some(actor.0), "started"),
            OrbEvent::Stopped { orb, actor, why } => (orb, Some(actor.0), stop_key(why)),
            OrbEvent::Taken { orb, actor } => (orb, Some(actor.0), "taken"),
            OrbEvent::Vanished { orb } => (orb, None, "vanished"),
            OrbEvent::Restored { orb } => (orb, None, "restored"),
        };
        book.recent_orbs.push(OrbEventStat {
            tick,
            orb: orb as u32,
            slot,
            change: change.to_owned(),
        });
    }
    let extra = book.recent_orbs.len().saturating_sub(RECENT_ORBS);
    book.recent_orbs.drain(..extra);
}

fn stop_key(why: Stop) -> &'static str {
    match why {
        Stop::Released => "released",
        Stop::Moved => "moved",
        Stop::Died => "died",
        Stop::TakenByOther => "taken_by_other",
        Stop::Gone => "gone",
    }
}

pub(super) fn write_stats(stats: &mut PowerStats, powers: &Powers, book: &Book, tick: u64) {
    stats.orbs = orb_stats(powers, tick);
    stats.orb_problem.clone_from(&book.orb_problem);
    stats.recent_orbs.clone_from(&book.recent_orbs);
}

fn orb_stats(powers: &Powers, tick: u64) -> Vec<OrbStat> {
    powers
        .orbs()
        .iter()
        .map(|orb| {
            let (state, by) = match orb.state {
                OrbState::Waiting => ("waiting", None),
                OrbState::Taken { by, .. } => ("taken", Some(by.0)),
                OrbState::Gone => ("gone", None),
            };
            OrbStat {
                name: orb.name.clone(),
                at: orb.at.to_array(),
                state: state.to_owned(),
                by,
                channels: orb
                    .channels
                    .iter()
                    .map(|channel| OrbChannelStat {
                        slot: channel.actor.0,
                        progress: powers
                            .orb_progress(channel.actor.0, tick)
                            .map_or(0.0, |(_, share)| share),
                    })
                    .collect(),
            }
        })
        .collect()
}
