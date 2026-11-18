use std::fmt::Debug;

use engine::prelude::*;

use super::{BotStat, BotsStats, SquadStat, WishStat};
use crate::play::bots::BotMinds;
use crate::play::paths::Paths;

fn name(value: impl Debug) -> String {
    let text = format!("{value:?}");
    let mut out = String::with_capacity(text.len() + 4);
    for (index, letter) in text.chars().enumerate() {
        if letter.is_ascii_uppercase() {
            if index > 0 {
                out.push('_');
            }
            out.push(letter.to_ascii_lowercase());
        } else {
            out.push(letter);
        }
    }
    out
}

pub(super) fn report(minds: Res<BotMinds>, paths: Res<Paths>, mut stats: ResMut<BotsStats>) {
    let squads = minds
        .squads
        .iter()
        .enumerate()
        .filter_map(|(team, squad)| {
            squad.map(|squad| SquadStat {
                team,
                approach: name(squad.approach),
                site: squad.site,
                step: name(squad.step),
                why: name(squad.why),
                since_tick: squad.since_tick,
            })
        })
        .collect();
    let bots = minds
        .bots
        .iter()
        .map(|bot| BotStat {
            slot: bot.slot,
            level: name(bot.mind.level),
            duty: name(bot.mind.duty),
            step: name(bot.mind.step),
            activity: name(bot.mind.activity),
            goal: bot.goal.to_array(),
            goal_place: paths.place_at(bot.goal).map(|place| place.name.clone()),
            target: bot.mind.target,
            why: name(bot.mind.why),
            since_tick: bot.mind.since_tick,
            wish: WishStat {
                direction: [bot.wish.direction.x, bot.wish.direction.z],
                walk: bot.wish.walk,
                fire: bot.wish.fire,
                yaw: bot.wish.yaw_radians,
                pitch: bot.wish.pitch_radians,
            },
        })
        .collect();
    stats.set_if_neq(BotsStats {
        problem: minds.problem.clone(),
        tick: minds.tick,
        frozen: minds.frozen,
        queries: minds.queries,
        expanded: minds.expanded,
        squads,
        bots,
    });
}
