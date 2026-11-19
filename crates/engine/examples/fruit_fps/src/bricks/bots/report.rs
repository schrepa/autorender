use std::fmt::Debug;

use engine::prelude::*;

use super::{Book, BotStat, BotsStats, BotsTest, SquadStat, ThrowStat, WishStat};
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

pub(super) fn report(
    minds: Res<BotMinds>,
    paths: Res<Paths>,
    book: Res<Book>,
    (test, mut stats): (DevSetting<BotsTest>, Stats<BotsStats>),
) {
    let places = book.places.as_deref();
    let (queries, expanded) = paths.counts();
    let mut throws = ThrowStat::default();
    for bot in &minds.bots {
        throws.solved += bot.throws.solved;
        throws.refused += bot.throws.refused;
        throws.thrown += bot.throws.thrown;
    }
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
            line_up: bot
                .line_up
                .zip(places)
                .map(|((line, _), places)| line.name(places)),
            line_up_state: bot.line_up.map(|(_, state)| state.key().to_owned()),
        })
        .collect();
    stats.show(|| BotsStats {
        problem: minds.problem.clone(),
        tick: minds.tick,
        frozen: test.get().is_some_and(|test| test.frozen),
        queries,
        expanded,
        throws,
        drops: minds.drops,
        gifts: minds.gifts,
        buy_calls: minds.buy_calls.map(|call| call.map(str::to_owned)),
        squads,
        bots,
    });
}
