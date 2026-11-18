use engine::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::play::round::Round;
use crate::rules::{RoundEndReason, Side};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Credits {
    pub slot: u32,
    pub amount: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Winner {
    Attack,
    Defend,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum By {
    Elimination,
    Time,
    Device,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EndRound {
    pub winner: Winner,
    pub by: By,
}

settings! {
    #[section("match_test")]
    #[serde(deny_unknown_fields)]
    pub struct MatchTest {
        pub credits: Option<Credits> = None => any,
        pub end_round: Option<EndRound> = None => any,
        pub ready: Option<u32> = None => any,
        pub hold: Option<bool> = None => any,
    }
}

pub(super) fn door(test: Res<MatchTest>, mut round: ResMut<Round>) {
    if !test.is_changed() {
        return;
    }
    let Some(game) = round.game.as_mut() else {
        return;
    };
    if let Some(credits) = test.credits {
        game.grant_credits(credits.slot as usize, credits.amount);
    }
    if let Some(held) = test.hold {
        game.hold(held);
    }
    if let Some(slot) = test.ready {
        game.mark_ready(slot as usize);
    }
    if let Some(end) = test.end_round {
        let side = match end.winner {
            Winner::Attack => Side::Attack,
            Winner::Defend => Side::Defend,
        };
        let reason = match (end.by, side) {
            (By::Elimination, _) => RoundEndReason::Elimination,
            (By::Time, _) => RoundEndReason::Time,
            (By::Device, Side::Attack) => RoundEndReason::Exploded,
            (By::Device, Side::Defend) => RoundEndReason::Defused,
        };
        game.end_round_now(side, reason);
    }
}
