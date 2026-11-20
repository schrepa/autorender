use engine::prelude::*;
use engine_core::DevSetting;
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
pub enum By {
    Elimination,
    Time,
    Device,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EndRound {
    pub winner: Side,
    pub by: By,
}

settings! {
    #[dev]
    #[section("match_test")]
    #[serde(deny_unknown_fields)]
    pub struct MatchTest {
        pub skip_buy: bool = false => any,
        pub credits: Option<Credits> = None => any,
        pub end_round: Option<EndRound> = None => any,
        pub ready: Option<u32> = None => any,
        pub hold: Option<bool> = None => any,
    }
}

pub(super) fn door(test: DevSetting<MatchTest>, mut round: ResMut<Round>) {
    let Some(test) = test.changed() else {
        return;
    };
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
        let side = end.winner;
        let reason = match (end.by, side) {
            (By::Elimination, _) => RoundEndReason::Elimination,
            (By::Time, _) => RoundEndReason::Time,
            (By::Device, Side::Attack) => RoundEndReason::Exploded,
            (By::Device, Side::Defend) => RoundEndReason::Defused,
        };
        game.end_round_now(side, reason);
    }
}
