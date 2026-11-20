engine::brick!(viewmodel);

mod hands;
mod place;

use engine::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::play::puppet::{Holding, ViewModel};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct HandsStats {
    pub shown: bool,
    pub ability: Option<String>,
    pub phase: Option<String>,
    pub clip: Option<String>,
    pub seconds: f32,
    pub prop: Option<String>,
    pub release_tick: Option<u64>,
    pub item_r: Option<[f32; 3]>,
    pub missing: Option<String>,
    pub refusal: Option<String>,
}

stats! {
    #[section("viewmodel")]
    pub struct ViewModelStats {
        pub weapon: Option<String>,
        pub clip: String,
        pub offset: [f32; 3],
        pub scale: f32,
        pub raise: f32,
        pub scope: f32,
        pub lower: f32,
        pub kick: f32,
        pub muzzle: Option<[f32; 3]>,
        pub bound: Option<String>,
        pub hidden: Option<String>,
        pub refusal: Option<String>,
        pub hands: HandsStats,
    }
}

#[derive(Component, Clone, Copy, Debug, Default, PartialEq)]
pub struct FirstPersonWeapon;

#[derive(Component, Clone, Copy, Debug, Default, PartialEq)]
pub struct FirstPersonHands;

pub struct ViewModelBrick;

impl Brick for ViewModelBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .resource::<ViewModel>()
            .resource::<Holding>()
            .resource::<place::Held>()
            .resource::<hands::Hands>()
            .moves::<FirstPersonWeapon>(Pace::Tick)
            .moves::<FirstPersonHands>(Pace::Tick)
            .systems(
                Schedule::FixedSimulation,
                (hands::decide, place::place, hands::draw, place::warm),
            )
            .systems(Schedule::Presentation, place::veil)
            .dev(|dev| {
                dev.stats::<ViewModelStats>();
            });
    }
}
