#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

use brick_present::PresentBrick;
use brick_sound::SoundBrick;
use engine::Start;
use engine::prelude::wgpu::Features;
use engine::prelude::*;

pub mod abilities;
pub mod bots;
pub mod names;
pub mod play;
pub mod rules;
pub mod weapons;

pub mod bricks {
    pub mod arms;
    pub mod body;
    pub mod bots;
    pub mod cues;
    pub mod effects;
    pub mod figures;
    pub mod flow;
    pub mod frame;
    pub mod level;
    pub mod lobby;
    pub mod look;
    pub mod paths;
    pub mod powers;
    pub mod puppet;
    pub mod round;
    pub mod screen;
    pub mod skinned;
    pub mod viewmodel;
}

use crate::bricks::arms::ArmsBrick;
use crate::bricks::body::BodyBrick;
use crate::bricks::bots::BotsBrick;
use crate::bricks::cues::CuesBrick;
use crate::bricks::effects::EffectsBrick;
use crate::bricks::figures::FiguresBrick;
use crate::bricks::flow::FlowBrick;
use crate::bricks::frame::FrameBrick;
use crate::bricks::level::LevelBrick;
use crate::bricks::lobby::LobbyBrick;
use crate::bricks::look::LookBrick;
use crate::bricks::paths::PathsBrick;
use crate::bricks::powers::PowersBrick;
use crate::bricks::puppet::PuppetBrick;
use crate::bricks::round::RoundBrick;
use crate::bricks::screen::{ScreenBrick, ScreenInterface};
use crate::bricks::skinned::SkinnedBrick;
use crate::bricks::viewmodel::ViewModelBrick;
use crate::play::world::Arena;

#[derive(Resource, Default)]
pub struct FruitFps;

impl Game for FruitFps {
    fn start() -> Start {
        let start = Start::default();
        Start {
            catch_up_seconds: 2.0 / f64::from(start.ticks_per_second),
            groups: &[names::TITLE_GROUP, names::MATCH_GROUP],
            variant: Some(texture_variant),
            ..start
        }
    }
}

fn texture_variant(features: Features) -> &'static str {
    if features.contains(Features::TEXTURE_COMPRESSION_BC) {
        "bc"
    } else {
        "rgba8"
    }
}

pub fn bricks<L: Listing>(list: L) -> L {
    list.with::<FlowBrick>()
        .with::<LobbyBrick>()
        .with::<PathsBrick>()
        .with::<BotsBrick>()
        .with::<BodyBrick>()
        .with::<ArmsBrick>()
        .with::<PowersBrick>()
        .with::<RoundBrick>()
        .with::<LookBrick>()
        .with::<PuppetBrick>()
        .with::<ViewModelBrick>()
        .with::<CuesBrick>()
        .with::<SoundBrick<Arena>>()
        .with::<FrameBrick>()
        .with::<PresentBrick<play::view::Controls>>()
        .with::<LevelBrick>()
        .with::<SkinnedBrick>()
        .with::<FiguresBrick>()
        .with::<EffectsBrick>()
        .with::<ScreenBrick>()
        .with::<ScreenInterface>()
}

engine::game!(
    pub struct FruitFpsGame(FruitFps),
    "Pulp Vendetta",
    bricks,
    scene = "fruit_fps.scene.json"
);
