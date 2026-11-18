#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

use engine::prelude::*;

pub mod abilities;
pub mod bots;
pub mod names;
pub mod nav;
pub mod play;
pub mod rules;
pub mod solids;
pub mod weapons;

pub mod bricks {
    pub mod arms;
    pub mod body;
    pub mod bots;
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
    pub mod sound;
    pub mod viewmodel;
}

use crate::bricks::arms::ArmsBrick;
use crate::bricks::body::BodyBrick;
use crate::bricks::bots::BotsBrick;
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
use crate::bricks::screen::ScreenBrick;
use crate::bricks::sound::SoundBrick;
use crate::bricks::viewmodel::ViewModelBrick;

#[derive(Resource, Default)]
pub struct FruitFps;

impl Game for FruitFps {}

pub fn bricks<L: Listing>(list: L) -> L {
    list.with::<FlowBrick>()
        .with::<LobbyBrick>()
        .with::<PathsBrick>()
        .with::<BotsBrick>()
        .with::<BodyBrick>()
        .with::<ArmsBrick>()
        .with::<PowersBrick>()
        .with::<RoundBrick>()
        .with::<PuppetBrick>()
        .with::<ViewModelBrick>()
        .with::<SoundBrick>()
        .with::<FrameBrick>()
        .with::<LevelBrick>()
        .with::<FiguresBrick>()
        .with::<EffectsBrick>()
        .with::<LookBrick>()
        .with::<ScreenBrick>()
}

engine::game!(
    pub struct FruitFpsGame(FruitFps),
    "Pulp Vendetta",
    bricks,
    scene = "fruit_fps.scene.json"
);
