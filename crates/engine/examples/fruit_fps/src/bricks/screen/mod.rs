engine::brick!(screen);

mod control;
mod demo;
mod facts;
mod hud;
mod live;
mod menu;
mod more;
mod paths;
mod pieces;
mod tactics;

use std::collections::BTreeMap;

use engine::prelude::*;
use engine::{Error, ErrorCode};
use engine_bindings::Input;
use engine_core::Stats;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::play::camera::TestCamera;
use crate::play::flow::Flow;
use crate::play::screen::{Options, ScreenBuys, ScreenState};

use hud::{HudFacts, Shown};

pub type ScreenInterface = brick_interface::InterfaceBrick<HudFacts>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ScreenKeys {
    pub take: Vec<String>,
    pub give_back: Vec<String>,
    pub pick: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Lows {
    pub kill_seconds: f32,
    pub time_seconds: f32,
    pub fuse_seconds: f32,
    pub health: f32,
    pub magazine_share: f32,
    pub buy_seconds: f32,
    pub go_seconds: f32,
    pub point_seconds: f32,
    pub lost_seconds: f32,
    pub ult_seconds: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FeedNumbers {
    pub lines: usize,
    pub seconds: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ArmorMark {
    pub icon: String,
    pub size_px: f32,
    pub offset_px: [f32; 2],
    pub seconds: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DamageEdge {
    pub tip: String,
    pub tip_scale: f32,
    pub radius_share: f32,
    pub length_deg: f32,
    pub band_px: f32,
    pub rim_px: f32,
    pub color: String,
    pub fade_seconds: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Marks {
    pub armor: ArmorMark,
    pub outline_color: String,
    pub crosshair_outline: String,
    pub atlas: String,
    pub damage: DamageEdge,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Texts {
    pub abilities: BTreeMap<String, String>,
    pub tips: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Words {
    pub kills: Vec<String>,
    pub attackers: String,
    pub defenders: String,
    pub round_won: String,
    pub round_lost: String,
    pub victory: String,
    pub defeat: String,
    pub elimination: String,
    pub exploded: String,
    pub defused: String,
    pub time_out: String,
    pub planting: String,
    pub defusing: String,
    pub free: String,
    pub armor_kind: String,
    pub light: String,
    pub heavy: String,
    pub per_charge: String,
    pub hidden: String,
    pub mvp_round: String,
    pub mvp_match: String,
    pub mvp_stats: String,
    pub mvp_char: String,
    pub signature_free: String,
    pub shop_zone: String,
    pub shop_dead: String,
    pub shop_closed: String,
    pub point_kill: String,
    pub point_death: String,
    pub point_plant: String,
    pub point_defuse: String,
    pub point_orb: String,
    pub status_slowed: String,
    pub status_sight: String,
    pub status_vulnerable: String,
    pub status_revealed: String,
    pub status_boosted: String,
    pub status_chained: String,
}

pub(super) struct Keys {
    take: Vec<Input>,
    give_back: Vec<Input>,
    pick: Vec<Input>,
}

pub(super) struct Loaded {
    keys: Keys,
    menu_keys: menu::MenuInputs,
    demo: HudFacts,
}

#[derive(Component, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ScreenSource {
    pub menu: menu::MenuSource,
    pub keys: ScreenKeys,
    pub key_names: menu::KeyNames,
    pub low: Lows,
    pub feed: FeedNumbers,
    pub marks: Marks,
    pub words: Words,
    pub texts: Texts,
}

fn inputs(names: &[String]) -> Result<Vec<Input>, Error> {
    (names.iter())
        .map(|name| {
            Input::named(name).ok_or_else(|| {
                Error::new(ErrorCode::UnknownVariant, "$.screen.keys").got(name.as_str())
            })
        })
        .collect()
}

fn above_zero(value: f32) -> bool {
    value > 0.0
}

impl ScreenSource {
    fn checked(&self) -> Result<(), String> {
        if !above_zero(self.marks.armor.seconds) {
            return Err("marks.armor.seconds must be above 0".to_owned());
        }
        if self.words.kills.is_empty() {
            return Err("words.kills must name at least one kill".to_owned());
        }
        if !above_zero(self.feed.seconds) {
            return Err("feed.seconds must be above 0".to_owned());
        }
        Ok(())
    }

    fn read(&self) -> Result<Loaded, Error> {
        let refused =
            |problem: String| Error::new(ErrorCode::InvalidRecord, "$.screen").got(problem);
        self.checked().map_err(refused)?;
        let keys = &self.keys;
        let menu = &self.menu.keys;
        Ok(Loaded {
            keys: Keys {
                take: inputs(&keys.take)?,
                give_back: inputs(&keys.give_back)?,
                pick: inputs(&keys.pick)?,
            },
            menu_keys: menu::MenuInputs {
                confirm: inputs(&menu.confirm)?,
                back: inputs(&menu.back)?,
                up: inputs(&menu.up)?,
                down: inputs(&menu.down)?,
                left: inputs(&menu.left)?,
                right: inputs(&menu.right)?,
                levels: inputs(&menu.levels)?,
            },
            demo: demo::demo(&self.words),
        })
    }
}

#[derive(Resource, Default)]
pub(super) struct ScreenLoad {
    loaded: Option<Loaded>,
    problem: Option<String>,
}

impl ScreenLoad {
    pub(super) fn loaded(&self) -> Option<&Loaded> {
        self.loaded.as_ref()
    }
}

fn load(sources: Query<Ref<ScreenSource>>, mut held: ResMut<ScreenLoad>) {
    let Some(source) = sources.iter().next() else {
        if held.loaded.is_some() {
            *held = ScreenLoad::default();
        }
        return;
    };
    if !source.is_added() {
        return;
    }
    *held = match source.read() {
        Ok(loaded) => ScreenLoad {
            loaded: Some(loaded),
            problem: None,
        },
        Err(error) => ScreenLoad {
            loaded: None,
            problem: Some(error.to_string()),
        },
    };
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Showing {
    shown: Shown,
    buy_open: bool,
    board_held: bool,
}

const IN_ROUND: [Shown; 4] = [Shown::Hud, Shown::Buy, Shown::Scoreboard, Shown::RoundEnd];

fn hud_hidden(camera: Option<&TestCamera>, shown: Shown) -> bool {
    camera.is_some_and(|camera| camera.hud_hidden) && IN_ROUND.contains(&shown)
}

settings! {
    #[dev]
    #[section("screen_show")]
    #[serde(deny_unknown_fields)]
    pub struct ScreenSetup {
        pub show: Option<Shown> = None => any,
        pub demo: bool = false => any,
        pub demo_blind: f32 = 0.0 => range(0.0, 1.0),
        pub demo_channel: bool = false => any,
        pub demo_spectate: bool = false => any,
        pub demo_scope: f32 = 0.0 => range(0.0, 1.0),
        pub demo_kills: u32 = 0 => range(0, 5),
        pub demo_armor: bool = false => any,
        pub demo_poor: bool = false => any,
    }
}

stats! {
    #[section("screen")]
    pub struct ScreenStats {
        pub shown: String,
        pub live: bool,
        pub pointer: Option<[f32; 2]>,
        pub hovered: Option<String>,
        pub buys: u32,
        pub flow: String,
        pub flow_settings_open: bool,
        pub character: String,
        pub level: String,
        pub quit_asked: u32,
        pub hud_hidden: bool,
        pub options_problem: Option<String>,
        pub problem: Option<String>,
    }
}

#[derive(Resource, Default)]
pub(super) struct ScreenBook {
    buys: u32,
    hovered: Option<(control::CardList, usize)>,
    pointer: Option<Vec2>,
}

pub struct ScreenBrick;

impl Brick for ScreenBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .join::<SceneComponents>(
                ComponentKind::of::<ScreenSource>("screen")
                    .schema(engine_core::schema_of::<ScreenSource>),
            )
            .resource::<ScreenLoad>()
            .resource::<HudFacts>()
            .resource::<ScreenState>()
            .resource::<Showing>()
            .resource::<ScreenBuys>()
            .resource::<ScreenBook>()
            .resource::<facts::Recent>()
            .settings::<Options>()
            .resource::<menu::OptionsWrite>()
            .resource::<menu::MenuState>()
            .systems(Schedule::BeforeTicks, load)
            .systems(Schedule::FixedSimulation, facts::collect)
            .systems(Schedule::Presentation, facts::fill)
            .systems(Schedule::Presentation, more::fill_more)
            .systems(Schedule::Presentation, tactics::fill_tactics)
            .systems(Schedule::Presentation, control::control)
            .systems(Schedule::Presentation, more::fill_menu)
            .systems(Schedule::Presentation, facts::fill_board)
            .dev(|dev| {
                dev.settings::<ScreenSetup>()
                    .stats::<ScreenStats>()
                    .systems(Schedule::Presentation, report);
            })
            .systems(Schedule::Presentation, menu::write_options)
            .systems(Schedule::Presentation, stamp);
    }
}

fn stamp(mut facts: ResMut<HudFacts>, mut last: Local<HudFacts>) {
    if *facts != *last {
        facts.version = last.version.wrapping_add(1);
        last.clone_from(&facts);
    }
}

type Reports<'w> = (
    Option<Res<'w, Flow>>,
    Res<'w, menu::MenuState>,
    Res<'w, menu::OptionsWrite>,
    (Res<'w, Showing>, Res<'w, HudFacts>, Res<'w, ScreenBook>),
    (Res<'w, ScreenLoad>, Option<Res<'w, TestCamera>>),
);

fn report(
    (flow, state, writes, (showing, facts, book), (load, camera)): Reports,
    sources: Query<(), With<ScreenSource>>,
    mut stats: Stats<ScreenStats>,
) {
    let problem = match (sources.is_empty(), &load.loaded) {
        (true, _) => Some("no entity carries the `screen` component".to_owned()),
        (false, None) => Some(
            load.problem
                .clone()
                .unwrap_or_else(|| "the screen files are not read yet".to_owned()),
        ),
        (false, Some(_)) => None,
    };
    stats.edit(|stats| {
        live::set(&mut stats.shown, showing.shown.key());
        stats.live = facts.live;
        stats.hud_hidden = hud_hidden(camera.as_deref(), showing.shown);
        stats.pointer = book.pointer.map(|pointer| [pointer.x, pointer.y]);
        match book
            .hovered
            .and_then(|hovered| control::card(&facts, hovered))
        {
            Some(card) => live::set(stats.hovered.get_or_insert_with(String::new), &card.name),
            None => stats.hovered = None,
        }
        stats.buys = book.buys;
        if let Some(flow) = flow.as_deref() {
            live::set(&mut stats.flow, flow.stage.key());
            stats.flow_settings_open = flow.settings_open;
            live::set(&mut stats.character, &flow.character);
            live::set(&mut stats.level, &flow.level);
        }
        stats.quit_asked = state.quit_asked;
        if stats.options_problem != writes.problem {
            stats.options_problem.clone_from(&writes.problem);
        }
        if stats.problem != problem {
            stats.problem = problem;
        }
    });
}

