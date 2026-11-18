engine::brick!(bots);

mod report;
mod senses;
mod think;
mod world;

use std::sync::Arc;

use engine::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::abilities::{AbilityBook, AbilitySpec, Action, Delivery, TrapShape};
use crate::bots::{BotNumbers, Level, MapLayout, Places};
use crate::play::bots::BotMinds;
use crate::play::numbers::Numbers;
use crate::play::order::Step;
use crate::play::paths::Paths;
use crate::weapons::{StandJoints, WeaponNumbers};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum LevelChoice {
    #[default]
    Seat,
    Easy,
    Normal,
    Hard,
}

settings! {
    #[section("bots_test")]
    #[serde(deny_unknown_fields)]
    pub struct BotsTest {
        pub frozen: bool = false => any,
        pub level: LevelChoice = LevelChoice::Seat => any,
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SquadStat {
    pub team: usize,
    pub approach: String,
    pub site: Option<usize>,
    pub step: String,
    pub why: String,
    pub since_tick: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct WishStat {
    #[serde(rename = "move")]
    pub direction: [f32; 2],
    pub walk: bool,
    pub fire: bool,
    pub yaw: f32,
    pub pitch: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BotStat {
    pub slot: u32,
    pub level: String,
    pub duty: String,
    pub step: String,
    pub activity: String,
    pub goal: [f32; 3],
    pub goal_place: Option<String>,
    pub target: Option<u32>,
    pub why: String,
    pub since_tick: u64,
    pub wish: WishStat,
}

stats! {
    #[section("bots")]
    pub struct BotsStats {
        pub problem: Option<String>,
        pub tick: u64,
        pub frozen: bool,
        pub queries: u32,
        pub expanded: u32,
        pub squads: Vec<SquadStat>,
        pub bots: Vec<BotStat>,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct SenseNumbers {
    view_half_angle_deg: f32,
    touch_m: f32,
    shot_heard_m: f32,
    reload_heard_m: f32,
    ability_heard_m: f32,
    device_heard_m: f32,
}

#[derive(Deserialize)]
struct SensesPart {
    senses: SenseNumbers,
}

#[derive(Deserialize)]
struct SensesRoot {
    bots: SensesPart,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Reach {
    range_m: f32,
    radius_m: f32,
}

#[derive(Clone, Debug)]
struct Loaded {
    bots: BotNumbers,
    senses: SenseNumbers,
    view_cos: f32,
    weapons: WeaponNumbers,
    joints: StandJoints,
    leg_radius_m: f32,
    eye_height_m: f32,
    reaches: Vec<(String, [Reach; 4])>,
}

impl Loaded {
    fn reach(&self, character: &str) -> [Reach; 4] {
        self.reaches
            .iter()
            .find(|(key, _)| key == character)
            .map_or([Reach::default(); 4], |(_, reach)| *reach)
    }
}

#[derive(Resource, Default)]
struct Book {
    numbers_revision: Option<u64>,
    paths_revision: Option<u64>,
    loaded: Option<Arc<Loaded>>,
    places: Option<Arc<Places>>,
    numbers_problem: Option<String>,
    places_problem: Option<String>,
}

impl Book {
    fn problem(&self) -> Option<String> {
        self.numbers_problem
            .clone()
            .or_else(|| self.places_problem.clone())
    }
}

pub struct BotsBrick;

impl BotsBrick {
    pub const STEP: Step = Step::Think;
}

impl Brick for BotsBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<BotsTest>()
            .stats::<BotsStats>()
            .resource::<BotMinds>()
            .resource::<Book>()
            .resource::<think::Minds>()
            .systems(Schedule::FixedSimulation, adopt)
            .systems(Schedule::FixedSimulation, think::think)
            .systems(Schedule::Presentation, report::report);
    }
}

fn ticks_per_second(time: &FixedTime) -> Option<u32> {
    (time.delta_seconds > 0.0).then(|| (1.0 / time.delta_seconds).round() as u32)
}

fn thrown_range(speed: f32, gravity: f32) -> f32 {
    if gravity > 0.0 {
        speed * speed / gravity
    } else {
        speed
    }
}

fn reach_of(spec: &AbilitySpec) -> Reach {
    let radius_m = match spec.action {
        Action::Flash(flash) | Action::FlashCone { flash, .. } => flash.range_m,
        Action::Explosion(falloff) | Action::Strike { falloff, .. } => falloff.radius_m,
        Action::Smoke(smoke) => smoke.radius_m,
        Action::Zone(zone) => zone.radius_m,
        Action::Wall(wall) => wall.length_m * 0.5,
        Action::Trap(trap) => match trap.shape {
            TrapShape::Radius(radius) => radius,
            TrapShape::Wire { max_length_m } => max_length_m * 0.5,
        },
        Action::Sensor(sensor) => sensor.radius_m,
        Action::Pulse { radius_m, .. } | Action::TeamHeal { radius_m, .. } => radius_m,
        Action::Dash { distance_m, .. } => distance_m,
        Action::Transform { .. } => 0.0,
    };
    let range_m = match spec.delivery {
        Delivery::Caster => match spec.action {
            Action::Dash { distance_m, .. } => distance_m,
            Action::Flash(_)
            | Action::FlashCone { .. }
            | Action::Explosion(_)
            | Action::Strike { .. }
            | Action::Smoke(_)
            | Action::Zone(_)
            | Action::Wall(_)
            | Action::Trap(_)
            | Action::Sensor(_)
            | Action::Pulse { .. }
            | Action::TeamHeal { .. }
            | Action::Transform { .. } => 0.0,
        },
        Delivery::Throw(throw) => throw
            .max_travel_m
            .unwrap_or_else(|| thrown_range(throw.speed, throw.gravity)),
        Delivery::Place(place) => place.range_m,
    };
    Reach { range_m, radius_m }
}

fn load(text: &str, tps: u32, game: &crate::play::numbers::GameNumbers) -> Result<Loaded, String> {
    let bots =
        BotNumbers::from_numbers_json(text, tps).map_err(|error| format!("bots: {error}"))?;
    let senses = serde_json::from_str::<SensesRoot>(text)
        .map_err(|error| format!("bots.senses: {error}"))?
        .bots
        .senses;
    let book = AbilityBook::parse(text, tps)
        .map_err(|error| format!("abilities {}: {}", error.character, error.message))?;
    let reaches = book
        .characters
        .iter()
        .map(|character| {
            let reach = [0, 1, 2, 3].map(|index| reach_of(&character.abilities[index]));
            (character.key.clone(), reach)
        })
        .collect();
    let half = senses.view_half_angle_deg.to_radians();
    Ok(Loaded {
        bots,
        senses,
        view_cos: engine::math::cos(half),
        weapons: game.weapons.clone(),
        joints: game.hit_shapes.stand_joints_m,
        leg_radius_m: game.hit_shapes.leg_radius_m,
        eye_height_m: game.player.eye_height_m,
        reaches,
    })
}

fn level_of(name: &str, choice: LevelChoice) -> Level {
    match choice {
        LevelChoice::Easy => Level::Easy,
        LevelChoice::Normal => Level::Normal,
        LevelChoice::Hard => Level::Hard,
        LevelChoice::Seat => match name {
            "easy" => Level::Easy,
            "hard" => Level::Hard,
            _ => Level::Normal,
        },
    }
}

fn places_of(paths: &Paths) -> Result<Option<Places>, String> {
    let Some(text) = paths.layout() else {
        return Ok(None);
    };
    let layout: MapLayout =
        serde_json::from_str(text).map_err(|error| format!("layout: {error}"))?;
    Places::from_layout(&layout)
        .map(Some)
        .map_err(|error| format!("layout places: {error:?}"))
}

fn adopt(time: Res<FixedTime>, numbers: Res<Numbers>, paths: Res<Paths>, mut book: ResMut<Book>) {
    if book.numbers_revision != Some(numbers.revision())
        && let (Some(game), Some(tps)) = (numbers.get(), ticks_per_second(&time))
    {
        book.numbers_revision = Some(numbers.revision());
        match load(&game.text, tps, game) {
            Ok(loaded) => {
                book.loaded = Some(Arc::new(loaded));
                book.numbers_problem = None;
            }
            Err(problem) => {
                book.loaded = None;
                book.numbers_problem = Some(problem);
            }
        }
    }
    if book.paths_revision != Some(paths.revision()) {
        book.paths_revision = Some(paths.revision());
        match places_of(&paths) {
            Ok(places) => {
                book.places = places.map(Arc::new);
                book.places_problem = None;
            }
            Err(problem) => {
                book.places = None;
                book.places_problem = Some(problem);
            }
        }
    }
}
