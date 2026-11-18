engine::brick!(round);

mod apply;
mod facts;
mod start;
mod test;

use engine::prelude::*;

use crate::play::events::round::RoundEvents;
use crate::play::round::{DeviceSpot, Round};
use crate::rules::{Control, Event, Phase, Side, Team};

const RECENT_EVENTS: usize = 10;

settings! {
    #[section("match")]
    #[serde(deny_unknown_fields)]
    pub struct MatchSetup {
        pub seed: u64 = 1 => any,
        pub skip_buy: bool = false => any,
    }
}

stats! {
    #[section("round")]
    pub struct RoundStats {
        pub tick: u64,
        pub matches: u32,
        pub seed: u64,
        pub phase: String,
        pub ticks_left: u32,
        pub round: u32,
        pub score: [u32; 2],
        pub sides: Vec<String>,
        pub human_credits: u32,
        pub alive: [u32; 2],
        pub device: String,
        pub device_at: Option<[f32; 3]>,
        pub last_events: Vec<String>,
        pub waiting: Option<String>,
        pub unplaced: u32,
        pub seated: u32,
        pub overtime: bool,
        pub decider: bool,
        pub credits: Vec<u32>,
        pub carried: Vec<Option<String>>,
    }
}

#[derive(Resource, Default)]
struct RoundBook {
    applied: Option<MatchSetup>,
    numbers_revision: u64,
    start: Option<(u64, Result<start::StartNumbers, String>)>,
    recent: Vec<&'static str>,
    waiting: Option<String>,
    unplaced: u32,
    flow_game: Option<u64>,
}

pub struct RoundBrick;

impl Brick for RoundBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<MatchSetup>()
            .settings::<test::MatchTest>()
            .stats::<RoundStats>()
            .resource::<Round>()
            .resource::<RoundEvents>()
            .resource::<RoundBook>()
            .systems(Schedule::FixedSimulation, test::door)
            .systems(Schedule::FixedSimulation, (facts::advance, apply::apply))
            .systems(Schedule::Presentation, report);
    }
}

fn event_kind(event: &Event) -> &'static str {
    match event {
        Event::Phase { .. } => "phase",
        Event::Respawn { .. } => "respawn",
        Event::GiveWeapon { .. } => "give_weapon",
        Event::TakeWeapon { .. } => "take_weapon",
        Event::Armor { .. } => "armor",
        Event::Charges { .. } => "charges",
        Event::Credits { .. } => "credits",
        Event::BuyRefused { .. } => "buy_refused",
        Event::AbilityUsed { .. } => "ability_used",
        Event::AbilityRefused { .. } => "ability_refused",
        Event::DeviceGiven { .. } => "device_given",
        Event::DeviceDropped { .. } => "device_dropped",
        Event::DevicePickedUp { .. } => "device_picked_up",
        Event::PlantStarted { .. } => "plant_started",
        Event::PlantCancelled { .. } => "plant_cancelled",
        Event::PlantCompleted { .. } => "plant_completed",
        Event::DefuseStarted { .. } => "defuse_started",
        Event::DefuseHalf { .. } => "defuse_half",
        Event::DefuseCancelled { .. } => "defuse_cancelled",
        Event::DefuseCompleted { .. } => "defuse_completed",
        Event::DeviceExploded { .. } => "device_exploded",
        Event::KillFeed { .. } => "kill_feed",
        Event::UltimatePoint { .. } => "ultimate_point",
        Event::UltimateReady { .. } => "ultimate_ready",
        Event::RoundEnd { .. } => "round_end",
        Event::SideSwap { .. } => "side_swap",
        Event::MatchEnd { .. } => "match_end",
    }
}

fn phase_name(phase: Phase) -> &'static str {
    match phase {
        Phase::Buy => "buy",
        Phase::Live => "live",
        Phase::Planted => "planted",
        Phase::RoundEnd => "round_end",
        Phase::Over => "over",
    }
}

fn side_name(side: Side) -> String {
    match side {
        Side::Attack => "attack",
        Side::Defend => "defend",
    }
    .to_owned()
}

fn device_name(device: &DeviceSpot) -> &'static str {
    match device {
        DeviceSpot::Nowhere => "nowhere",
        DeviceSpot::Carried { .. } => "carried",
        DeviceSpot::Dropped { .. } => "dropped",
        DeviceSpot::Planted { .. } => "planted",
        DeviceSpot::Exploded { .. } => "exploded",
        DeviceSpot::Defused { .. } => "defused",
    }
}

fn report(round: Res<Round>, book: Res<RoundBook>, mut stats: ResMut<RoundStats>) {
    let mut next = RoundStats {
        matches: round.matches,
        seed: round.seed,
        phase: "waiting".to_owned(),
        device: device_name(&round.device).to_owned(),
        device_at: round.device.at().map(|at| at.to_array()),
        last_events: book.recent.iter().map(|kind| (*kind).to_owned()).collect(),
        waiting: book.waiting.clone(),
        unplaced: book.unplaced,
        seated: round
            .seats
            .iter()
            .filter(|seat| seat.entity.is_some())
            .count() as u32,
        ..RoundStats::default()
    };
    if let Some(view) = round.view() {
        next.tick = view.tick;
        next.phase = phase_name(view.phase).to_owned();
        next.ticks_left = view.ticks_left;
        next.round = view.round;
        next.score = view.score;
        next.overtime = view.overtime;
        next.decider = view.decider;
        next.credits = view.players.iter().map(|p| p.credits).collect();
        next.carried = view
            .players
            .iter()
            .map(|p| {
                p.primary
                    .and_then(|weapon| view.catalog.weapon(weapon))
                    .map(|entry| entry.key.clone())
            })
            .collect();
        next.sides = view.sides.iter().map(|side| side_name(*side)).collect();
        next.human_credits = view
            .players
            .iter()
            .find(|p| p.control == Control::Human)
            .map_or(0, |p| p.credits);
        for team in Team::BOTH {
            let alive = view
                .players
                .iter()
                .filter(|p| p.team == team && p.alive)
                .count();
            next.alive[team.index()] = alive as u32;
        }
    }
    stats.set_if_neq(next);
}
