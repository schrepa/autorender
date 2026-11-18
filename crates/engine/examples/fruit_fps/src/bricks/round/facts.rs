use engine::prelude::*;

use super::start::{SeatNumbers, StartNumbers, seats, start_numbers};
use super::{MatchSetup, RoundBook};
use crate::play::actors::{Actor, Body, Health, Human, Wish};
use crate::play::arms::{Arms, Held};
use crate::play::events::arms::{ArmsEvents, Weapon};
use crate::play::events::powers::PowerEvents;
use crate::play::events::round::RoundEvents;
use crate::play::flow::{Flow, Stage};
use crate::play::numbers::{GameNumbers, Numbers};
use crate::play::round::{DeviceSpot, Round, Seat};
use crate::play::world::{Arena, Zone};
use crate::rules::{
    Catalog, Device, Event, Holding, Match, Phase, PlayerFacts, Side, TickInput, WeaponId, ticks,
};

pub(super) fn ticks_per_second(time: &FixedTime) -> u32 {
    if time.delta_seconds > 0.0 {
        (1.0 / time.delta_seconds).round() as u32
    } else {
        0
    }
}

type ActorRow<'a> = (
    Entity,
    &'a Actor,
    &'a Body,
    &'a Wish,
    &'a Health,
    Has<Human>,
    Option<&'a Arms>,
);

fn fresh_match(
    numbers: &GameNumbers,
    setup: &MatchSetup,
    seats: Vec<Seat>,
    seed: u64,
    tps: u32,
    round: &mut Round,
) -> Result<(), String> {
    let mut rules = numbers.rules.clone();
    if setup.skip_buy {
        rules.round.buy_seconds = 0.0;
        rules.round.buy_seconds_first_round_of_half = 0.0;
    }
    let roster: Vec<_> = seats.iter().map(|seat| seat.slot.clone()).collect();
    let game = Match::new(&rules, &roster, seed, tps).map_err(|error| error.to_string())?;
    round.game = Some(game);
    round.seed = seed;
    round.matches += 1;
    round.seats = seats;
    round.restart_ticks = 0;
    round.buy_lock = false;
    round.device = DeviceSpot::Nowhere;
    Ok(())
}

fn wants_fresh(
    round: &Round,
    book: &RoundBook,
    setup: &MatchSetup,
    revision: u64,
    seats: &[Seat],
    flow: Option<&Flow>,
) -> bool {
    let over = flow.is_none() && round.phase() == Some(Phase::Over) && round.restart_ticks == 0;
    round.game.is_none()
        || over
        || flow.is_some_and(|flow| book.flow_game != Some(flow.game))
        || book.applied.as_ref() != Some(setup)
        || book.numbers_revision != revision
        || round
            .seats
            .iter()
            .map(|s| s.entity)
            .ne(seats.iter().map(|s| s.entity))
}

fn prepare(
    numbers: &GameNumbers,
    revision: u64,
    setup: &MatchSetup,
    tps: u32,
    (round, book): (&mut Round, &mut RoundBook),
    actors: &Query<ActorRow<'_>>,
    flow: Option<&Flow>,
) -> Result<StartNumbers, String> {
    let start = match &book.start {
        Some((read, start)) if *read == revision => start.clone()?,
        Some(_) | None => {
            let start = start_numbers(&numbers.text);
            book.start = Some((revision, start.clone()));
            start?
        }
    };
    let mut chosen = start.clone();
    if let Some(flow) = flow.filter(|flow| !flow.roster.is_empty()) {
        chosen.roster = flow
            .roster
            .iter()
            .map(|seat| SeatNumbers {
                team: seat.team,
                character: seat.character.clone(),
                control: seat.control,
                level: seat.level.clone(),
            })
            .collect();
    }
    let seated = seats(&chosen, actors.iter().map(|row| (row.0, row.1, row.5)))?;
    if !wants_fresh(round, book, setup, revision, &seated, flow) {
        return Ok(start);
    }
    book.flow_game = flow.map(|flow| flow.game);
    let fresh_setup = book.applied.as_ref() != Some(setup) || book.numbers_revision != revision;
    let seed = if fresh_setup || round.game.is_none() {
        setup.seed
    } else {
        setup.seed.wrapping_add(u64::from(round.matches))
    };
    if fresh_setup {
        round.matches = 0;
    }
    fresh_match(numbers, setup, seated, seed, tps, round)?;
    book.applied = Some(setup.clone());
    book.numbers_revision = revision;
    book.recent.clear();
    book.unplaced = 0;
    Ok(start)
}

fn facts_of(
    row: Option<(&Body, &Wish, &Health, Option<&Arms>)>,
    side: Option<Side>,
    arena: &Arena,
    device: Option<Vec3>,
    reach: f32,
) -> PlayerFacts {
    let Some((body, wish, health, arms)) = row else {
        return PlayerFacts::default();
    };
    let device_out = arms.is_some_and(|arms| arms.device_out);
    let feet = body.feet;
    let buy_zone = match side {
        Some(Side::Attack) => Some(Zone::BuyAttack),
        Some(Side::Defend) => Some(Zone::BuyDefend),
        None => None,
    };
    PlayerFacts {
        position: feet.to_array(),
        site: arena.site(feet),
        in_buy_zone: buy_zone.is_some_and(|zone| arena.zone(zone, feet)),
        near_device: device.is_some_and(|at| at.distance(feet) <= reach),
        plant: wish.interact_held || (device_out && wish.fire_held),
        defuse: wish.interact_held,
        armor_left: Some(health.armour.ceil() as u32),
    }
}

fn spot_of(device: &Device, seats: &[Seat]) -> DeviceSpot {
    match *device {
        Device::Carried { player, .. } => DeviceSpot::Carried {
            slot: player,
            entity: seats.get(player).and_then(|seat| seat.entity),
        },
        Device::Dropped { at } => DeviceSpot::Dropped {
            at: Vec3::from_array(at),
        },
        Device::Planted {
            site,
            at,
            fuse_ticks_left,
            ..
        } => DeviceSpot::Planted {
            site,
            at: Vec3::from_array(at),
            fuse_ticks_left,
        },
        Device::Exploded { site, at } => DeviceSpot::Exploded {
            site,
            at: Vec3::from_array(at),
        },
        Device::Defused { site, at } => DeviceSpot::Defused {
            site,
            at: Vec3::from_array(at),
        },
    }
}

fn input_of(
    round: &Round,
    arena: &Arena,
    arms: &ArmsEvents,
    powers: Option<&PowerEvents>,
    reach: f32,
    actors: &Query<ActorRow<'_>>,
) -> TickInput {
    let Some(game) = round.game.as_ref() else {
        return TickInput::default();
    };
    let device = round.device.at();
    let mut input = TickInput::default();
    for (slot, seat) in round.seats.iter().enumerate() {
        let row = seat.entity.and_then(|entity| actors.get(entity).ok());
        let side = round.side_of_slot(slot);
        input.players.push(facts_of(
            row.map(|r| (r.2, r.3, r.4, r.6)),
            side,
            arena,
            device,
            reach,
        ));
        if let Some((_, _, _, wish, _, _, arms)) = row {
            input.buys.extend(wish.buys.iter().map(|buy| (slot, *buy)));
            let device_held = arms.is_some_and(|arms| arms.hand.held == Held::Device && !arms.busy);
            if wish.drop && device_held {
                input.device_drops.push(slot);
            }
        }
    }
    let catalog = game.catalog();
    input.deaths = arms
        .deaths
        .iter()
        .filter_map(|death| death.rules_death(catalog))
        .collect();
    if let Some(powers) = powers {
        input.abilities.clone_from(&powers.accepted);
    }
    let mut changed: Vec<u32> = (arms.pickups.iter().map(|pickup| pickup.slot))
        .chain(arms.drops.iter().map(|drop| drop.slot))
        .collect();
    changed.sort_unstable();
    changed.dedup();
    input.holdings = changed
        .into_iter()
        .filter_map(|slot| {
            let player = slot as usize;
            let held = actors.get(round.seats.get(player)?.entity?).ok()?.6?;
            Some(Holding {
                player,
                primary: weapon_id(catalog, held.loadout.primary.as_ref()),
                sidearm: weapon_id(catalog, Some(&held.loadout.sidearm)),
            })
        })
        .collect();
    input
}

fn weapon_id(catalog: &Catalog, gun: Option<&crate::weapons::Gun>) -> Option<WeaponId> {
    catalog.weapon_id(Weapon::Gun(gun?.kind()).key())
}

fn remember(book: &mut RoundBook, events: &[Event]) {
    for event in events {
        book.recent.push(super::event_kind(event));
    }
    let keep = super::RECENT_EVENTS;
    if book.recent.len() > keep {
        let cut = book.recent.len() - keep;
        book.recent.drain(..cut);
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "one system reads every world fact the rules need"
)]
pub(super) fn advance(
    time: Res<FixedTime>,
    numbers: Res<Numbers>,
    setup: Res<MatchSetup>,
    arena: Res<Arena>,
    arms: Res<ArmsEvents>,
    powers: Option<Res<PowerEvents>>,
    mut round: ResMut<Round>,
    mut out: ResMut<RoundEvents>,
    (mut book, flow): (ResMut<RoundBook>, Option<Res<Flow>>),
    actors: Query<ActorRow<'_>>,
) {
    out.events.clear();
    let flow = flow.as_deref();
    if let Some(flow) = flow.filter(|flow| flow.stage != Stage::Playing) {
        book.waiting = Some(format!("flow stage is {}", flow.stage.key()));
        return;
    }
    let Some(game_numbers) = numbers.get() else {
        book.waiting = Some("numbers.json not read yet".to_owned());
        return;
    };
    let tps = ticks_per_second(&time);
    let start = match prepare(
        game_numbers,
        numbers.revision(),
        &setup,
        tps,
        (&mut round, &mut book),
        &actors,
        flow,
    ) {
        Ok(start) => start,
        Err(reason) => {
            book.waiting = Some(reason);
            return;
        }
    };
    book.waiting = None;
    if round.phase() == Some(Phase::Over) {
        round.restart_ticks = round.restart_ticks.saturating_sub(1);
        return;
    }
    let input = input_of(
        &round,
        &arena,
        &arms,
        powers.as_deref(),
        start.device_reach_m,
        &actors,
    );
    let Some(game) = round.game.as_mut() else {
        return;
    };
    let events = game.tick(&input);
    let Some(view) = round.view() else {
        return;
    };
    let rules_tick = view.tick;
    let phase = view.phase;
    let device = spot_of(view.device, &round.seats);
    round.buy_lock = phase == Phase::Buy;
    round.device = device;
    if events
        .iter()
        .any(|event| matches!(event, Event::MatchEnd { .. }))
    {
        round.restart_ticks = ticks(start.next_match_seconds, tps);
    }
    remember(&mut book, &events);
    out.tick = rules_tick;
    out.events = events;
}
