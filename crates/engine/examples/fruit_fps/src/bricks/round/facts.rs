use std::sync::Arc;

use engine::prelude::*;
use engine_core::DevSetting;

use super::start::{Listed, StartNumbers, seated, seats, start_numbers};
use super::test::MatchTest;
use super::{Chosen, MatchSetup, RoundBook};
use crate::play::actors::{Actor, Body, Health, Human, Wish};
use crate::play::arms::{Arms, Held};
use crate::play::events::arms::{ArmsEvents, Weapon};
use crate::play::events::powers::PowerEvents;
use crate::play::events::round::RoundEvents;
use crate::play::flow::{Flow, Stage};
use crate::play::numbers::{GameNumbers, Numbers};
use crate::play::round::{DeviceSpot, Round, Seat};
use crate::play::skinned::FigureClock;
use crate::play::world::{Arena, Zone};
use crate::rules::{
    Catalog, Device, Event, Holding, Match, Phase, PlayerFacts, Side, TickInput, WeaponId, ticks,
};

type ActorRow<'a> = (
    Entity,
    &'a Actor,
    &'a Body,
    &'a Wish,
    &'a Health,
    Has<Human>,
    Option<&'a Arms>,
    Option<&'a FigureClock>,
);

fn fresh_match(
    numbers: &GameNumbers,
    skip_buy: bool,
    seats: Vec<Seat>,
    seed: u64,
    tps: u32,
    round: &mut Round,
) -> Result<(), String> {
    let mut rules = numbers.rules.clone();
    if skip_buy {
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
    chosen: Chosen,
    revision: u64,
    seated: &[Option<Entity>],
    flow: Option<&Flow>,
) -> bool {
    let over = flow.is_none() && round.phase() == Some(Phase::Over) && round.restart_ticks == 0;
    round.game.is_none()
        || over
        || flow.is_some_and(|flow| book.flow_game != Some(flow.game))
        || book.applied != Some(chosen)
        || book.numbers_revision != revision
        || round
            .seats
            .iter()
            .map(|s| s.entity)
            .ne(seated.iter().copied())
}

fn prepare(
    numbers: &GameNumbers,
    revision: u64,
    setup: Chosen,
    tps: u32,
    (round, book): (&mut Round, &mut RoundBook),
    actors: &Query<ActorRow<'_>>,
    flow: Option<&Flow>,
) -> Result<Arc<StartNumbers>, String> {
    let start = match &book.start {
        Some((read, start)) if *read == revision => start.clone()?,
        Some(_) | None => {
            let start = start_numbers(&numbers.whole).map(Arc::new);
            book.start = Some((revision, start.clone()));
            start?
        }
    };
    let roster: Vec<Listed<'_>> = match flow.filter(|flow| !flow.roster.is_empty()) {
        Some(flow) => (flow.roster.iter())
            .map(|seat| (seat.team, seat.character.as_str(), seat.control, seat.level.as_str()))
            .collect(),
        None => (start.roster.iter())
            .map(|seat| (seat.team, seat.character.as_str(), seat.control, seat.level.as_str()))
            .collect(),
    };
    let seated = seated(&roster, actors.iter().map(|row| (row.0, row.1, row.5)))?;
    if !wants_fresh(round, book, setup, revision, &seated, flow) {
        return Ok(start.clone());
    }
    book.flow_game = flow.map(|flow| flow.game);
    let fresh_setup = book.applied != Some(setup) || book.numbers_revision != revision;
    let seed = if fresh_setup || round.game.is_none() {
        setup.seed
    } else {
        setup.seed.wrapping_add(u64::from(round.matches))
    };
    if fresh_setup {
        round.matches = 0;
    }
    let seats = seats(&roster, &seated);
    fresh_match(numbers, setup.skip_buy, seats, seed, tps, round)?;
    book.applied = Some(setup);
    book.numbers_revision = revision;
    book.recent.clear();
    book.planted = None;
    book.unplaced = 0;
    Ok(start.clone())
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
        yaw: wish.yaw_radians,
        set_down: feet.to_array(),
        site: arena.site(feet),
        in_buy_zone: buy_zone.is_some_and(|zone| arena.zone(zone, feet)),
        near_device: device.is_some_and(|at| at.distance(feet) <= reach),
        plant: wish.interact_held || (device_out && wish.fire_held),
        defuse: wish.interact_held,
        armor_left: Some(health.armour.ceil() as u32),
    }
}

fn set_down(
    arena: &Arena,
    start: &StartNumbers,
    (body, wish, posed): (&Body, &Wish, Option<&FigureClock>),
    site: usize,
) -> Vec3 {
    let place = &start.device_set_down;
    let fit = posed.map_or(1.0, |posed| posed.fit);
    let fit = if fit > 0.0 { fit } else { 1.0 };
    let ahead = crate::weapons::forward(wish.yaw_radians, 0.0);
    let right = Vec3::new(-ahead.z, 0.0, ahead.x);
    let reach = (ahead * place.ahead_m + right * place.right_m) * fit;
    let feet = body.feet;
    let lift = Vec3::Y * place.lift_m;
    let flat = engine::math::cos(start.footing.slope_limit_deg.to_radians());
    let length = reach.length();
    let steps = engine::math::max((length / place.pull_step_m).ceil(), 1.0) as u32;
    (0..steps)
        .map(|pulled| reach * ((steps - pulled) as f32 / steps as f32))
        .find_map(|offset| {
            let spot = feet + offset;
            let along = offset.length();
            let clear = arena
                .ray(feet + lift, offset.normalize_or_zero(), along + place.clear_m)
                .is_none();
            let ground = arena
                .ground_under(spot + lift, place.lift_m + start.footing.step_height_m)
                .filter(|hit| hit.normal.y >= flat)
                .filter(|hit| (hit.point.y - feet.y).abs() <= start.footing.step_height_m)?;
            let spot = Vec3::new(spot.x, ground.point.y, spot.z);
            (clear && arena.site(spot) == Some(site)).then_some(spot)
        })
        .unwrap_or(feet)
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
            yaw,
            fuse_ticks_left,
            ..
        } => DeviceSpot::Planted {
            site,
            at: Vec3::from_array(at),
            yaw,
            fuse_ticks_left,
        },
        Device::Exploded { site, at, yaw } => DeviceSpot::Exploded {
            site,
            at: Vec3::from_array(at),
            yaw,
        },
        Device::Defused { site, at, yaw } => DeviceSpot::Defused {
            site,
            at: Vec3::from_array(at),
            yaw,
        },
    }
}

fn input_of(
    round: &Round,
    arena: &Arena,
    arms: &ArmsEvents,
    powers: Option<&PowerEvents>,
    start: &StartNumbers,
    actors: &Query<ActorRow<'_>>,
) -> TickInput {
    let Some(game) = round.game.as_ref() else {
        return TickInput::default();
    };
    let device = round.device.at();
    let carrier = match round.device {
        DeviceSpot::Carried { slot, .. } => Some(slot),
        DeviceSpot::Nowhere
        | DeviceSpot::Dropped { .. }
        | DeviceSpot::Planted { .. }
        | DeviceSpot::Exploded { .. }
        | DeviceSpot::Defused { .. } => None,
    };
    let mut input = TickInput::default();
    for (slot, seat) in round.seats.iter().enumerate() {
        let row = seat.entity.and_then(|entity| actors.get(entity).ok());
        let side = round.side_of_slot(slot);
        let mut facts = facts_of(
            row.map(|r| (r.2, r.3, r.4, r.6)),
            side,
            arena,
            device,
            start.device_reach_m,
        );
        if let (Some(row), Some(site)) = (row, facts.site)
            && carrier == Some(slot)
            && facts.plant
        {
            facts.set_down = set_down(arena, start, (row.2, row.3, row.7), site).to_array();
        }
        input.players.push(facts);
        if let Some((_, _, _, wish, _, _, arms, _)) = row {
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
                sidearm: weapon_id(catalog, held.loadout.sidearm.as_ref()),
            })
        })
        .collect();
    input
}

fn weapon_id(catalog: &Catalog, gun: Option<&crate::weapons::Gun>) -> Option<WeaponId> {
    catalog.weapon_id(Weapon::Gun(gun?.kind()).key())
}

fn wait_for(waiting: &mut Option<String>, [head, tail]: [&str; 2]) {
    if (waiting.as_deref()).and_then(|held| held.strip_prefix(head)) != Some(tail) {
        *waiting = Some([head, tail].concat());
    }
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
    (setup, test): (Res<MatchSetup>, DevSetting<MatchTest>),
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
        wait_for(&mut book.waiting, ["flow stage is ", flow.stage.key()]);
        return;
    }
    let Some(game_numbers) = numbers.get() else {
        wait_for(&mut book.waiting, ["numbers.json not read yet", ""]);
        return;
    };
    let tps = time.rate;
    let chosen = Chosen {
        seed: setup.seed,
        skip_buy: test.get().is_some_and(|test| test.skip_buy),
    };
    let start = match prepare(
        game_numbers,
        numbers.revision(),
        chosen,
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
        &start,
        &actors,
    );
    let Round {
        game, seats: held, ..
    } = &mut *round;
    let Some(game) = game.as_mut() else {
        return;
    };
    for (slot, seat) in held.iter().enumerate() {
        let row = seat.entity.and_then(|entity| actors.get(entity).ok());
        if row.is_some_and(|row| row.3.ready) {
            game.mark_ready(slot);
        }
    }
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
    for event in &events {
        if let Event::PlantCompleted { player, at, .. } = event
            && let Some(facts) = input.players.get(*player)
        {
            book.planted = Some(super::Planted {
                from: facts.position,
                yaw: facts.yaw,
                at: *at,
            });
        }
    }
    out.tick = rules_tick;
    out.events = events;
}
