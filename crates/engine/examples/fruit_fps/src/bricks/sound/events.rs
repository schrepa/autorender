use glam::Vec3;

use crate::abilities::{CueKind, Event as PowerEvent, Slot};
use crate::play::events::arms::{ArmsEvents, Handling, Target, Weapon};
use crate::play::events::body::BodyEvents;
use crate::play::events::powers::PowerEvents;
use crate::play::sound::{Anchor, Cue};
use crate::rules::{CreditReason, Event, Phase, Team};
use crate::solids::Surface;
use crate::weapons::BodyPart;

use super::table::Table;

fn cue(event: impl Into<String>, anchor: Anchor, source: Option<u32>) -> Cue {
    Cue {
        event: event.into(),
        anchor,
        source,
        gain: 1.0,
        hold_ticks: None,
    }
}

fn surface(surface: Surface) -> &'static str {
    match surface {
        Surface::Stone => "stone",
        Surface::Metal => "metal",
        Surface::Wood => "wood",
        Surface::Dirt => "dirt",
    }
}

fn point(at: [f32; 3]) -> Anchor {
    Anchor::Point(Vec3::from_array(at))
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Human {
    pub slot: Option<u32>,
    pub team: Option<Team>,
}

impl Human {
    fn is(&self, player: usize) -> bool {
        self.slot.is_some_and(|slot| slot as usize == player)
    }
}

pub(super) fn arms(events: &ArmsEvents, human: Human, out: &mut Vec<Cue>) {
    for shot in &events.shots {
        out.push(cue(
            format!("shot.{}", shot.weapon.key()),
            Anchor::Actor(shot.shooter),
            Some(shot.shooter),
        ));
        if let Target::World { surface: hit } = shot.target {
            out.push(cue(
                format!("impact.{}", surface(hit)),
                Anchor::Point(shot.end),
                Some(shot.shooter),
            ));
        }
    }
    for hit in &events.hits {
        let event = match (hit.weapon, hit.part) {
            (Weapon::Melee, _) => "hit.melee",
            (Weapon::Gun(_), BodyPart::Head) => "hit.head",
            (Weapon::Gun(_), BodyPart::Body) => "hit.body",
            (Weapon::Gun(_), BodyPart::Leg) => "hit.leg",
        };
        out.push(cue(event, Anchor::Actor(hit.victim), Some(hit.shooter)));
    }
    for handling in &events.handling {
        let key = handling.weapon.key();
        let here = (Anchor::Actor(handling.slot), Some(handling.slot));
        let names: &[&str] = match handling.kind {
            Handling::Equip => &["equip"],
            Handling::ReloadStarted => &["mag_out"],
            Handling::ReloadDone => &["mag_in", "bolt"],
            Handling::DryFire => &["dry"],
        };
        for name in names {
            out.push(cue(format!("{name}.{key}"), here.0, here.1));
        }
    }
    for death in &events.deaths {
        if death.killer.is_some_and(|killer| human.is(killer as usize)) {
            out.push(cue("kill.confirm", Anchor::Listener, death.killer));
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Repeat {
    anchor: Anchor,
    source: Option<u32>,
    next: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Beep {
    at: Vec3,
    from: u64,
    fuse: u32,
    next: u64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct Repeaters {
    plant: Option<Repeat>,
    defuse: Option<Repeat>,
    beep: Option<Beep>,
}

fn ticks(seconds: f32, tps: u32) -> u64 {
    ((seconds * tps as f32).round() as u64).max(1)
}

impl Repeaters {
    fn clear(&mut self) {
        *self = Repeaters::default();
    }

    pub fn tick(&mut self, now: u64, tps: u32, table: &Table, out: &mut Vec<Cue>) {
        let timing = &table.device;
        for (held, event, seconds) in [
            (
                &mut self.plant,
                "device.plant_tick",
                timing.plant_repeat_seconds,
            ),
            (
                &mut self.defuse,
                "device.defuse_tick",
                timing.defuse_repeat_seconds,
            ),
        ] {
            if let Some(repeat) = held
                && now >= repeat.next
            {
                out.push(cue(event, repeat.anchor, repeat.source));
                repeat.next = now + ticks(seconds, tps);
            }
        }
        if let Some(beep) = &mut self.beep
            && now >= beep.next
        {
            out.push(cue("device.beep", Anchor::Point(beep.at), None));
            let progress =
                engine::math::min((now - beep.from) as f32 / beep.fuse.max(1) as f32, 1.0);
            let seconds = timing.beep_slow_seconds
                + (timing.beep_fast_seconds - timing.beep_slow_seconds) * progress;
            beep.next = now + ticks(seconds, tps);
        }
    }
}

fn outcome(won: bool, win: &'static str, loss: &'static str) -> &'static str {
    if won { win } else { loss }
}

pub(super) fn round(
    events: &[Event],
    human: Human,
    repeaters: &mut Repeaters,
    now: u64,
    out: &mut Vec<Cue>,
) {
    let actor = |player: usize| {
        let slot = u32::try_from(player).unwrap_or(u32::MAX);
        (Anchor::Actor(slot), Some(slot))
    };
    for event in events {
        match event {
            Event::Phase { phase, .. } => match phase {
                Phase::Live => out.push(cue("round.start", Anchor::Listener, None)),
                Phase::Buy => repeaters.clear(),
                Phase::Planted | Phase::RoundEnd | Phase::Over => {}
            },
            Event::RoundEnd { winner, .. } => {
                repeaters.clear();
                if let Some(team) = human.team {
                    let event = outcome(team == *winner, "round.win", "round.loss");
                    out.push(cue(event, Anchor::Listener, None));
                }
            }
            Event::MatchEnd { winner, .. } => {
                if let Some(team) = human.team {
                    let event = outcome(team == *winner, "match.win", "match.loss");
                    out.push(cue(event, Anchor::Listener, None));
                }
            }
            Event::PlantStarted { player, .. } => {
                let (anchor, source) = actor(*player);
                repeaters.plant = Some(Repeat {
                    anchor,
                    source,
                    next: now,
                });
            }
            Event::PlantCancelled { .. } => repeaters.plant = None,
            Event::PlantCompleted { at, fuse_ticks, .. } => {
                repeaters.plant = None;
                out.push(cue("device.planted", point(*at), None));
                out.push(cue("device.alert", Anchor::Listener, None));
                repeaters.beep = Some(Beep {
                    at: Vec3::from_array(*at),
                    from: now,
                    fuse: *fuse_ticks,
                    next: now,
                });
            }
            Event::DefuseStarted { player, .. } => {
                let (anchor, source) = actor(*player);
                repeaters.defuse = Some(Repeat {
                    anchor,
                    source,
                    next: now,
                });
            }
            Event::DefuseCancelled { .. } => repeaters.defuse = None,
            Event::DefuseCompleted { player } => {
                let anchor = repeaters
                    .beep
                    .map_or(actor(*player).0, |beep| Anchor::Point(beep.at));
                repeaters.clear();
                out.push(cue("device.defused", anchor, None));
            }
            Event::DeviceExploded { at, .. } => {
                repeaters.clear();
                out.push(cue("device.explode", point(*at), None));
            }
            Event::BuyRefused { player, .. } if human.is(*player) => {
                out.push(cue("ui.deny", Anchor::Listener, None));
            }
            Event::Credits {
                player,
                reason: CreditReason::Bought,
                ..
            } if human.is(*player) => out.push(cue("ui.buy", Anchor::Listener, None)),
            Event::UltimateReady { player } if human.is(*player) => {
                out.push(cue("ult.ready", Anchor::Listener, None));
            }
            Event::Respawn { .. }
            | Event::GiveWeapon { .. }
            | Event::TakeWeapon { .. }
            | Event::Armor { .. }
            | Event::Charges { .. }
            | Event::Credits { .. }
            | Event::BuyRefused { .. }
            | Event::AbilityUsed { .. }
            | Event::AbilityRefused { .. }
            | Event::DeviceGiven { .. }
            | Event::DeviceDropped { .. }
            | Event::DevicePickedUp { .. }
            | Event::DefuseHalf { .. }
            | Event::KillFeed { .. }
            | Event::UltimatePoint { .. }
            | Event::UltimateReady { .. }
            | Event::SideSwap { .. } => {}
        }
    }
}

pub(super) fn bodies(
    events: &BodyEvents,
    listener: Option<u32>,
    table: &Table,
    out: &mut Vec<Cue>,
) {
    for step in &events.footsteps {
        let own = listener == Some(step.slot);
        if !step.loud && !own {
            continue;
        }
        let mut heard = cue(
            format!("step.{}", surface(step.surface)),
            Anchor::Point(step.position),
            Some(step.slot),
        );
        if !step.loud {
            heard.gain = table.quiet_step_gain;
        }
        out.push(heard);
    }
    for jump in &events.jumps {
        out.push(cue("jump", Anchor::Point(jump.position), Some(jump.slot)));
    }
    for landing in &events.landings {
        let event = if landing.hard {
            "land.hard"
        } else {
            "land.soft"
        };
        out.push(cue(
            event,
            Anchor::Point(landing.position),
            Some(landing.slot),
        ));
    }
}

pub(super) fn countdown(ticks_left: u32, tps: u32, table: &Table, out: &mut Vec<Cue>) {
    if ticks_left > 0
        && ticks_left <= table.countdown_seconds * tps
        && ticks_left.is_multiple_of(tps)
    {
        out.push(cue("round.countdown", Anchor::Listener, None));
    }
}

fn cue_name(kind: CueKind, slot: Slot) -> &'static str {
    match kind {
        CueKind::Cast if slot == Slot::X => "power.ultimate",
        CueKind::Cast => "power.cast",
        CueKind::Throw => "power.throw",
        CueKind::Bounce => "power.bounce",
        CueKind::FlashBurst => "power.flash_burst",
        CueKind::Explosion => "power.explosion",
        CueKind::SmokeBurst => "power.smoke_burst",
        CueKind::ZoneStart => "power.zone_start",
        CueKind::WallUp => "power.wall_up",
        CueKind::WallDown => "power.wall_down",
        CueKind::TrapPlaced => "power.trap_placed",
        CueKind::TrapArmed => "power.trap_armed",
        CueKind::TrapTriggered => "power.trap_triggered",
        CueKind::SensorPlaced => "power.sensor_placed",
        CueKind::Destroyed => "power.destroyed",
        CueKind::StrikeWarning => "power.strike_warning",
        CueKind::StrikeImpact => "power.strike_impact",
        CueKind::Pulse => "power.pulse",
        CueKind::Dash => "power.dash",
        CueKind::TransformStart => "power.transform_start",
        CueKind::TransformEnd => "power.transform_end",
        CueKind::HealStart => "power.heal_start",
        CueKind::Glare => "power.glare",
    }
}

fn held(event: &'static str, anchor: Anchor, source: Option<u32>, ticks: u64) -> Cue {
    Cue {
        hold_ticks: Some(u32::try_from(ticks).unwrap_or(u32::MAX)),
        ..cue(event, anchor, source)
    }
}

pub(super) fn powers(events: &PowerEvents, human: Human, now: u64, out: &mut Vec<Cue>) {
    for event in &events.events {
        match event {
            PowerEvent::Cue {
                cue: kind,
                ability,
                at,
                ticks,
                ..
            } => {
                let anchor = Anchor::Point(*at);
                let hold = u64::from(*ticks);
                match kind {
                    CueKind::ZoneStart => out.push(held("power.zone_start", anchor, None, hold)),
                    CueKind::HealStart => {
                        out.push(cue("power.heal_start", anchor, None));
                        out.push(held("power.heal_loop", anchor, None, hold));
                    }
                    CueKind::Cast
                    | CueKind::Throw
                    | CueKind::Bounce
                    | CueKind::FlashBurst
                    | CueKind::Explosion
                    | CueKind::SmokeBurst
                    | CueKind::WallUp
                    | CueKind::WallDown
                    | CueKind::TrapPlaced
                    | CueKind::TrapArmed
                    | CueKind::TrapTriggered
                    | CueKind::SensorPlaced
                    | CueKind::Destroyed
                    | CueKind::StrikeWarning
                    | CueKind::StrikeImpact
                    | CueKind::Pulse
                    | CueKind::Dash
                    | CueKind::TransformStart
                    | CueKind::TransformEnd
                    | CueKind::Glare => out.push(cue(cue_name(*kind, ability.slot), anchor, None)),
                }
            }
            PowerEvent::Blinded { actor, until, .. } if human.slot == Some(actor.0) => {
                let hold = until.saturating_sub(now);
                out.push(held("power.blinded", Anchor::Listener, Some(actor.0), hold));
            }
            PowerEvent::Damage { .. }
            | PowerEvent::Heal { .. }
            | PowerEvent::Shield { .. }
            | PowerEvent::BuffStarted { .. }
            | PowerEvent::BuffEnded { .. }
            | PowerEvent::Blinded { .. }
            | PowerEvent::RevealStarted { .. }
            | PowerEvent::RevealEnded { .. }
            | PowerEvent::Moved { .. }
            | PowerEvent::SolidAdded { .. }
            | PowerEvent::SolidRemoved { .. }
            | PowerEvent::TrapArmed { .. }
            | PowerEvent::TrapTriggered { .. }
            | PowerEvent::Destroyed { .. }
            | PowerEvent::Transformed { .. }
            | PowerEvent::TransformEnded { .. }
            | PowerEvent::Ended { .. } => {}
        }
    }
}
