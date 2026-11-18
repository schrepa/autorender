use std::collections::BTreeMap;

use engine::math;
use glam::Vec3;

use crate::abilities::{ActorId, CueKind, EffectId, Event as PowerEvent, Orb, OrbEvent, Slot};
use crate::play::arms::Drops;
use crate::play::events::arms::{ArmsEvents, Handling, Weapon};
use crate::play::events::body::BodyEvents;
use crate::play::events::powers::PowerEvents;
use crate::play::sound::{Anchor, Cue};
use crate::play::world::{Arena, Barrier};
use crate::rules::{AbilitySlot, CreditReason, Event, Phase, Team};
use crate::weapons::BodyPart;

use super::table::Table;

pub(super) fn cue(event: impl Into<String>, anchor: Anchor, source: Option<u32>) -> Cue {
    Cue {
        event: event.into(),
        anchor,
        source,
        gain: 1.0,
        hold_ticks: None,
        tag: None,
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

pub(super) fn arms(
    events: &ArmsEvents,
    (human, round_kills): (Human, Option<u32>),
    table: &Table,
    out: &mut Vec<Cue>,
) {
    for shot in &events.shots {
        let weight = if shot.heavy { "_heavy" } else { "" };
        out.push(cue(
            format!("shot.{}{weight}", shot.weapon.key()),
            Anchor::Actor(shot.shooter),
            Some(shot.shooter),
        ));
        let entry = shot.pierced.zip(shot.pierced_kind);
        let end = shot.kind.map(|kind| (shot.end, kind));
        for (at, kind) in entry.into_iter().chain(end) {
            out.push(cue(
                format!("impact.{}", table.kind(kind)),
                Anchor::Point(at),
                Some(shot.shooter),
            ));
        }
    }
    for hit in &events.hits {
        let heavy = events
            .shots
            .iter()
            .any(|shot| shot.heavy && shot.shooter == hit.shooter);
        let event = match (hit.weapon, hit.part) {
            (Weapon::Melee, _) if heavy => "hit.melee_heavy",
            (Weapon::Melee, _) => "hit.melee",
            (Weapon::Gun(_), BodyPart::Head) => "hit.head",
            (Weapon::Gun(_), BodyPart::Body) => "hit.body",
            (Weapon::Gun(_), BodyPart::Leg) => "hit.leg",
        };
        out.push(cue(event, Anchor::Actor(hit.victim), Some(hit.shooter)));
        if hit.armor_broke && human.is(hit.shooter as usize) {
            out.push(cue(
                "armor.break.dealt",
                Anchor::Listener,
                Some(hit.shooter),
            ));
        }
        if hit.armor_broke && human.is(hit.victim as usize) {
            out.push(cue("armor.break.taken", Anchor::Listener, Some(hit.victim)));
        }
    }
    let mut equipped: Vec<u32> = Vec::new();
    for handling in &events.handling {
        if handling.kind == Handling::Equip {
            if equipped.contains(&handling.slot) {
                continue;
            }
            equipped.push(handling.slot);
        }
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
            let step = round_kills.unwrap_or(1).clamp(1, table.kill_steps);
            out.push(cue(
                format!("kill.confirm.{step}"),
                Anchor::Listener,
                death.killer,
            ));
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
    slowed: impl Fn(u32) -> bool,
    table: &Table,
    out: &mut Vec<Cue>,
) {
    for step in &events.footsteps {
        let own = listener == Some(step.slot);
        if (!step.loud && !own) || slowed(step.slot) {
            continue;
        }
        let mut heard = cue(
            format!("step.{}", table.kind(step.kind)),
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
        let weight = if landing.hard { "hard" } else { "soft" };
        out.push(cue(
            format!("land.{weight}.{}", table.kind(landing.kind)),
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
        CueKind::BounceBody => "power.bounce_body",
        CueKind::FlashBurst => "power.flash_burst",
        CueKind::Explosion => "power.explosion",
        CueKind::SmokeBurst => "power.smoke_burst",
        CueKind::ZoneStart => "power.zone_start",
        CueKind::WallUp => "power.wall_up",
        CueKind::WallDown => "power.wall_down",
        CueKind::TrapPlaced => "power.trap_placed",
        CueKind::TrapArmed => "power.trap_armed",
        CueKind::TrapTriggered => "power.trap_triggered",
        CueKind::WireChained => "power.wire_chained",
        CueKind::WireZap => "power.wire_zap",
        CueKind::WireReleased => "power.wire_released",
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

pub(super) struct Heard<'a> {
    pub human: Human,
    pub listener: Option<u32>,
    pub casters: &'a [(u32, Team, String)],
    pub now: u64,
    pub table: &'a Table,
    pub orbs: &'a [Orb],
}

fn wire_tag(effect: EffectId) -> u64 {
    (1 << 61) | u64::from(effect.0)
}

fn reveal_tag(actor: ActorId, team: usize) -> u64 {
    (1 << 62) | ((team as u64) << 32) | u64::from(actor.0)
}

fn orb_tag(orb: usize, actor: ActorId) -> u64 {
    (1 << 63) | ((orb as u64) << 32) | u64::from(actor.0)
}

fn orbs(events: &PowerEvents, heard: &Heard<'_>, out: &mut Vec<Cue>, stops: &mut Vec<u64>) {
    for event in &events.orbs {
        let (orb, actor, taken) = match event {
            OrbEvent::Started { orb, actor } => {
                if let Some(at) = heard.orbs.get(*orb).map(|orb| orb.at) {
                    out.push(Cue {
                        hold_ticks: Some(heard.table.orb_hold_ticks),
                        tag: Some(orb_tag(*orb, *actor)),
                        ..cue("orb.channel", Anchor::Point(at), Some(actor.0))
                    });
                }
                continue;
            }
            OrbEvent::Stopped { orb, actor, .. } => (*orb, *actor, false),
            OrbEvent::Taken { orb, actor } => (*orb, *actor, true),
            OrbEvent::Vanished { .. } | OrbEvent::Restored { .. } => continue,
        };
        stops.push(orb_tag(orb, actor));
        if let (true, Some(at)) = (taken, heard.orbs.get(orb).map(|orb| orb.at)) {
            out.push(cue("orb.pickup", Anchor::Point(at), Some(actor.0)));
        }
    }
}

pub(super) fn powers(
    events: &PowerEvents,
    heard: &Heard<'_>,
    out: &mut Vec<Cue>,
    stops: &mut Vec<u64>,
) {
    orbs(events, heard, out, stops);
    ultimates(events, heard, out);
    for event in &events.events {
        match event {
            PowerEvent::Cue {
                cue: kind,
                ability,
                at,
                size,
                ticks,
                ..
            } => cued(*kind, ability.slot, (*at, *size, *ticks), heard.table, out),
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
            | PowerEvent::WireChained { .. }
            | PowerEvent::WireReleased { .. }
            | PowerEvent::Destroyed { .. }
            | PowerEvent::Transformed { .. }
            | PowerEvent::TransformEnded { .. }
            | PowerEvent::Ended { .. } => status(event, heard, out, stops),
        }
    }
}

fn ultimates(events: &PowerEvents, heard: &Heard<'_>, out: &mut Vec<Cue>) {
    for cast in events
        .casts
        .iter()
        .filter(|cast| cast.ability == AbilitySlot::X)
    {
        let Some((_, team, character)) = heard.casters.iter().find(|row| row.0 == cast.slot) else {
            continue;
        };
        let side = if heard.human.team == Some(*team) {
            "ally"
        } else {
            "enemy"
        };
        out.push(cue(
            format!("ult.{character}.{side}"),
            Anchor::Listener,
            None,
        ));
    }
}

fn cued(
    kind: CueKind,
    slot: Slot,
    (at, size, ticks): (Vec3, f32, u32),
    table: &Table,
    out: &mut Vec<Cue>,
) {
    let anchor = Anchor::Point(at);
    let hold = u64::from(ticks);
    match kind {
        CueKind::ZoneStart => out.push(held("power.zone_start", anchor, None, hold)),
        CueKind::WireChained => {}
        CueKind::HealStart => {
            out.push(cue("power.heal_start", anchor, None));
            out.push(held("power.heal_loop", anchor, None, hold));
        }
        CueKind::Destroyed if size >= table.break_large_m => {
            out.push(cue("power.destroyed.large", anchor, None));
        }
        CueKind::Destroyed => out.push(cue("power.destroyed.small", anchor, None)),
        CueKind::Cast
        | CueKind::Throw
        | CueKind::Bounce
        | CueKind::BounceBody
        | CueKind::FlashBurst
        | CueKind::Explosion
        | CueKind::SmokeBurst
        | CueKind::WallUp
        | CueKind::WallDown
        | CueKind::TrapPlaced
        | CueKind::TrapArmed
        | CueKind::TrapTriggered
        | CueKind::WireZap
        | CueKind::WireReleased
        | CueKind::SensorPlaced
        | CueKind::StrikeWarning
        | CueKind::StrikeImpact
        | CueKind::Pulse
        | CueKind::Dash
        | CueKind::TransformStart
        | CueKind::TransformEnd
        | CueKind::Glare => out.push(cue(cue_name(kind, slot), anchor, None)),
    }
}

fn status(event: &PowerEvent, heard: &Heard<'_>, out: &mut Vec<Cue>, stops: &mut Vec<u64>) {
    let (human, now) = (heard.human, heard.now);
    match event {
        PowerEvent::Blinded { actor, until, .. } if human.slot == Some(actor.0) => {
            let hold = until.saturating_sub(now);
            out.push(held("power.blinded", Anchor::Listener, Some(actor.0), hold));
        }
        PowerEvent::RevealStarted {
            actor, team, until, ..
        } if heard.listener == Some(actor.0) => {
            out.push(Cue {
                tag: Some(reveal_tag(*actor, *team)),
                ..held(
                    "status.revealed",
                    Anchor::Listener,
                    Some(actor.0),
                    until.saturating_sub(now),
                )
            });
        }
        PowerEvent::RevealEnded { actor, team } => stops.push(reveal_tag(*actor, *team)),
        PowerEvent::WireChained {
            effect,
            actor,
            until,
        } => out.push(Cue {
            tag: Some(wire_tag(*effect)),
            ..held(
                "wire.taut",
                Anchor::Actor(actor.0),
                None,
                until.saturating_sub(now),
            )
        }),
        PowerEvent::WireReleased { effect, .. } => stops.push(wire_tag(*effect)),
        PowerEvent::Destroyed { team, .. }
            if human.team.is_some_and(|own| own.index() == *team) =>
        {
            out.push(cue("power.destroyed.own", Anchor::Listener, None));
        }
        PowerEvent::Damage { .. }
        | PowerEvent::Heal { .. }
        | PowerEvent::Shield { .. }
        | PowerEvent::BuffStarted { .. }
        | PowerEvent::BuffEnded { .. }
        | PowerEvent::Blinded { .. }
        | PowerEvent::RevealStarted { .. }
        | PowerEvent::Moved { .. }
        | PowerEvent::SolidAdded { .. }
        | PowerEvent::SolidRemoved { .. }
        | PowerEvent::TrapArmed { .. }
        | PowerEvent::TrapTriggered { .. }
        | PowerEvent::Destroyed { .. }
        | PowerEvent::Transformed { .. }
        | PowerEvent::TransformEnded { .. }
        | PowerEvent::Ended { .. }
        | PowerEvent::Cue { .. } => {}
    }
}

pub(super) fn ready(was: &mut bool, now: bool, out: &mut Vec<Cue>) {
    if now && !*was {
        out.push(cue("ui.ready", Anchor::Listener, None));
    }
    *was = now;
}

pub(super) fn barrier_point(barrier: &Barrier, eye: Vec3) -> Option<Vec3> {
    barrier
        .faces
        .iter()
        .map(|face| {
            let [origin, along, _, up] = face.corners;
            let (u, v) = (along - origin, up - origin);
            let offset = eye - origin;
            let s = math::clamp(
                offset.dot(u) / math::max(u.length_squared(), f32::EPSILON),
                0.0,
                1.0,
            );
            let t = math::clamp(
                offset.dot(v) / math::max(v.length_squared(), f32::EPSILON),
                0.0,
                1.0,
            );
            origin + u * s + v * t
        })
        .min_by(|a, b| a.distance_squared(eye).total_cmp(&b.distance_squared(eye)))
}

pub(super) fn drops(
    drops: &Drops,
    seen: &mut Vec<u32>,
    arena: Option<&Arena>,
    table: &Table,
    out: &mut Vec<Cue>,
) {
    for item in &drops.items {
        if seen.contains(&item.id) {
            continue;
        }
        let lift = Vec3::new(0.0, table.drop_probe_m, 0.0);
        let kind = arena
            .and_then(|arena| arena.ground_under(item.position + lift, table.drop_probe_m * 2.0))
            .map_or(table.default_kind.as_str(), |hit| table.kind(hit.kind));
        out.push(cue(
            format!("drop.{kind}"),
            Anchor::Point(item.position),
            None,
        ));
    }
    *seen = drops.items.iter().map(|item| item.id).collect();
}

pub(super) fn scopes(
    now: &[(u32, Option<(Weapon, usize)>)],
    was: &mut BTreeMap<u32, (Weapon, usize)>,
    fired: &[u32],
    out: &mut Vec<Cue>,
) {
    for (slot, scope) in now {
        let before = was.get(slot).copied();
        let here = (Anchor::Actor(*slot), Some(*slot));
        match (before, *scope) {
            (Some((weapon, from)), Some((held, to))) if weapon == held && to > from => {
                out.push(cue(format!("scope.in.{}", held.key()), here.0, here.1));
            }
            (Some((weapon, from)), Some((held, 0)))
                if weapon == held && from > 0 && !fired.contains(slot) =>
            {
                out.push(cue(format!("scope.out.{}", held.key()), here.0, here.1));
            }
            (None, Some((held, to))) if to > 0 => {
                out.push(cue(format!("scope.in.{}", held.key()), here.0, here.1));
            }
            (Some(_) | None, Some(_) | None) => {}
        }
    }
    *was = now
        .iter()
        .filter_map(|(slot, scope)| scope.map(|scope| (*slot, scope)))
        .collect();
}
