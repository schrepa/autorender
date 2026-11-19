use super::cast::{Asked, SLOTS, slot_key};
use super::{
    ActorPowers, Book, CastStat, Count, CueStat, HeldStat, PowerSetup, PowerStats, PropStat,
    SlotStat,
};
use crate::abilities::{
    AbilityId, Action, CueKind, DangerKind, DrawKind, DrawShape, Drawable, Event, PropPose,
};
use crate::play::effects::{
    Cue, CueLook, EffectsFeed, Form, HumanScreen, Look, PropPlace, PropShown, Shown,
};
use crate::play::events::powers::PowerEvents;
use crate::play::paths::{Hazard, HazardKind, Hazards, SmokeCloud};
use crate::play::powers::{Held, PowerActor, Powers};
use crate::rules::Team;

pub(super) struct Sheet<'a> {
    pub(super) book: &'a Book,
    pub(super) powers: &'a Powers,
    pub(super) asked: &'a Asked<'a>,
    pub(super) tick: u64,
    pub(super) human: Option<(u32, String)>,
    pub(super) human_damage: f32,
    pub(super) events: &'a [crate::abilities::Event],
}

fn team_of(index: usize) -> Team {
    if index == Team::First.index() {
        Team::First
    } else {
        Team::Second
    }
}

fn form_of(shape: DrawShape) -> Form {
    match shape {
        DrawShape::Sphere { centre, radius } => Form::Sphere { centre, radius },
        DrawShape::Box {
            centre,
            half_extents,
            axes,
        } => Form::Box {
            centre,
            half_extents,
            axes,
        },
        DrawShape::Segment { a, b } => Form::Segment { a, b },
    }
}

fn cue_look(cue: CueKind) -> Option<CueLook> {
    match cue {
        CueKind::Explosion => Some(CueLook::Explosion),
        CueKind::FlashBurst | CueKind::Glare => Some(CueLook::Flash),
        CueKind::Pulse => Some(CueLook::Pulse),
        CueKind::StrikeImpact => Some(CueLook::StrikeImpact),
        CueKind::Cast
        | CueKind::Throw
        | CueKind::Bounce
        | CueKind::BounceBody
        | CueKind::SmokeBurst
        | CueKind::ZoneStart
        | CueKind::WallUp
        | CueKind::WallDown
        | CueKind::TrapPlaced
        | CueKind::TrapArmed
        | CueKind::TrapTriggered
        | CueKind::WireChained
        | CueKind::WireZap
        | CueKind::WireReleased
        | CueKind::SensorPlaced
        | CueKind::Destroyed
        | CueKind::StrikeWarning
        | CueKind::Dash
        | CueKind::TransformStart
        | CueKind::TransformEnd
        | CueKind::HealStart => None,
    }
}

pub(super) fn draw_key(kind: DrawKind) -> &'static str {
    match kind {
        DrawKind::Projectile => "projectile",
        DrawKind::Smoke => "smoke",
        DrawKind::Zone => "zone",
        DrawKind::Curtain => "curtain",
        DrawKind::Wall => "wall",
        DrawKind::Trap => "trap",
        DrawKind::Wire => "wire",
        DrawKind::Sensor => "sensor",
        DrawKind::StrikeMark => "strike_mark",
    }
}

pub(super) fn event_key(event: &Event) -> &'static str {
    match event {
        Event::Damage { .. } => "damage",
        Event::Heal { .. } => "heal",
        Event::Shield { .. } => "shield",
        Event::BuffStarted { .. } => "buff_started",
        Event::BuffEnded { .. } => "buff_ended",
        Event::Blinded { .. } => "blinded",
        Event::RevealStarted { .. } => "reveal_started",
        Event::RevealEnded { .. } => "reveal_ended",
        Event::Moved { .. } => "moved",
        Event::SolidAdded { .. } => "solid_added",
        Event::SolidRemoved { .. } => "solid_removed",
        Event::TrapArmed { .. } => "trap_armed",
        Event::TrapTriggered { .. } => "trap_triggered",
        Event::WireChained { .. } => "wire_chained",
        Event::WireReleased { .. } => "wire_released",
        Event::Destroyed { .. } => "destroyed",
        Event::Transformed { .. } => "transformed",
        Event::TransformEnded { .. } => "transform_ended",
        Event::Ended { .. } => "ended",
        Event::Cue { .. } => "cue",
    }
}

fn hazard_kind(kind: DangerKind) -> Option<HazardKind> {
    match kind {
        DangerKind::Damage => Some(HazardKind::Damage),
        DangerKind::Strike => Some(HazardKind::Strike),
        DangerKind::Slow => Some(HazardKind::Slow),
        DangerKind::NearSight => Some(HazardKind::NearSight),
        DangerKind::Reveal => Some(HazardKind::Reveal),
        DangerKind::Smoke => None,
    }
}

fn count(counts: &mut Vec<Count>, kind: &str) {
    match counts.iter_mut().find(|held| held.kind == kind) {
        Some(held) => held.count += 1,
        None => counts.push(Count {
            kind: kind.to_owned(),
            count: 1,
        }),
    }
}

impl Sheet<'_> {
    fn zone_look(&self, ability: AbilityId) -> Look {
        let Some(book) = self.book.abilities.as_ref() else {
            return Look::SlowZone;
        };
        let Action::Zone(spec) = book.ability(ability).action else {
            return Look::SlowZone;
        };
        if spec.damage_per_tick.is_some() {
            Look::DamageZone
        } else if spec.heal_per_tick.is_some() {
            Look::HealZone
        } else if spec.sight_m.is_some() {
            Look::SightZone
        } else if spec.reveal {
            Look::RevealZone
        } else {
            Look::SlowZone
        }
    }

    fn shown(&self, drawable: &Drawable) -> Shown {
        let look = match drawable.kind {
            DrawKind::Projectile => Look::Projectile,
            DrawKind::Smoke => Look::Smoke,
            DrawKind::Zone => self.zone_look(drawable.ability),
            DrawKind::Curtain => Look::Curtain,
            DrawKind::Wall => Look::Wall,
            DrawKind::Trap => Look::Trap,
            DrawKind::Wire => Look::Wire,
            DrawKind::Sensor => Look::Sensor,
            DrawKind::StrikeMark => Look::StrikeMark,
        };
        Shown {
            effect: drawable.effect.0,
            look,
            form: form_of(drawable.shape),
            team: team_of(drawable.team),
            age_ticks: drawable.age,
            lifetime_ticks: drawable.lifetime,
            strength: drawable.strength,
            armed: drawable.armed,
        }
    }

    pub(super) fn feed(&self) -> EffectsFeed {
        let cues = self
            .events
            .iter()
            .filter_map(|event| match event {
                Event::Cue {
                    cue,
                    at,
                    size,
                    ticks,
                    ..
                } => cue_look(*cue).map(|look| Cue {
                    look,
                    at: *at,
                    size: *size,
                    ticks: *ticks,
                }),
                Event::Damage { .. }
                | Event::Heal { .. }
                | Event::Shield { .. }
                | Event::BuffStarted { .. }
                | Event::BuffEnded { .. }
                | Event::Blinded { .. }
                | Event::RevealStarted { .. }
                | Event::RevealEnded { .. }
                | Event::Moved { .. }
                | Event::SolidAdded { .. }
                | Event::SolidRemoved { .. }
                | Event::TrapArmed { .. }
                | Event::TrapTriggered { .. }
                | Event::WireChained { .. }
                | Event::WireReleased { .. }
                | Event::Destroyed { .. }
                | Event::Transformed { .. }
                | Event::TransformEnded { .. }
                | Event::Ended { .. } => None,
            })
            .collect();
        let human = self
            .human
            .as_ref()
            .map_or(HumanScreen::default(), |(slot, _)| HumanScreen {
                blindness: self.powers.blindness(*slot),
                sight_limit_m: self.powers.sight_limit(*slot),
                damage: self.human_damage,
            });
        EffectsFeed {
            tick: self.tick,
            shown: self
                .powers
                .drawables()
                .iter()
                .map(|drawable| self.shown(drawable))
                .collect(),
            cues,
            human,
            props: self.props(),
        }
    }

    pub(super) fn props(&self) -> Vec<PropShown> {
        let Some(book) = self.book.abilities.as_ref() else {
            return Vec::new();
        };
        let tps = self.book.ticks_per_second as f32;
        self.powers
            .effects
            .props()
            .into_iter()
            .filter_map(|state| {
                let spec = book.ability(state.ability);
                let prop = spec.prop.clone()?;
                let place = match state.pose {
                    PropPose::Flying {
                        centre,
                        velocity_per_tick,
                        resting,
                    } => PropPlace::Thrown {
                        centre,
                        velocity_mps: velocity_per_tick * tps,
                        resting,
                    },
                    PropPose::Placed {
                        contact,
                        normal,
                        facing,
                    } => PropPlace::Placed {
                        contact,
                        normal,
                        facing,
                    },
                    PropPose::Wire {
                        a,
                        b,
                        normal_a,
                        normal_b,
                    } => PropPlace::Wire {
                        a,
                        b,
                        normal_a,
                        normal_b,
                    },
                    PropPose::Falling {
                        target,
                        impact_tick,
                    } => {
                        let drop = spec.drop?;
                        PropPlace::Drop {
                            target,
                            height_m: drop.height_m,
                            fall_starts_tick: impact_tick.saturating_sub(u64::from(drop.ticks)),
                            impact_tick,
                        }
                    }
                };
                Some(PropShown {
                    effect: state.effect.0,
                    prop,
                    team: team_of(state.team),
                    place,
                    enemy_share: spec.enemy_visibility,
                })
            })
            .collect()
    }

    pub(super) fn hazards(&self) -> Hazards {
        let mut hazards = Hazards::default();
        for team in Team::BOTH {
            for danger in self.powers.dangers(team) {
                match hazard_kind(danger.kind) {
                    Some(kind) => hazards.teams[team.index()].push(Hazard {
                        kind,
                        centre: danger.centre,
                        radius: danger.radius,
                    }),
                    None if team == Team::First => hazards.smokes.push(SmokeCloud {
                        centre: danger.centre,
                        radius: danger.radius,
                    }),
                    None => {}
                }
            }
        }
        hazards
    }

    pub(super) fn actor_views(&self, actors: &[(u32, Team, String, bool)]) -> Vec<PowerActor> {
        actors
            .iter()
            .map(|(slot, team, character, _)| PowerActor {
                slot: *slot,
                team: *team,
                blindness: self.powers.blindness(*slot),
                sight_limit_m: self.powers.sight_limit(*slot),
                modifiers: self.powers.modifiers(*slot),
                revealed_to: Team::BOTH.map(|to| self.powers.revealed(*slot, to)),
                slots: self.asked.slots(*slot, character),
                held: self.powers.held(*slot).map(|held| held.ability),
                weapon_down_until: self.powers.weapon_down_until(*slot),
            })
            .collect()
    }

    pub(super) fn human_slots(&self) -> Vec<SlotStat> {
        let Some((slot, character)) = self.human.as_ref() else {
            return Vec::new();
        };
        SLOTS
            .into_iter()
            .zip(self.asked.slots(*slot, character))
            .map(|(ability, view)| SlotStat {
                slot: slot_key(ability).to_owned(),
                name: view.name,
                block: view.block,
                charges: view.charges,
                max: view.max,
                cooldown_ticks: view.cooldown_ticks,
                ready: view.ready,
            })
            .collect()
    }

    pub(super) fn live(&self) -> Vec<Count> {
        let mut live = Vec::new();
        for drawable in self.powers.drawables() {
            count(&mut live, draw_key(drawable.kind));
        }
        live
    }
}

pub(super) fn remember_cues(recent: &mut Vec<CueStat>, events: &[Event], tick: u64) {
    for event in events {
        if let Event::Cue {
            cue,
            ability,
            at,
            size,
            ..
        } = event
        {
            recent.push(CueStat {
                tick,
                cue: format!("{cue:?}"),
                slot: ability.slot.key().to_owned(),
                character: ability.character,
                at: at.to_array(),
                size: *size,
            });
        }
    }
    let keep = super::RECENT_CUES;
    if recent.len() > keep {
        let cut = recent.len() - keep;
        recent.drain(..cut);
    }
}

pub(super) fn write_stats(
    stats: &mut PowerStats,
    book: &Book,
    setup: &PowerSetup,
    live: &[Count],
    tick: u64,
) {
    stats.refusal = book.refusal.clone();
    stats.tick = tick;
    stats.free = setup.free;
    stats.live = live.to_vec();
    stats.casts_total = book.casts_total;
    stats.refused_total = book.refused_total;
    stats.last_cast = book.last_cast.clone();
    stats.last_refusal = book.last_refusal.clone();
    stats.recent_cues.clone_from(&book.recent_cues);
}

pub(super) fn prop_stats(props: &[PropShown]) -> Vec<PropStat> {
    props
        .iter()
        .map(|shown| {
            let (place, at) = match shown.place {
                PropPlace::Thrown {
                    centre, resting, ..
                } => (if resting { "resting" } else { "thrown" }, centre),
                PropPlace::Placed { contact, .. } => ("placed", contact),
                PropPlace::Wire { a, .. } => ("wire", a),
                PropPlace::Drop { target, .. } => ("drop", target),
            };
            PropStat {
                effect: shown.effect,
                prop: shown.prop.clone(),
                place: place.to_owned(),
                at: at.to_array(),
            }
        })
        .collect()
}

fn shape_key(shape: &DrawShape) -> &'static str {
    match shape {
        DrawShape::Sphere { .. } => "sphere",
        DrawShape::Box { .. } => "box",
        DrawShape::Segment { .. } => "segment",
    }
}

fn held_stat(held: &Held) -> HeldStat {
    HeldStat {
        slot: held.slot,
        ability: slot_key(held.ability).to_owned(),
        name: held.name.clone(),
        since: held.since,
        preview: held.preview.is_some(),
        valid: held.preview.as_ref().is_some_and(|preview| preview.valid),
        at: held
            .preview
            .as_ref()
            .map_or([0.0; 3], |preview| preview.at.to_array()),
        shape: held
            .preview
            .as_ref()
            .map_or("", |preview| shape_key(&preview.shape))
            .to_owned(),
        path_points: held
            .preview
            .as_ref()
            .map_or(0, |preview| preview.path.len() as u32),
    }
}

pub(super) fn write_tick_stats(
    stats: &mut PowerStats,
    events: &PowerEvents,
    powers: &Powers,
    human_slots: Vec<SlotStat>,
) {
    stats.accepted = events
        .casts
        .iter()
        .map(|cast| CastStat {
            tick: events.tick,
            slot: cast.slot,
            ability: slot_key(cast.ability).to_owned(),
            reason: None,
        })
        .collect();
    stats.refused = events
        .refused
        .iter()
        .map(|refused| CastStat {
            tick: events.tick,
            slot: refused.slot,
            ability: slot_key(refused.ability).to_owned(),
            reason: Some(refused.reason.key().to_owned()),
        })
        .collect();
    let mut counts = Vec::new();
    for event in &events.events {
        count(&mut counts, event_key(event));
    }
    stats.events = counts;
    stats.human_slots = human_slots;
    stats.holds = events
        .holds
        .iter()
        .map(|hold| CastStat {
            tick: events.tick,
            slot: hold.slot,
            ability: slot_key(hold.ability).to_owned(),
            reason: Some(hold.change.key().to_owned()),
        })
        .collect();
    stats.held = powers.held.iter().map(held_stat).collect();
    stats.actors = powers
        .actors
        .iter()
        .map(|actor| ActorPowers {
            slot: actor.slot,
            blindness: actor.blindness,
            sight_limit_m: actor.sight_limit_m,
            speed: actor.modifiers.speed,
            fire_rate: actor.modifiers.fire_rate,
            reload: actor.modifiers.reload,
            damage_taken: actor.modifiers.damage_taken,
            spread: actor.modifiers.spread,
            revealed_to: actor.revealed_to,
            held: actor.held.map(|ability| slot_key(ability).to_owned()),
            weapon_down_until: actor.weapon_down_until,
        })
        .collect();
}
