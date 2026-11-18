use engine::prelude::*;

use super::apply::{Row, power_actor};
use super::cast::Asked;
use super::{Parts, Relay};
use crate::abilities::{self, AbilitySpec, Aim, Event, Preview, needs_aim};
use crate::play::events::powers::{Cast, Hold, HoldChange, PowerEvents, Refused};
use crate::play::powers::{Held, Powers};
use crate::play::world::Arena;
use crate::rules::{AbilitySlot, AbilityUse};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Move {
    Stay,
    Equip(AbilitySlot),
    Cancel,
    Cast(AbilitySlot),
}

pub(super) struct Casting<'a> {
    pub(super) relay: Relay,
    pub(super) asked: &'a Asked<'a>,
    pub(super) parts: &'a Parts,
    pub(super) fallen: &'a [u32],
    pub(super) free: bool,
    pub(super) tick: u64,
}

struct Seat<'a> {
    slot: u32,
    character: &'a str,
    alive: bool,
    aims: bool,
    pressed: Option<AbilitySlot>,
    held: Option<AbilitySlot>,
    fire: bool,
    switched: bool,
}

impl Casting<'_> {
    fn checked(
        &self,
        seat: &Seat<'_>,
        ability: AbilitySlot,
        events: &mut PowerEvents,
    ) -> Option<&AbilitySpec> {
        match self
            .asked
            .check(seat.slot, seat.character, seat.alive, ability)
        {
            Ok(spec) => Some(spec),
            Err(reason) => {
                events.refused.push(Refused {
                    slot: seat.slot,
                    ability,
                    reason,
                });
                None
            }
        }
    }

    fn decide(&self, seat: &Seat<'_>, events: &mut PowerEvents) -> Move {
        if !seat.alive {
            return match (seat.held, seat.pressed) {
                (Some(_), _) => Move::Cancel,
                (None, Some(ability)) => {
                    self.checked(seat, ability, events);
                    Move::Stay
                }
                (None, None) => Move::Stay,
            };
        }
        if let Some(pressed) = seat.pressed {
            if seat.held == Some(pressed) {
                return Move::Cancel;
            }
            return match self.checked(seat, pressed, events) {
                Some(spec) if seat.aims && needs_aim(spec) => Move::Equip(pressed),
                Some(_) => Move::Cast(pressed),
                None => Move::Stay,
            };
        }
        let Some(held) = seat.held else {
            return Move::Stay;
        };
        if seat.switched {
            return Move::Cancel;
        }
        if !seat.fire {
            return Move::Stay;
        }
        match self.checked(seat, held, events) {
            Some(_) => Move::Cast(held),
            None => Move::Cancel,
        }
    }

    fn release(powers: &mut Powers, events: &mut PowerEvents, slot: u32, change: Hold) {
        if let Some(index) = powers.held.iter().position(|held| held.slot == slot) {
            let held = powers.held.remove(index);
            events.holds.push(HoldChange {
                slot,
                ability: held.ability,
                change,
            });
        }
    }

    fn cast(
        &self,
        caster: &abilities::Actor,
        spec: &AbilitySpec,
        ability: AbilitySlot,
        (powers, events, ticked): (&mut Powers, &mut PowerEvents, &mut Vec<Event>),
    ) {
        let slot = caster.id.0;
        Self::release(powers, events, slot, Hold::Cast);
        let aim = Aim {
            origin: caster.eye,
            direction: caster.view,
        };
        ticked.extend(powers.effects.cast(caster, spec, aim, self.tick));
        events.casts.push(Cast { slot, ability });
        if !self.free {
            events.accepted.push(AbilityUse {
                player: slot as usize,
                slot: ability,
            });
        }
    }

    pub(super) fn run(
        &self,
        rows: &Query<Row<'_>>,
        order: &[(u32, Entity)],
        (powers, events, ticked): (&mut Powers, &mut PowerEvents, &mut Vec<Event>),
    ) {
        let human_cast = order.iter().find_map(|(_, entity)| {
            let row = rows.get(*entity).ok()?;
            row.9.then_some(row.2.cast)
        });
        for (_, entity) in order {
            let Ok((_, actor, wish, body, health, _, _, offsets, alive, human)) = rows.get(*entity)
            else {
                continue;
            };
            let pressed = match self.relay {
                Relay::Own => wish.cast,
                Relay::Slot(slot) if slot == actor.slot => human_cast.flatten(),
                Relay::Slot(_) if human => None,
                Relay::Slot(_) => wish.cast,
            };
            let seat = Seat {
                slot: actor.slot,
                character: &actor.character,
                alive: alive && health.health > 0.0 && !self.fallen.contains(&actor.slot),
                aims: human && self.relay == Relay::Own,
                pressed,
                held: powers.held(actor.slot).map(|held| held.ability),
                fire: wish.fire_pressed,
                switched: wish.equip.is_some(),
            };
            match self.decide(&seat, events) {
                Move::Stay => {}
                Move::Cancel => Self::release(powers, events, actor.slot, Hold::Cancelled),
                Move::Equip(ability) => {
                    Self::release(powers, events, actor.slot, Hold::Cancelled);
                    powers.held.push(Held {
                        slot: actor.slot,
                        ability,
                        name: self
                            .asked
                            .spec(&actor.character, ability)
                            .map(|spec| spec.name.clone())
                            .unwrap_or_default(),
                        since: self.tick,
                        preview: None,
                    });
                    events.holds.push(HoldChange {
                        slot: actor.slot,
                        ability,
                        change: Hold::Equipped,
                    });
                }
                Move::Cast(ability) => {
                    if let Some(spec) = self.asked.spec(&actor.character, ability) {
                        let caster = power_actor(actor, body, wish, offsets, self.parts);
                        self.cast(&caster, spec, ability, (powers, events, ticked));
                    }
                }
            }
        }
    }

    pub(super) fn previews(
        &self,
        arena: &Arena,
        rows: &Query<Row<'_>>,
        order: &[(u32, Entity)],
        powers: &mut Powers,
    ) {
        let found: Vec<Option<Preview>> = powers
            .held
            .iter()
            .map(|held| {
                let (_, entity) = order.iter().find(|(slot, _)| *slot == held.slot)?;
                let (_, actor, wish, body, _, _, _, offsets, _, _) = rows.get(*entity).ok()?;
                let spec = self.asked.spec(&actor.character, held.ability)?;
                let caster = power_actor(actor, body, wish, offsets, self.parts);
                let aim = Aim {
                    origin: caster.eye,
                    direction: caster.view,
                };
                powers.landing(arena, &caster, spec, aim)
            })
            .collect();
        for (held, preview) in powers.held.iter_mut().zip(found) {
            held.preview = preview;
        }
    }
}
