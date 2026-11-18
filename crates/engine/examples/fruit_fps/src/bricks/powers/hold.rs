use engine::prelude::*;

use super::apply::{Row, entity_of, power_actor};
use super::cast::Asked;
use super::{CastAt, Parts, Relay, TestSlot};
use crate::abilities::{self, AbilitySpec, Aim, Event, Preview, needs_aim};
use crate::play::events::powers::{Cast, Hold, HoldChange, PowerEvents, Refusal, Refused};
use crate::play::powers::{Held, Powers};
use crate::play::world::Arena;
use crate::rules::{AbilitySlot, AbilityUse};
use crate::weapons::forward;

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
    pub(super) cast_at: Option<CastAt>,
}

fn test_slot(slot: TestSlot) -> AbilitySlot {
    match slot {
        TestSlot::C => AbilitySlot::C,
        TestSlot::Q => AbilitySlot::Q,
        TestSlot::E => AbilitySlot::E,
        TestSlot::X => AbilitySlot::X,
    }
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
        if seat.switched
            || self
                .asked
                .check(seat.slot, seat.character, seat.alive, held)
                .is_err()
        {
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
        if let Ok(index) = powers.held.binary_search_by_key(&slot, |held| held.slot) {
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
        let until = self.tick + 1 + u64::from(spec.recover_ticks);
        match powers
            .recovering
            .binary_search_by_key(&slot, |(held, _)| *held)
        {
            Ok(index) => powers.recovering[index].1 = until,
            Err(index) => powers.recovering.insert(index, (slot, until)),
        }
        events.casts.push(Cast { slot, ability });
        if !self.free {
            events.accepted.push(AbilityUse {
                player: slot as usize,
                slot: ability,
            });
        }
    }

    fn placed_cast(
        &self,
        rows: &Query<Row<'_>>,
        order: &[(u32, Entity)],
        (powers, events, ticked): (&mut Powers, &mut PowerEvents, &mut Vec<Event>),
    ) {
        let Some(request) = self.cast_at else {
            return;
        };
        let Some(Ok((_, actor, wish, body, health, _, _, offsets, alive, _))) =
            entity_of(order, request.slot).map(|entity| rows.get(entity))
        else {
            return;
        };
        let ability = test_slot(request.ability);
        let seat = Seat {
            slot: actor.slot,
            character: &actor.character,
            alive: alive && health.health > 0.0,
            aims: false,
            pressed: Some(ability),
            held: None,
            fire: false,
            switched: false,
        };
        let Some(spec) = self.checked(&seat, ability, events) else {
            return;
        };
        let mut caster = power_actor(actor, body, wish, offsets, self.parts);
        let eye = glam::Vec3::from_array(request.eye);
        caster.feet = eye - (body.eye - body.feet);
        caster.eye = eye;
        caster.view = forward(request.yaw_deg.to_radians(), request.pitch_deg.to_radians());
        self.cast(&caster, spec, ability, (powers, events, ticked));
    }

    fn inside_buy_zone(
        &self,
        arena: &Arena,
        powers: &Powers,
        caster: &abilities::Actor,
        spec: &AbilitySpec,
    ) -> bool {
        let Some(zone) = self.asked.buy_zone(caster.id.0) else {
            return true;
        };
        if !needs_aim(spec) {
            return true;
        }
        let aim = Aim {
            origin: caster.eye,
            direction: caster.view,
        };
        powers
            .landing(arena, caster, spec, aim)
            .is_some_and(|preview| preview.valid && arena.zone(zone, preview.at))
    }

    pub(super) fn run(
        &self,
        arena: &Arena,
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
                alive: alive
                    && health.health > 0.0
                    && self.fallen.binary_search(&actor.slot).is_err(),
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
                    let at = powers.held.partition_point(|held| held.slot < actor.slot);
                    powers.held.insert(
                        at,
                        Held {
                            slot: actor.slot,
                            ability,
                            name: self
                                .asked
                                .spec(&actor.character, ability)
                                .map(|spec| spec.name.clone())
                                .unwrap_or_default(),
                            since: self.tick,
                            preview: None,
                        },
                    );
                    events.holds.push(HoldChange {
                        slot: actor.slot,
                        ability,
                        change: Hold::Equipped,
                    });
                }
                Move::Cast(ability) => {
                    if let Some(spec) = self.asked.spec(&actor.character, ability) {
                        let caster = power_actor(actor, body, wish, offsets, self.parts);
                        if self.inside_buy_zone(arena, powers, &caster, spec) {
                            self.cast(&caster, spec, ability, (powers, events, ticked));
                        } else {
                            events.refused.push(Refused {
                                slot: actor.slot,
                                ability,
                                reason: Refusal::OutsideBuyZone,
                            });
                        }
                    }
                }
            }
        }
        self.placed_cast(rows, order, (powers, events, ticked));
    }

    pub(super) fn previews(
        &self,
        arena: &Arena,
        rows: &Query<Row<'_>>,
        (order, watched): (&[(u32, Entity)], Option<u32>),
        powers: &mut Powers,
    ) {
        let viewers: Vec<u32> = order
            .iter()
            .filter(|(slot, entity)| {
                watched == Some(*slot) || rows.get(*entity).is_ok_and(|row| row.9)
            })
            .map(|(slot, _)| *slot)
            .collect();
        let found: Vec<Option<Preview>> = powers
            .held
            .iter()
            .map(|held| {
                if !viewers.contains(&held.slot) {
                    return None;
                }
                let (_, actor, wish, body, _, _, _, offsets, _, _) =
                    rows.get(entity_of(order, held.slot)?).ok()?;
                let spec = self.asked.spec(&actor.character, held.ability)?;
                let caster = power_actor(actor, body, wish, offsets, self.parts);
                let aim = Aim {
                    origin: caster.eye,
                    direction: caster.view,
                };
                let mut preview = powers.landing(arena, &caster, spec, aim)?;
                if let Some(zone) = self.asked.buy_zone(held.slot) {
                    preview.valid = preview.valid && arena.zone(zone, preview.at);
                }
                Some(preview)
            })
            .collect();
        for (held, preview) in powers.held.iter_mut().zip(found) {
            held.preview = preview;
        }
    }
}
