use crate::abilities::{AbilityBook, AbilitySpec};
use crate::play::events::powers::Refusal;
use crate::play::powers::SlotView;
use crate::play::world::Zone;
use crate::rules::{AbilitySlot, Phase, Player, Side, View};

fn phase_allows(phase: Phase) -> bool {
    match phase {
        Phase::Buy | Phase::Live | Phase::Planted | Phase::RoundEnd => true,
        Phase::Over => false,
    }
}

fn ready_in(player: &Player, slot: AbilitySlot) -> Result<(), Refusal> {
    if !player.alive {
        return Err(Refusal::Dead);
    }
    match player.ability(slot) {
        Some(ability) if ability.charges > 0 => Ok(()),
        Some(_) => Err(Refusal::NoCharges),
        None if player.ultimate.points >= player.ultimate.cost => Ok(()),
        None => Err(Refusal::UltimateNotReady),
    }
}

pub(super) struct Asked<'a> {
    pub(super) book: Option<&'a AbilityBook>,
    pub(super) view: Option<View<'a>>,
    pub(super) free: bool,
}

impl<'a> Asked<'a> {
    pub(super) fn check(
        &self,
        slot: u32,
        character: &str,
        alive: bool,
        ability: AbilitySlot,
    ) -> Result<&'a AbilitySpec, Refusal> {
        let book = self.book.ok_or(Refusal::NotLoaded)?;
        if !alive {
            return Err(Refusal::Dead);
        }
        let index = book.character(character).ok_or(Refusal::UnknownCharacter)?;
        let spec = &book.characters[index].abilities[ability.index()];
        if self.free {
            return Ok(spec);
        }
        let view = self.view.as_ref().ok_or(Refusal::NoMatch)?;
        if !phase_allows(view.phase) {
            return Err(Refusal::Phase);
        }
        let player = view.players.get(slot as usize).ok_or(Refusal::NoMatch)?;
        ready_in(player, ability)?;
        Ok(spec)
    }

    pub(super) fn buy_zone(&self, slot: u32) -> Option<Zone> {
        let view = self.view.as_ref()?;
        if view.phase != Phase::Buy {
            return None;
        }
        let player = view.players.get(slot as usize)?;
        Some(match view.sides[player.team.index()] {
            Side::Attack => Zone::BuyAttack,
            Side::Defend => Zone::BuyDefend,
        })
    }

    pub(super) fn spec(&self, character: &str, ability: AbilitySlot) -> Option<&'a AbilitySpec> {
        let book = self.book?;
        let index = book.character(character)?;
        Some(&book.characters[index].abilities[ability.index()])
    }

    pub(super) fn slots(&self, slot: u32, character: &str, views: &mut [SlotView; 4]) {
        let index = self.book.and_then(|book| book.character(character));
        let player = self
            .view
            .as_ref()
            .and_then(|view| view.players.get(slot as usize));
        let phase_open = self.free
            || self
                .view
                .as_ref()
                .is_some_and(|view| phase_allows(view.phase));
        for (ability, view) in AbilitySlot::ALL.into_iter().zip(views) {
            let spec = self
                .book
                .zip(index)
                .map(|(book, index)| &book.characters[index].abilities[ability.index()]);
            let (charges, max, cooldown_ticks) = match player {
                Some(player) => match player.ability(ability) {
                    Some(held) => (held.charges, held.max, held.cooldown_ticks),
                    None => (
                        player.ultimate.points,
                        player.ultimate.cost,
                        player.ultimate.active_ticks,
                    ),
                },
                None => (0, 0, 0),
            };
            let ready = phase_open
                && (self.free || player.is_some_and(|player| ready_in(player, ability).is_ok()));
            match spec {
                Some(spec) => {
                    view.name.clone_from(&spec.name);
                    view.block.clone_from(&spec.block);
                }
                None => {
                    view.name.clear();
                    view.block.clear();
                }
            }
            view.charges = charges;
            view.max = max;
            view.cooldown_ticks = cooldown_ticks;
            view.ready = ready;
        }
    }
}
