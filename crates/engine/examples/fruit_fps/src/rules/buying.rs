use super::{
    AbilitySlot, Armor, ArmorTier, BuyRefusal, BuyRequest, CreditReason, Event, Item, Match, Phase,
    PlayerId, TickInput, WeaponId, WeaponSlot,
};

impl Match {
    pub(super) fn apply_buy(
        &mut self,
        player: PlayerId,
        request: BuyRequest,
        input: &TickInput,
        events: &mut Vec<Event>,
    ) {
        let outcome = self
            .buy_context(player, input)
            .and_then(|()| match request {
                BuyRequest::Buy(Item::Weapon(weapon)) => self.buy_weapon(player, weapon, events),
                BuyRequest::Buy(Item::Armor(tier)) => self.buy_armor(player, tier, events),
                BuyRequest::Buy(Item::Ability(slot)) => self.buy_charge(player, slot, events),
                BuyRequest::Sell(Item::Weapon(weapon)) => self.sell_weapon(player, weapon, events),
                BuyRequest::Sell(Item::Armor(tier)) => self.sell_armor(player, tier, events),
                BuyRequest::Sell(Item::Ability(slot)) => self.sell_charge(player, slot, events),
            });
        if let Err(reason) = outcome {
            events.push(Event::BuyRefused {
                player,
                request,
                reason,
            });
        }
    }

    fn buy_context(&self, player: PlayerId, input: &TickInput) -> Result<(), BuyRefusal> {
        if self.phase != Phase::Buy {
            return Err(BuyRefusal::WrongPhase);
        }
        let p = self.players.get(player).ok_or(BuyRefusal::UnknownPlayer)?;
        if !p.alive {
            return Err(BuyRefusal::Dead);
        }
        if !input.players.get(player).is_some_and(|f| f.in_buy_zone) {
            return Err(BuyRefusal::OutsideBuyZone);
        }
        Ok(())
    }

    fn pay(&mut self, player: PlayerId, cost: u32, events: &mut Vec<Event>) {
        self.credit(player, -i64::from(cost), CreditReason::Bought, events);
    }

    fn refund(&mut self, player: PlayerId, amount: u32, events: &mut Vec<Event>) {
        self.credit(player, i64::from(amount), CreditReason::Refunded, events);
    }

    fn buy_weapon(
        &mut self,
        player: PlayerId,
        weapon: WeaponId,
        events: &mut Vec<Event>,
    ) -> Result<(), BuyRefusal> {
        let entry = self.catalog.weapon(weapon).ok_or(BuyRefusal::NotForSale)?;
        if entry.slot != WeaponSlot::Primary {
            return Err(BuyRefusal::NotForSale);
        }
        let cost = entry.cost.unwrap_or(0);
        let held = self.players[player].primary;
        if held == Some(weapon) {
            return Err(BuyRefusal::AlreadyOwned);
        }
        let paid = self.ledgers[player].primary_paid;
        if self.players[player].credits + paid.unwrap_or(0) < cost {
            return Err(BuyRefusal::TooFewCredits);
        }
        if let Some(old) = held {
            events.push(Event::TakeWeapon {
                player,
                weapon: old,
            });
            match paid {
                Some(amount) => self.refund(player, amount, events),
                None => self.ledgers[player].replaced_primary = Some(old),
            }
        }
        self.pay(player, cost, events);
        self.players[player].primary = Some(weapon);
        self.ledgers[player].primary_paid = Some(cost);
        events.push(Event::GiveWeapon { player, weapon });
        Ok(())
    }

    fn sell_weapon(
        &mut self,
        player: PlayerId,
        weapon: WeaponId,
        events: &mut Vec<Event>,
    ) -> Result<(), BuyRefusal> {
        let paid = self.ledgers[player].primary_paid;
        let (Some(amount), true) = (paid, self.players[player].primary == Some(weapon)) else {
            return Err(BuyRefusal::NothingToSell);
        };
        events.push(Event::TakeWeapon { player, weapon });
        self.refund(player, amount, events);
        let ledger = &mut self.ledgers[player];
        ledger.primary_paid = None;
        let back = ledger.replaced_primary.take();
        self.players[player].primary = back;
        if let Some(weapon) = back {
            events.push(Event::GiveWeapon { player, weapon });
        }
        Ok(())
    }

    fn buy_armor(
        &mut self,
        player: PlayerId,
        tier: ArmorTier,
        events: &mut Vec<Event>,
    ) -> Result<(), BuyRefusal> {
        let numbers = *self.catalog.armor(tier);
        let held = self.players[player].armor;
        if held.is_some_and(|a| a.points >= numbers.points) {
            return Err(BuyRefusal::AlreadyOwned);
        }
        let paid = self.ledgers[player].armor_paid;
        if self.players[player].credits + paid.unwrap_or(0) < numbers.cost {
            return Err(BuyRefusal::TooFewCredits);
        }
        if let Some(old) = held {
            match paid {
                Some(amount) => self.refund(player, amount, events),
                None => self.ledgers[player].replaced_armor = Some(old),
            }
        }
        self.pay(player, numbers.cost, events);
        let armor = Some(Armor {
            tier,
            points: numbers.points,
        });
        self.players[player].armor = armor;
        self.ledgers[player].armor_paid = Some(numbers.cost);
        events.push(Event::Armor { player, armor });
        Ok(())
    }

    fn sell_armor(
        &mut self,
        player: PlayerId,
        tier: ArmorTier,
        events: &mut Vec<Event>,
    ) -> Result<(), BuyRefusal> {
        let paid = self.ledgers[player].armor_paid;
        let held = self.players[player].armor;
        let (Some(amount), true) = (paid, held.is_some_and(|a| a.tier == tier)) else {
            return Err(BuyRefusal::NothingToSell);
        };
        self.refund(player, amount, events);
        let ledger = &mut self.ledgers[player];
        ledger.armor_paid = None;
        let armor = ledger.replaced_armor.take();
        self.players[player].armor = armor;
        events.push(Event::Armor { player, armor });
        Ok(())
    }

    fn bought_index(slot: AbilitySlot) -> Result<usize, BuyRefusal> {
        match slot {
            AbilitySlot::C => Ok(0),
            AbilitySlot::Q => Ok(1),
            AbilitySlot::E | AbilitySlot::X => Err(BuyRefusal::NotForSale),
        }
    }

    fn buy_charge(
        &mut self,
        player: PlayerId,
        slot: AbilitySlot,
        events: &mut Vec<Event>,
    ) -> Result<(), BuyRefusal> {
        let index = Self::bought_index(slot)?;
        let character = self.players[player].character;
        let cost = self.catalog.characters[character].bought[index].cost;
        let ability = self.players[player].abilities[index];
        if ability.charges >= ability.max {
            return Err(BuyRefusal::AtMaximum);
        }
        if self.players[player].credits < cost {
            return Err(BuyRefusal::TooFewCredits);
        }
        self.pay(player, cost, events);
        let charges = ability.charges + 1;
        self.players[player].abilities[index].charges = charges;
        self.ledgers[player].charges_bought[index] += 1;
        events.push(Event::Charges {
            player,
            slot,
            charges,
        });
        Ok(())
    }

    fn sell_charge(
        &mut self,
        player: PlayerId,
        slot: AbilitySlot,
        events: &mut Vec<Event>,
    ) -> Result<(), BuyRefusal> {
        let index = Self::bought_index(slot).map_err(|_| BuyRefusal::NothingToSell)?;
        let charges = self.players[player].abilities[index].charges;
        if self.ledgers[player].charges_bought[index] == 0 || charges == 0 {
            return Err(BuyRefusal::NothingToSell);
        }
        let character = self.players[player].character;
        let cost = self.catalog.characters[character].bought[index].cost;
        self.refund(player, cost, events);
        self.players[player].abilities[index].charges = charges - 1;
        self.ledgers[player].charges_bought[index] -= 1;
        events.push(Event::Charges {
            player,
            slot,
            charges: charges - 1,
        });
        Ok(())
    }
}
