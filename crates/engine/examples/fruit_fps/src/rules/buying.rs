use super::{
    AbilitySlot, Armor, ArmorTier, BuyRefusal, BuyRequest, CreditReason, Event, Item, Match, Phase,
    Player, PlayerId, Team, TickInput, View, WeaponId, WeaponSlot,
};

fn gift_cost(
    catalog: &super::Catalog,
    players: &[Player],
    payer: PlayerId,
    (weapon, receiver): (WeaponId, PlayerId),
) -> Result<u32, BuyRefusal> {
    let entry = catalog.weapon(weapon).ok_or(BuyRefusal::NotForSale)?;
    if entry.slot != WeaponSlot::Primary {
        return Err(BuyRefusal::NotForSale);
    }
    let to = players.get(receiver).ok_or(BuyRefusal::UnknownPlayer)?;
    if receiver == payer || to.team != players[payer].team {
        return Err(BuyRefusal::NotForSale);
    }
    if !to.alive {
        return Err(BuyRefusal::Dead);
    }
    if to.primary.is_some() {
        return Err(BuyRefusal::AlreadyOwned);
    }
    let cost = entry.cost.unwrap_or(0);
    if players[payer].credits < cost {
        return Err(BuyRefusal::TooFewCredits);
    }
    Ok(cost)
}

fn shop_open(
    phase: Phase,
    players: &[Player],
    player: PlayerId,
    in_zone: bool,
) -> Result<&Player, BuyRefusal> {
    if phase != Phase::Buy {
        return Err(BuyRefusal::WrongPhase);
    }
    let p = players.get(player).ok_or(BuyRefusal::UnknownPlayer)?;
    if !p.alive {
        return Err(BuyRefusal::Dead);
    }
    if !in_zone {
        return Err(BuyRefusal::OutsideBuyZone);
    }
    Ok(p)
}

impl View<'_> {
    pub fn loss_bonus(&self, team: Team) -> u32 {
        self.catalog.loss_reward(self.loss_streak[team.index()])
    }

    pub fn asked(&self, player: PlayerId) -> Option<WeaponId> {
        let ask = self.asks.get(player).copied().flatten();
        ask.filter(|_| self.phase == Phase::Buy)
    }

    pub fn price_for(
        &self,
        payer: PlayerId,
        weapon: WeaponId,
        receiver: PlayerId,
    ) -> Result<u32, BuyRefusal> {
        self.can_buy(payer)?;
        gift_cost(self.catalog, self.players, payer, (weapon, receiver))
    }

    pub fn can_buy(&self, player: PlayerId) -> Result<(), BuyRefusal> {
        let in_zone = self.in_zone.get(player).copied().unwrap_or(false);
        shop_open(self.phase, self.players, player, in_zone).map(|_| ())
    }

    pub fn price(&self, player: PlayerId, item: Item) -> Result<u32, BuyRefusal> {
        let in_zone = self.in_zone.get(player).copied().unwrap_or(false);
        let p = shop_open(self.phase, self.players, player, in_zone)?;
        let refund = self.refund(player);
        let (cost, back) = match item {
            Item::Weapon(weapon) if weapon == self.catalog.sidearm => {
                if p.sidearm.is_some() {
                    return Err(BuyRefusal::AlreadyOwned);
                }
                (0, 0)
            }
            Item::Weapon(weapon) => {
                let entry = self.catalog.weapon(weapon).ok_or(BuyRefusal::NotForSale)?;
                if entry.slot != WeaponSlot::Primary {
                    return Err(BuyRefusal::NotForSale);
                }
                if p.primary == Some(weapon) {
                    return Err(BuyRefusal::AlreadyOwned);
                }
                (entry.cost.unwrap_or(0), refund.primary.unwrap_or(0))
            }
            Item::Armor(tier) => {
                let numbers = self.catalog.armor(tier);
                if p.armor.is_some_and(|a| a.points >= numbers.points) {
                    return Err(BuyRefusal::AlreadyOwned);
                }
                (numbers.cost, refund.armor.unwrap_or(0))
            }
            Item::Ability(slot) => {
                let index = Match::bought_index(slot)?;
                let ability = p.abilities[index];
                if ability.charges >= ability.max {
                    return Err(BuyRefusal::AtMaximum);
                }
                (self.catalog.characters[p.character].bought[index].cost, 0)
            }
        };
        if p.credits + back < cost {
            return Err(BuyRefusal::TooFewCredits);
        }
        Ok(cost)
    }
}

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
                BuyRequest::Ask(weapon) => self.ask(player, weapon),
                BuyRequest::BuyFor(weapon, receiver) => {
                    self.buy_for(player, (weapon, receiver), events)
                }
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
        let in_zone = input.players.get(player).is_some_and(|f| f.in_buy_zone);
        shop_open(self.phase, &self.players, player, in_zone).map(|_| ())
    }

    fn ask(&mut self, player: PlayerId, weapon: WeaponId) -> Result<(), BuyRefusal> {
        let entry = self.catalog.weapon(weapon).ok_or(BuyRefusal::NotForSale)?;
        if entry.slot != WeaponSlot::Primary {
            return Err(BuyRefusal::NotForSale);
        }
        if self.players[player].primary == Some(weapon) {
            return Err(BuyRefusal::AlreadyOwned);
        }
        self.asks[player] = Some(weapon);
        Ok(())
    }

    fn buy_for(
        &mut self,
        payer: PlayerId,
        (weapon, receiver): (WeaponId, PlayerId),
        events: &mut Vec<Event>,
    ) -> Result<(), BuyRefusal> {
        let cost = gift_cost(&self.catalog, &self.players, payer, (weapon, receiver))?;
        self.pay(payer, cost, events);
        self.players[receiver].primary = Some(weapon);
        let ledger = &mut self.ledgers[receiver];
        ledger.primary_paid = Some(cost);
        ledger.paid_by = Some(payer);
        ledger.replaced_primary = None;
        self.asks[receiver] = None;
        events.push(Event::GiveWeapon {
            player: receiver,
            weapon,
        });
        Ok(())
    }

    fn take_free_sidearm(
        &mut self,
        player: PlayerId,
        events: &mut Vec<Event>,
    ) -> Result<(), BuyRefusal> {
        let weapon = self.catalog.sidearm;
        let p = &mut self.players[player];
        if p.sidearm.is_some() {
            return Err(BuyRefusal::AlreadyOwned);
        }
        p.sidearm = Some(weapon);
        events.push(Event::GiveWeapon { player, weapon });
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
        if weapon == self.catalog.sidearm {
            return self.take_free_sidearm(player, events);
        }
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
                Some(amount) => {
                    let payer = self.ledgers[player].paid_by.unwrap_or(player);
                    self.refund(payer, amount, events);
                }
                None => self.ledgers[player].replaced_primary = Some(old),
            }
        }
        self.pay(player, cost, events);
        self.players[player].primary = Some(weapon);
        self.ledgers[player].primary_paid = Some(cost);
        self.ledgers[player].paid_by = None;
        self.asks[player] = None;
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
        let payer = self.ledgers[player].paid_by.unwrap_or(player);
        self.refund(payer, amount, events);
        let ledger = &mut self.ledgers[player];
        ledger.primary_paid = None;
        ledger.paid_by = None;
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
