use super::{
    AbilitySlot, ArmorTier, BuyRequest, Catalog, Item, Phase, Player, PlayerId, Team, View,
    WeaponId,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Plan {
    Opening,
    Save,
    Half,
    Full,
}

fn full_buy_price(catalog: &Catalog) -> u32 {
    let primaries = catalog.primaries();
    let standard = match primaries.len() {
        0 => 0,
        1 => catalog.weapon_cost(primaries[0]),
        n => catalog.weapon_cost(primaries[n - 2]),
    };
    standard + catalog.economy.heavy_armor.cost
}

fn team_plan(view: &View<'_>, team: Team) -> Plan {
    if view.first_round_of_half {
        return Plan::Opening;
    }
    let catalog = view.catalog;
    let mates: Vec<&Player> = view.players.iter().filter(|p| p.team == team).collect();
    let count = mates.len().max(1) as u64;
    let mean = mates.iter().map(|p| u64::from(p.credits)).sum::<u64>() / count;
    let full = u64::from(full_buy_price(catalog));
    let decider = view.score.iter().any(|&s| s + 1 >= catalog.rounds_to_win);
    if mean >= full || view.last_round_of_half || decider {
        return Plan::Full;
    }
    let next_loss = u64::from(catalog.loss_reward(view.loss_streak[team.index()]));
    if mean + next_loss >= full {
        Plan::Save
    } else {
        Plan::Half
    }
}

fn richest_on_team(view: &View<'_>, player: PlayerId) -> bool {
    let Some(me) = view.players.get(player) else {
        return false;
    };
    view.players
        .iter()
        .enumerate()
        .filter(|(_, p)| p.team == me.team && p.alive)
        .max_by_key(|(i, p)| (p.credits, std::cmp::Reverse(*i)))
        .is_some_and(|(i, _)| i == player)
}

struct Cart<'a> {
    catalog: &'a Catalog,
    me: &'a Player,
    credits: u32,
    requests: Vec<BuyRequest>,
}

impl Cart<'_> {
    fn primary(&mut self, reserve: u32, allow_top: bool) {
        if self.me.primary.is_some() {
            return;
        }
        let primaries = self.catalog.primaries();
        let top = primaries.last().copied();
        let choice = primaries
            .iter()
            .copied()
            .filter(|&w| allow_top || Some(w) != top || primaries.len() == 1)
            .filter(|&w| self.catalog.weapon_cost(w) + reserve <= self.credits)
            .max_by_key(|&w| self.catalog.weapon_cost(w));
        if let Some(weapon) = choice {
            self.take(Item::Weapon(weapon), self.catalog.weapon_cost(weapon));
        }
    }

    fn cheapest_primary(&mut self, reserve: u32) {
        if self.me.primary.is_some() {
            return;
        }
        let first: Option<WeaponId> = self.catalog.primaries().first().copied();
        if let Some(weapon) = first {
            let cost = self.catalog.weapon_cost(weapon);
            if cost + reserve <= self.credits {
                self.take(Item::Weapon(weapon), cost);
            }
        }
    }

    fn armor(&mut self, tiers: &[ArmorTier]) {
        let held = self.me.armor.map_or(0, |a| a.points);
        for &tier in tiers {
            let numbers = *self.catalog.armor(tier);
            if numbers.points > held && numbers.cost <= self.credits {
                self.take(Item::Armor(tier), numbers.cost);
                return;
            }
        }
    }

    fn abilities(&mut self, per_slot: u32) {
        let entry = &self.catalog.characters[self.me.character];
        let mut slots = [(AbilitySlot::C, 0usize), (AbilitySlot::Q, 1usize)];
        slots.sort_by_key(|&(_, i)| (entry.bought[i].cost, i));
        for (slot, index) in slots {
            let cost = entry.bought[index].cost;
            let have = self.me.abilities[index].charges;
            let room = self.me.abilities[index]
                .max
                .saturating_sub(have)
                .min(per_slot);
            for _ in 0..room {
                if cost > self.credits {
                    break;
                }
                self.take(Item::Ability(slot), cost);
            }
        }
    }

    fn take(&mut self, item: Item, cost: u32) {
        self.credits -= cost;
        self.requests.push(BuyRequest::Buy(item));
    }
}

pub fn suggest_buy(view: &View<'_>, player: PlayerId) -> Vec<BuyRequest> {
    let Some(me) = view.players.get(player) else {
        return Vec::new();
    };
    if view.phase != Phase::Buy || !me.alive {
        return Vec::new();
    }
    let catalog = view.catalog;
    let mut cart = Cart {
        catalog,
        me,
        credits: me.credits,
        requests: Vec::new(),
    };
    let heavy = catalog.economy.heavy_armor.cost;
    let light = catalog.economy.light_armor.cost;
    match team_plan(view, me.team) {
        Plan::Opening => {
            cart.armor(&[ArmorTier::Light]);
            cart.abilities(u32::MAX);
        }
        Plan::Save => cart.abilities(1),
        Plan::Half => {
            cart.cheapest_primary(light);
            cart.armor(&[ArmorTier::Light]);
            cart.abilities(1);
        }
        Plan::Full => {
            if me.credits >= full_buy_price(catalog) {
                cart.primary(heavy, richest_on_team(view, player));
            } else {
                cart.primary(light, false);
            }
            cart.armor(&[ArmorTier::Heavy, ArmorTier::Light]);
            cart.abilities(u32::MAX);
        }
    }
    cart.requests
}
