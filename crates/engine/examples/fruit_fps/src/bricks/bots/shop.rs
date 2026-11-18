use super::Reach;
use super::senses::{self, Who};
use crate::bots::{BuyCall, Cart, Goods, PlayNumbers, Role, Shop, Wallet, team_carts};
use crate::play::powers::Powers;
use crate::rules::{AbilitySlot, ArmorTier, BuyRequest, Item, Phase, Team, View, WeaponId};

#[derive(Default)]
pub(super) struct Shopping {
    round: Option<u32>,
    pub calls: [Option<BuyCall>; 2],
    carts: Vec<Option<Cart>>,
    guns: Vec<WeaponId>,
}

pub(super) struct Counter<'a> {
    pub view: &'a View<'a>,
    pub powers: Option<&'a Powers>,
    pub whos: &'a [Who],
    pub reaches: &'a [[Reach; 4]],
    pub roles: &'a dyn Fn(u32) -> Role,
    pub carrier: Option<u32>,
    pub play: &'a PlayNumbers,
}

impl Shopping {
    pub fn refresh(&mut self, counter: &Counter<'_>) {
        let view = counter.view;
        if view.phase != Phase::Buy {
            self.round = None;
            return;
        }
        if self.round == Some(view.round) {
            return;
        }
        self.round = Some(view.round);
        let catalog = view.catalog;
        self.guns = catalog.primaries();
        let light = catalog.armor(ArmorTier::Light);
        let heavy = catalog.armor(ArmorTier::Heavy);
        self.carts = vec![None; view.players.len()];
        for team in [Team::First, Team::Second] {
            let seats: Vec<usize> = (0..view.players.len())
                .filter(|&s| view.players[s].team == team)
                .collect();
            let decider = view.score.iter().any(|&s| s + 1 >= catalog.rounds_to_win);
            let shop = Shop {
                guns: self.guns.iter().map(|&g| catalog.weapon_cost(g)).collect(),
                armor: [(light.cost, light.points), (heavy.cost, heavy.points)],
                loss_next: catalog.loss_reward(view.loss_streak[team.index()]),
                pistol: view.first_round_of_half && !view.overtime,
                must_win: view.last_round_of_half || view.decider || decider,
                won_last: view.loss_streak[team.index()] == 0,
            };
            let wallets: Vec<Wallet> = seats.iter().map(|&s| self.wallet(counter, s)).collect();
            let (call, carts) = team_carts(&shop, &wallets, counter.play);
            self.calls[team.index()] = Some(call);
            for (k, mut cart) in carts.into_iter().enumerate() {
                cart.gift = cart.gift.map(|(receiver, gun)| (seats[receiver], gun));
                self.carts[seats[k]] = Some(cart);
            }
        }
    }

    fn wallet(&self, counter: &Counter<'_>, seat: usize) -> Wallet {
        let view = counter.view;
        let player = &view.players[seat];
        let catalog = view.catalog;
        let who = counter.whos.iter().position(|w| w.slot as usize == seat);
        let blocks = who.map_or([None, None], |at| {
            let senses = senses::abilities_of(counter.powers, seat as u32, counter.reaches[at]);
            [senses[0].block, senses[1].block]
        });
        let bought = catalog.characters.get(player.character);
        let cost = |i: usize| bought.map_or(u32::MAX, |c| c.bought[i].cost);
        Wallet {
            credits: player.credits,
            gun: player
                .primary
                .and_then(|g| self.guns.iter().position(|&id| id == g)),
            armor: player.armor.map_or(0, |a| a.points),
            charges: [player.abilities[0].charges, player.abilities[1].charges],
            max: [player.abilities[0].max, player.abilities[1].max],
            ability_cost: [cost(0), cost(1)],
            ability_block: blocks,
            role: (counter.roles)(seat as u32),
            bot: who.is_some_and(|at| !counter.whos[at].human)
                && counter.carrier != Some(seat as u32),
        }
    }

    pub fn gift_to(&self, seat: usize) -> Option<usize> {
        self.carts
            .get(seat)?
            .as_ref()?
            .gift
            .map(|(receiver, _)| receiver)
    }

    pub fn requests(&self, seat: usize, gift: bool) -> Vec<BuyRequest> {
        let Some(Some(cart)) = self.carts.get(seat) else {
            return Vec::new();
        };
        if gift {
            return cart
                .gift
                .and_then(|(_, gun)| self.guns.get(gun))
                .map(|&g| vec![BuyRequest::Buy(Item::Weapon(g))])
                .unwrap_or_default();
        }
        cart.goods
            .iter()
            .filter_map(|goods| match *goods {
                Goods::Gun(g) => self.guns.get(g).map(|&id| Item::Weapon(id)),
                Goods::Armor(0) => Some(Item::Armor(ArmorTier::Light)),
                Goods::Armor(_) => Some(Item::Armor(ArmorTier::Heavy)),
                Goods::Ability(0) => Some(Item::Ability(AbilitySlot::C)),
                Goods::Ability(_) => Some(Item::Ability(AbilitySlot::Q)),
            })
            .map(BuyRequest::Buy)
            .collect()
    }
}
