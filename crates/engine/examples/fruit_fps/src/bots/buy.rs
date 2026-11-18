use super::numbers::PlayNumbers;
use super::sense::Block;
use super::team::Role;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuyCall {
    Pistol,
    Save,
    Half,
    Full,
    Force,
}

impl BuyCall {
    pub fn key(self) -> &'static str {
        match self {
            BuyCall::Pistol => "pistol",
            BuyCall::Save => "save",
            BuyCall::Half => "half",
            BuyCall::Full => "full",
            BuyCall::Force => "force",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Goods {
    Gun(usize),
    Armor(usize),
    Ability(usize),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Shop {
    pub guns: Vec<u32>,
    pub armor: [(u32, u32); 2],
    pub loss_next: u32,
    pub pistol: bool,
    pub must_win: bool,
    pub won_last: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Wallet {
    pub credits: u32,
    pub gun: Option<usize>,
    pub armor: u32,
    pub charges: [u32; 2],
    pub max: [u32; 2],
    pub ability_cost: [u32; 2],
    pub ability_block: [Option<Block>; 2],
    pub role: Role,
    pub bot: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Cart {
    pub goods: Vec<Goods>,
    pub spent: u32,
    pub gift: Option<(usize, usize)>,
    pub awaits_gift: bool,
}

impl Shop {
    pub fn rifle(&self) -> Option<usize> {
        match self.guns.len() {
            0 => None,
            1 => Some(0),
            n => Some(n - 2),
        }
    }

    fn need(&self, wallet: &Wallet) -> u32 {
        let rifle = self.rifle();
        let gun = match (wallet.gun, rifle) {
            (Some(held), Some(rifle)) if held >= rifle => 0,
            (_, Some(rifle)) => self.guns[rifle],
            (_, None) => 0,
        };
        let armor = if wallet.armor >= self.armor[1].1 {
            0
        } else {
            self.armor[1].0
        };
        gun + armor
    }
}

pub fn team_call(shop: &Shop, wallets: &[Wallet]) -> BuyCall {
    if shop.pistol {
        return BuyCall::Pistol;
    }
    if shop.must_win {
        return BuyCall::Force;
    }
    let credits: u32 = wallets.iter().map(|w| w.credits).sum();
    let need: u32 = wallets.iter().map(|w| shop.need(w)).sum();
    if credits >= need {
        return BuyCall::Full;
    }
    let next = credits + shop.loss_next * wallets.len() as u32;
    let half = shop.guns.first().copied().unwrap_or(0) + shop.armor[0].0;
    let mean = credits / wallets.len().max(1) as u32;
    if mean < half || (next >= need && !shop.won_last) {
        BuyCall::Save
    } else {
        BuyCall::Half
    }
}

fn priority(role: Role) -> &'static [Block] {
    match role {
        Role::Controller => &[Block::Smoke, Block::Wall],
        Role::Initiator => &[Block::Flash, Block::Reveal],
        Role::Denial => &[Block::Trap, Block::Wall],
        Role::Entry => &[Block::Flash, Block::Dash, Block::DamageZone],
        Role::Support => &[Block::Heal, Block::Buff, Block::Smoke],
    }
}

fn ability_order(wallet: &Wallet) -> [usize; 2] {
    let rank = |slot: usize| {
        wallet.ability_block[slot]
            .and_then(|block| priority(wallet.role).iter().position(|b| *b == block))
            .unwrap_or(usize::MAX)
    };
    if rank(1) < rank(0) { [1, 0] } else { [0, 1] }
}

struct Basket<'a> {
    shop: &'a Shop,
    wallet: Wallet,
    left: u32,
    cart: Cart,
}

impl Basket<'_> {
    fn take(&mut self, goods: Goods, cost: u32) {
        self.left -= cost;
        self.cart.spent += cost;
        self.cart.goods.push(goods);
    }

    fn gun(&mut self, best: Option<usize>, reserve: u32) {
        if self
            .wallet
            .gun
            .is_some_and(|held| best.is_none_or(|b| held >= b))
        {
            return;
        }
        let pick = (0..self.shop.guns.len())
            .rev()
            .filter(|&g| best.is_none_or(|b| g <= b))
            .find(|&g| self.shop.guns[g] + reserve <= self.left);
        if let Some(g) = pick
            && self.wallet.gun.is_none_or(|held| g > held)
        {
            self.take(Goods::Gun(g), self.shop.guns[g]);
            self.wallet.gun = Some(g);
        }
    }

    fn armor(&mut self, heavy: bool) {
        for tier in if heavy { [1, 0] } else { [0, 0] } {
            let (cost, points) = self.shop.armor[tier];
            if points > self.wallet.armor && cost <= self.left {
                self.take(Goods::Armor(tier), cost);
                self.wallet.armor = points;
                return;
            }
        }
    }

    fn abilities(&mut self, per_slot: u32, budget: u32) {
        let mut spent = 0u32;
        for slot in ability_order(&self.wallet) {
            let cost = self.wallet.ability_cost[slot];
            let room = self.wallet.max[slot]
                .saturating_sub(self.wallet.charges[slot])
                .min(per_slot);
            for _ in 0..room {
                if cost > self.left || spent + cost > budget {
                    break;
                }
                spent += cost;
                self.take(Goods::Ability(slot), cost);
                self.wallet.charges[slot] += 1;
            }
        }
    }
}

fn own_cart(
    shop: &Shop,
    wallet: &Wallet,
    call: BuyCall,
    play: &PlayNumbers,
    awaits_gift: bool,
) -> Cart {
    let mut basket = Basket {
        shop,
        wallet: *wallet,
        left: wallet.credits,
        cart: Cart {
            awaits_gift,
            ..Cart::default()
        },
    };
    let light = shop.armor[0].0;
    let heavy = shop.armor[1].0;
    let rifle = shop.rifle();
    match call {
        BuyCall::Pistol => {
            basket.armor(false);
            basket.abilities(u32::MAX, u32::MAX);
        }
        BuyCall::Save => basket.abilities(1, play.save_spend),
        BuyCall::Half => {
            if !awaits_gift {
                basket.gun(Some(0), light);
            }
            basket.armor(false);
            basket.abilities(1, u32::MAX);
        }
        BuyCall::Full | BuyCall::Force => {
            let top = shop.guns.len().checked_sub(1);
            let sniper = wallet.role == Role::Denial
                && top.is_some_and(|t| Some(t) != rifle)
                && top.is_some_and(|t| wallet.credits >= shop.guns[t] + heavy + play.sniper_spare);
            if !awaits_gift {
                let best = if sniper { top } else { rifle };
                basket.gun(best, light);
            }
            basket.armor(true);
            basket.abilities(u32::MAX, u32::MAX);
        }
    }
    basket.cart
}

pub fn team_carts(shop: &Shop, wallets: &[Wallet], play: &PlayNumbers) -> (BuyCall, Vec<Cart>) {
    let call = team_call(shop, wallets);
    let mut carts: Vec<Cart> = wallets
        .iter()
        .map(|w| own_cart(shop, w, call, play, false))
        .collect();
    let Some(rifle) = shop.rifle() else {
        return (call, carts);
    };
    if !matches!(call, BuyCall::Full | BuyCall::Force) {
        return (call, carts);
    }
    let rifle_cost = shop.guns[rifle];
    let mut poor: Vec<usize> = (0..wallets.len())
        .filter(|&m| {
            wallets[m].gun.is_none_or(|g| g < rifle)
                && !carts[m]
                    .goods
                    .iter()
                    .any(|g| matches!(g, Goods::Gun(r) if *r >= rifle))
        })
        .collect();
    poor.sort_by_key(|&m| (wallets[m].credits, m));
    let mut rich: Vec<usize> = (0..wallets.len())
        .filter(|&m| wallets[m].bot && wallets[m].credits >= carts[m].spent + rifle_cost)
        .collect();
    rich.sort_by_key(|&m| (std::cmp::Reverse(wallets[m].credits - carts[m].spent), m));
    for (&giver, &receiver) in rich.iter().zip(poor.iter()) {
        if giver == receiver {
            continue;
        }
        carts[giver].gift = Some((receiver, rifle));
        carts[giver].spent += rifle_cost;
        carts[receiver] = own_cart(shop, &wallets[receiver], call, play, true);
    }
    (call, carts)
}
