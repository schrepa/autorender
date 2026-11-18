use std::fmt::Write;

use engine::math;

use crate::play::screen::{
    AbilityFacts, BoardRow, Card, HudFacts, Mate, PlateFacts, Tone, WeaponStats,
};
use crate::rules::{
    AbilitySlot, ArmorTier, BuyRefusal, BuyRequest, Catalog, Defusing, Device, Item, Phase, Player,
    PlayerId, Refund, Side, Team, View, WeaponId, WeaponSlot,
};

use crate::weapons::{GunKind, GunNumbers, WeaponNumbers};

use super::{Lows, Words};

const RIFLE_KEY: &str = "rifle";

const GUNS: [GunKind; 4] = [
    GunKind::Sidearm,
    GunKind::Smg,
    GunKind::Rifle,
    GunKind::Sniper,
];

pub(super) fn set(out: &mut String, text: &str) {
    if out != text {
        out.clear();
        out.push_str(text);
    }
}

pub(super) fn named(out: &mut String, prefix: &str, name: &str) {
    out.clear();
    out.push_str(prefix);
    for character in name.chars() {
        if character == ' ' {
            out.push('_');
        } else {
            out.extend(character.to_lowercase());
        }
    }
}

pub(super) fn credits_into(out: &mut String, value: u32) {
    out.clear();
    let mut place = 1_u32;
    while place.saturating_mul(1000) <= value && place < 1_000_000_000 {
        place *= 1000;
    }
    let mut first = true;
    while place > 0 {
        let group = (value / place) % 1000;
        let _ = if first {
            write!(out, "{group}")
        } else {
            write!(out, ",{group:03}")
        };
        first = false;
        place /= 1000;
    }
}

pub(super) fn worded(out: &mut String, template: &str, args: &[(&str, &str)]) {
    out.clear();
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        out.push_str(&rest[..open]);
        let after = &rest[open + 1..];
        match after.find('}') {
            Some(close) => {
                let name = &after[..close];
                match args.iter().find(|(known, _)| *known == name) {
                    Some((_, value)) => out.push_str(value),
                    None => {
                        out.push('{');
                        out.push_str(name);
                        out.push('}');
                    }
                }
                rest = &after[close + 1..];
            }
            None => {
                out.push_str(&rest[open..]);
                rest = "";
            }
        }
    }
    out.push_str(rest);
}

pub(super) fn tone(side: Side) -> Tone {
    match side {
        Side::Attack => Tone::Attack,
        Side::Defend => Tone::Defend,
    }
}

pub(super) fn side_name(words: &Words, side: Side) -> &str {
    match side {
        Side::Attack => &words.attackers,
        Side::Defend => &words.defenders,
    }
}

pub(super) fn seconds(ticks: u32, catalog: &Catalog) -> f32 {
    ticks as f32 / catalog.ticks_per_second.max(1) as f32
}

pub(super) fn character_key<'a>(catalog: &'a Catalog, player: &Player) -> &'a str {
    catalog
        .characters
        .get(player.character)
        .map_or("", |entry| entry.key.as_str())
}

pub(super) fn character_name<'a>(catalog: &'a Catalog, player: &Player) -> &'a str {
    catalog
        .characters
        .get(player.character)
        .map_or("", |entry| entry.name.as_str())
}

pub(super) fn device_part(facts: &mut HudFacts, view: &View<'_>, human: usize) {
    let state = match view.device {
        Device::Carried { player, .. } if *player == human => "carried",
        Device::Dropped { .. } => "dropped",
        Device::Carried { .. }
        | Device::Planted { .. }
        | Device::Exploded { .. }
        | Device::Defused { .. } => "",
    };
    set(&mut facts.round.device, state);
}

pub(super) fn round_part(
    facts: &mut HudFacts,
    view: &View<'_>,
    team: Team,
    (lows, words): (&Lows, &Words),
) {
    let catalog = view.catalog;
    let round = &mut facts.round;
    round.number = view.round;
    let planted = matches!(view.device, Device::Planted { .. });
    round.planted = planted;
    round.time = match view.phase {
        Phase::Buy | Phase::Live => seconds(view.ticks_left, catalog),
        Phase::Planted | Phase::RoundEnd | Phase::Over => 0.0,
    };
    round.fuse = if let Device::Planted {
        fuse_ticks_left, ..
    } = view.device
    {
        seconds(*fuse_ticks_left, catalog)
    } else {
        0.0
    };
    let site = match view.device {
        Device::Planted { site, .. }
        | Device::Exploded { site, .. }
        | Device::Defused { site, .. } => catalog.sites.get(*site).map_or("", String::as_str),
        Device::Carried { .. } | Device::Dropped { .. } => "",
    };
    set(&mut round.site, site);
    round.time_low = view.phase == Phase::Live && round.time <= lows.time_seconds;
    round.fuse_low = planted && round.fuse <= lows.fuse_seconds;
    round.overtime = view.overtime;
    round.decider = view.decider;
    round.swapped = view.first_round_of_half && view.round > 1;
    round.last_of_half = view.last_round_of_half;
    round.buying = view.phase == Phase::Buy;
    facts.phase = Some(view.phase);
    for (index, team) in [team, team.other()].into_iter().enumerate() {
        facts.score[index] = view.score[team.index()];
        let side = view.side_of(team);
        facts.teams[index].color = tone(side);
        set(&mut facts.teams[index].name, side_name(words, side));
    }
}

pub(super) fn plates_part(
    facts: &mut HudFacts,
    view: &View<'_>,
    team: Team,
    healths: &[Option<f32>],
) {
    let catalog = view.catalog;
    for (index, team) in [team, team.other()].into_iter().enumerate() {
        let plates = &mut facts.players[index];
        let count = view
            .players
            .iter()
            .filter(|player| player.team == team)
            .count();
        plates.resize_with(count, PlateFacts::default);
        let members = view
            .players
            .iter()
            .enumerate()
            .filter(|(_, player)| player.team == team);
        for (plate, (slot, player)) in plates.iter_mut().zip(members) {
            named(&mut plate.icon, "char_", character_key(catalog, player));
            plate.alive = player.alive;
            plate.health = if player.alive {
                healths
                    .get(slot)
                    .copied()
                    .flatten()
                    .unwrap_or(catalog.health as f32)
            } else {
                0.0
            };
            plate.ult_ready = player.ultimate.points >= player.ultimate.cost;
        }
    }
}

pub(super) fn abilities_part(
    facts: &mut HudFacts,
    view: &View<'_>,
    (player, me): (&Player, &Player),
) {
    let catalog = view.catalog;
    let Some(entry) = catalog.characters.get(player.character) else {
        return;
    };
    for (index, slot) in [AbilitySlot::C, AbilitySlot::Q, AbilitySlot::E]
        .into_iter()
        .enumerate()
    {
        let shown = &mut facts.abilities[index];
        named(&mut shown.icon, "ability_", &entry.ability_names[index]);
        if let Some(ability) = player.ability(slot) {
            shown.max = ability.max;
            shown.charges = ability.charges;
            shown.empty = ability.charges == 0;
            shown.cooling = ability.cooldown_ticks > 0;
            shown.cooldown = seconds(ability.cooldown_ticks, catalog);
            shown.cooldown_frac = cooldown_share(catalog, player, (index, ability.cooldown_ticks));
        }
    }
    let Some(own) = catalog.characters.get(me.character) else {
        return;
    };
    let ultimate: &mut AbilityFacts = &mut facts.abilities[3];
    named(&mut ultimate.icon, "ability_", &own.ability_names[3]);
    ultimate.points = me.ultimate.points;
    ultimate.points_needed = me.ultimate.cost;
    ultimate.ready = me.ultimate.points >= me.ultimate.cost;
}

pub(super) fn cooldown_share(
    catalog: &Catalog,
    player: &Player,
    (index, ticks): (usize, u32),
) -> f32 {
    if ticks == 0 {
        return 0.0;
    }
    let full = catalog
        .characters
        .get(player.character)
        .filter(|_| index == 2)
        .and_then(|entry| entry.signature.cooldown_ticks)
        .filter(|full| *full > 0);
    full.map_or(1.0, |full| math::min(ticks as f32 / full as f32, 1.0))
}

const PLANT: &str = "plant";
const DEFUSE: &str = "defuse";

pub(super) fn action_part(facts: &mut HudFacts, view: &View<'_>, human: usize, words: &Words) {
    let catalog = view.catalog;
    let action = &mut facts.action;
    let device = catalog.device_name.as_str();
    let doing = if let Device::Carried {
        player,
        planting: Some(planting),
    } = view.device
        && *player == human
    {
        Some((
            (&words.planting, PLANT),
            planting.progress,
            planting.needed,
            Tone::Attack,
            None,
        ))
    } else if let Device::Planted {
        defusing:
            Defusing {
                player: Some(player),
                progress,
                needed,
                ..
            },
        ..
    } = view.device
        && *player == human
    {
        Some((
            (&words.defusing, DEFUSE),
            *progress,
            *needed,
            Tone::Defend,
            catalog.timings.defuse_checkpoint,
        ))
    } else {
        None
    };
    action.active = doing.is_some();
    if let Some(((label, kind), progress, needed, color, checkpoint)) = doing {
        worded(&mut action.label, label, &[("device", device)]);
        set(&mut action.kind, kind);
        action.left = seconds(needed.saturating_sub(progress), catalog);
        action.progress = progress as f32 / needed.max(1) as f32;
        action.color = color;
        action.checkpoint = checkpoint.is_some_and(|half| progress >= half);
    }
}

fn raw_stats(gun: &GunNumbers) -> WeaponStats {
    let scoped = gun
        .scope
        .as_ref()
        .map_or(gun.spread_deg.stand, |scope| scope.spread_deg.stand);
    WeaponStats {
        damage: gun.damage.first().map_or(0.0, |band| band.body),
        rate: gun.shots_per_second,
        range: 1.0 / (1.0 + 4.0 * math::min(gun.spread_deg.stand, scoped)),
    }
}

fn stats_of(numbers: Option<&WeaponNumbers>, name: &str) -> WeaponStats {
    let Some(numbers) = numbers else {
        return WeaponStats::default();
    };
    let mut most = WeaponStats::default();
    let mut found = WeaponStats::default();
    for kind in GUNS {
        let gun = numbers.gun(kind);
        let stats = raw_stats(gun);
        most.damage = math::max(most.damage, stats.damage);
        most.rate = math::max(most.rate, stats.rate);
        most.range = math::max(most.range, stats.range);
        if gun.name == name {
            found = stats;
        }
    }
    let share = |value: f32, top: f32| if top > 0.0 { value / top } else { 0.0 };
    WeaponStats {
        damage: share(found.damage, most.damage),
        rate: share(found.rate, most.rate),
        range: share(found.range, most.range),
    }
}

fn gated(refusal: BuyRefusal) -> bool {
    match refusal {
        BuyRefusal::WrongPhase
        | BuyRefusal::Dead
        | BuyRefusal::OutsideBuyZone
        | BuyRefusal::UnknownPlayer => true,
        BuyRefusal::NotForSale
        | BuyRefusal::TooFewCredits
        | BuyRefusal::AlreadyOwned
        | BuyRefusal::AtMaximum
        | BuyRefusal::NothingToSell => false,
    }
}

fn settle(
    card: &mut Card,
    (view, human, player): (&View<'_>, PlayerId, &Player),
    (item, cost, back): (Item, u32, u32),
    refund: Option<u32>,
) {
    let quote = view.price(human, item);
    card.afford = card.owned || quote.is_ok();
    card.closed = quote.is_err_and(gated);
    card.need = if quote == Err(BuyRefusal::TooFewCredits) {
        cost.saturating_sub(player.credits + back)
    } else {
        0
    };
    credits_into(&mut card.price, quote.unwrap_or(cost));
    card.buy = quote.is_ok().then_some(BuyRequest::Buy(item));
    card.ask = false;
    card.asked = false;
    if let Item::Weapon(weapon) = item
        && view.catalog.weapon(weapon).map(|entry| entry.slot) == Some(WeaponSlot::Primary)
    {
        card.asked = view.asked(human) == Some(weapon);
        card.ask = quote == Err(BuyRefusal::TooFewCredits) && !card.asked;
        if card.ask {
            card.buy = Some(BuyRequest::Ask(weapon));
        }
    }
    let refund = refund.filter(|_| !card.closed);
    card.sell_credits = refund.unwrap_or(0);
    card.sell = refund.map(|_| BuyRequest::Sell(item));
}

fn weapon_card(
    card: &mut Card,
    (view, numbers): (&View<'_>, Option<&WeaponNumbers>),
    id: WeaponId,
    (human, player, refund): (PlayerId, &Player, &Refund),
) {
    let Some(entry) = view.catalog.weapon(id) else {
        return;
    };
    let cost = entry.cost.unwrap_or(0);
    set(&mut card.kind, &entry.key);
    set(&mut card.name, &entry.name);
    named(&mut card.icon, "weapon_", &entry.name);
    card.max = 0;
    card.charges = 0;
    card.points = 0;
    card.stats = stats_of(numbers, &entry.name);
    let sold = entry.slot == WeaponSlot::Primary;
    card.owned = player.primary == Some(id) || (!sold && player.sidearm == Some(id));
    let back = if sold { refund.primary.unwrap_or(0) } else { 0 };
    let refund = refund
        .primary
        .filter(|_| sold && player.primary == Some(id));
    settle(
        card,
        (view, human, player),
        (Item::Weapon(id), cost, back),
        refund,
    );
}

fn armor_card(
    card: &mut Card,
    view: &View<'_>,
    tier: ArmorTier,
    (human, player, refund): (PlayerId, &Player, &Refund),
    words: &Words,
) {
    let numbers = view.catalog.armor(tier);
    card.price.clear();
    let _ = write!(card.price, "{}", numbers.points);
    worded(
        &mut card.kind,
        &words.armor_kind,
        &[("points", &card.price)],
    );
    let (name, icon) = match tier {
        ArmorTier::Light => (&words.light, "armor_light"),
        ArmorTier::Heavy => (&words.heavy, "armor_heavy"),
    };
    set(&mut card.name, name);
    set(&mut card.icon, icon);
    let held = player.armor;
    card.owned = held.is_some_and(|armor| armor.points >= numbers.points);
    card.max = 0;
    card.charges = 0;
    card.points = numbers.points;
    let mine = refund
        .armor
        .filter(|_| held.is_some_and(|armor| armor.tier == tier));
    settle(
        card,
        (view, human, player),
        (Item::Armor(tier), numbers.cost, refund.armor.unwrap_or(0)),
        mine,
    );
}

fn ability_card(
    card: &mut Card,
    view: &View<'_>,
    index: usize,
    (human, player, refund): (PlayerId, &Player, &Refund),
    words: &Words,
) {
    let Some(entry) = view.catalog.characters.get(player.character) else {
        return;
    };
    let slot = [AbilitySlot::C, AbilitySlot::Q][index];
    let bought = entry.bought[index];
    set(&mut card.kind, &words.per_charge);
    set(&mut card.name, &entry.ability_names[index]);
    named(&mut card.icon, "ability_", &entry.ability_names[index]);
    let ability = player.ability(slot);
    card.max = ability.map_or(bought.charges, |ability| ability.max);
    card.charges = ability.map_or(0, |ability| ability.charges);
    card.owned = card.charges >= card.max;
    let sellable = refund.charges[index] > 0 && card.charges > 0;
    settle(
        card,
        (view, human, player),
        (Item::Ability(slot), bought.cost, 0),
        sellable.then_some(bought.cost),
    );
}

fn signature_card(card: &mut Card, catalog: &Catalog, player: &Player, words: &Words) {
    let Some(entry) = catalog.characters.get(player.character) else {
        return;
    };
    set(&mut card.kind, &words.signature_free);
    set(&mut card.name, &entry.ability_names[2]);
    named(&mut card.icon, "ability_", &entry.ability_names[2]);
    set(&mut card.price, &words.free);
    let ability = player.ability(AbilitySlot::E);
    card.max = ability.map_or(0, |ability| ability.max);
    card.charges = ability.map_or(0, |ability| ability.charges);
    card.owned = true;
    card.afford = true;
    card.need = 0;
    card.sell_credits = 0;
    card.buy = None;
    card.sell = None;
}

fn team_part(team: &mut Vec<Mate>, view: &View<'_>, (human, player): (usize, &Player)) {
    let catalog = view.catalog;
    let rifle = catalog
        .weapon_id(RIFLE_KEY)
        .and_then(|id| catalog.weapon(id))
        .and_then(|entry| entry.cost);
    let mates = view
        .players
        .iter()
        .enumerate()
        .filter(|(slot, mate)| *slot != human && mate.team == player.team);
    team.resize_with(mates.clone().count(), Mate::default);
    for (shown, (slot, mate)) in team.iter_mut().zip(mates) {
        set(&mut shown.name, character_name(catalog, mate));
        named(&mut shown.icon, "char_", character_key(catalog, mate));
        shown.credits = mate.credits;
        shown.can_rifle = rifle.is_some_and(|cost| mate.credits >= cost);
        let asked = view.asked(slot);
        shown.asked = asked.is_some();
        let entry = asked.and_then(|weapon| catalog.weapon(weapon));
        match entry {
            Some(entry) => named(&mut shown.ask_icon, "weapon_", &entry.name),
            None => shown.ask_icon.clear(),
        }
        let quote = asked.map(|weapon| (weapon, view.price_for(human, weapon, slot)));
        shown.ask_price = entry.and_then(|entry| entry.cost).unwrap_or(0);
        shown.giveable = matches!(quote, Some((_, Ok(_))));
        shown.give = quote
            .filter(|(_, price)| price.is_ok())
            .map(|(weapon, _)| BuyRequest::BuyFor(weapon, slot));
    }
}

pub(super) fn buy_part(
    facts: &mut HudFacts,
    view: &View<'_>,
    (human, player): (usize, &Player),
    (lows, words): (&Lows, &Words),
    (numbers, order): (Option<&WeaponNumbers>, &mut Vec<(u32, WeaponId)>),
) {
    let catalog = view.catalog;
    let buy = &mut facts.buy;
    buy.time = if view.phase == Phase::Buy {
        seconds(view.ticks_left, catalog)
    } else {
        0.0
    };
    buy.time_low = view.phase == Phase::Buy && buy.time <= lows.buy_seconds;
    order.clear();
    order.push((0, catalog.sidearm));
    let start = order.len();
    for (index, entry) in catalog.weapons.iter().enumerate() {
        if entry.slot == WeaponSlot::Primary {
            order.push((entry.cost.unwrap_or(0), WeaponId(index)));
        }
    }
    order[start..].sort_unstable();
    let refund = view.refund(human);
    let mine = (human, player, &refund);
    buy.weapons.resize_with(order.len(), Card::default);
    for (card, (_, id)) in buy.weapons.iter_mut().zip(order.iter()) {
        weapon_card(card, (view, numbers), *id, mine);
    }
    let tiers = [ArmorTier::Light, ArmorTier::Heavy];
    buy.armor.resize_with(tiers.len(), Card::default);
    for (card, tier) in buy.armor.iter_mut().zip(tiers) {
        armor_card(card, view, tier, mine, words);
    }
    buy.abilities.resize_with(2, Card::default);
    for (index, card) in buy.abilities.iter_mut().enumerate() {
        ability_card(card, view, index, mine, words);
    }
    let gate = match view.can_buy(human) {
        Ok(()) => "",
        Err(BuyRefusal::OutsideBuyZone) => words.shop_zone.as_str(),
        Err(BuyRefusal::Dead) => words.shop_dead.as_str(),
        Err(
            BuyRefusal::WrongPhase
            | BuyRefusal::UnknownPlayer
            | BuyRefusal::NotForSale
            | BuyRefusal::TooFewCredits
            | BuyRefusal::AlreadyOwned
            | BuyRefusal::AtMaximum
            | BuyRefusal::NothingToSell,
        ) => words.shop_closed.as_str(),
    };
    set(&mut buy.gate, gate);
    signature_card(&mut buy.signature, catalog, player, words);
    buy.reward_win = catalog.economy.round_win;
    buy.reward_loss = catalog.loss_reward(view.loss_streak[player.team.index()]);
    team_part(&mut buy.team, view, (human, player));
}

pub(super) fn adr(view: &View<'_>, dealt: &[f32], slot: usize) -> u32 {
    let rounds = view.round.max(1) as f32;
    (dealt.get(slot).copied().unwrap_or(0.0) / rounds).round() as u32
}

fn gear(out: &mut String, catalog: &Catalog, player: &Player) {
    match player
        .primary
        .or(player.sidearm)
        .and_then(|id| catalog.weapon(id))
    {
        Some(entry) => named(out, "weapon_", &entry.name),
        None => out.clear(),
    }
}

pub(super) fn board_part(
    facts: &mut HudFacts,
    view: &View<'_>,
    (human, team, dealt): (usize, Team, &[f32]),
    words: &Words,
) {
    let catalog = view.catalog;
    for (index, listed) in [team, team.other()].into_iter().enumerate() {
        let rows = &mut facts.board[index];
        let count = view
            .players
            .iter()
            .filter(|player| player.team == listed)
            .count();
        rows.resize_with(count, BoardRow::default);
        let members = view
            .players
            .iter()
            .enumerate()
            .filter(|(_, player)| player.team == listed);
        for (row, (slot, player)) in rows.iter_mut().zip(members) {
            set(&mut row.name, character_name(catalog, player));
            named(&mut row.icon, "char_", character_key(catalog, player));
            row.k = player.kills;
            row.d = player.deaths;
            row.a = player.assists;
            row.adr = adr(view, dealt, slot);
            gear(&mut row.loadout, catalog, player);
            if listed == team {
                credits_into(&mut row.credits, player.credits);
            } else {
                set(&mut row.credits, &words.hidden);
            }
            row.ult = player.ultimate.points;
            row.ult_need = player.ultimate.cost;
            row.ult_ready = player.ultimate.points >= player.ultimate.cost;
            row.alive = player.alive;
            row.me = slot == human;
        }
    }
}
