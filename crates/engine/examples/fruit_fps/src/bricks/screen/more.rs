use engine::prelude::*;

use crate::play::actors::{Actor, Body, Human};
use crate::play::arms::{Arms, Drops, Held as Holding};
use crate::play::flow::Flow;
use crate::play::numbers::{GameNumbers, Numbers};
use crate::play::paths::Paths;
use crate::play::powers::{Held, Powers};
use crate::play::round::Round;
use crate::play::screen::{HudFacts, SelectCard};
use crate::play::view::{Watch, Watched};
use crate::rules::{AbilitySlot, Player, View};
use crate::weapons::GunKind;

use super::ScreenSource;
use super::facts::Recent;
use super::live::{character_key, character_name, cooldown_share, named, seconds, set, tone};
use super::menu::{KeyNames, binding_of, spoken_binding};

const ABILITY_KEYS: [&str; 4] = ["ability_c", "ability_q", "ability_e", "ability_x"];
const ABILITY_SLOTS: [AbilitySlot; 4] = [
    AbilitySlot::C,
    AbilitySlot::Q,
    AbilitySlot::E,
    AbilitySlot::X,
];

type Mine<'a> = (&'a Actor, Option<&'a Arms>, Option<&'a Body>);

type InRound<'w> = (
    Option<Res<'w, Round>>,
    Option<Res<'w, Watched>>,
    Option<Res<'w, Paths>>,
    Option<Res<'w, Powers>>,
    Res<'w, Recent>,
    (Option<Res<'w, Numbers>>, Option<Res<'w, Drops>>),
);

fn spoken_key(keys: &[(String, String)], name: &str, out: &mut String) {
    let spoken = keys
        .iter()
        .find(|(known, _)| known == name)
        .map_or("", |(_, spoken)| spoken.as_str());
    set(out, spoken);
}

fn pickup_part(
    facts: &mut HudFacts,
    (arms, numbers, drops): (Option<&Arms>, Option<&GameNumbers>, Option<&Drops>),
) {
    let pickup = &mut facts.pickup;
    let dropped = arms
        .and_then(|arms| arms.pickable)
        .zip(drops)
        .and_then(|(id, drops)| drops.get(id));
    let (Some(dropped), Some(numbers)) = (dropped, numbers) else {
        pickup.active = false;
        pickup.key.clear();
        pickup.weapon.clear();
        pickup.icon.clear();
        return;
    };
    pickup.active = true;
    spoken_key(&facts.keys, "interact", &mut pickup.key);
    let weapon = &numbers.weapons.gun(dropped.weapon).name;
    set(&mut pickup.weapon, weapon);
    named(&mut pickup.icon, "weapon_", weapon);
}

fn scope_ease((arms, numbers): (&Arms, &GameNumbers)) -> f32 {
    if !arms.scoped || arms.hand.held != Holding::Gun(GunKind::Sniper) {
        return 0.0;
    }
    let first = numbers
        .weapons
        .gun(GunKind::Sniper)
        .scope
        .as_ref()
        .and_then(|scope| scope.zoom.first().copied())
        .unwrap_or(1.0);
    if first <= 1.0 {
        return 1.0;
    }
    engine::math::clamp((arms.zoom - 1.0) / (first - 1.0), 0.0, 1.0)
}

fn cast_part(facts: &mut HudFacts, held: Option<&Held>) {
    let cast = &mut facts.cast;
    let Some(held) = held else {
        cast.active = false;
        cast.key.clear();
        cast.icon.clear();
        return;
    };
    let index = ABILITY_SLOTS
        .iter()
        .position(|slot| *slot == held.ability)
        .unwrap_or(0);
    cast.active = true;
    spoken_key(&facts.keys, ABILITY_KEYS[index], &mut cast.key);
    named(&mut cast.icon, "ability_", &held.name);
}

type Watchee<'a> = (&'a Actor, Option<&'a Arms>);

fn spectate_part(facts: &mut HudFacts, view: &View<'_>, watching: Option<&Player>) {
    let spectate = &mut facts.spectate;
    spectate.active = watching.is_some();
    let Some(player) = watching else {
        spectate.name.clear();
        spectate.icon.clear();
        spectate.color.clear();
        return;
    };
    set(&mut spectate.name, character_name(view.catalog, player));
    named(
        &mut spectate.icon,
        "char_",
        character_key(view.catalog, player),
    );
    set(&mut spectate.color, tone(view.side_of(player.team)).key());
}

pub(super) fn fill_more(
    (round, watched, paths, powers, recent, (numbers, drops)): InRound,
    humans: Query<Mine<'_>, With<Human>>,
    others: Query<Watchee<'_>>,
    time: Res<FixedTime>,
    mut facts: ResMut<HudFacts>,
) {
    if recent.shows_demo() {
        return;
    }
    let (Some(view), Some((actor, arms, body))) =
        (round.as_deref().and_then(Round::view), humans.iter().next())
    else {
        return;
    };
    let facts = &mut *facts;
    let game = numbers.as_deref().and_then(Numbers::get);
    let watching = watched
        .as_deref()
        .filter(|watched| watched.watch == Watch::Teammate)
        .and_then(|watched| watched.slot)
        .and_then(|slot| Some((slot, view.players.get(slot as usize)?)));
    spectate_part(facts, &view, watching.map(|(_, player)| player));
    let viewer = watching.map_or(actor.slot, |(slot, _)| slot);
    let seen_arms = match watching {
        Some((slot, _)) => others
            .iter()
            .find(|(other, _)| other.slot == slot)
            .and_then(|(_, arms)| arms),
        None => arms,
    };
    let place = paths
        .as_deref()
        .zip(body)
        .and_then(|(paths, body)| paths.place_at(body.feet));
    set(
        &mut facts.place.callout,
        place.map_or("", |place| place.callout.as_str()),
    );
    facts.scope = seen_arms
        .and_then(|arms| arms.loadout.active_gun())
        .map_or(0, |gun| gun.scope_level() as u32);
    facts.scope_ease = seen_arms.zip(game).map_or(0.0, scope_ease);
    pickup_part(facts, (arms, game, drops.as_deref()));
    cast_part(
        facts,
        powers.as_deref().and_then(|powers| powers.held(viewer)),
    );
    let until = powers
        .as_deref()
        .and_then(|powers| powers.weapon_down_until(viewer));
    facts.weapon.down_left = until.map_or(0.0, |until| {
        until.saturating_sub(time.tick) as f32 * time.delta_seconds
    });
    let (Some(powered), Some(own)) = (
        powers.as_deref().and_then(|powers| powers.actor(viewer)),
        powers
            .as_deref()
            .and_then(|powers| powers.actor(actor.slot)),
    ) else {
        return;
    };
    let me = view.players.get(viewer as usize);
    for (index, (shown, slot)) in facts.abilities.iter_mut().zip(&powered.slots).enumerate() {
        if index < 3 {
            shown.equipped = powered.held == Some(ABILITY_SLOTS[index]);
            shown.charges = slot.charges;
            shown.max = slot.max;
            shown.empty = slot.charges == 0;
            shown.cooling = slot.cooldown_ticks > 0;
            shown.cooldown = seconds(slot.cooldown_ticks, view.catalog);
            shown.cooldown_frac = me.map_or(0.0, |player| {
                cooldown_share(view.catalog, player, (index, slot.cooldown_ticks))
            });
        } else if let Some(slot) = own.slots.get(index) {
            shown.equipped = own.held == Some(ABILITY_SLOTS[index]);
            shown.ready = slot.ready;
        }
    }
}
fn card_of(
    card: &mut SelectCard,
    (key, character): (&str, &crate::rules::CharacterNumbers),
    (numbers, spoken): (&GameNumbers, &KeyNames),
    texts: &std::collections::BTreeMap<String, String>,
    images: &std::collections::BTreeMap<String, String>,
) {
    set(&mut card.key, key);
    set(&mut card.name, &character.name);
    set(&mut card.role, &character.role);
    named(&mut card.image, "char_", key);
    let image = images.get(&card.image).map_or("", String::as_str);
    set(&mut card.image, image);
    let names = [
        &character.c.name,
        &character.q.name,
        &character.e.name,
        &character.x.name,
    ];
    for (index, (shown, name)) in card.abilities.iter_mut().zip(names).enumerate() {
        set(&mut shown.name, name);
        named(&mut shown.icon, "ability_", name);
        set(
            &mut shown.desc,
            texts.get(name.as_str()).map_or("", String::as_str),
        );
        match binding_of(&numbers.keys, ABILITY_KEYS[index]) {
            Some(binding) => spoken_binding(binding, spoken, &mut shown.key),
            None => shown.key.clear(),
        }
        shown.ultimate = index == 3;
    }
}

type MenuReads<'w, 's> = (
    Option<Res<'w, Flow>>,
    Option<Res<'w, Numbers>>,
    Query<'w, 's, &'static ScreenSource>,
);

pub(super) fn fill_menu((flow, numbers, sources): MenuReads, mut facts: ResMut<HudFacts>) {
    let Some(source) = sources.iter().next() else {
        return;
    };
    let Some(loaded) = source.loaded() else {
        return;
    };
    facts.result.next_seconds = flow.as_deref().map_or(0.0, |flow| flow.next_match_left);
    let menu = &mut facts.menu;
    if let Some(game) = numbers.as_deref().and_then(Numbers::get) {
        let characters = &game.rules.characters;
        menu.select
            .cards
            .resize_with(characters.len(), SelectCard::default);
        for (card, (key, character)) in menu.select.cards.iter_mut().zip(characters) {
            card_of(
                card,
                (key, character),
                (game, &source.key_names),
                &loaded.layout.ability_texts,
                &loaded.select_images,
            );
        }
    }
    if let Some(flow) = flow.as_deref() {
        for card in &mut menu.select.cards {
            card.selected = card.key == flow.character;
        }
        set(&mut menu.select.level, &flow.level);
        menu.loading.progress = flow.progress;
        menu.loading.percent = (flow.progress * 100.0).round() as u32;
    }
    let words = &source.menu.loading;
    let loading = &mut menu.loading;
    set(&mut loading.map, &words.map);
    set(&mut loading.mode, &words.mode);
    set(&mut loading.site_a_note, &words.site_a_note);
    set(&mut loading.site_b_note, &words.site_b_note);
    let tips = &loaded.layout.tips;
    loading.tips.resize_with(tips.len(), String::new);
    for (shown, tip) in loading.tips.iter_mut().zip(tips) {
        set(shown, tip);
    }
}
