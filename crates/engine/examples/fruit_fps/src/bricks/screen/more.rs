use engine::prelude::*;

use crate::play::actors::{Actor, Body, Human};
use crate::play::arms::{Arms, Drops};
use crate::play::flow::Flow;
use crate::play::numbers::{GameNumbers, KeyMap, Numbers};
use crate::play::paths::Paths;
use crate::play::powers::{Held, Powers};
use crate::play::round::Round;
use crate::play::screen::{HudFacts, SelectCard};
use crate::play::view::{Watch, Watched};
use crate::rules::{AbilitySlot, Player, View};

use super::ScreenSource;
use super::facts::Recent;
use super::live::{character_key, character_name, named, seconds, set, tone};
use super::menu::{binding_of, spoken_binding};

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

fn spoken_key(keys: &KeyMap, name: &str, out: &mut String) {
    match binding_of(keys, name) {
        Some(binding) => spoken_binding(binding, out),
        None => out.clear(),
    }
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
    spoken_key(&numbers.keys, "interact", &mut pickup.key);
    let weapon = &numbers.weapons.gun(dropped.weapon).name;
    set(&mut pickup.weapon, weapon);
    named(&mut pickup.icon, "weapon_", weapon);
}

fn cast_part(facts: &mut HudFacts, held: Option<&Held>, numbers: Option<&GameNumbers>) {
    let cast = &mut facts.cast;
    let (Some(held), Some(numbers)) = (held, numbers) else {
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
    spoken_key(&numbers.keys, ABILITY_KEYS[index], &mut cast.key);
    named(&mut cast.icon, "ability_", &held.name);
}

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
        .and_then(|slot| view.players.get(slot as usize));
    spectate_part(facts, &view, watching);
    let place = paths
        .as_deref()
        .zip(body)
        .and_then(|(paths, body)| paths.place_at(body.feet));
    set(
        &mut facts.place.name,
        place.map_or("", |place| place.name.as_str()),
    );
    set(
        &mut facts.place.callout,
        place.map_or("", |place| place.callout.as_str()),
    );
    facts.scope = arms
        .and_then(|arms| arms.loadout.active_gun())
        .map_or(0, |gun| gun.scope_level() as u32);
    pickup_part(facts, (arms, game, drops.as_deref()));
    cast_part(
        facts,
        powers.as_deref().and_then(|powers| powers.held(actor.slot)),
        game,
    );
    let Some(powered) = powers
        .as_deref()
        .and_then(|powers| powers.actor(actor.slot))
    else {
        return;
    };
    for (index, (shown, slot)) in facts.abilities.iter_mut().zip(&powered.slots).enumerate() {
        if index < 3 {
            shown.charges = slot.charges;
            shown.max = slot.max;
            shown.empty = slot.charges == 0;
            shown.cooling = slot.cooldown_ticks > 0;
            shown.cooldown = seconds(slot.cooldown_ticks, view.catalog);
        } else {
            shown.ready = slot.ready;
        }
    }
}
fn card_of(
    card: &mut SelectCard,
    (key, character): (&str, &crate::rules::CharacterNumbers),
    numbers: &crate::play::numbers::GameNumbers,
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
            Some(binding) => spoken_binding(binding, &mut shown.key),
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
                game,
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
