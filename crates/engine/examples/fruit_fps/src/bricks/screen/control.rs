use engine::prelude::*;
use engine::{CursorMode, Devices};
use engine_core::Closing;

use crate::play::flow::{Flow, Stage};
use crate::play::numbers::{KeyMap, Numbers};
use crate::play::screen::{Card, HudFacts, Reticle, ScreenBuys, ScreenDraw, ScreenState, Shown};
use crate::play::sound::Volume;
use crate::play::view::Controls;
use crate::rules::{BuyRequest, Phase};

use super::layout::{CardList, Layout};
use super::menu::{self, MenuState, Nav, Options, OptionsWrite, Values};
use super::{Loaded, ScreenBook, ScreenSetup, ScreenSource, Showing};

fn automatic(facts: &HudFacts, state: ScreenState) -> Shown {
    match facts.phase {
        Some(Phase::Over) => Shown::MatchEnd,
        Some(Phase::Buy | Phase::Live | Phase::Planted | Phase::RoundEnd) | None => {
            if state.board_held {
                Shown::Scoreboard
            } else if state.buy_open {
                Shown::Buy
            } else if facts.phase == Some(Phase::RoundEnd) {
                Shown::RoundEnd
            } else {
                Shown::Hud
            }
        }
    }
}

fn chosen(show: Showing) -> Option<Shown> {
    match show {
        Showing::Auto => None,
        Showing::Hud => Some(Shown::Hud),
        Showing::Buy => Some(Shown::Buy),
        Showing::Scoreboard => Some(Shown::Scoreboard),
        Showing::RoundEnd => Some(Shown::RoundEnd),
        Showing::MatchEnd => Some(Shown::MatchEnd),
        Showing::Title => Some(Shown::Title),
        Showing::CharacterSelect => Some(Shown::Select),
        Showing::Loading => Some(Shown::Loading),
        Showing::Pause => Some(Shown::Pause),
        Showing::Settings => Some(Shown::Settings),
    }
}

pub(super) fn card(facts: &HudFacts, (list, index): (CardList, usize)) -> Option<&Card> {
    match list {
        CardList::Weapons => facts.buy.weapons.get(index),
        CardList::Armor => facts.buy.armor.get(index),
        CardList::Abilities => facts.buy.abilities.get(index),
        CardList::Team => None,
    }
}

fn actions(
    facts: &HudFacts,
    (list, index): (CardList, usize),
) -> (Option<BuyRequest>, Option<BuyRequest>) {
    match list {
        CardList::Team => (facts.buy.team.get(index).and_then(|mate| mate.give), None),
        CardList::Weapons | CardList::Armor | CardList::Abilities => {
            card(facts, (list, index)).map_or((None, None), |found| (found.buy, found.sell))
        }
    }
}

fn hovered(layout: &Layout, draw: &ScreenDraw, pointer: Option<Vec2>) -> Option<(CardList, usize)> {
    let pointer = pointer?;
    let screen = layout.screens.get(&Shown::Buy)?;
    screen.elements.iter().rev().find_map(|element| {
        let card = element.card?;
        draw.place(element.rect, element.anchor)
            .contains(pointer)
            .then_some(card)
    })
}

fn buying(
    (devices, loaded, draw): (&Devices, &Loaded, &ScreenDraw),
    facts: &HudFacts,
    buys: &mut ScreenBuys,
    book: &mut ScreenBook,
) {
    let keys = &loaded.keys;
    let over = hovered(&loaded.layout, draw, book.pointer);
    book.hovered = over;
    let mut asked = |request| {
        if let Some(request) = request {
            buys.requests.push(request);
            book.buys += 1;
        }
    };
    if let Some((take, give_back)) = over.map(|over| actions(facts, over)) {
        if KeyMap::pressed(devices, &keys.take) {
            asked(take);
        }
        if KeyMap::pressed(devices, &keys.give_back) {
            asked(give_back);
        }
    }
    for (binding, found) in keys.pick.iter().zip(facts.cards()) {
        if KeyMap::pressed(devices, std::slice::from_ref(binding)) {
            asked(found.buy);
        }
    }
}

fn menus(
    nav: &Nav<'_>,
    (flow, state): (&mut Flow, &mut ScreenState),
    facts: &mut HudFacts,
    (menu_state, values): (&mut MenuState, &mut Values),
) {
    let escape = KeyMap::pressed(nav.devices, &nav.keys.back);
    if flow.stage == Stage::Playing && !flow.settings_open {
        if escape && state.buy_open {
            state.buy_open = false;
        } else if escape {
            flow.stage = Stage::Paused;
        }
        facts.menu.screen = None;
        return;
    }
    menu::navigate(nav, (menu_state, flow, facts), values);
}

fn options_facts(
    facts: &mut HudFacts,
    (source, flow): (&ScreenSource, Option<&Flow>),
    values: &Values,
    numbers: Option<&Numbers>,
) {
    let settings = &mut facts.menu.settings;
    if let Some(flow) = flow {
        super::live::set(&mut settings.level, &flow.level);
    }
    menu::slider_facts(&mut settings.sliders, &source.menu.sliders, values);
    settings.crosshair = values.options.crosshair as usize;
    settings.invert = values.controls.invert_pitch;
    let options = &values.options;
    settings.reticle = Reticle {
        gap: options.crosshair_gap,
        length: options.crosshair_length,
        thickness: options.crosshair_thickness,
        outline: options.crosshair_outline,
        dot: options.crosshair_dot,
    };
    let color = source
        .menu
        .crosshairs
        .get(settings.crosshair)
        .map_or("", String::as_str);
    super::live::set(&mut settings.crosshair_color, color);
    let Some(numbers) = numbers.and_then(Numbers::get) else {
        return;
    };
    let names = &source.key_names;
    menu::key_rows(
        &mut settings.keys,
        (&source.menu.key_rows, names),
        &numbers.keys,
    );
    let shop = source.loaded().map(|loaded| {
        [
            ("take", &loaded.keys.take),
            ("give_back", &loaded.keys.give_back),
        ]
    });
    let all = menu::bindings(&numbers.keys)
        .into_iter()
        .chain(shop.into_iter().flatten());
    menu::key_facts(&mut facts.keys, all, names);
}

type Reads<'w> = (
    Res<'w, Devices>,
    Res<'w, ScreenSetup>,
    Res<'w, ScreenDraw>,
    Option<Res<'w, Numbers>>,
    (Res<'w, Controls>, Res<'w, Options>, Res<'w, Volume>),
);

type Writes<'w> = (
    ResMut<'w, CursorMode>,
    ResMut<'w, ScreenState>,
    ResMut<'w, ScreenBuys>,
    ResMut<'w, ScreenBook>,
    ResMut<'w, MenuState>,
    ResMut<'w, OptionsWrite>,
);

pub(super) fn control(
    (devices, setup, draw, numbers, (controls, options, volume)): Reads,
    sources: Query<&ScreenSource>,
    (mut cursor, mut state, mut buys, mut book, mut menu_state, mut writes): Writes,
    (mut flow, mut closing): (Option<ResMut<Flow>>, ResMut<Closing>),
    mut facts: ResMut<HudFacts>,
) {
    let Some(source) = sources.iter().next() else {
        return;
    };
    let Some(loaded) = source.loaded() else {
        return;
    };
    let facts = &mut *facts;
    let map = numbers
        .as_deref()
        .and_then(Numbers::get)
        .map(|numbers| &numbers.keys);
    let mut next = *state;
    let mut values = Values {
        controls: controls.clone(),
        options: options.clone(),
        volume: volume.clone(),
    };
    if let Some(flow) = flow.as_deref_mut() {
        let nav = Nav {
            devices: &devices,
            keys: &loaded.menu_keys,
            layout: &loaded.layout,
            draw: &draw,
            source: &source.menu,
            can_quit: closing.offered(),
        };
        let asked = menu_state.quit_asked;
        menus(
            &nav,
            (flow, &mut next),
            facts,
            (&mut menu_state, &mut values),
        );
        if menu_state.quit_asked != asked
            && let Err(error) = closing.ask()
        {
            writes.problem = Some(error.to_string());
        }
    }
    facts.menu.can_quit = closing.offered();
    if values.controls != *controls {
        writes.controls_wanted(values.controls.clone());
    }
    if values.volume != *volume {
        writes.volume_wanted(values.volume.clone());
    }
    if values.options != *options {
        writes.options_wanted(values.options.clone());
    }
    options_facts(
        facts,
        (source, flow.as_deref()),
        &values,
        numbers.as_deref(),
    );
    let playing = flow
        .as_deref()
        .is_none_or(|flow| flow.stage == Stage::Playing && !flow.settings_open);
    if facts.phase != Some(Phase::Buy) || !playing {
        next.buy_open = false;
    } else if map.is_some_and(|map| KeyMap::pressed(&devices, &map.buy_menu)) {
        next.buy_open = !next.buy_open;
    }
    next.board_held = playing && map.is_some_and(|map| KeyMap::held(&devices, &map.scoreboard));
    let flowing = flow.as_deref().and_then(menu::chosen_screen);
    next.shown = chosen(setup.show)
        .or(flowing)
        .unwrap_or_else(|| automatic(facts, next));
    next.holds_pointer =
        next.shown == Shown::Buy || flow.as_deref().is_some_and(Flow::holds_pointer);
    if next.holds_pointer != state.holds_pointer {
        cursor.set_if_neq(if next.holds_pointer {
            CursorMode::Free
        } else {
            CursorMode::Captured
        });
    }
    state.set_if_neq(next);
    book.pointer = devices.mouse.position_pixels;
    book.hovered = None;
    if next.shown == Shown::Buy {
        buying((&devices, loaded, &draw), facts, &mut buys, &mut book);
    }
    let hovered = book.hovered;
    let lists = [
        (CardList::Weapons, &mut facts.buy.weapons),
        (CardList::Armor, &mut facts.buy.armor),
        (CardList::Abilities, &mut facts.buy.abilities),
    ];
    for (list, cards) in lists {
        for (index, card) in cards.iter_mut().enumerate() {
            card.focus = hovered == Some((list, index));
        }
    }
    for (index, mate) in facts.buy.team.iter_mut().enumerate() {
        mate.focus = hovered == Some((CardList::Team, index));
    }
}
