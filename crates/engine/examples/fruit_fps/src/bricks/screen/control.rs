use brick_interface::{Element, InterfaceDraw};
use engine::prelude::*;
use engine::{CursorMode, Devices, Pointer};
use engine_core::{Closing, DevSetting};
use engine_touch::TouchState;

use crate::play::camera::TestCamera;
use crate::play::flow::{Flow, Stage};
use crate::play::numbers::Numbers;
use crate::play::screen::{Options, ScreenBuys, ScreenSounds, ScreenState, UiSound};
use crate::play::sound::Volume;
use crate::play::touch;
use crate::play::view::Controls;
use crate::rules::{BuyRequest, Phase};

use super::hud::{Card, HudFacts, Reticle, Shown};
use super::menu::{self, MenuState, Nav, OptionsWrite, Values};
use super::{
    Loaded, ScreenBook, ScreenLoad, ScreenSetup, ScreenSource, Showing, hud_hidden, paths,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CardList {
    Weapons,
    Armor,
    Abilities,
    Team,
}

fn card_of(element: &Element) -> Option<(CardList, usize)> {
    element.style_conditions().find_map(paths::card)
}

fn automatic(facts: &HudFacts, state: Showing) -> Shown {
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

fn hovered(draw: &InterfaceDraw, pointer: &Pointer) -> Option<(CardList, usize)> {
    draw.under_pointer(Shown::Buy as usize, pointer)
        .find_map(card_of)
}

fn buying(
    (devices, pointer, loaded, draw): (&Devices, &Pointer, &Loaded, &InterfaceDraw),
    facts: &HudFacts,
    buys: &mut ScreenBuys,
    book: &mut ScreenBook,
    sounds: &mut ScreenSounds,
    prev_hovered: Option<(CardList, usize)>,
) {
    let keys = &loaded.keys;
    let over = hovered(draw, pointer);
    let shown_hover = over.filter(|_| pointer.hovering);
    if shown_hover.is_some() && shown_hover != prev_hovered {
        sounds.play(UiSound::Hover);
    }
    book.hovered = shown_hover;
    let mut asked = |request| {
        if let Some(request) = request {
            buys.requests.push(request);
            book.buys += 1;
        }
    };
    if let Some((take, give_back)) = over.map(|over| actions(facts, over)) {
        if pointer.pressed || engine_bindings::pressed(devices, &keys.take) {
            asked(take);
        }
        if engine_bindings::pressed(devices, &keys.give_back) {
            asked(give_back);
        }
    }
    for (binding, found) in keys.pick.iter().zip(facts.cards()) {
        if engine_bindings::pressed(devices, std::slice::from_ref(binding)) {
            asked(found.buy);
        }
    }
}

fn menus(
    nav: &Nav<'_>,
    touch_pause: bool,
    (flow, state): (&mut Flow, &mut Showing),
    facts: &mut HudFacts,
    (menu_state, values, sounds): (&mut MenuState, &mut Values, &mut ScreenSounds),
) {
    let escape = engine_bindings::pressed(nav.devices, &nav.keys.back);
    if flow.stage == Stage::Playing && !flow.settings_open {
        if escape && state.buy_open {
            state.buy_open = false;
            sounds.play(UiSound::Click);
        } else if escape || touch_pause {
            flow.stage = Stage::Paused;
            sounds.play(UiSound::Click);
        }
        facts.menu.screen = None;
        return;
    }
    menu::navigate(nav, (menu_state, flow, facts), values, sounds);
}

fn options_facts(
    facts: &mut HudFacts,
    (source, loaded, flow): (&ScreenSource, &Loaded, Option<&Flow>),
    (values, shown): (&Values, bool),
    numbers: Option<&Numbers>,
) {
    let settings = &mut facts.menu.settings;
    if shown {
        if let Some(flow) = flow {
            settings.level = source.menu.levels.iter().position(|level| *level == flow.level);
        }
        menu::slider_facts(&mut settings.sliders, &source.menu.sliders, values);
    }
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
    if shown {
        menu::key_rows(
            &mut settings.keys,
            (&source.menu.key_rows, names),
            &numbers.keys,
        );
    }
    let shop = [&loaded.keys.take, &loaded.keys.give_back];
    let all = menu::bindings(&numbers.keys).into_iter().chain(shop);
    menu::key_facts(&mut facts.keys, all, names);
}

type Reads<'w> = (
    (Res<'w, Devices>, Res<'w, Pointer>, Option<Res<'w, TouchState>>),
    DevSetting<'w, ScreenSetup>,
    Res<'w, InterfaceDraw>,
    Option<Res<'w, Numbers>>,
    (Res<'w, Controls>, Res<'w, Options>, Res<'w, Volume>),
    (Res<'w, ScreenLoad>, Option<Res<'w, TestCamera>>),
);

type Writes<'w> = (
    ResMut<'w, CursorMode>,
    (ResMut<'w, ScreenState>, ResMut<'w, Showing>),
    (ResMut<'w, ScreenBuys>, ResMut<'w, ScreenSounds>),
    ResMut<'w, ScreenBook>,
    ResMut<'w, MenuState>,
    ResMut<'w, OptionsWrite>,
);

pub(super) fn control(
    ((devices, pointer, touch_state), setup, draw, numbers, (controls, options, volume), (load, camera)): Reads,
    sources: Query<&ScreenSource>,
    (
        mut cursor,
        (mut state, mut showing),
        (mut buys, mut sounds),
        mut book,
        mut menu_state,
        mut writes,
    ): Writes,
    (mut flow, mut closing): (Option<ResMut<Flow>>, ResMut<Closing>),
    mut facts: ResMut<HudFacts>,
) {
    let Some(source) = sources.iter().next() else {
        return;
    };
    let Some(loaded) = load.loaded() else {
        return;
    };
    let facts = &mut *facts;
    let map = numbers
        .as_deref()
        .and_then(Numbers::get)
        .map(|numbers| &numbers.keys);
    let touch_pause = touch_state
        .as_deref()
        .is_some_and(|state| state.pressed(touch::PAUSE));
    let touch_buy = touch_state
        .as_deref()
        .is_some_and(|state| state.pressed(touch::BUY));
    let touch_close = touch_state
        .as_deref()
        .is_some_and(|state| state.pressed(touch::CLOSE));
    let mut next = *showing;
    let mut values = Values {
        controls: controls.clone(),
        options: options.clone(),
        volume: volume.clone(),
    };
    if let Some(flow) = flow.as_deref_mut() {
        let nav = Nav {
            devices: &devices,
            pointer: &pointer,
            keys: &loaded.menu_keys,
            draw: &draw,
            source: &source.menu,
            can_quit: closing.offered(),
        };
        let asked = menu_state.quit_asked;
        menus(
            &nav,
            touch_pause,
            (flow, &mut next),
            facts,
            (&mut menu_state, &mut values, &mut sounds),
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
    let playing = flow
        .as_deref()
        .is_none_or(|flow| flow.stage == Stage::Playing && !flow.settings_open);
    if facts.phase != Some(Phase::Buy) || !playing {
        next.buy_open = false;
    } else if touch_close && next.buy_open {
        next.buy_open = false;
        sounds.play(UiSound::Click);
    } else if touch_buy || map.is_some_and(|map| engine_bindings::pressed(&devices, &map.buy_menu))
    {
        next.buy_open = !next.buy_open;
        sounds.play(UiSound::Click);
    }
    next.board_held = !touch_close
        && playing
        && map.is_some_and(|map| engine_bindings::held(&devices, &map.scoreboard));
    let flowing = flow.as_deref().and_then(menu::chosen_screen);
    next.shown = setup
        .get()
        .and_then(|setup| setup.show)
        .or(flowing)
        .unwrap_or_else(|| automatic(facts, next));
    let holds_pointer =
        next.shown == Shown::Buy || flow.as_deref().is_some_and(Flow::holds_pointer);
    if holds_pointer != state.holds_pointer {
        cursor.set_if_neq(if holds_pointer {
            CursorMode::Free
        } else {
            CursorMode::Captured
        });
    }
    state.set_if_neq(ScreenState { holds_pointer });
    showing.set_if_neq(next);
    options_facts(
        facts,
        (source, loaded, flow.as_deref()),
        (&values, next.shown == Shown::Settings),
        numbers.as_deref(),
    );
    facts.screen = (!hud_hidden(camera.as_deref(), next.shown)).then_some(next.shown);
    book.pointer = pointer.position_pixels;
    let prev_hovered = book.hovered;
    book.hovered = None;
    if next.shown == Shown::Buy && !touch_close {
        buying(
            (&devices, &pointer, loaded, &draw),
            facts,
            &mut buys,
            &mut book,
            &mut sounds,
            prev_hovered,
        );
    }
    let hovered = book.hovered;
    apply_buy_focus(facts, hovered);
}

fn apply_buy_focus(facts: &mut HudFacts, hovered: Option<(CardList, usize)>) {
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
