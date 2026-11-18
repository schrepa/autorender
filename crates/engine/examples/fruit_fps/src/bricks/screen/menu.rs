use std::fmt::Write;

use engine::prelude::*;
use engine::{Devices, MouseButton};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::play::flow::{Flow, Stage};
use crate::play::numbers::{Binding, Input, KeyMap};
use crate::play::screen::{Area, HudFacts, KeyRow, ScreenDraw, Shown, Slider};
use crate::play::sound::Volume;
use crate::play::view::Controls;

use super::layout::Layout;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MenuKeys {
    pub confirm: Vec<String>,
    pub back: Vec<String>,
    pub up: Vec<String>,
    pub down: Vec<String>,
    pub left: Vec<String>,
    pub right: Vec<String>,
    pub levels: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SliderNumbers {
    pub min: f32,
    pub max: f32,
    pub step: f32,
    pub shown_scale: f32,
    pub digits: usize,
    pub suffix: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SliderSet {
    pub sensitivity: SliderNumbers,
    pub fov: SliderNumbers,
    pub volume: SliderNumbers,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct KeyName {
    pub action: String,
    pub key: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ScopeNumbers {
    pub ring_share: f32,
    pub width_px: f32,
    pub line_px: f32,
    pub color: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LoadingNumbers {
    pub map: String,
    pub mode: String,
    pub site_a_note: String,
    pub site_b_note: String,
    pub min_seconds: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MenuSource {
    pub keys: MenuKeys,
    pub levels: Vec<String>,
    pub sliders: SliderSet,
    pub key_rows: Vec<KeyName>,
    pub crosshairs: Vec<String>,
    pub scope: ScopeNumbers,
    pub backdrop: String,
    pub loading: LoadingNumbers,
}

pub(super) struct MenuInputs {
    pub confirm: Vec<Input>,
    pub back: Vec<Input>,
    pub up: Vec<Input>,
    pub down: Vec<Input>,
    pub left: Vec<Input>,
    pub right: Vec<Input>,
    pub levels: Vec<Input>,
}

settings! {
    #[section("options")]
    #[serde(deny_unknown_fields)]
    pub struct Options {
        pub crosshair: u32 = 0 => any,
    }
}

#[derive(Resource, Default)]
pub(super) struct OptionsWrite {
    controls: Option<Controls>,
    options: Option<Options>,
    volume: Option<Volume>,
    pub problem: Option<String>,
}

impl OptionsWrite {
    pub(super) fn controls_wanted(&mut self, controls: Controls) {
        self.controls = Some(controls);
    }

    pub(super) fn options_wanted(&mut self, options: Options) {
        self.options = Some(options);
    }

    pub(super) fn volume_wanted(&mut self, volume: Volume) {
        self.volume = Some(volume);
    }
}

pub(super) fn write_options(world: &mut World) {
    let (controls, options, volume) = {
        let mut held = world.resource_mut::<OptionsWrite>();
        (
            held.controls.take(),
            held.options.take(),
            held.volume.take(),
        )
    };
    let mut problem = None;
    if let Some(controls) = controls
        && let Err(error) = world.section::<Controls>().set(|held| *held = controls)
    {
        problem = Some(error.to_string());
    }
    if let Some(options) = options
        && let Err(error) = world.section::<Options>().set(|held| *held = options)
    {
        problem = Some(error.to_string());
    }
    if let Some(volume) = volume
        && let Err(error) = world.section::<Volume>().set(|held| *held = volume)
    {
        problem = Some(error.to_string());
    }
    if problem.is_some() {
        world.resource_mut::<OptionsWrite>().problem = problem;
    }
}

const TITLE: [&str; 3] = ["play", "settings", "quit"];
const TITLE_NO_QUIT: [&str; 2] = ["play", "settings"];
const PAUSE: [&str; 3] = ["resume", "settings", "quit"];
const MATCH_END: [&str; 2] = ["play", "quit"];
const SETTINGS_ROWS: usize = 8;
const CROSSHAIR_ROW: usize = 5;
const INVERT_ROW: usize = 6;
const BACK_ROW: usize = 7;

#[derive(Resource, Default)]
pub(super) struct MenuState {
    focus: usize,
    on: Option<Shown>,
    pub quit_asked: u32,
}

pub(super) struct Nav<'a> {
    pub devices: &'a Devices,
    pub keys: &'a MenuInputs,
    pub layout: &'a Layout,
    pub draw: &'a ScreenDraw,
    pub source: &'a MenuSource,
    pub can_quit: bool,
}

impl Nav<'_> {
    fn pressed(&self, binding: &[Input]) -> bool {
        KeyMap::pressed(self.devices, binding)
    }

    fn clicked(&self) -> bool {
        self.devices
            .mouse
            .buttons
            .pressed
            .contains(&MouseButton::Left)
    }

    fn area(&self, shown: Shown, id: &str) -> Option<Area> {
        let screen = self.layout.screens.get(&shown)?;
        let element = screen.elements.iter().find(|element| element.id == id)?;
        Some(
            self.draw
                .place(element.rect, element.anchor, element.unscaled),
        )
    }

    fn over(&self, shown: Shown, id: &str) -> Option<Area> {
        let pointer = self.devices.mouse.position_pixels?;
        self.area(shown, id).filter(|area| area.contains(pointer))
    }

    fn hovered<const N: usize>(&self, shown: Shown, ids: [&str; N]) -> Option<usize> {
        ids.iter().position(|id| self.over(shown, id).is_some())
    }

    fn step(&self, focus: usize, count: usize) -> usize {
        if count == 0 {
            return 0;
        }
        if self.pressed(&self.keys.down) || self.pressed(&self.keys.right) {
            (focus + 1) % count
        } else if self.pressed(&self.keys.up) || self.pressed(&self.keys.left) {
            (focus + count - 1) % count
        } else {
            focus.min(count - 1)
        }
    }
}

fn buttons<const N: usize>(
    nav: &Nav<'_>,
    state: &mut MenuState,
    shown: Shown,
    (names, ids): ([&'static str; N], [&str; N]),
) -> Option<&'static str> {
    if let Some(hover) = nav.hovered(shown, ids) {
        state.focus = hover;
        if nav.clicked() {
            return Some(names[hover]);
        }
    }
    state.focus = nav.step(state.focus, N);
    nav.pressed(&nav.keys.confirm).then(|| names[state.focus])
}

pub(super) fn start(flow: &mut Flow) {
    flow.stage = Stage::Loading;
    flow.game += 1;
}

fn title(nav: &Nav<'_>, state: &mut MenuState, flow: &mut Flow) {
    let chosen = if nav.can_quit {
        buttons(
            nav,
            state,
            Shown::Title,
            (TITLE, ["button_play", "button_settings", "button_quit"]),
        )
    } else {
        buttons(
            nav,
            state,
            Shown::Title,
            (TITLE_NO_QUIT, ["button_play", "button_settings"]),
        )
    };
    match chosen {
        Some("play") => flow.stage = Stage::Select,
        Some("settings") => flow.settings_open = true,
        Some(_) => state.quit_asked += 1,
        None => {}
    }
}

fn pause(nav: &Nav<'_>, state: &mut MenuState, flow: &mut Flow) {
    if nav.pressed(&nav.keys.back) {
        flow.stage = Stage::Playing;
        return;
    }
    let chosen = buttons(
        nav,
        state,
        Shown::Pause,
        (PAUSE, ["button_resume", "button_settings", "button_quit"]),
    );
    match chosen {
        Some("resume") => flow.stage = Stage::Playing,
        Some("settings") => flow.settings_open = true,
        Some(_) => flow.stage = Stage::Title,
        None => {}
    }
}

fn match_end(nav: &Nav<'_>, state: &mut MenuState, flow: &mut Flow) {
    let chosen = buttons(
        nav,
        state,
        Shown::MatchEnd,
        (MATCH_END, ["play_again", "quit"]),
    );
    match chosen {
        Some("play") => flow.stage = Stage::Select,
        Some(_) => flow.stage = Stage::Title,
        None => {}
    }
}

fn select(nav: &Nav<'_>, facts: &HudFacts, flow: &mut Flow) {
    if nav.pressed(&nav.keys.back) {
        flow.stage = Stage::Title;
        return;
    }
    let cards = &facts.menu.select.cards;
    let mut at = cards
        .iter()
        .position(|card| card.key == flow.character)
        .unwrap_or(0);
    if nav.pressed(&nav.keys.right) && !cards.is_empty() {
        at = (at + 1) % cards.len();
    } else if nav.pressed(&nav.keys.left) && !cards.is_empty() {
        at = (at + cards.len() - 1) % cards.len();
    }
    let mut id = String::new();
    for index in 0..cards.len() {
        id.clear();
        let _ = write!(id, "card_{index}");
        if nav.clicked() && nav.over(Shown::Select, &id).is_some() {
            at = index;
        }
    }
    if let Some(card) = cards.get(at) {
        flow.character.clone_from(&card.key);
    }
    let levels = &nav.source.levels;
    for (index, level) in levels.iter().enumerate() {
        id.clear();
        let _ = write!(id, "bots_{level}");
        let keyed = nav
            .keys
            .levels
            .get(index)
            .is_some_and(|key| nav.pressed(std::slice::from_ref(key)));
        if keyed || (nav.clicked() && nav.over(Shown::Select, &id).is_some()) {
            flow.level.clone_from(level);
        }
    }
    if nav.pressed(&nav.keys.confirm)
        || (nav.clicked() && nav.over(Shown::Select, "play").is_some())
    {
        start(flow);
    }
}

fn frac_of(value: f32, numbers: &SliderNumbers) -> f32 {
    let span = numbers.max - numbers.min;
    if span > 0.0 {
        engine::math::clamp((value - numbers.min) / span, 0.0, 1.0)
    } else {
        0.0
    }
}

fn value_of(frac: f32, numbers: &SliderNumbers) -> f32 {
    let value = numbers.min + engine::math::clamp(frac, 0.0, 1.0) * (numbers.max - numbers.min);
    if numbers.step > 0.0 {
        engine::math::clamp(
            numbers.min + ((value - numbers.min) / numbers.step).round() * numbers.step,
            numbers.min,
            numbers.max,
        )
    } else {
        value
    }
}

pub(super) struct Values {
    pub controls: Controls,
    pub options: Options,
    pub volume: Volume,
}

impl Values {
    fn slider(&self, row: usize) -> f32 {
        match row {
            0 => self.controls.radians_per_pixel,
            1 => self.controls.fov_degrees,
            2 => self.volume.master,
            3 => self.volume.effects,
            _ => self.volume.interface,
        }
    }

    fn set_slider(&mut self, row: usize, value: f32) {
        match row {
            0 => self.controls.radians_per_pixel = value,
            1 => self.controls.fov_degrees = value,
            2 => self.volume.master = value,
            3 => self.volume.effects = value,
            _ => self.volume.interface = value,
        }
    }
}

fn numbers_of(sliders: &SliderSet, row: usize) -> &SliderNumbers {
    match row {
        0 => &sliders.sensitivity,
        1 => &sliders.fov,
        _ => &sliders.volume,
    }
}

const SLIDER_IDS: [&str; 5] = [
    "sensitivity_slider",
    "fov_slider",
    "master_slider",
    "effects_slider",
    "interface_slider",
];

fn settings(nav: &Nav<'_>, state: &mut MenuState, flow: &mut Flow, values: &mut Values) {
    if nav.pressed(&nav.keys.back) {
        flow.settings_open = false;
        return;
    }
    if nav.pressed(&nav.keys.down) {
        state.focus = (state.focus + 1) % SETTINGS_ROWS;
    } else if nav.pressed(&nav.keys.up) {
        state.focus = (state.focus + SETTINGS_ROWS - 1) % SETTINGS_ROWS;
    }
    let sliders = &nav.source.sliders;
    let turn = i32::from(nav.pressed(&nav.keys.right)) - i32::from(nav.pressed(&nav.keys.left));
    if turn != 0 && state.focus < CROSSHAIR_ROW {
        let numbers = numbers_of(sliders, state.focus);
        let next = values.slider(state.focus) + turn as f32 * numbers.step;
        values.set_slider(
            state.focus,
            engine::math::clamp(next, numbers.min, numbers.max),
        );
    }
    let count = nav.source.crosshairs.len() as u32;
    if turn != 0 && state.focus == CROSSHAIR_ROW && count > 0 {
        values.options.crosshair =
            (values.options.crosshair + count).wrapping_add_signed(turn) % count;
    }
    let confirmed = nav.pressed(&nav.keys.confirm);
    if state.focus == INVERT_ROW && (turn != 0 || confirmed) {
        values.controls.invert_pitch = !values.controls.invert_pitch;
    }
    if confirmed && state.focus == BACK_ROW {
        flow.settings_open = false;
    }
    if !nav.clicked() {
        return;
    }
    if nav.over(Shown::Settings, "invert").is_some() {
        values.controls.invert_pitch = !values.controls.invert_pitch;
        state.focus = INVERT_ROW;
    }
    for (row, id) in SLIDER_IDS.iter().enumerate() {
        if let (Some(area), Some(pointer)) = (
            nav.over(Shown::Settings, id),
            nav.devices.mouse.position_pixels,
        ) {
            let frac = if area.w > 0.0 {
                (pointer.x - area.x) / area.w
            } else {
                0.0
            };
            values.set_slider(row, value_of(frac, numbers_of(sliders, row)));
            state.focus = row;
        }
    }
    let mut id = String::new();
    for index in 0..count {
        id.clear();
        let _ = write!(id, "swatch_{index}");
        if nav.over(Shown::Settings, &id).is_some() {
            values.options.crosshair = index;
            state.focus = CROSSHAIR_ROW;
        }
    }
    if nav.over(Shown::Settings, "back").is_some() {
        flow.settings_open = false;
    }
}

pub(super) fn chosen_screen(flow: &Flow) -> Option<Shown> {
    if !flow.restored {
        return Some(Shown::Loading);
    }
    if flow.settings_open {
        return Some(Shown::Settings);
    }
    match flow.stage {
        Stage::Title => Some(Shown::Title),
        Stage::Select => Some(Shown::Select),
        Stage::Loading => Some(Shown::Loading),
        Stage::Paused => Some(Shown::Pause),
        Stage::MatchEnd => Some(Shown::MatchEnd),
        Stage::Playing => None,
    }
}

pub(super) fn navigate(
    nav: &Nav<'_>,
    (state, flow, facts): (&mut MenuState, &mut Flow, &mut HudFacts),
    values: &mut Values,
) {
    let shown = chosen_screen(flow);
    if shown != state.on {
        state.on = shown;
        state.focus = 0;
    }
    match shown {
        Some(Shown::Settings) => settings(nav, state, flow, values),
        Some(Shown::Title) => title(nav, state, flow),
        Some(Shown::Select) => select(nav, facts, flow),
        Some(Shown::Pause) => pause(nav, state, flow),
        Some(Shown::MatchEnd) => match_end(nav, state, flow),
        Some(Shown::Loading | Shown::Hud | Shown::Buy | Shown::Scoreboard | Shown::RoundEnd)
        | None => {}
    }
    let menu = &mut facts.menu;
    menu.screen = shown;
    menu.focus.clear();
    let focus = match shown {
        Some(Shown::Title) => TITLE.get(state.focus).copied(),
        Some(Shown::Pause) => PAUSE.get(state.focus).copied(),
        Some(Shown::MatchEnd) => MATCH_END.get(state.focus).copied(),
        Some(Shown::Settings) => match state.focus {
            INVERT_ROW => Some("invert"),
            BACK_ROW => Some("back"),
            _ => None,
        },
        Some(
            Shown::Select
            | Shown::Loading
            | Shown::Hud
            | Shown::Buy
            | Shown::Scoreboard
            | Shown::RoundEnd,
        )
        | None => None,
    };
    menu.focus.push_str(focus.unwrap_or(""));
}

pub(super) fn slider_facts(sliders: &mut [Slider; 5], source: &SliderSet, values: &Values) {
    for (row, slider) in sliders.iter_mut().enumerate() {
        let numbers = numbers_of(source, row);
        let value = values.slider(row);
        slider.frac = frac_of(value, numbers);
        slider.text.clear();
        let _ = write!(
            slider.text,
            "{:.*}{}",
            numbers.digits,
            value * numbers.shown_scale,
            numbers.suffix
        );
    }
}

const MOUSE_NAMES: [(MouseButton, &str); 5] = [
    (MouseButton::Left, "Mouse Left"),
    (MouseButton::Middle, "Mouse Middle"),
    (MouseButton::Right, "Mouse Right"),
    (MouseButton::Back, "Mouse Back"),
    (MouseButton::Forward, "Mouse Forward"),
];

fn spoken_input(input: &Input, out: &mut String) {
    match input {
        Input::Key(code) => {
            let short = code
                .strip_prefix("Key")
                .or_else(|| code.strip_prefix("Digit"))
                .or_else(|| code.strip_prefix("Arrow"))
                .unwrap_or(code);
            out.push_str(short);
        }
        Input::Mouse(button) => {
            let name = MOUSE_NAMES
                .iter()
                .find(|(known, _)| known == button)
                .map_or("Mouse", |(_, name)| name);
            out.push_str(name);
        }
    }
}

pub(super) fn binding_of<'a>(keys: &'a KeyMap, name: &str) -> Option<&'a Binding> {
    Some(match name {
        "forward" => &keys.forward,
        "back" => &keys.back,
        "left" => &keys.left,
        "right" => &keys.right,
        "walk" => &keys.walk,
        "crouch" => &keys.crouch,
        "jump" => &keys.jump,
        "fire" => &keys.fire,
        "scope" => &keys.scope,
        "reload" => &keys.reload,
        "primary" => &keys.primary,
        "sidearm" => &keys.sidearm,
        "melee" => &keys.melee,
        "device" => &keys.device,
        "interact" => &keys.interact,
        "ability_c" => &keys.ability_c,
        "ability_q" => &keys.ability_q,
        "ability_e" => &keys.ability_e,
        "ability_x" => &keys.ability_x,
        "buy_menu" => &keys.buy_menu,
        "scoreboard" => &keys.scoreboard,
        "capture" => &keys.capture,
        "release" => &keys.release,
        _ => return None,
    })
}

pub(super) fn spoken_binding(binding: &[Input], out: &mut String) {
    out.clear();
    for (index, input) in binding.iter().enumerate() {
        if index > 0 {
            out.push_str(" / ");
        }
        spoken_input(input, out);
    }
}

pub(super) fn key_rows(rows: &mut Vec<KeyRow>, names: &[KeyName], keys: &KeyMap) {
    rows.resize_with(names.len(), KeyRow::default);
    for (row, name) in rows.iter_mut().zip(names) {
        if row.action != name.action {
            row.action.clone_from(&name.action);
        }
        match binding_of(keys, &name.key) {
            Some(binding) => spoken_binding(binding, &mut row.binding),
            None => row.binding.clear(),
        }
    }
}
