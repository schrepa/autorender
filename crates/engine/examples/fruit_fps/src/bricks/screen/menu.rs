use std::collections::BTreeMap;
use std::fmt::Write;

use brick_interface::{Area, InterfaceDraw, Slot};
use engine::prelude::*;
use engine::{Devices, Pointer};
use engine_bindings::{Binding, Input};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::play::flow::{Flow, Stage};
use crate::play::numbers::KeyMap;
use crate::play::screen::{Options, ScreenSounds, UiSound};
use crate::play::sound::Volume;
use crate::play::view::Controls;

use super::ScreenSource;
use super::hud::{HudFacts, KeyRow, Shown, Slider};

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
    pub render_scale: SliderNumbers,
    pub crosshair_gap: SliderNumbers,
    pub crosshair_length: SliderNumbers,
    pub crosshair_thickness: SliderNumbers,
    pub crosshair_outline: SliderNumbers,
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
    pub shade: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LoadingNumbers {
    pub map: String,
    pub mode: String,
    pub site_a_note: String,
    pub site_b_note: String,
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

pub(super) const TITLE: [&str; 3] = ["play", "settings", "quit"];
const TITLE_NO_QUIT: [&str; 2] = ["play", "settings"];
pub(super) const PAUSE: [&str; 3] = ["resume", "settings", "quit"];
pub(super) const MATCH_END: [&str; 2] = ["play", "quit"];
pub(super) const SELECT: [&str; 1] = ["play"];
pub(super) const SETTINGS_ROWS: [&str; 16] = [
    "master",
    "effects",
    "footsteps",
    "interface",
    "sensitivity",
    "invert",
    "fov",
    "render_scale",
    "bots",
    "crosshair",
    "crosshair_gap",
    "crosshair_length",
    "crosshair_thickness",
    "crosshair_outline",
    "crosshair_dot",
    "back",
];
const SLIDER_ROWS: [usize; 11] = [4, 6, 0, 1, 3, 7, 10, 11, 12, 13, 2];
const INVERT_ROW: usize = 5;
const CROSSHAIR_ROW: usize = 9;
const DOT_ROW: usize = 14;
const BOTS_ROW: usize = 8;
const BACK_ROW: usize = 15;

const BUTTONS: [&str; 8] = [
    "button_play",
    "button_settings",
    "button_quit",
    "button_resume",
    "button_again",
    "button_back",
    "invert_off",
    "dot_off",
];
const PLAY: Slot = 0;
const SETTINGS: Slot = 1;
const QUIT: Slot = 2;
const RESUME: Slot = 3;
const AGAIN: Slot = 4;
const BACK: Slot = 5;
const INVERT: Slot = 6;
const DOT: Slot = 7;
const SLIDER: Slot = 100;
const ROSTER: Slot = 200;
const BOTS: Slot = 300;
const CROSSHAIR: Slot = 400;
const RANGE: usize = 100;

fn ranged(base: Slot, index: Option<usize>) -> Option<Slot> {
    let index = index.filter(|index| *index < RANGE)?;
    Some(base + index as Slot)
}

pub(super) fn spot(look: &ScreenSource, id: &str) -> Option<Slot> {
    let menu = &look.menu;
    let numbered = |prefix: &str| id.strip_prefix(prefix)?.parse::<usize>().ok();
    let named = |names: &[&str]| names.iter().position(|name| *name == id);
    let level = || {
        let level = id.strip_prefix("bots_")?;
        menu.levels.iter().position(|known| known == level)
    };
    if id == "invert_on" {
        return Some(INVERT);
    }
    if id == "dot_on" {
        return Some(DOT);
    }
    ranged(0, named(&BUTTONS))
        .or_else(|| ranged(SLIDER, named(&SLIDER_IDS)))
        .or_else(|| ranged(ROSTER, numbered("roster_")))
        .or_else(|| ranged(BOTS, level()))
        .or_else(|| ranged(CROSSHAIR, numbered("crosshair_")))
}

fn spot_of(base: Slot, index: usize) -> Slot {
    ranged(base, Some(index)).unwrap_or(Slot::MAX)
}

#[derive(Resource, Default)]
pub(super) struct MenuState {
    focus: usize,
    hover: Option<usize>,
    press: Option<usize>,
    dragging_slider: Option<usize>,
    last_hover: Option<u64>,
    on: Option<Shown>,
    under: Option<(Option<Shown>, usize)>,
    pub quit_asked: u32,
}

pub(super) struct Nav<'a> {
    pub devices: &'a Devices,
    pub pointer: &'a Pointer,
    pub keys: &'a MenuInputs,
    pub draw: &'a InterfaceDraw,
    pub source: &'a MenuSource,
    pub can_quit: bool,
}

impl Nav<'_> {
    fn pressed(&self, binding: &[Input]) -> bool {
        engine_bindings::pressed(self.devices, binding)
    }

    fn clicked(&self) -> bool {
        self.pointer.pressed
    }

    fn held(&self) -> bool {
        self.pointer.down
    }

    fn hovering(&self) -> bool {
        self.pointer.hovering
    }

    fn over(&self, shown: Shown, spot: Slot) -> Option<Area> {
        self.draw.element_hit(shown as usize, spot, self.pointer)
    }

    fn over_slider(&self, shown: Shown, spot: Slot) -> Option<Area> {
        let pointer = self.pointer.position_pixels?;
        let coarse_pad = if self.pointer.is_coarse() {
            22.0 * self.draw.pixel_ratio()
        } else {
            0.0
        };
        self.draw.element(shown as usize, spot).filter(|area| {
            let pad_x = engine::math::max(area.h, coarse_pad);
            let pad_y = engine::math::max(area.h * 1.4, coarse_pad);
            pointer.x >= area.x - pad_x
                && pointer.x <= area.x + area.w + pad_x
                && pointer.y >= area.y - pad_y
                && pointer.y <= area.y + area.h + pad_y
        })
    }

    fn hovered<const N: usize>(&self, shown: Shown, spots: [Slot; N]) -> Option<usize> {
        spots.iter().position(|spot| self.over(shown, *spot).is_some())
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

fn hover_tag(shown: Shown, group: u64, index: usize) -> u64 {
    ((shown as u64) << 32) | (group << 16) | (index as u64)
}

fn note_hover(state: &mut MenuState, key: Option<u64>, sounds: &mut ScreenSounds) {
    if key.is_some() && key != state.last_hover {
        sounds.play(UiSound::Hover);
    }
    state.last_hover = key;
}

fn buttons<const N: usize>(
    nav: &Nav<'_>,
    state: &mut MenuState,
    shown: Shown,
    (names, spots): ([&'static str; N], [Slot; N]),
    sounds: &mut ScreenSounds,
) -> Option<&'static str> {
    let hit = nav.hovered(shown, spots);
    let hover = hit.filter(|_| nav.hovering());
    state.hover = hover;
    state.press = hit.filter(|_| nav.held());
    note_hover(state, hover.map(|at| hover_tag(shown, 1, at)), sounds);
    if let Some(hit) = hit {
        state.focus = hit;
        if nav.clicked() {
            sounds.play(UiSound::Click);
            return Some(names[hit]);
        }
    }
    let before = state.focus;
    state.focus = nav.step(state.focus, N);
    if state.focus != before {
        sounds.play(UiSound::Hover);
    }
    if nav.pressed(&nav.keys.confirm) {
        sounds.play(UiSound::Click);
        return Some(names[state.focus]);
    }
    None
}

pub(super) fn start(flow: &mut Flow) {
    flow.stage = Stage::Loading;
    flow.game += 1;
}

fn title(nav: &Nav<'_>, state: &mut MenuState, flow: &mut Flow, sounds: &mut ScreenSounds) {
    let chosen = if nav.can_quit {
        buttons(
            nav,
            state,
            Shown::Title,
            (TITLE, [PLAY, SETTINGS, QUIT]),
            sounds,
        )
    } else {
        buttons(
            nav,
            state,
            Shown::Title,
            (TITLE_NO_QUIT, [PLAY, SETTINGS]),
            sounds,
        )
    };
    match chosen {
        Some("play") => flow.stage = Stage::Select,
        Some("settings") => flow.settings_open = true,
        Some(_) => state.quit_asked += 1,
        None => {}
    }
}

fn pause(nav: &Nav<'_>, state: &mut MenuState, flow: &mut Flow, sounds: &mut ScreenSounds) {
    if nav.pressed(&nav.keys.back) {
        sounds.play(UiSound::Click);
        flow.stage = Stage::Playing;
        return;
    }
    let chosen = buttons(
        nav,
        state,
        Shown::Pause,
        (PAUSE, [RESUME, SETTINGS, QUIT]),
        sounds,
    );
    match chosen {
        Some("resume") => flow.stage = Stage::Playing,
        Some("settings") => flow.settings_open = true,
        Some(_) => flow.stage = Stage::Title,
        None => {}
    }
}

fn match_end(nav: &Nav<'_>, state: &mut MenuState, flow: &mut Flow, sounds: &mut ScreenSounds) {
    let chosen = buttons(
        nav,
        state,
        Shown::MatchEnd,
        (MATCH_END, [AGAIN, QUIT]),
        sounds,
    );
    match chosen {
        Some("play") => flow.stage = Stage::Select,
        Some(_) => flow.stage = Stage::Title,
        None => {}
    }
}

fn select(
    nav: &Nav<'_>,
    state: &mut MenuState,
    facts: &mut HudFacts,
    flow: &mut Flow,
    sounds: &mut ScreenSounds,
) {
    if nav.pressed(&nav.keys.back) {
        sounds.play(UiSound::Click);
        flow.stage = Stage::Title;
        return;
    }
    let cards_len = facts.menu.select.cards.len();
    let mut at = facts
        .menu
        .select
        .cards
        .iter()
        .position(|card| card.key == flow.character)
        .unwrap_or(0);
    let before = at;
    if nav.pressed(&nav.keys.right) && cards_len > 0 {
        at = (at + 1) % cards_len;
    } else if nav.pressed(&nav.keys.left) && cards_len > 0 {
        at = (at + cards_len - 1) % cards_len;
    }
    if at != before {
        sounds.play(UiSound::Click);
    }
    let hover_card = (0..cards_len)
        .find(|&index| nav.over(Shown::Select, spot_of(ROSTER, index)).is_some());
    if let Some(index) = hover_card
        && nav.clicked()
    {
        at = index;
        sounds.play(UiSound::Click);
    }
    if let Some(card) = facts.menu.select.cards.get(at) {
        flow.character.clone_from(&card.key);
    }
    let levels = &nav.source.levels;
    let hover_level = (0..levels.len())
        .find(|&index| nav.over(Shown::Select, spot_of(BOTS, index)).is_some());
    for (index, level) in levels.iter().enumerate() {
        let keyed = nav
            .keys
            .levels
            .get(index)
            .is_some_and(|key| nav.pressed(std::slice::from_ref(key)));
        if keyed || (nav.clicked() && hover_level == Some(index)) {
            flow.level.clone_from(level);
            sounds.play(UiSound::Click);
        }
    }
    let hit_play = nav.over(Shown::Select, PLAY).is_some();
    let held = nav.held();
    let hovering = nav.hovering();
    let hover_card_ui = hover_card.filter(|_| hovering);
    let hover_level_ui = hover_level.filter(|_| hovering);
    let hover_play_ui = hit_play && hovering;
    let key = hover_card_ui
        .map(|index| hover_tag(Shown::Select, 1, index))
        .or_else(|| hover_level_ui.map(|index| hover_tag(Shown::Select, 2, index)))
        .or_else(|| hover_play_ui.then(|| hover_tag(Shown::Select, 3, 0)));
    note_hover(state, key, sounds);
    state.hover = hover_play_ui.then_some(0);
    state.press = (hit_play && held).then_some(0);
    let select_facts = &mut facts.menu.select;
    select_facts.hover_card = hover_card_ui;
    select_facts.press_card = hover_card.filter(|_| held);
    select_facts.hover_level = hover_level_ui;
    select_facts.press_level = hover_level.filter(|_| held);
    select_facts.hover_play = hover_play_ui;
    select_facts.press_play = hit_play && held;
    if nav.pressed(&nav.keys.confirm) || (nav.clicked() && hit_play) {
        sounds.play(UiSound::Ready);
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
            4 => self.volume.interface,
            5 => self.controls.render_scale,
            6 => self.options.crosshair_gap,
            7 => self.options.crosshair_length,
            8 => self.options.crosshair_thickness,
            9 => self.options.crosshair_outline,
            _ => self.volume.footsteps,
        }
    }

    fn set_slider(&mut self, row: usize, value: f32) {
        match row {
            0 => self.controls.radians_per_pixel = value,
            1 => self.controls.fov_degrees = value,
            2 => self.volume.master = value,
            3 => self.volume.effects = value,
            4 => self.volume.interface = value,
            5 => self.controls.render_scale = value,
            6 => self.options.crosshair_gap = value,
            7 => self.options.crosshair_length = value,
            8 => self.options.crosshair_thickness = value,
            9 => self.options.crosshair_outline = value,
            _ => self.volume.footsteps = value,
        }
    }
}

fn numbers_of(sliders: &SliderSet, row: usize) -> &SliderNumbers {
    match row {
        0 => &sliders.sensitivity,
        1 => &sliders.fov,
        2..=4 => &sliders.volume,
        5 => &sliders.render_scale,
        6 => &sliders.crosshair_gap,
        7 => &sliders.crosshair_length,
        8 => &sliders.crosshair_thickness,
        9 => &sliders.crosshair_outline,
        _ => &sliders.volume,
    }
}

const SLIDER_IDS: [&str; 11] = [
    "slider_sensitivity",
    "slider_fov",
    "slider_master",
    "slider_effects",
    "slider_interface",
    "slider_render_scale",
    "slider_crosshair_gap",
    "slider_crosshair_length",
    "slider_crosshair_thickness",
    "slider_crosshair_outline",
    "slider_footsteps",
];

fn level_step(flow: &mut Flow, levels: &[String], turn: i32) {
    let count = levels.len();
    if count == 0 {
        return;
    }
    let at = levels
        .iter()
        .position(|level| *level == flow.level)
        .unwrap_or(0);
    let next = (at as i32 + turn).rem_euclid(count as i32) as usize;
    flow.level.clone_from(&levels[next]);
}

fn settings(
    nav: &Nav<'_>,
    state: &mut MenuState,
    facts: &mut HudFacts,
    flow: &mut Flow,
    values: &mut Values,
    sounds: &mut ScreenSounds,
) {
    if nav.pressed(&nav.keys.back) {
        sounds.play(UiSound::Click);
        state.dragging_slider = None;
        flow.settings_open = false;
        return;
    }
    let rows = SETTINGS_ROWS.len();
    let before = state.focus;
    if nav.pressed(&nav.keys.down) {
        state.focus = (state.focus + 1) % rows;
    } else if nav.pressed(&nav.keys.up) {
        state.focus = (state.focus + rows - 1) % rows;
    }
    if state.focus != before {
        sounds.play(UiSound::Hover);
    }
    let sliders = &nav.source.sliders;
    let turn = i32::from(nav.pressed(&nav.keys.right)) - i32::from(nav.pressed(&nav.keys.left));
    if let Some(slider) = SLIDER_ROWS.iter().position(|row| *row == state.focus)
        && turn != 0
    {
        let numbers = numbers_of(sliders, slider);
        let next = values.slider(slider) + turn as f32 * numbers.step;
        values.set_slider(slider, engine::math::clamp(next, numbers.min, numbers.max));
        sounds.play(UiSound::Click);
    }
    let count = nav.source.crosshairs.len() as u32;
    if turn != 0 && state.focus == CROSSHAIR_ROW && count > 0 {
        values.options.crosshair =
            (values.options.crosshair + count).wrapping_add_signed(turn) % count;
        sounds.play(UiSound::Click);
    }
    if turn != 0 && state.focus == BOTS_ROW {
        level_step(flow, &nav.source.levels, turn);
        sounds.play(UiSound::Click);
    }
    let confirmed = nav.pressed(&nav.keys.confirm);
    if state.focus == INVERT_ROW && (turn != 0 || confirmed) {
        values.controls.invert_pitch = !values.controls.invert_pitch;
        sounds.play(UiSound::Click);
    }
    if state.focus == DOT_ROW && (turn != 0 || confirmed) {
        values.options.crosshair_dot = !values.options.crosshair_dot;
        sounds.play(UiSound::Click);
    }
    if confirmed && state.focus == BACK_ROW {
        sounds.play(UiSound::Click);
        flow.settings_open = false;
    }
    settings_pointer(nav, state, facts, flow, values, sounds);
}

fn slider_drag(nav: &Nav<'_>, slider: usize, values: &mut Values) {
    let (Some(area), Some(pointer)) = (
        nav.draw.element(Shown::Settings as usize, spot_of(SLIDER, slider)),
        nav.pointer.position_pixels,
    ) else {
        return;
    };
    let frac = if area.w > 0.0 {
        (pointer.x - area.x) / area.w
    } else {
        0.0
    };
    let numbers = numbers_of(&nav.source.sliders, slider);
    values.set_slider(slider, value_of(frac, numbers));
}

fn settings_pointer(
    nav: &Nav<'_>,
    state: &mut MenuState,
    facts: &mut HudFacts,
    flow: &mut Flow,
    values: &mut Values,
    sounds: &mut ScreenSounds,
) {
    let hover_back = nav.over(Shown::Settings, BACK).is_some();
    let hover_invert = nav.over(Shown::Settings, INVERT).is_some();
    let hover_dot = nav.over(Shown::Settings, DOT).is_some();
    let hover_slider = SLIDER_ROWS
        .iter()
        .enumerate()
        .find(|&(slider, _)| nav.over_slider(Shown::Settings, spot_of(SLIDER, slider)).is_some())
        .map(|(slider, &row)| (slider, row));
    let hover_crosshair = (0..nav.source.crosshairs.len())
        .find(|&index| nav.over(Shown::Settings, spot_of(CROSSHAIR, index)).is_some());
    let hover_level = (0..nav.source.levels.len())
        .find(|&index| nav.over(Shown::Settings, spot_of(BOTS, index)).is_some());
    let held = nav.held();
    if !held {
        state.dragging_slider = None;
    }
    if nav.clicked() {
        settings_click(
            (hover_back, hover_invert, hover_dot),
            (hover_slider, hover_crosshair, hover_level),
            (nav, state, flow, values, sounds),
        );
    }
    if let Some(slider) = state.dragging_slider {
        slider_drag(nav, slider, values);
        state.focus = SLIDER_ROWS[slider];
    }
    let hovering = nav.hovering();
    let hover_row = if hovering {
        if hover_back {
            Some(BACK_ROW)
        } else if hover_invert {
            Some(INVERT_ROW)
        } else if hover_dot {
            Some(DOT_ROW)
        } else if hover_crosshair.is_some() {
            Some(CROSSHAIR_ROW)
        } else if hover_level.is_some() {
            Some(BOTS_ROW)
        } else {
            hover_slider.map(|(_, row)| row)
        }
    } else {
        None
    };
    let key = if hovering {
        hover_back
            .then(|| hover_tag(Shown::Settings, 1, 0))
            .or_else(|| hover_invert.then(|| hover_tag(Shown::Settings, 2, 0)))
            .or_else(|| hover_dot.then(|| hover_tag(Shown::Settings, 3, 0)))
            .or_else(|| hover_level.map(|index| hover_tag(Shown::Settings, 4, index)))
            .or_else(|| hover_crosshair.map(|index| hover_tag(Shown::Settings, 5, index)))
            .or_else(|| hover_slider.map(|(slider, _)| hover_tag(Shown::Settings, 6, slider)))
    } else {
        None
    };
    note_hover(state, key, sounds);
    state.hover = hover_row;
    state.press = (hover_back && held).then_some(BACK_ROW);
    let shown = &mut facts.menu.settings;
    shown.hover_row = hover_row;
    shown.hover_level = hover_level.filter(|_| hovering);
    shown.press_level = hover_level.filter(|_| held);
    shown.hover_crosshair = hover_crosshair.filter(|_| hovering);
    shown.hover_back = hover_back && hovering;
    shown.press_back = hover_back && held;
    shown.hover_invert = hover_invert && hovering;
    shown.hover_dot = hover_dot && hovering;
}

fn settings_click(
    (back, invert, dot): (bool, bool, bool),
    (slider, crosshair, level): (Option<(usize, usize)>, Option<usize>, Option<usize>),
    (nav, state, flow, values, sounds): (
        &Nav<'_>,
        &mut MenuState,
        &mut Flow,
        &mut Values,
        &mut ScreenSounds,
    ),
) {
    if invert {
        values.controls.invert_pitch = !values.controls.invert_pitch;
        state.focus = INVERT_ROW;
        sounds.play(UiSound::Click);
    }
    if dot {
        values.options.crosshair_dot = !values.options.crosshair_dot;
        state.focus = DOT_ROW;
        sounds.play(UiSound::Click);
    }
    if let Some((slider, row)) = slider {
        state.dragging_slider = Some(slider);
        slider_drag(nav, slider, values);
        state.focus = row;
        sounds.play(UiSound::Click);
    }
    if let Some(index) = crosshair {
        values.options.crosshair = index as u32;
        state.focus = CROSSHAIR_ROW;
        sounds.play(UiSound::Click);
    }
    if let Some(index) = level
        && let Some(chosen) = nav.source.levels.get(index)
    {
        flow.level.clone_from(chosen);
        state.focus = BOTS_ROW;
        sounds.play(UiSound::Click);
    }
    if back {
        state.dragging_slider = None;
        flow.settings_open = false;
        sounds.play(UiSound::Click);
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
    sounds: &mut ScreenSounds,
) {
    let shown = chosen_screen(flow);
    if shown != state.on {
        let back = state
            .under
            .filter(|(under, _)| *under == shown)
            .map(|(_, focus)| focus);
        state.under = (shown == Some(Shown::Settings)).then_some((state.on, state.focus));
        state.on = shown;
        state.focus = back.unwrap_or(0);
        state.hover = None;
        state.press = None;
        state.dragging_slider = None;
        state.last_hover = None;
    }
    match shown {
        Some(Shown::Settings) => settings(nav, state, facts, flow, values, sounds),
        Some(Shown::Title) => title(nav, state, flow, sounds),
        Some(Shown::Select) => select(nav, state, facts, flow, sounds),
        Some(Shown::Pause) => pause(nav, state, flow, sounds),
        Some(Shown::MatchEnd) => match_end(nav, state, flow, sounds),
        Some(Shown::Loading | Shown::Hud | Shown::Buy | Shown::Scoreboard | Shown::RoundEnd)
        | None => {}
    }
    let menu = &mut facts.menu;
    menu.screen = shown;
    menu.focus = match shown {
        Some(Shown::Title | Shown::Pause | Shown::MatchEnd | Shown::Settings) => Some(state.focus),
        Some(Shown::Select) => Some(0),
        Some(Shown::Loading | Shown::Hud | Shown::Buy | Shown::Scoreboard | Shown::RoundEnd)
        | None => None,
    };
    menu.hover = state.hover;
    menu.press = state.press;
}

pub(super) fn slider_facts(sliders: &mut [Slider; 11], source: &SliderSet, values: &Values) {
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

pub(super) type KeyNames = BTreeMap<String, String>;

pub(super) const ACTIONS: [&str; 30] = [
    "forward",
    "back",
    "left",
    "right",
    "walk",
    "crouch",
    "jump",
    "fire",
    "scope",
    "reload",
    "inspect",
    "drop",
    "next_item",
    "previous_item",
    "ready",
    "primary",
    "sidearm",
    "melee",
    "device",
    "interact",
    "ability_c",
    "ability_q",
    "ability_e",
    "ability_x",
    "buy_menu",
    "scoreboard",
    "capture",
    "release",
    "take",
    "give_back",
];

pub(super) fn bindings(keys: &KeyMap) -> [&Binding; 28] {
    [
        &keys.forward,
        &keys.back,
        &keys.left,
        &keys.right,
        &keys.walk,
        &keys.crouch,
        &keys.jump,
        &keys.fire,
        &keys.scope,
        &keys.reload,
        &keys.inspect,
        &keys.drop,
        &keys.next_item,
        &keys.previous_item,
        &keys.ready,
        &keys.primary,
        &keys.sidearm,
        &keys.melee,
        &keys.device,
        &keys.interact,
        &keys.ability_c,
        &keys.ability_q,
        &keys.ability_e,
        &keys.ability_x,
        &keys.buy_menu,
        &keys.scoreboard,
        &keys.capture,
        &keys.release,
    ]
}

pub(super) fn action(name: &str) -> Option<usize> {
    ACTIONS.iter().position(|known| *known == name)
}

pub(super) fn binding_of<'a>(keys: &'a KeyMap, name: &str) -> Option<&'a Binding> {
    bindings(keys).get(action(name)?).copied()
}

pub(super) fn spoken_binding(binding: &[Input], names: &KeyNames, out: &mut String) {
    out.clear();
    let Some(input) = binding.first() else {
        return;
    };
    let code = input.code();
    match names.get(code) {
        Some(name) => out.push_str(name),
        None => out.push_str(
            code.strip_prefix("Key")
                .or_else(|| code.strip_prefix("Digit"))
                .unwrap_or(code),
        ),
    }
}

pub(super) fn key_facts<'a>(
    out: &mut Vec<String>,
    all: impl Iterator<Item = &'a Vec<Input>>,
    names: &KeyNames,
) {
    let mut count = 0;
    for (index, binding) in all.enumerate() {
        if out.len() == index {
            out.push(String::new());
        }
        spoken_binding(binding, names, &mut out[index]);
        count = index + 1;
    }
    out.truncate(count);
}

pub(super) fn key_rows(
    rows: &mut Vec<KeyRow>,
    (names, spoken): (&[KeyName], &KeyNames),
    keys: &KeyMap,
) {
    rows.resize_with(names.len(), KeyRow::default);
    for (row, name) in rows.iter_mut().zip(names) {
        if row.action != name.action {
            row.action.clone_from(&name.action);
        }
        match binding_of(keys, &name.key) {
            Some(binding) => spoken_binding(binding, spoken, &mut row.binding),
            None => row.binding.clear(),
        }
    }
}
