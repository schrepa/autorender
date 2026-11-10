use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::hash::Hash;

use bevy_ecs::prelude::{Resource, World};
use bevy_ecs::system::{Commands, Res, SystemParamItem};
use engine_registry::MembersOf;
use glam::Vec2;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{
    EntityIds, Error, ErrorCode, Event, Feed, FeedSettings, Hold, Interest, InvalidSetting, List,
    Operation, Raises, Requests, Rule, Section, Setting, Ticket, settings,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum VerbShape {
    Button,
    Value,
    Value2,
    Delta,
    Delta2,
}

impl VerbShape {
    const fn flat(self) -> bool {
        matches!(self, Self::Button | Self::Value | Self::Delta)
    }
}

#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum MouseButton {
    Left,
    Middle,
    Right,
    Back,
    Forward,
}

impl MouseButton {
    pub const ALL: [Self; 5] = [
        Self::Left,
        Self::Middle,
        Self::Right,
        Self::Back,
        Self::Forward,
    ];

    pub fn of(index: u32) -> Option<Self> {
        Self::ALL.get(index as usize).copied()
    }
}

#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum PadButton {
    South,
    East,
    West,
    North,
    LeftShoulder,
    RightShoulder,
    Select,
    Start,
    LeftStick,
    RightStick,
    Up,
    Down,
    Left,
    Right,
}

impl PadButton {
    pub const ALL: [(Self, u32); 14] = [
        (Self::South, 0),
        (Self::East, 1),
        (Self::West, 2),
        (Self::North, 3),
        (Self::LeftShoulder, 4),
        (Self::RightShoulder, 5),
        (Self::Select, 8),
        (Self::Start, 9),
        (Self::LeftStick, 10),
        (Self::RightStick, 11),
        (Self::Up, 12),
        (Self::Down, 13),
        (Self::Left, 14),
        (Self::Right, 15),
    ];

    pub fn held(buttons: u32) -> impl Iterator<Item = Self> {
        Self::ALL
            .into_iter()
            .filter(move |&(_, bit)| buttons & (1 << bit) != 0)
            .map(|(button, _)| button)
    }
}

#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum Stick {
    Left,
    Right,
}

#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum Trigger {
    Left,
    Right,
}

#[derive(
    Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum Button {
    Key(Cow<'static, str>),
    Mouse(MouseButton),
    Pad(PadButton),
}

impl Button {
    pub const fn key(code: &'static str) -> Self {
        Self::Key(Cow::Borrowed(code))
    }
}

#[derive(
    Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum BindingSource {
    Key(Cow<'static, str>),
    Mouse(MouseButton),
    Pad(PadButton),
    Keys {
        right: Cow<'static, str>,
        left: Cow<'static, str>,
        up: Cow<'static, str>,
        down: Cow<'static, str>,
    },
    MouseMotion,
    TouchMotion,
    Wheel,
    Stick(Stick),
    Trigger(Trigger),
}

impl BindingSource {
    pub const fn key(code: &'static str) -> Self {
        Self::Key(Cow::Borrowed(code))
    }

    pub const fn keys(
        right: &'static str,
        left: &'static str,
        up: &'static str,
        down: &'static str,
    ) -> Self {
        Self::Keys {
            right: Cow::Borrowed(right),
            left: Cow::Borrowed(left),
            up: Cow::Borrowed(up),
            down: Cow::Borrowed(down),
        }
    }

    const fn flat(&self) -> bool {
        match self {
            Self::Key(_) | Self::Mouse(_) | Self::Pad(_) | Self::Trigger(_) => true,
            Self::Keys { .. }
            | Self::MouseMotion
            | Self::TouchMotion
            | Self::Wheel
            | Self::Stick(_) => false,
        }
    }

    fn parts(&self) -> Vec<Self> {
        match self {
            Self::Keys {
                right,
                left,
                up,
                down,
            } => [right, left, up, down]
                .into_iter()
                .map(|code| Self::Key(code.clone()))
                .collect(),
            Self::Key(_)
            | Self::Mouse(_)
            | Self::Pad(_)
            | Self::MouseMotion
            | Self::TouchMotion
            | Self::Wheel
            | Self::Stick(_)
            | Self::Trigger(_) => vec![self.clone()],
        }
    }

    fn codes(&self) -> Vec<&str> {
        match self {
            Self::Key(code) => vec![&**code],
            Self::Keys {
                right,
                left,
                up,
                down,
            } => vec![&**right, &**left, &**up, &**down],
            Self::Mouse(_)
            | Self::Pad(_)
            | Self::MouseMotion
            | Self::TouchMotion
            | Self::Wheel
            | Self::Stick(_)
            | Self::Trigger(_) => Vec::new(),
        }
    }
}

#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(transparent)]
pub struct Player(pub u8);

impl Player {
    pub const ONE: Self = Self(1);
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Binding {
    pub verb: Cow<'static, str>,
    pub source: BindingSource,
    #[serde(default = "unit", skip_serializing_if = "is_unit")]
    pub scale: f32,
    #[serde(default, skip_serializing_if = "is_false")]
    pub invert: bool,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub dead_zone: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modifier: Option<Button>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<Cow<'static, str>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub player: Option<Player>,
}

impl Binding {
    pub const fn new(verb: &'static str, source: BindingSource) -> Self {
        Self::of(verb, source, None, None)
    }

    pub const fn held(verb: &'static str, source: BindingSource, modifier: Button) -> Self {
        Self::of(verb, source, Some(modifier), None)
    }

    pub const fn of(
        verb: &'static str,
        source: BindingSource,
        modifier: Option<Button>,
        context: Option<&'static str>,
    ) -> Self {
        Self {
            verb: Cow::Borrowed(verb),
            source,
            scale: 1.0,
            invert: false,
            dead_zone: 0.0,
            modifier,
            context: match context {
                Some(context) => Some(Cow::Borrowed(context)),
                None => None,
            },
            player: None,
        }
    }

    #[must_use]
    pub const fn scaled(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    #[must_use]
    pub const fn inverted(mut self) -> Self {
        self.invert = true;
        self
    }

    #[must_use]
    pub const fn dead_zone(mut self, dead_zone: f32) -> Self {
        self.dead_zone = dead_zone;
        self
    }
}

fn unit() -> f32 {
    1.0
}

fn is_unit(value: &f32) -> bool {
    *value == 1.0
}

fn is_zero(value: &f32) -> bool {
    *value == 0.0
}

fn is_false(value: &bool) -> bool {
    !*value
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PlayerDevices {
    #[serde(default, skip_serializing_if = "is_false")]
    pub keyboard_mouse: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pads: Vec<u32>,
}

const DEAD_ZONE: Rule = Rule::range(0.0, 1.0);

settings! {
    #[section("input")]
    #[check(check_settings)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneInput")]
    pub struct InputSettings {
        pub players: Vec<PlayerDevices> = Vec::new() => any,
        pub bindings: Vec<Binding> = Vec::new() => any,
    }
}

fn check_settings(settings: &InputSettings) -> Result<(), InvalidSetting> {
    if settings.players.len() > usize::from(u8::MAX) {
        return Err(InvalidSetting::broken(
            "players",
            Rule::count(0.0, f64::from(u8::MAX)),
        )
        .got(settings.players.len()));
    }
    for (index, binding) in settings.bindings.iter().enumerate() {
        check_binding(binding).map_err(|invalid| invalid.within("bindings").indexed("bindings", index))?;
    }
    Ok(())
}

fn check_binding(binding: &Binding) -> Result<(), InvalidSetting> {
    binding.verb.to_string().check("verb", Rule::id())?;
    binding.scale.check("scale", Rule::finite())?;
    binding.dead_zone.check("dead_zone", DEAD_ZONE)?;
    if let Some(context) = &binding.context {
        context.to_string().check("context", Rule::id())?;
    }
    if binding.player.is_some_and(|player| player.0 == 0) {
        return Err(InvalidSetting::broken("player", Rule::range(1.0, f64::from(u8::MAX))).got(0));
    }
    for code in binding.source.codes() {
        code.to_owned().check("source", Rule::id())?;
    }
    if let Some(Button::Key(code)) = &binding.modifier {
        code.to_string().check("modifier", Rule::id())?;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub struct Verb {
    pub name: &'static str,
    pub shape: VerbShape,
    pub bindings: &'static [Binding],
}

pub struct Verbs;

impl List for Verbs {
    const NAME: &'static str = "verbs";
    type Member = Verb;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VerbId(u32);

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Buttons<B: Ord> {
    pub held: BTreeSet<B>,
    pub pressed: BTreeSet<B>,
    pub released: BTreeSet<B>,
}

impl<B: Ord> Default for Buttons<B> {
    fn default() -> Self {
        Self {
            held: BTreeSet::new(),
            pressed: BTreeSet::new(),
            released: BTreeSet::new(),
        }
    }
}

impl<B: Ord + Clone> Buttons<B> {
    pub fn set(&mut self, button: B, down: bool) {
        if down {
            if self.held.insert(button.clone()) {
                self.pressed.insert(button);
            }
        } else if self.held.remove(&button) {
            self.released.insert(button);
        }
    }

    pub fn drop_edges(&mut self) {
        self.pressed.clear();
        self.released.clear();
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Mouse {
    pub buttons: Buttons<MouseButton>,
    #[cfg_attr(feature = "schema", schemars(with = "Option<[f32; 2]>"))]
    pub position_pixels: Option<Vec2>,
    pub inside: bool,
    #[cfg_attr(feature = "schema", schemars(with = "[f32; 2]"))]
    pub motion_pixels: Vec2,
    #[cfg_attr(feature = "schema", schemars(with = "[f32; 2]"))]
    pub wheel_notches: Vec2,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Pad {
    pub index: u32,
    pub buttons: Buttons<PadButton>,
    #[cfg_attr(feature = "schema", schemars(with = "[[f32; 2]; 2]"))]
    pub sticks: [Vec2; 2],
    pub triggers: [f32; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
#[repr(u32)]
pub enum TouchPhase {
    Began = 0,
    Moved = 1,
    Ended = 2,
    Cancelled = 3,
}

impl TouchPhase {
    pub const ALL: [(Self, &'static str); 4] = [
        (Self::Began, "began"),
        (Self::Moved, "moved"),
        (Self::Ended, "ended"),
        (Self::Cancelled, "cancelled"),
    ];

    pub fn of(index: u32) -> Option<Self> {
        Self::ALL.get(index as usize).map(|&(phase, _)| phase)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Touch {
    pub id: u32,
    #[cfg_attr(feature = "schema", schemars(with = "[f32; 2]"))]
    pub position_pixels: Vec2,
    #[cfg_attr(feature = "schema", schemars(with = "[f32; 2]"))]
    pub motion_pixels: Vec2,
    pub phase: TouchPhase,
}

#[derive(Resource, Clone, Debug, Default, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[component(immutable)]
pub struct Devices {
    pub keyboard: Buttons<String>,
    pub mouse: Mouse,
    pub pads: Vec<Pad>,
    pub touches: Vec<Touch>,
}

#[derive(Clone, Copy)]
enum Edge {
    Held,
    Pressed,
    Released,
}

fn edged<B: Ord>(buttons: &Buttons<B>, edge: Edge) -> &BTreeSet<B> {
    match edge {
        Edge::Held => &buttons.held,
        Edge::Pressed => &buttons.pressed,
        Edge::Released => &buttons.released,
    }
}

impl Devices {
    #[must_use]
    pub fn held(&self) -> Self {
        let mut held = self.clone();
        held.keyboard.drop_edges();
        held.mouse.buttons.drop_edges();
        held.mouse.motion_pixels = Vec2::ZERO;
        held.mouse.wheel_notches = Vec2::ZERO;
        for pad in &mut held.pads {
            pad.buttons.drop_edges();
        }
        held.touches
            .retain(|touch| matches!(touch.phase, TouchPhase::Began | TouchPhase::Moved));
        for touch in &mut held.touches {
            touch.phase = TouchPhase::Moved;
            touch.motion_pixels = Vec2::ZERO;
        }
        held
    }

    fn button(&self, button: &Button, owned: &Owned, edge: Edge) -> bool {
        match button {
            Button::Key(code) => {
                owned.keyboard_mouse && edged(&self.keyboard, edge).contains(&**code)
            }
            Button::Mouse(mouse) => {
                owned.keyboard_mouse && edged(&self.mouse.buttons, edge).contains(mouse)
            }
            Button::Pad(pad) => self
                .pads
                .iter()
                .any(|at| owned.pad(at.index) && edged(&at.buttons, edge).contains(pad)),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Gathered {
    pub devices: Devices,
    pub driven: Vec<Driven>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Driven {
    pub player: Player,
    pub verb: String,
    pub value: Vec2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ContextAsk {
    pub by: &'static str,
    pub context: &'static str,
    pub blocks: bool,
}

#[derive(Resource, Debug, Default, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Contexts {
    #[serde(skip)]
    requests: Requests<ContextAsk, Ticket>,
    applied: Vec<ContextAsk>,
}

impl List for Contexts {
    const NAME: &'static str = "contexts";
    type Member = ContextAsk;
}

impl Contexts {
    pub fn ask(
        &mut self,
        by: &'static str,
        context: &'static str,
        blocks: bool,
    ) -> Hold {
        let interest = self.requests.ask(ContextAsk { by, context, blocks });
        stamp(&mut self.requests);
        interest.into()
    }

    pub fn asked(&self) -> Vec<ContextAsk> {
        stacked(&self.requests)
    }

    pub fn applied(&self) -> &[ContextAsk] {
        &self.applied
    }

    #[doc(hidden)]
    pub fn apply(&mut self) -> &[ContextAsk] {
        self.requests.settle(0.0);
        stamp(&mut self.requests);
        self.applied = stacked(&self.requests);
        &self.applied
    }
}

pub(crate) fn stamp<K: Clone + Eq + Hash>(requests: &mut Requests<K, Ticket>) {
    let asked: Vec<(K, Ticket)> = requests
        .asked()
        .map(|(key, ticket)| (key.clone(), ticket))
        .collect();
    for (key, ticket) in asked {
        requests.ready(&key, ticket, ticket);
    }
}

pub(crate) fn stacked<K: Clone + Eq + Hash>(requests: &Requests<K, Ticket>) -> Vec<K> {
    let mut held: Vec<(Ticket, K)> = requests
        .iter()
        .filter_map(|(key, request)| Some((*request.ready()?, key.clone())))
        .collect();
    held.sort_unstable_by_key(|(ticket, _)| *ticket);
    held.into_iter().map(|(_, key)| key).collect()
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Conflict {
    pub player: Player,
    pub context: String,
    pub source: BindingSource,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modifier: Option<Button>,
    pub verbs: Vec<&'static str>,
}

impl Conflict {
    fn error(&self) -> Error {
        Error::new(
            ErrorCode::BindingConflict,
            format!("$.input.{}", self.context),
        )
        .got(json!({
            "player": self.player,
            "source": self.source,
            "modifier": self.modifier,
            "verbs": self.verbs,
        }))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
struct VerbView {
    name: &'static str,
    shape: VerbShape,
    readings: Vec<[f32; 2]>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
struct BindingView {
    context: String,
    game: bool,
    binding: Binding,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
struct ContextView {
    name: &'static str,
    by: &'static str,
    blocks: bool,
    blocked: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
struct PlayerView {
    player: Player,
    keyboard_mouse: bool,
    pads: Option<Vec<u32>>,
}

#[derive(Clone, Debug, Default, PartialEq)]
struct Edges {
    pressed: BTreeSet<u32>,
    released: BTreeSet<u32>,
    sums: BTreeMap<u32, Vec2>,
}

impl Edges {
    fn is_empty(&self) -> bool {
        self.pressed.is_empty() && self.released.is_empty() && self.sums.is_empty()
    }

    fn add(&mut self, pressed: &BTreeSet<u32>, released: &BTreeSet<u32>, sums: &BTreeMap<u32, Vec2>) {
        self.pressed.extend(pressed.iter().copied());
        self.released.extend(released.iter().copied());
        for (&slot, &sum) in sums {
            *self.sums.entry(slot).or_default() += sum;
        }
    }
}

#[derive(Clone, Debug)]
struct Readings {
    held: BTreeSet<u32>,
    values: BTreeMap<u32, Vec2>,
    consumer: u64,
    next_tick: Edges,
    next_frame: Edges,
    shown: Edges,
}

impl Default for Readings {
    fn default() -> Self {
        Self {
            held: BTreeSet::new(),
            values: BTreeMap::new(),
            consumer: 1,
            next_tick: Edges::default(),
            next_frame: Edges::default(),
            shown: Edges::default(),
        }
    }
}

impl Readings {
    fn ingest(&mut self, resolved: Resolved) {
        let pressed: BTreeSet<u32> = resolved
            .pressed
            .iter()
            .filter(|slot| !self.held.contains(*slot))
            .chain(resolved.held.difference(&self.held))
            .copied()
            .collect();
        let released: BTreeSet<u32> = resolved
            .released
            .iter()
            .filter(|slot| !resolved.held.contains(*slot))
            .chain(self.held.difference(&resolved.held))
            .copied()
            .collect();
        let sums: BTreeMap<u32, Vec2> = resolved
            .sums
            .into_iter()
            .filter(|(_, sum)| *sum != Vec2::ZERO)
            .collect();
        for edges in [&mut self.next_tick, &mut self.next_frame] {
            edges.add(&pressed, &released, &sums);
        }
        self.held = resolved.held;
        self.values = resolved.values;
    }

    fn begin_tick(&mut self, tick: u64) {
        let pending = std::mem::take(&mut self.next_tick);
        self.shown = if tick == self.consumer {
            pending
        } else {
            Edges::default()
        };
        self.consumer = tick.saturating_add(1);
    }

    fn pending_tick(&self) -> Option<u64> {
        (!self.next_tick.is_empty()).then_some(self.consumer)
    }
}

#[derive(Default)]
struct Resolved {
    held: BTreeSet<u32>,
    pressed: BTreeSet<u32>,
    released: BTreeSet<u32>,
    sums: BTreeMap<u32, Vec2>,
    values: BTreeMap<u32, Vec2>,
    used: BTreeSet<Button>,
    conflicts: BTreeSet<Conflict>,
}

impl Resolved {
    fn add(&mut self, slot: u32, shape: VerbShape, reading: &Reading) {
        match shape {
            VerbShape::Button => {
                if reading.value != Vec2::ZERO {
                    self.held.insert(slot);
                }
                if reading.pressed {
                    self.pressed.insert(slot);
                }
                if reading.released {
                    self.released.insert(slot);
                }
            }
            VerbShape::Value | VerbShape::Value2 => {
                *self.values.entry(slot).or_default() += reading.value;
            }
            VerbShape::Delta | VerbShape::Delta2 => {
                *self.sums.entry(slot).or_default() += reading.value;
            }
        }
    }

    fn reading(&self, slot: u32, shape: VerbShape) -> Vec2 {
        match shape {
            VerbShape::Button if self.held.contains(&slot) => Vec2::X,
            VerbShape::Button => Vec2::ZERO,
            VerbShape::Value | VerbShape::Value2 => self.values.get(&slot).copied().unwrap_or_default(),
            VerbShape::Delta | VerbShape::Delta2 => self.sums.get(&slot).copied().unwrap_or_default(),
        }
    }
}

#[derive(Default)]
struct Reading {
    value: Vec2,
    pressed: bool,
    released: bool,
}

struct Row<'a> {
    verb: VerbId,
    shape: VerbShape,
    binding: &'a Binding,
    game: bool,
    context: &'a str,
}

struct Owned {
    keyboard_mouse: bool,
    pads: Option<Vec<u32>>,
}

impl Owned {
    fn pad(&self, index: u32) -> bool {
        self.pads.as_ref().is_none_or(|pads| pads.contains(&index))
    }
}

type Claim = (BindingSource, Option<Button>);

fn slot(verb: VerbId, player: Player) -> u32 {
    (u32::from(player.0) << 16) | (verb.0 & 0xFFFF)
}

fn owner(name: &str) -> &str {
    name.split_once('.').map_or(name, |(owner, _)| owner)
}

fn zoned(value: Vec2, dead_zone: f32) -> Vec2 {
    if dead_zone <= 0.0 {
        return value;
    }
    let length = value.length();
    let span = 1.0 - dead_zone;
    if length <= dead_zone || span <= 0.0 {
        return Vec2::ZERO;
    }
    let reach = crate::math::min(length, 1.0) - dead_zone;
    value * (reach / span / length)
}

fn rows<'a>(declared: &[Verb], settings: &'a InputSettings) -> Vec<Row<'a>> {
    let mut rows = Vec::new();
    for (index, verb) in declared.iter().enumerate() {
        let id = VerbId(u32::try_from(index).unwrap_or(u32::MAX));
        let game: Vec<&'a Binding> = settings
            .bindings
            .iter()
            .filter(|binding| binding.verb == verb.name)
            .collect();
        let from_game = !game.is_empty();
        let listed: Vec<&'a Binding> = if from_game {
            game
        } else {
            verb.bindings.iter().collect()
        };
        for binding in listed {
            rows.push(Row {
                verb: id,
                shape: verb.shape,
                binding,
                game: from_game,
                context: binding.context.as_deref().unwrap_or_else(|| owner(verb.name)),
            });
        }
    }
    rows
}

fn owners(settings: &InputSettings) -> Vec<Owned> {
    if settings.players.is_empty() {
        return vec![Owned {
            keyboard_mouse: true,
            pads: None,
        }];
    }
    settings
        .players
        .iter()
        .take(usize::from(u8::MAX))
        .map(|player| Owned {
            keyboard_mouse: player.keyboard_mouse,
            pads: Some(player.pads.clone()),
        })
        .collect()
}

fn visible(asked: &[ContextAsk]) -> (Vec<(&'static str, usize)>, Vec<ContextView>) {
    let mut ranks = Vec::new();
    let mut views = Vec::new();
    let mut blocked = false;
    for (at, ask) in asked.iter().enumerate().rev() {
        views.push(ContextView {
            name: ask.context,
            by: ask.by,
            blocks: ask.blocks,
            blocked,
        });
        if !blocked {
            ranks.push((ask.context, at + 1));
        }
        blocked |= ask.blocks;
    }
    (ranks, views)
}

fn rank(ranks: &[(&'static str, usize)], context: &str) -> Option<usize> {
    ranks
        .iter()
        .filter(|(name, _)| *name == context)
        .map(|&(_, rank)| rank)
        .max()
}

fn pressing(
    button: &Button,
    (devices, owned): (&Devices, &Owned),
    reading: &mut Reading,
    used: &mut BTreeSet<Button>,
) -> f32 {
    if owned.keyboard_mouse || matches!(button, Button::Pad(_)) {
        used.insert(button.clone());
    }
    reading.pressed |= devices.button(button, owned, Edge::Pressed);
    reading.released |= devices.button(button, owned, Edge::Released);
    f32::from(u8::from(devices.button(button, owned, Edge::Held)))
}

fn read(
    (row, rank): (&Row<'_>, usize),
    winners: &BTreeMap<Claim, usize>,
    held: (&Devices, &Owned),
    used: &mut BTreeSet<Button>,
) -> Reading {
    let (devices, owned) = held;
    let binding = row.binding;
    let wins = |part: BindingSource| winners.get(&(part, binding.modifier.clone())) == Some(&rank);
    let mut reading = Reading::default();
    if !binding.source.parts().into_iter().any(wins) {
        return reading;
    }
    let dead_zone = binding.dead_zone;
    let pads = || devices.pads.iter().filter(|pad| owned.pad(pad.index));
    let raw = match &binding.source {
        BindingSource::Key(code) => {
            let pressed = pressing(&Button::Key(code.clone()), held, &mut reading, used);
            zoned(Vec2::new(pressed, 0.0), dead_zone)
        }
        BindingSource::Mouse(button) => {
            let pressed = pressing(&Button::Mouse(*button), held, &mut reading, used);
            zoned(Vec2::new(pressed, 0.0), dead_zone)
        }
        BindingSource::Pad(button) => {
            let pressed = pressing(&Button::Pad(*button), held, &mut reading, used);
            zoned(Vec2::new(pressed, 0.0), dead_zone)
        }
        BindingSource::Keys {
            right,
            left,
            up,
            down,
        } => {
            let mut axes = [0.0; 4];
            for (axis, code) in axes.iter_mut().zip([right, left, up, down]) {
                if wins(BindingSource::Key(code.clone())) {
                    *axis = pressing(&Button::Key(code.clone()), held, &mut reading, used);
                }
            }
            let [right, left, up, down] = axes;
            zoned(Vec2::new(right - left, up - down), dead_zone)
        }
        BindingSource::MouseMotion if owned.keyboard_mouse => {
            zoned(devices.mouse.motion_pixels, dead_zone)
        }
        BindingSource::TouchMotion if owned.keyboard_mouse => zoned(
            devices
                .touches
                .iter()
                .map(|touch| touch.motion_pixels)
                .sum(),
            dead_zone,
        ),
        BindingSource::Wheel if owned.keyboard_mouse => zoned(devices.mouse.wheel_notches, dead_zone),
        BindingSource::MouseMotion | BindingSource::TouchMotion | BindingSource::Wheel => {
            Vec2::ZERO
        }
        BindingSource::Stick(stick) => pads()
            .map(|pad| zoned(pad.sticks[*stick as usize], dead_zone))
            .sum(),
        BindingSource::Trigger(trigger) => pads()
            .map(|pad| zoned(Vec2::new(pad.triggers[*trigger as usize], 0.0), dead_zone))
            .sum(),
    };
    let modified = binding.modifier.as_ref().is_none_or(|modifier| {
        if owned.keyboard_mouse || matches!(modifier, Button::Pad(_)) {
            used.insert(modifier.clone());
        }
        devices.button(modifier, owned, Edge::Held)
    });
    let mut value = raw * binding.scale;
    if binding.invert {
        if binding.source.flat() {
            value.x = -value.x;
        } else {
            value.y = -value.y;
        }
    }
    if !modified {
        value = Vec2::ZERO;
        reading.pressed = false;
    }
    if row.shape.flat() {
        value.y = 0.0;
    }
    reading.value = value;
    reading
}

fn resolve(
    declared: &[Verb],
    settings: &InputSettings,
    ranks: &[(&'static str, usize)],
    (devices, driven): (&Devices, &[Driven]),
) -> (Resolved, Vec<BindingView>, Vec<PlayerView>) {
    let rows = rows(declared, settings);
    let owners = owners(settings);
    let mut resolved = Resolved::default();
    for (at, owned) in owners.iter().enumerate() {
        let player = Player(u8::try_from(at + 1).unwrap_or(u8::MAX));
        let mine: Vec<(&Row<'_>, usize)> = rows
            .iter()
            .filter(|row| row.binding.player.is_none_or(|bound| bound == player))
            .filter_map(|row| Some((row, rank(ranks, row.context)?)))
            .collect();
        let mut winners: BTreeMap<Claim, usize> = BTreeMap::new();
        let mut claims: BTreeMap<(&str, Claim), BTreeSet<VerbId>> = BTreeMap::new();
        for &(row, rank) in &mine {
            for part in row.binding.source.parts() {
                let claim = (part, row.binding.modifier.clone());
                let best = winners.entry(claim.clone()).or_insert(rank);
                *best = (*best).max(rank);
                claims.entry((row.context, claim)).or_default().insert(row.verb);
            }
        }
        for ((context, (source, modifier)), verbs) in claims {
            if verbs.len() > 1 {
                resolved.conflicts.insert(Conflict {
                    player,
                    context: context.to_owned(),
                    source,
                    modifier,
                    verbs: verbs
                        .iter()
                        .filter_map(|verb| declared.get(verb.0 as usize))
                        .map(|verb| verb.name)
                        .collect(),
                });
            }
        }
        for &(row, rank) in &mine {
            let reading = read((row, rank), &winners, (devices, owned), &mut resolved.used);
            resolved.add(slot(row.verb, player), row.shape, &reading);
        }
    }
    for drive in driven {
        let Some(index) = declared.iter().position(|verb| verb.name == drive.verb) else {
            continue;
        };
        let shape = declared[index].shape;
        let mut value = drive.value;
        if shape.flat() {
            value.y = 0.0;
        }
        let verb = VerbId(u32::try_from(index).unwrap_or(u32::MAX));
        resolved.add(
            slot(verb, drive.player),
            shape,
            &Reading {
                value,
                pressed: false,
                released: false,
            },
        );
    }
    let bindings = rows
        .iter()
        .map(|row| BindingView {
            context: row.context.to_owned(),
            game: row.game,
            binding: row.binding.clone(),
        })
        .collect();
    let players = owners
        .into_iter()
        .enumerate()
        .map(|(at, owned)| PlayerView {
            player: Player(u8::try_from(at + 1).unwrap_or(u8::MAX)),
            keyboard_mouse: owned.keyboard_mouse,
            pads: owned.pads,
        })
        .collect();
    (resolved, bindings, players)
}

#[derive(Resource, Default, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Input {
    verbs: Vec<VerbView>,
    bindings: Vec<BindingView>,
    contexts: Vec<ContextView>,
    players: Vec<PlayerView>,
    conflicts: Vec<Conflict>,
    #[serde(skip)]
    declared: Vec<Verb>,
    #[serde(skip)]
    readings: Readings,
    #[serde(skip)]
    used: BTreeSet<Button>,
    #[serde(skip)]
    reported: Requests<Conflict, ()>,
    #[serde(skip)]
    raised: BTreeMap<Conflict, Interest<Conflict>>,
}

impl Input {
    pub fn declare(&mut self, verb: &Verb) -> VerbId {
        if let Some(known) = self.find(verb.name) {
            return known;
        }
        self.declared.push(*verb);
        self.verbs.push(VerbView {
            name: verb.name,
            shape: verb.shape,
            readings: Vec::new(),
        });
        VerbId(u32::try_from(self.declared.len() - 1).unwrap_or(u32::MAX))
    }

    pub fn find(&self, name: &str) -> Option<VerbId> {
        self.declared
            .iter()
            .position(|verb| verb.name == name)
            .and_then(|at| u32::try_from(at).ok())
            .map(VerbId)
    }

    pub fn shape(&self, verb: VerbId) -> Option<VerbShape> {
        self.declared.get(verb.0 as usize).map(|verb| verb.shape)
    }

    pub fn names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.declared.iter().map(|verb| verb.name)
    }

    pub fn players(&self) -> u8 {
        u8::try_from(self.players.len().max(1)).unwrap_or(u8::MAX)
    }

    pub fn held(&self, verb: VerbId, player: Player) -> bool {
        self.readings.held.contains(&slot(verb, player))
    }

    pub fn pressed(&self, verb: VerbId, player: Player) -> bool {
        self.readings.shown.pressed.contains(&slot(verb, player))
    }

    pub fn released(&self, verb: VerbId, player: Player) -> bool {
        self.readings.shown.released.contains(&slot(verb, player))
    }

    pub fn value(&self, verb: VerbId, player: Player) -> f32 {
        self.value2(verb, player).x
    }

    pub fn value2(&self, verb: VerbId, player: Player) -> Vec2 {
        let slot = slot(verb, player);
        match self.shape(verb) {
            Some(VerbShape::Button) if self.readings.held.contains(&slot) => Vec2::X,
            Some(VerbShape::Button) | None => Vec2::ZERO,
            Some(VerbShape::Value | VerbShape::Value2) => {
                self.readings.values.get(&slot).copied().unwrap_or_default()
            }
            Some(VerbShape::Delta | VerbShape::Delta2) => {
                self.readings.shown.sums.get(&slot).copied().unwrap_or_default()
            }
        }
    }

    pub fn uses_key(&self, code: &str) -> bool {
        self.used
            .iter()
            .any(|button| matches!(button, Button::Key(key) if key == code))
    }

    pub fn uses_mouse(&self, button: MouseButton) -> bool {
        self.used.contains(&Button::Mouse(button))
    }

    pub fn conflicts(&self) -> &[Conflict] {
        &self.conflicts
    }

    pub fn pending_tick(&self) -> Option<u64> {
        self.readings.pending_tick()
    }

    #[doc(hidden)]
    pub fn begin_tick(&mut self, tick: u64) {
        self.readings.begin_tick(tick);
    }

    #[doc(hidden)]
    pub fn begin_frame(&mut self) {
        self.readings.shown = std::mem::take(&mut self.readings.next_frame);
    }

    #[doc(hidden)]
    pub fn hide_edges(&mut self) {
        self.readings.shown = Edges::default();
    }

    #[doc(hidden)]
    pub fn drop_tick_edges(&mut self) {
        self.readings.next_tick = Edges::default();
    }

    #[doc(hidden)]
    pub fn drop_edges(&mut self) {
        self.readings.next_tick = Edges::default();
        self.readings.next_frame = Edges::default();
        self.readings.shown = Edges::default();
    }

    fn resolve(
        &mut self,
        (devices, driven): (&Devices, &[Driven]),
        settings: &InputSettings,
        asked: &[ContextAsk],
    ) {
        let (ranks, contexts) = visible(asked);
        let (mut resolved, bindings, players) =
            resolve(&self.declared, settings, &ranks, (devices, driven));
        let count = u8::try_from(players.len()).unwrap_or(u8::MAX);
        self.verbs = self
            .declared
            .iter()
            .enumerate()
            .map(|(index, verb)| {
                let id = VerbId(u32::try_from(index).unwrap_or(u32::MAX));
                VerbView {
                    name: verb.name,
                    shape: verb.shape,
                    readings: (1..=count)
                        .map(|player| resolved.reading(slot(id, Player(player)), verb.shape).to_array())
                        .collect(),
                }
            })
            .collect();
        self.used = std::mem::take(&mut resolved.used);
        let conflicts = std::mem::take(&mut resolved.conflicts);
        self.readings.ingest(resolved);
        self.bindings = bindings;
        self.contexts = contexts;
        self.players = players;
        self.report(conflicts);
    }

    fn report(&mut self, conflicts: BTreeSet<Conflict>) {
        self.raised.retain(|conflict, _| conflicts.contains(conflict));
        for conflict in &conflicts {
            if self.raised.contains_key(conflict) {
                continue;
            }
            let interest = self.reported.ask(conflict.clone());
            if let Some(ticket) = self.reported.ticket(conflict) {
                self.reported.fail(conflict, ticket, conflict.error());
            }
            self.raised.insert(conflict.clone(), interest);
        }
        self.reported.settle(0.0);
        self.conflicts = conflicts.into_iter().collect();
    }
}

pub(crate) fn install(world: &mut World) -> Vec<Error> {
    let Some(members) = world.get_resource::<MembersOf<Verbs>>() else {
        return Vec::new();
    };
    let twice = members.names_twice(|verb| verb.name);
    let listed = members.all().to_vec();
    if let Some(mut input) = world.get_resource_mut::<Input>() {
        for verb in &listed {
            input.declare(verb);
        }
    }
    twice
}

#[derive(Clone, Debug, PartialEq, Serialize, schemars::JsonSchema)]
#[serde(tag = "device", rename_all = "snake_case")]
pub enum InputChange {
    Key {
        code: String,
        down: bool,
    },
    Mouse {
        button: MouseButton,
        down: bool,
    },
    Pad {
        pad: u32,
        button: PadButton,
        down: bool,
    },
    Pointer {
        position: Option<[f32; 2]>,
        inside: bool,
    },
    Touch {
        id: u32,
        position: [f32; 2],
        phase: TouchPhase,
    },
}

impl Event for InputChange {
    const NAME: &'static str = "input";
    type Row = Self;

    fn row(&self, _: &dyn EntityIds) -> Self {
        self.clone()
    }
}

#[derive(Resource)]
pub struct InputChanges {
    raised: Feed<InputChange>,
}

impl Default for InputChanges {
    fn default() -> Self {
        Self {
            raised: Feed::new(FeedSettings::default().events as usize),
        }
    }
}

impl Raises<InputChange> for InputChanges {
    fn raised(&self) -> &Feed<InputChange> {
        &self.raised
    }
}

fn edges<B: Ord + Clone>(buttons: &Buttons<B>, change: impl Fn(B, bool) -> InputChange) -> Vec<InputChange> {
    buttons
        .pressed
        .iter()
        .map(|button| change(button.clone(), true))
        .chain(buttons.released.iter().map(|button| change(button.clone(), false)))
        .collect()
}

fn changes(before: Option<&Devices>, after: &Devices) -> Vec<InputChange> {
    let mut changed = edges(&after.keyboard, |code, down| InputChange::Key { code, down });
    changed.extend(edges(&after.mouse.buttons, |button, down| InputChange::Mouse {
        button,
        down,
    }));
    for pad in &after.pads {
        let index = pad.index;
        changed.extend(edges(&pad.buttons, |button, down| InputChange::Pad {
            pad: index,
            button,
            down,
        }));
    }
    let moved = before.is_none_or(|before| {
        before.mouse.position_pixels != after.mouse.position_pixels
            || before.mouse.inside != after.mouse.inside
    });
    if moved {
        changed.push(InputChange::Pointer {
            position: after.mouse.position_pixels.map(|position| position.to_array()),
            inside: after.mouse.inside,
        });
    }
    changed.extend(
        after
            .touches
            .iter()
            .filter(|touch| touch.phase != TouchPhase::Moved)
            .map(|touch| InputChange::Touch {
                id: touch.id,
                position: touch.position_pixels.to_array(),
                phase: touch.phase,
            }),
    );
    changed
}

fn raise_changes(world: &mut World, devices: &Devices) {
    let changed = changes(world.get_resource::<Devices>(), devices);
    let size = world
        .get_resource::<FeedSettings>()
        .map_or(FeedSettings::default().events, |feeds| feeds.events);
    if let Some(mut raised) = world.get_resource_mut::<InputChanges>() {
        raised.raised.resize(size as usize);
        for change in changed {
            raised.raised.push(change);
        }
    }
}

#[doc(hidden)]
pub fn accept_input(world: &mut World, gathered: Gathered) {
    let Gathered { devices, driven } = gathered;
    raise_changes(world, &devices);
    let settings = world
        .get_resource::<InputSettings>()
        .cloned()
        .unwrap_or_default();
    let asked = world
        .get_resource_mut::<Contexts>()
        .map(|mut contexts| contexts.apply().to_vec())
        .unwrap_or_default();
    if let Some(mut input) = world.get_resource_mut::<Input>() {
        input.resolve((&devices, &driven), &settings, &asked);
    }
    world.insert_resource(devices);
}

#[doc(hidden)]
pub fn hold_input(world: &mut World) {
    if let Some(held) = world.get_resource::<Devices>().map(Devices::held) {
        world.insert_resource(held);
    }
}

#[doc(hidden)]
pub fn suspend_input(world: &mut World) {
    hold_input(world);
    if let Some(mut input) = world.get_resource_mut::<Input>() {
        input.drop_edges();
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct Rebind {
    #[serde(default)]
    bindings: Vec<Binding>,
    #[serde(default)]
    reset: Vec<String>,
}

impl Operation for Rebind {
    const NAME: &'static str = "rebind";
    type Params = (Res<'static, InputSettings>, Commands<'static, 'static>);
    type Request = Self;
    type Reply = RebindReply;

    fn run(
        (settings, mut commands): SystemParamItem<'_, '_, Self::Params>,
        request: Self,
    ) -> Result<RebindReply, Error> {
        let mut next = settings.clone();
        next.bindings.retain(|binding| {
            !request.reset.iter().any(|verb| verb.as_str() == &*binding.verb)
                && !request.bindings.iter().any(|bound| bound.verb == binding.verb)
        });
        next.bindings.extend(request.bindings);
        next.check()?;
        let bindings = next.bindings.len();
        commands.insert_resource(next);
        Ok(RebindReply { bindings })
    }
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub(crate) struct RebindReply {
    bindings: usize,
}
