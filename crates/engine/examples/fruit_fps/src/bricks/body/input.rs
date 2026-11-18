use std::sync::Arc;

use engine::math;
use engine::prelude::*;
use engine::{Error, ErrorCode};
use engine_core::SceneCx;
use glam::Vec3;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::names::PITCH_LIMIT_DEGREES;
use crate::play::actors::{Equip, Gait, Human, Wish};
use crate::play::numbers::{GameNumbers, Input, KeyMap, Numbers};
use crate::play::screen::{ScreenBuys, ScreenState};
use crate::play::view::Controls;
use crate::rules::AbilitySlot;

#[derive(Component, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(super) struct NumbersFile {
    pub file: String,
    #[serde(skip)]
    parsed: Option<Arc<GameNumbers>>,
}

fn part<T: serde::de::DeserializeOwned>(whole: &Value, key: &str) -> Result<T, Error> {
    let value = whole
        .get(key)
        .cloned()
        .ok_or_else(|| Error::new(ErrorCode::MissingField, format!("$.{key}")))?;
    engine::bricks::parse_value_at(value, &format!("$.{key}"))
}

fn parsed(text: &str) -> Result<GameNumbers, Error> {
    let whole: Value = engine::bricks::parse_json(text)?;
    Ok(GameNumbers {
        player: part(&whole, "player")?,
        keys: part(&whole, "keys")?,
        spectate: part(&whole, "spectate")?,
        weapons: part(&whole, "weapons")?,
        hit_shapes: part(&whole, "hit_shapes")?,
        rules: engine::bricks::parse_value_at(whole, "$")?,
        text: Arc::from(text),
    })
}

impl NumbersFile {
    pub(super) fn files(&self) -> Vec<String> {
        vec![self.file.clone()]
    }

    pub(super) fn prepared(mut self, cx: &SceneCx<'_>, path: &str) -> Result<Self, Error> {
        let at = format!("{path}.file");
        let Some(bytes) = cx.buffer(&self.file, &at)? else {
            return Ok(self);
        };
        let text = std::str::from_utf8(&bytes)
            .map_err(|_| Error::new(ErrorCode::InvalidRecord, at.clone()).got(&self.file))?;
        let numbers = parsed(text).map_err(|error| error.within(&at))?;
        self.parsed = Some(Arc::new(numbers));
        Ok(self)
    }
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Latched {
    fire: bool,
    jump: bool,
    reload: bool,
    scope: bool,
    equip: Option<Equip>,
    cast: Option<AbilitySlot>,
    buy_menu: bool,
}

pub(super) fn adopt_numbers(files: Query<Ref<NumbersFile>>, mut numbers: ResMut<Numbers>) {
    for file in &files {
        if file.is_changed() {
            numbers.adopt(file.parsed.clone());
        }
    }
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Swallow(bool);

fn menu_open(screen: Option<&ScreenState>) -> bool {
    screen.is_some_and(|screen| screen.holds_pointer)
}

pub(super) fn capture(
    devices: Res<Devices>,
    numbers: Res<Numbers>,
    screen: Option<Res<ScreenState>>,
    mut cursor: ResMut<CursorMode>,
    mut swallow: ResMut<Swallow>,
) {
    let Some(numbers) = numbers.get() else {
        return;
    };
    if menu_open(screen.as_deref()) || KeyMap::pressed(&devices, &numbers.keys.release) {
        cursor.set_if_neq(CursorMode::Free);
    } else if *cursor != CursorMode::Captured && KeyMap::pressed(&devices, &numbers.keys.capture) {
        *cursor = CursorMode::Captured;
        swallow.set_if_neq(Swallow(true));
    } else if swallow.0 && !KeyMap::held(&devices, &numbers.keys.capture) {
        *swallow = Swallow(false);
    }
}

pub(super) fn aim(
    devices: Res<Devices>,
    cursor: Res<CursorMode>,
    controls: Res<Controls>,
    mut humans: Query<&mut Wish, With<Human>>,
) {
    let motion = match *cursor {
        CursorMode::Captured => devices.mouse.motion_pixels,
        CursorMode::Free | CursorMode::Drag => return,
    };
    if motion == engine::glam::Vec2::ZERO {
        return;
    }
    let limit = PITCH_LIMIT_DEGREES.to_radians();
    let rate = controls.radians_per_pixel;
    let rise = if controls.invert_pitch { rate } else { -rate };
    for mut wish in &mut humans {
        wish.yaw_radians -= motion.x * rate;
        wish.pitch_radians = math::clamp(wish.pitch_radians + motion.y * rise, -limit, limit);
    }
}

pub(super) fn latch(
    devices: Res<Devices>,
    numbers: Res<Numbers>,
    swallow: Res<Swallow>,
    mut latched: ResMut<Latched>,
) {
    let Some(numbers) = numbers.get() else {
        return;
    };
    let keys = &numbers.keys;
    let pressed = |binding: &[Input]| KeyMap::pressed(&devices, binding);
    let equips = [
        (&keys.primary, Equip::Primary),
        (&keys.sidearm, Equip::Sidearm),
        (&keys.melee, Equip::Melee),
        (&keys.device, Equip::Device),
    ];
    let casts = [
        (&keys.ability_c, AbilitySlot::C),
        (&keys.ability_q, AbilitySlot::Q),
        (&keys.ability_e, AbilitySlot::E),
        (&keys.ability_x, AbilitySlot::X),
    ];
    let next = Latched {
        fire: latched.fire || (!swallow.0 && pressed(&keys.fire)),
        jump: latched.jump || pressed(&keys.jump),
        buy_menu: latched.buy_menu || pressed(&keys.buy_menu),
        reload: latched.reload || pressed(&keys.reload),
        scope: latched.scope || pressed(&keys.scope),
        equip: equips
            .iter()
            .find(|(binding, _)| pressed(binding))
            .map(|&(_, equip)| equip)
            .or(latched.equip),
        cast: casts
            .iter()
            .find(|(binding, _)| pressed(binding))
            .map(|&(_, slot)| slot)
            .or(latched.cast),
    };
    latched.set_if_neq(next);
}

pub(super) fn wish(
    devices: Res<Devices>,
    numbers: Res<Numbers>,
    screen: Option<Res<ScreenState>>,
    swallow: Res<Swallow>,
    mut latched: ResMut<Latched>,
    mut humans: Query<&mut Wish, With<Human>>,
    mut buys: Option<ResMut<ScreenBuys>>,
) {
    let Some(numbers) = numbers.get() else {
        return;
    };
    let keys = &numbers.keys;
    let bought = buys.as_deref_mut().map(|buys| std::mem::take(&mut buys.requests)).unwrap_or_default();
    let playing = !menu_open(screen.as_deref());
    let held = |binding: &[Input]| KeyMap::held(&devices, binding);
    let played = |binding: &[Input]| playing && held(binding);
    let axis = |plus: bool, minus: bool| f32::from(u8::from(plus)) - f32::from(u8::from(minus));
    let ahead = axis(played(&keys.forward), played(&keys.back));
    let aside = axis(played(&keys.right), played(&keys.left));
    let spent = std::mem::take(&mut *latched);
    for mut wish in &mut humans {
        let (sin, cos) = math::sin_cos(wish.yaw_radians);
        let forward = Vec3::new(-sin, 0.0, -cos);
        let right = Vec3::new(cos, 0.0, -sin);
        wish.direction = (forward * ahead + right * aside).normalize_or_zero();
        wish.gait = if played(&keys.walk) {
            Gait::Walk
        } else {
            Gait::Run
        };
        wish.crouch = played(&keys.crouch);
        wish.jump = playing && spent.jump;
        wish.fire_held = played(&keys.fire) && !swallow.0;
        wish.fire_pressed = playing && spent.fire;
        wish.reload = playing && spent.reload;
        wish.scope = playing && spent.scope;
        wish.equip = spent.equip.filter(|_| playing);
        wish.interact_held = played(&keys.interact);
        wish.cast = spent.cast.filter(|_| playing);
        wish.buy_menu = spent.buy_menu;
        wish.scoreboard = held(&keys.scoreboard);
        wish.buys.clone_from(&bought);
    }
}
