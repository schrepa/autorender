use std::sync::Arc;

use engine::math;
use engine::prelude::*;
use engine::{Error, ErrorCode};
use engine_bindings::Input;
use engine_core::{Asset, Assets};
use glam::Vec3;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::names::PITCH_LIMIT_DEGREES;
use crate::play::actors::{Equip, Gait, Human, Wish};
use crate::play::files;
use crate::play::numbers::{GameNumbers, Numbers};
use crate::play::round::Round;
use crate::play::screen::{ScreenBuys, ScreenState};
use crate::play::view::Controls;
use crate::rules::AbilitySlot;

#[derive(Component, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(super) struct NumbersFile {
    pub file: Asset<Arc<GameNumbers>>,
}

fn part<T: serde::de::DeserializeOwned>(whole: &Value, key: &str) -> Result<T, Error> {
    let value = whole
        .get(key)
        .ok_or_else(|| Error::new(ErrorCode::MissingField, format!("$.{key}")))?;
    engine::bricks::parse_value_at(value, &format!("$.{key}"))
}

fn parsed(whole: Value) -> Result<GameNumbers, Error> {
    Ok(GameNumbers {
        player: part(&whole, "player")?,
        keys: part(&whole, "keys")?,
        spectate: part(&whole, "spectate")?,
        weapons: part(&whole, "weapons")?,
        hit_shapes: part(&whole, "hit_shapes")?,
        rules: engine::bricks::parse_value_at(&whole, "$")?,
        nav: part(&whole, "nav")?,
        paths: whole.get("paths").map(|_| part(&whole, "paths")).transpose()?,
        whole,
    })
}

pub(super) fn numbers(bytes: Arc<[u8]>, at: &str) -> Result<Arc<GameNumbers>, Error> {
    parsed(files::json(bytes, at)?)
        .map(Arc::new)
        .map_err(|error| error.within(at))
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Spun(f32);

fn cycled(spun: f32, turned: f32) -> (i32, f32) {
    let spun = if turned * spun < 0.0 {
        turned
    } else {
        spun + turned
    };
    if spun.abs() >= 1.0 {
        (if spun > 0.0 { 1 } else { -1 }, spun.fract())
    } else {
        (0, spun)
    }
}

pub(super) fn adopt_numbers(
    files: Query<Ref<NumbersFile>>,
    assets: Res<Assets>,
    mut numbers: ResMut<Numbers>,
) {
    for file in &files {
        if file.is_changed() {
            numbers.adopt(assets.get(file.file).ok().cloned());
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
    if menu_open(screen.as_deref()) || engine_bindings::pressed(&devices, &numbers.keys.release) {
        cursor.set_if_neq(CursorMode::Free);
    } else if *cursor != CursorMode::Captured
        && engine_bindings::pressed(&devices, &numbers.keys.capture)
    {
        *cursor = CursorMode::Captured;
        swallow.set_if_neq(Swallow(true));
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

pub(super) fn wish(
    devices: Res<Devices>,
    numbers: Res<Numbers>,
    (screen, round): (Option<Res<ScreenState>>, Option<Res<Round>>),
    (mut swallow, mut spun): (ResMut<Swallow>, ResMut<Spun>),
    mut humans: Query<&mut Wish, With<Human>>,
    mut buys: Option<ResMut<ScreenBuys>>,
) {
    let Some(numbers) = numbers.get() else {
        return;
    };
    let keys = &numbers.keys;
    let bought = buys
        .as_deref_mut()
        .map(|buys| std::mem::take(&mut buys.requests))
        .unwrap_or_default();
    let playing = !menu_open(screen.as_deref());
    let buying = round.as_deref().is_some_and(|round| round.buy_lock);
    let held = |binding: &[Input]| engine_bindings::held(&devices, binding);
    let pressed = |binding: &[Input]| engine_bindings::pressed(&devices, binding);
    let played = |binding: &[Input]| playing && held(binding);
    let acted = |binding: &[Input]| playing && pressed(binding);
    let axis = |plus: bool, minus: bool| f32::from(u8::from(plus)) - f32::from(u8::from(minus));
    let ahead = axis(played(&keys.forward), played(&keys.back));
    let aside = axis(played(&keys.right), played(&keys.left));
    let turned = engine_bindings::turns(&devices, &keys.next_item)
        - engine_bindings::turns(&devices, &keys.previous_item);
    let (cycle, carried) = cycled(spun.0, turned);
    spun.set_if_neq(Spun(carried));
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
    let equip = (equips.iter())
        .find(|(binding, _)| acted(binding))
        .map(|&(_, equip)| equip);
    let cast = (casts.iter())
        .find(|(binding, _)| acted(binding))
        .map(|&(_, slot)| slot);
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
        wish.jump = acted(&keys.jump);
        wish.fire_held = played(&keys.fire) && !swallow.0;
        wish.fire_pressed = acted(&keys.fire) && !swallow.0;
        wish.reload = acted(&keys.reload);
        wish.inspect = acted(&keys.inspect);
        wish.ready = buying && acted(&keys.ready);
        wish.drop = acted(&keys.drop);
        wish.cycle = if playing { cycle } else { 0 };
        wish.scope = acted(&keys.scope);
        wish.equip = equip;
        wish.interact_held = played(&keys.interact);
        wish.cast = cast;
        wish.buy_menu = pressed(&keys.buy_menu);
        wish.scoreboard = held(&keys.scoreboard);
        wish.buys.clone_from(&bought);
    }
    if swallow.0 && !held(&keys.capture) {
        *swallow = Swallow(false);
    }
}
