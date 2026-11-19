use engine_core::{Error, ErrorCode, Expected, MouseButton, TouchPhase};
use engine_registry::{Kind, Lands, NameEntry, Reach, unavailable};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::host::DeviceGatherer;

#[derive(Default, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
struct Pointer {
    position_pixels: Option<[f32; 2]>,
    #[serde(default)]
    motion_pixels: [f32; 2],
    #[serde(default)]
    wheel_notches: [f32; 2],
    #[serde(default)]
    buttons: Vec<MouseButton>,
}

#[derive(Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
struct Touch {
    id: u32,
    position_pixels: [f32; 2],
    #[serde(default)]
    motion_pixels: [f32; 2],
    phase: TouchPhase,
}

#[derive(Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
struct Held {
    #[serde(default)]
    keys: Vec<String>,
    #[serde(default)]
    pointer: Pointer,
    #[serde(default)]
    touches: Vec<Touch>,
}

fn held(reach: &Reach<'_>) -> Value {
    match reach.part::<DeviceGatherer>() {
        Some(agent) => json!({
            "keys": agent.keys_held().collect::<Vec<_>>(),
            "buttons": agent.mouse_held().collect::<Vec<_>>(),
        }),
        None => unavailable("no_agent_input"),
    }
}

fn checked(asked: &Held) -> Result<(), Error> {
    for (at, key) in asked.keys.iter().enumerate() {
        if !engine_core::is_id(key) {
            return Err(
                Error::new(ErrorCode::InvalidInput, format!("$.args.keys[{at}]"))
                    .expected(Expected::Id)
                    .got(key.as_str()),
            );
        }
    }
    let pointer = &asked.pointer;
    let finite = [
        (
            "$.args.pointer.position_pixels",
            pointer.position_pixels.unwrap_or_default(),
        ),
        ("$.args.pointer.motion_pixels", pointer.motion_pixels),
        ("$.args.pointer.wheel_notches", pointer.wheel_notches),
    ]
    .into_iter()
    .chain(asked.touches.iter().flat_map(|touch| {
        [
            ("$.args.touches.position_pixels", touch.position_pixels),
            ("$.args.touches.motion_pixels", touch.motion_pixels),
        ]
    }));
    for (path, [x, y]) in finite {
        if !(x.is_finite() && y.is_finite()) {
            return Err(Error::new(ErrorCode::InvalidInput, path)
                .expected(Expected::Finite)
                .got([x, y]));
        }
    }
    Ok(())
}

fn hold(agent: &mut DeviceGatherer, asked: &Held) {
    let keys: Vec<String> = agent.keys_held().map(str::to_owned).collect();
    for key in keys.iter().filter(|key| !asked.keys.contains(key)) {
        agent.key(key, false);
    }
    for key in &asked.keys {
        agent.key(key, true);
    }
    let pressed: Vec<MouseButton> = agent.mouse_held().collect();
    for &button in pressed
        .iter()
        .filter(|button| !asked.pointer.buttons.contains(button))
    {
        agent.mouse_button(button, false);
    }
    for &button in &asked.pointer.buttons {
        agent.mouse_button(button, true);
    }
    match asked.pointer.position_pixels {
        Some([x, y]) => agent.pointer(x, y),
        None => agent.pointer_left(),
    }
    let [dx, dy] = asked.pointer.motion_pixels;
    agent.motion(dx, dy);
    let [right, up] = asked.pointer.wheel_notches;
    agent.wheel(right, up);
    for touch in &asked.touches {
        let [x, y] = touch.position_pixels;
        agent.touch(touch.id, x, y, touch.phase);
        let [dx, dy] = touch.motion_pixels;
        agent.touch_motion(touch.id, dx, dy);
    }
}

pub(super) fn entry() -> NameEntry {
    NameEntry::read("input", Kind::State, |reach, _| Ok(held(reach)))
        .set(Lands::NextFrame, |reach, args| {
            let asked: Held = engine_settings::parse_value_at(args.clone(), "$.args")?;
            checked(&asked)?;
            let before = reach.read(held);
            let agent = reach
                .part::<DeviceGatherer>()
                .ok_or_else(|| Error::new(ErrorCode::MissingResource, "$").got("agent_input"))?;
            hold(agent, &asked);
            Ok(json!({"before": before, "after": reach.read(held)}))
        })
        .args(
            serde_json::to_value(schemars::schema_for!(Held)).expect("schema_serializes"),
            json!({"keys": ["KeyW"], "pointer": {"buttons": ["left"]}}),
        )
}
