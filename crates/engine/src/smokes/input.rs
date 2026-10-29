use engine_core::smoke::{SmokeFailure, SmokeResult, SmokeSession};
use serde_json::{Value, json};

use super::requests::{control, grid, inspected, spawn};
use crate::camera::FlyCameraController;
use crate::host::InputGatherer;

pub(crate) fn input_keys(_: &mut dyn SmokeSession) -> SmokeResult {
    let built = InputGatherer::new(&[("Space", 0)]);
    engine_core::smoke_check!(built.is_ok(), "gatherer_builds", json!(false), json!(true));
    let Ok(mut keys) = built else {
        return Ok(());
    };
    engine_core::smoke_check!(
        keys.key("KeyW", true),
        "movement_key_used",
        json!(false),
        json!(true)
    );
    let held = [keys.take(), keys.take()].map(|input| input.state.move_forward);
    engine_core::smoke_eq!(json!(held), json!([1.0, 1.0]), "key_moves");
    keys.key("KeyW", false);
    engine_core::smoke_eq!(
        json!(keys.take().state.move_forward),
        json!(0.0),
        "release_stops"
    );
    keys.look(120.0, 40.0);
    let turned = keys.take().state;
    engine_core::smoke_eq!(
        json!([turned.look_yaw, turned.look_pitch]),
        json!([120.0, 40.0]),
        "drag_turns"
    );
    let after = keys.take().state;
    engine_core::smoke_eq!(
        json!([after.look_yaw, after.look_pitch]),
        json!([0.0, 0.0]),
        "turns_apply_once"
    );
    engine_core::smoke_eq!(
        json!([keys.key("KeyW", true), keys.key("KeyF", true)]),
        json!([true, false]),
        "unused_keys_stay"
    );
    keys.key("KeyW", false);
    keys.key("Space", true);
    let pressed = keys.take();
    engine_core::smoke_eq!(
        json!([pressed.held & 1, pressed.pressed & 1]),
        json!([1, 1]),
        "actions_press"
    );
    keys.key("KeyW", true);
    keys.clear();
    let blurred = keys.take();
    engine_core::smoke_eq!(
        json!([blurred.state.move_forward, blurred.held]),
        json!([0.0, 0]),
        "blur_releases"
    );
    Ok(())
}

fn axes(value: &Value) -> [f64; 3] {
    std::array::from_fn(|axis| value[axis].as_f64().unwrap_or(f64::NAN))
}

fn direction(camera: &Value) -> [f64; 3] {
    let (position, target) = (axes(&camera["position"]), axes(&camera["target"]));
    let delta: [f64; 3] = std::array::from_fn(|axis| target[axis] - position[axis]);
    let length = delta.iter().map(|value| value * value).sum::<f64>().sqrt();
    delta.map(|value| value / length)
}

fn moved(from: &Value, to: &Value) -> [f64; 3] {
    let (from, to) = (axes(&from["position"]), axes(&to["position"]));
    std::array::from_fn(|axis| to[axis] - from[axis])
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

fn camera(session: &mut dyn SmokeSession) -> Result<Value, SmokeFailure> {
    inspected(session, "Camera", "inspects_camera")
}

fn steps(session: &mut dyn SmokeSession, input: Value, ticks: u32) -> SmokeResult {
    control(session, "input", input, "takes_input")?;
    control(session, "step", json!({ "ticks": ticks }), "steps")?;
    Ok(())
}

pub(crate) fn input_camera(session: &mut dyn SmokeSession) -> SmokeResult {
    spawn(session, &grid(8, false))?;
    control(session, "pause", json!({}), "pauses")?;
    let flies = session
        .app()
        .world()
        .contains_resource::<FlyCameraController>();
    engine_core::smoke_check!(flies, "fly_camera_added", json!(flies), json!(true));
    let start = camera(session)?;
    steps(session, json!({ "forward": 1 }), 20)?;
    let forward = camera(session)?;
    let along = dot(moved(&start, &forward), direction(&start));
    engine_core::smoke_check!(
        along > 0.5,
        "key_moves",
        json!({ "start": start, "forward": forward }),
        json!({ "along": "> 0.5" })
    );
    steps(session, json!({}), 10)?;
    let still = camera(session)?;
    let drift = dot(moved(&forward, &still), moved(&forward, &still)).sqrt();
    engine_core::smoke_check!(
        drift < 1e-3,
        "release_stops",
        json!(drift),
        json!(1e-3)
    );
    steps(session, json!({ "yaw": 120, "pitch": 40 }), 10)?;
    let turned = camera(session)?;
    let cosine = dot(direction(&still), direction(&turned));
    let turn = if cosine < 1.0 { cosine.acos() } else { 0.0 };
    engine_core::smoke_check!(turn > 0.05, "drag_turns", json!(turn), json!(0.05));
    steps(session, json!({}), 1)?;
    Ok(())
}
