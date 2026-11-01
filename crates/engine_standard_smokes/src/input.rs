use engine_core::smoke::{SmokeResult, SmokeSession};
use serde_json::json;

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
