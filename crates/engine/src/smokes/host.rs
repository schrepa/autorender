use std::collections::BTreeSet;

use engine_core::{Runs, Smoke, SmokeFailure, SmokeSession, smoke_eq};
use serde_json::json;

use crate::host::{InputGatherer, MENU_KEYS, MOVE_KEYS, MoveDirection, SceneLoad};
use engine_registry::InputState;

pub(super) const SMOKES: &[Smoke] = &[
    Smoke {
        name: "input_keys",
        runs: Runs::Headless,
        run: input_keys,
    },
    Smoke {
        name: "host_scene_files",
        runs: Runs::Headless,
        run: scene_files,
    },
];

const ACTIONS: &[(&str, u32)] = &[("Space", 0), ("MouseRight", 1)];
const AXES: &[(&str, u32, f32)] = &[("KeyL", 0, 1.0), ("KeyJ", 0, -1.0), ("PadLeftX", 1, 0.5)];
const PAD_UP: u32 = 1 << 12;
const PAD_SOUTH: u32 = 1;

fn input_keys(_: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    let built = InputGatherer::new(ACTIONS, AXES, MOVE_KEYS, MENU_KEYS);
    smoke_eq!(built.is_ok(), true, "gatherer_builds");
    let Ok(mut keys) = built else {
        return Ok(());
    };
    keyboard(&mut keys)?;
    pointer(&mut keys)?;
    mouse_binding(&mut keys)?;
    axes(&mut keys)?;
    pad(&mut keys)?;
    menu(&mut keys)?;
    move_keys()?;
    edges()
}

fn move_keys() -> Result<(), SmokeFailure> {
    let built = InputGatherer::new(
        ACTIONS,
        AXES,
        &[("KeyW", MoveDirection::Forward)],
        MENU_KEYS,
    );
    smoke_eq!(built.is_ok(), true, "game_move_keys_build");
    let Ok(mut keys) = built else {
        return Ok(());
    };
    smoke_eq!(
        keys.key("KeyE", true, false),
        false,
        "unlisted_move_key_stays_free"
    );
    smoke_eq!(keys.key("KeyW", true, false), true, "listed_move_key_used");
    let state = keys.take(0.0).state;
    smoke_eq!(
        json!([state.move_forward, state.move_up]),
        json!([1.0, 0.0]),
        "only_listed_keys_move"
    );
    let unknown = InputGatherer::new(ACTIONS, AXES, &[("KeyNone", MoveDirection::Up)], MENU_KEYS);
    smoke_eq!(
        json!(unknown.err().map(|error| error.path)),
        json!("$.move_keys[0]"),
        "unknown_move_key_refused"
    );
    Ok(())
}

fn edges() -> Result<(), SmokeFailure> {
    let shown = |state: &InputState| json!([state.just_pressed(0), state.just_released(0)]);
    let mut state = InputState::default();
    let first = BTreeSet::from([0]);
    let none = BTreeSet::new();
    state.ingest(&first, &first, &none);
    state.begin_tick();
    smoke_eq!(
        shown(&state),
        json!([true, false]),
        "first_tick_sees_a_press"
    );
    state.hide_edges();
    state.begin_tick();
    smoke_eq!(shown(&state), json!([false, false]), "second_tick_does_not");
    state.hide_edges();
    state.begin_frame();
    smoke_eq!(
        shown(&state),
        json!([true, false]),
        "the_frame_sees_the_press"
    );
    state.hide_edges();
    state.begin_frame();
    smoke_eq!(
        shown(&state),
        json!([false, false]),
        "the_next_frame_does_not"
    );
    state.hide_edges();
    state.ingest(&none, &none, &first);
    state.begin_frame();
    smoke_eq!(
        shown(&state),
        json!([false, true]),
        "a_frame_without_ticks_sees_a_release"
    );
    state.hide_edges();
    state.begin_tick();
    smoke_eq!(
        shown(&state),
        json!([false, true]),
        "the_tick_after_still_sees_it"
    );
    Ok(())
}

fn keyboard(keys: &mut InputGatherer) -> Result<(), SmokeFailure> {
    smoke_eq!(keys.key("KeyW", true, false), true, "movement_key_used");
    let held = [keys.take(0.0), keys.take(0.0)].map(|input| input.state.move_forward);
    smoke_eq!(json!(held), json!([1.0, 1.0]), "key_moves");
    keys.key("KeyW", false, false);
    smoke_eq!(
        json!(keys.take(0.0).state.move_forward),
        json!(0.0),
        "release_stops"
    );
    keys.look(120.0, 40.0);
    let turned = keys.take(0.0).state;
    smoke_eq!(
        json!([turned.look_right_pixels, turned.look_down_pixels]),
        json!([120.0, 40.0]),
        "drag_turns"
    );
    let after = keys.take(0.0).state;
    smoke_eq!(
        json!([after.look_right_pixels, after.look_down_pixels]),
        json!([0.0, 0.0]),
        "turns_apply_once"
    );
    smoke_eq!(
        json!([keys.key("KeyW", true, false), keys.key("KeyF", true, false)]),
        json!([true, false]),
        "unused_keys_stay"
    );
    keys.key("KeyW", false, false);
    keys.key("Space", true, false);
    let pressed = keys.take(0.0);
    smoke_eq!(
        json!([pressed.held, pressed.pressed]),
        json!([[0], [0]]),
        "actions_press"
    );
    keys.key("KeyW", true, false);
    keys.clear();
    let blurred = keys.take(0.0);
    smoke_eq!(
        json!([blurred.state.move_forward, blurred.held]),
        json!([0.0, []]),
        "blur_releases"
    );
    Ok(())
}

fn pointer(keys: &mut InputGatherer) -> Result<(), SmokeFailure> {
    keys.pointer(10.0, 20.0);
    keys.pointer(13.0, 24.0);
    keys.wheel(0.0, 2.0);
    let moved = keys.take(0.0).pointer;
    smoke_eq!(
        json!([
            moved.position_pixels.map(|position| position.to_array()),
            moved.delta_pixels.to_array(),
            [moved.wheel_right_notches, moved.wheel_up_notches],
            moved.inside
        ]),
        json!([[13.0, 24.0], [3.0, 4.0], [0.0, 2.0], true]),
        "pointer_moves"
    );
    let still = keys.take(0.0).pointer;
    smoke_eq!(
        json!([
            still.delta_pixels.to_array(),
            [still.wheel_right_notches, still.wheel_up_notches]
        ]),
        json!([[0.0, 0.0], [0.0, 0.0]]),
        "pointer_motion_covers_one_frame"
    );
    keys.pointer_left();
    smoke_eq!(
        json!(keys.take(0.0).pointer.inside),
        json!(false),
        "pointer_left"
    );
    keys.motion(5.0, 6.0);
    let captured = keys.take(0.0);
    smoke_eq!(
        json!([
            captured.pointer.delta_pixels.to_array(),
            captured.state.look_right_pixels,
            captured.state.look_down_pixels
        ]),
        json!([[5.0, 6.0], 5.0, 6.0]),
        "motion_looks"
    );
    keys.pointer(1.0, 2.0);
    keys.wheel(1.0, 0.0);
    keys.clear();
    let cleared = keys.take(0.0).pointer;
    smoke_eq!(
        json!([
            cleared.position_pixels.is_none(),
            cleared.delta_pixels.to_array(),
            [cleared.wheel_right_notches, cleared.wheel_up_notches]
        ]),
        json!([true, [0.0, 0.0], [0.0, 0.0]]),
        "pointer_clears"
    );
    Ok(())
}

fn mouse_binding(keys: &mut InputGatherer) -> Result<(), SmokeFailure> {
    smoke_eq!(
        json!([
            keys.key("MouseRight", true, false),
            keys.key("MouseLeft", true, false)
        ]),
        json!([true, true]),
        "mouse_bindings_used"
    );
    let pressed = keys.take(0.0);
    smoke_eq!(
        json!([pressed.held, pressed.pressed]),
        json!([[1], [1]]),
        "mouse_presses_action"
    );
    smoke_eq!(
        json!([pressed.pointer.held, pressed.pointer.pressed]),
        json!([true, true]),
        "mouse_left_presses_pointer"
    );
    keys.key("MouseRight", false, false);
    keys.key("MouseLeft", false, false);
    let released = keys.take(0.0);
    smoke_eq!(
        json!([released.held, released.released]),
        json!([[], [1]]),
        "mouse_releases_action"
    );
    smoke_eq!(
        json!([
            released.pointer.held,
            released.pointer.pressed,
            released.pointer.released
        ]),
        json!([false, false, true]),
        "mouse_left_releases_pointer"
    );
    let after = keys.take(0.0).pointer;
    smoke_eq!(
        json!([after.held, after.pressed, after.released]),
        json!([false, false, false]),
        "pointer_press_covers_one_frame"
    );
    Ok(())
}

fn axes(keys: &mut InputGatherer) -> Result<(), SmokeFailure> {
    keys.key("KeyL", true, false);
    smoke_eq!(
        json!(keys.take(0.0).axes.values()[0]),
        json!(1.0),
        "axis_key_counts"
    );
    keys.key("KeyJ", true, false);
    smoke_eq!(
        json!(keys.take(0.0).axes.values()[0]),
        json!(0.0),
        "axis_sums"
    );
    keys.key("KeyL", false, false);
    keys.key("KeyJ", false, false);
    keys.pad(0, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    smoke_eq!(
        json!(keys.take(0.0).axes.values()[1]),
        json!(0.5),
        "axis_analog_scales"
    );
    keys.pad(0, [0.1, 0.0, 0.0, 0.0, 0.0, 0.0]);
    smoke_eq!(
        json!(keys.take(0.0).axes.values()[1]),
        json!(0.0),
        "stick_dead_zone"
    );
    Ok(())
}

fn pad(keys: &mut InputGatherer) -> Result<(), SmokeFailure> {
    keys.take(0.0);
    keys.pad(0, [0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
    keys.pad(0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let moved = keys.take(1000.0).state;
    smoke_eq!(
        json!([moved.move_forward, moved.move_up, moved.look_right_pixels]),
        json!([1.0, 1.0, 600.0]),
        "pads_merge_and_move"
    );
    let unplugged = keys.take(2000.0).state;
    smoke_eq!(
        json!([unplugged.move_forward, unplugged.look_right_pixels]),
        json!([0.0, 0.0]),
        "pad_needs_polling"
    );
    Ok(())
}

fn menu(keys: &mut InputGatherer) -> Result<(), SmokeFailure> {
    smoke_eq!(
        json!(keys.key("Tab", true, false)),
        json!(false),
        "menu_key_stays_with_browser"
    );
    keys.key("Tab", false, false);
    smoke_eq!(
        json!(keys.take(0.0).menu.down),
        json!(true),
        "menu_gathered_unwanted"
    );
    smoke_eq!(
        json!(keys.key("Tab", true, true)),
        json!(true),
        "menu_key_taken_while_wanted"
    );
    smoke_eq!(
        json!(keys.take(0.0).menu.down),
        json!(true),
        "menu_pressed_wanted"
    );
    smoke_eq!(
        json!(keys.take(0.0).menu.down),
        json!(false),
        "menu_pressed_once"
    );
    smoke_eq!(
        json!(keys.key("Tab", false, false)),
        json!(true),
        "menu_release_taken_after_unwanted"
    );
    keys.pad(PAD_UP | PAD_SOUTH, [0.0; 6]);
    let pressed = keys.take(0.0).menu;
    smoke_eq!(
        json!([pressed.up, pressed.accept]),
        json!([true, true]),
        "pad_menu"
    );
    keys.pad(PAD_UP | PAD_SOUTH, [0.0; 6]);
    let held = keys.take(0.0).menu;
    smoke_eq!(
        json!([held.up, held.accept]),
        json!([false, false]),
        "pad_menu_once"
    );
    keys.key("Enter", true, true);
    keys.clear();
    smoke_eq!(
        json!(keys.take(0.0).menu.accept),
        json!(false),
        "menu_clears"
    );
    Ok(())
}

const ALPHA_PIXELS: [u8; 16] = [
    201, 77, 13, 3, 10, 250, 128, 128, 255, 255, 255, 0, 33, 66, 99, 254,
];
const ALPHA_URI: &str = "host-parity/alpha.png";

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0_u32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

fn adler32(bytes: &[u8]) -> u32 {
    let (mut low, mut high) = (1_u32, 0_u32);
    for &byte in bytes {
        low = (low + u32::from(byte)) % 65_521;
        high = (high + low) % 65_521;
    }
    (high << 16) | low
}

fn chunk(png: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    png.extend((data.len() as u32).to_be_bytes());
    let start = png.len();
    png.extend(kind);
    png.extend(data);
    let crc = crc32(&png[start..]);
    png.extend(crc.to_be_bytes());
}

fn png(width: u32, pixels: &[u8]) -> Vec<u8> {
    let height = pixels.len() as u32 / 4 / width;
    let mut rows = Vec::with_capacity(pixels.len() + height as usize);
    for row in pixels.chunks(width as usize * 4) {
        rows.push(0);
        rows.extend_from_slice(row);
    }
    let stored = rows.len() as u16;
    let mut deflated = vec![0x78, 0x01, 0x01];
    deflated.extend(stored.to_le_bytes());
    deflated.extend((!stored).to_le_bytes());
    deflated.extend(&rows);
    deflated.extend(adler32(&rows).to_be_bytes());
    let mut header = Vec::with_capacity(13);
    header.extend(width.to_be_bytes());
    header.extend(height.to_be_bytes());
    header.extend([8, 6, 0, 0, 0]);
    let mut png = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    chunk(&mut png, b"IHDR", &header);
    chunk(&mut png, b"IDAT", &deflated);
    chunk(&mut png, b"IEND", &[]);
    png
}

fn loaded(source: &str, image: &[u8]) -> crate::Result<engine_scene::SceneFiles> {
    let mut load = SceneLoad::new(source, &engine_scene::SceneKinds::default())?;
    loop {
        let reads = load.take_reads();
        if reads.is_empty() {
            break;
        }
        for read in reads {
            if let Some(bytes) = load.read(&read.uri, image.to_vec())? {
                load.decode(&read.uri, &bytes)?;
            }
        }
    }
    load.finish()
}

fn scene_files(_: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    let image = png(2, &ALPHA_PIXELS);
    let decoded = crate::host::decode_image(ALPHA_URI, &image);
    smoke_eq!(
        json!(
            decoded
                .as_ref()
                .map(|image| (image.width, image.height, image.pixels.clone()))
                .ok()
        ),
        json!([2, 2, ALPHA_PIXELS]),
        "png_decodes_exactly"
    );
    let source = json!({
        "id": "host-parity-texture", "revision": 1,
        "camera": { "position": [0, 2, 4], "target": [0, 0, 0], "fov_degrees": 60, "near": 0.1, "far": 100 },
        "textures": [{ "id": "alpha", "uri": ALPHA_URI }],
        "materials": [{ "id": "alpha", "color": [1, 1, 1, 1], "texture": "alpha" }],
        "meshes": [{ "id": "panel", "source": { "kind": "cube" } }],
        "entities": [{ "id": "panel", "mesh": "panel", "material": "alpha", "transform": { "translation": [2, 0, 0] } }],
    })
    .to_string();
    let files = loaded(&source, &image);
    smoke_eq!(
        json!(
            files
                .as_ref()
                .err()
                .map(|error| json!({ "code": error.code, "path": error.path }))
        ),
        json!(null),
        "scene_files_load"
    );
    Ok(())
}
