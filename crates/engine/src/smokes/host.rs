use bevy_ecs::world::World;
use engine_core::{
    Binding, BindingSource, Button, ContextAsk, Contexts, Devices, Gathered, Input, InputSettings,
    Interest, MouseButton, PadButton, Player, PlayerDevices, Runs, Smoke, SmokeFailure,
    SmokeSession, Stick, TouchPhase, Verb, VerbId, VerbShape, accept_input, smoke_eq,
};
use serde_json::json;

use crate::host::{DeviceGatherer, SceneLoad};

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

const VERBS: [Verb; 4] = [
    Verb {
        name: "test.move",
        shape: VerbShape::Value2,
        bindings: &[
            Binding::new("test.move", BindingSource::keys("KeyD", "KeyA", "KeyW", "KeyS")),
            Binding::new("test.move", BindingSource::Stick(Stick::Left)).dead_zone(0.15),
        ],
    },
    Verb {
        name: "test.look",
        shape: VerbShape::Delta2,
        bindings: &[Binding::held(
            "test.look",
            BindingSource::MouseMotion,
            Button::Mouse(MouseButton::Left),
        )],
    },
    Verb {
        name: "test.jump",
        shape: VerbShape::Button,
        bindings: &[
            Binding::new("test.jump", BindingSource::key("Space")),
            Binding::new("test.jump", BindingSource::key("Enter")),
            Binding::new("test.jump", BindingSource::Pad(PadButton::South)),
        ],
    },
    Verb {
        name: "menu.accept",
        shape: VerbShape::Button,
        bindings: &[Binding::new("menu.accept", BindingSource::key("Enter"))],
    },
];
const PAD_SOUTH: u32 = 1;
const ONE: Player = Player::ONE;

fn rig() -> (World, [VerbId; 4]) {
    let mut world = World::new();
    world.init_resource::<Input>();
    world.init_resource::<Contexts>();
    world.init_resource::<InputSettings>();
    world.init_resource::<Devices>();
    let ids = {
        let mut input = world.resource_mut::<Input>();
        VERBS.map(|verb| input.declare(&verb))
    };
    (world, ids)
}

fn ask(world: &mut World, context: &'static str, blocks: bool) -> Interest<ContextAsk> {
    world
        .resource_mut::<Contexts>()
        .ask("smoke", context, blocks)
}

fn accept(world: &mut World, keys: &mut DeviceGatherer) {
    accept_input(
        world,
        Gathered {
            devices: keys.take(),
            driven: Vec::new(),
        },
    );
}

fn frame<'w>(world: &'w mut World, keys: &mut DeviceGatherer) -> &'w Input {
    accept(world, keys);
    world.resource_mut::<Input>().begin_frame();
    world.resource::<Input>()
}

fn input_keys(_: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    keyboard()?;
    pointer()?;
    pads()?;
    contexts()?;
    players()?;
    touches()?;
    edges()
}

fn keyboard() -> Result<(), SmokeFailure> {
    let (mut world, [moving, look, jump, _]) = rig();
    let _asked = ask(&mut world, "test", false);
    let mut keys = DeviceGatherer::default();
    keys.key("KeyW", true);
    let first = frame(&mut world, &mut keys).value2(moving, ONE).to_array();
    let second = frame(&mut world, &mut keys).value2(moving, ONE).to_array();
    smoke_eq!(json!([first, second]), json!([[0.0, 1.0], [0.0, 1.0]]), "keys_hold_a_value");
    keys.key("KeyW", false);
    let released = frame(&mut world, &mut keys).value2(moving, ONE).to_array();
    smoke_eq!(json!(released), json!([0.0, 0.0]), "release_stops");
    keys.motion(120.0, 40.0);
    let unheld = frame(&mut world, &mut keys).value2(look, ONE).to_array();
    smoke_eq!(json!(unheld), json!([0.0, 0.0]), "motion_waits_for_its_modifier");
    keys.mouse_button(MouseButton::Left, true);
    keys.motion(120.0, 40.0);
    let turned = frame(&mut world, &mut keys).value2(look, ONE).to_array();
    smoke_eq!(json!(turned), json!([120.0, 40.0]), "drag_turns");
    let after = frame(&mut world, &mut keys).value2(look, ONE).to_array();
    smoke_eq!(json!(after), json!([0.0, 0.0]), "turns_apply_once");
    keys.mouse_button(MouseButton::Left, false);
    keys.key("Space", true);
    let input = frame(&mut world, &mut keys);
    let pressed = json!([input.held(jump, ONE), input.pressed(jump, ONE)]);
    smoke_eq!(pressed, json!([true, true]), "buttons_press");
    keys.key("Enter", true);
    keys.key("Space", false);
    let input = frame(&mut world, &mut keys);
    let still = json!([
        input.held(jump, ONE),
        input.pressed(jump, ONE),
        input.released(jump, ONE)
    ]);
    smoke_eq!(still, json!([true, false, false]), "one_key_still_holds_the_verb");
    keys.key("Enter", false);
    let input = frame(&mut world, &mut keys);
    let last = json!([input.held(jump, ONE), input.released(jump, ONE)]);
    smoke_eq!(last, json!([false, true]), "the_last_key_releases_the_verb");
    keys.key("Space", true);
    keys.drop_edges();
    let kept = keys.take();
    smoke_eq!(
        json!([kept.keyboard.held.contains("Space"), kept.keyboard.pressed.is_empty()]),
        json!([true, true]),
        "a_suspend_keeps_keys_and_drops_edges"
    );
    keys.clear();
    smoke_eq!(
        json!(keys.take().keyboard.held.is_empty()),
        json!(true),
        "blur_releases"
    );
    Ok(())
}

fn pointer() -> Result<(), SmokeFailure> {
    let mut keys = DeviceGatherer::default();
    keys.pointer(13.0, 24.0);
    keys.wheel(0.0, 2.0);
    keys.motion(3.0, 4.0);
    let moved = keys.take().mouse;
    smoke_eq!(
        json!([
            moved.position_pixels.map(|position| position.to_array()),
            moved.motion_pixels.to_array(),
            moved.wheel_notches.to_array(),
            moved.inside
        ]),
        json!([[13.0, 24.0], [3.0, 4.0], [0.0, 2.0], true]),
        "pointer_moves"
    );
    let still = keys.take().mouse;
    smoke_eq!(
        json!([still.motion_pixels.to_array(), still.wheel_notches.to_array()]),
        json!([[0.0, 0.0], [0.0, 0.0]]),
        "pointer_motion_covers_one_frame"
    );
    keys.pointer_left();
    smoke_eq!(json!(keys.take().mouse.inside), json!(false), "pointer_left");
    keys.pointer(1.0, 2.0);
    keys.clear();
    smoke_eq!(
        json!(keys.take().mouse.position_pixels.is_none()),
        json!(true),
        "pointer_clears"
    );
    Ok(())
}

fn pads() -> Result<(), SmokeFailure> {
    let (mut world, [moving, _, jump, _]) = rig();
    let _asked = ask(&mut world, "test", false);
    let mut keys = DeviceGatherer::default();
    keys.pad(0, 0, [0.1, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let zoned = frame(&mut world, &mut keys).value2(moving, ONE).to_array();
    smoke_eq!(json!(zoned), json!([0.0, 0.0]), "the_binding_holds_the_dead_zone");
    keys.pad(0, 0, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
    keys.pad(1, PAD_SOUTH, [0.0; 6]);
    let input = frame(&mut world, &mut keys);
    let both = json!([input.value2(moving, ONE).to_array(), input.pressed(jump, ONE)]);
    smoke_eq!(both, json!([[0.0, 1.0], true]), "each_pad_by_index");
    let input = frame(&mut world, &mut keys);
    let unplugged = json!([input.value2(moving, ONE).to_array(), input.released(jump, ONE)]);
    smoke_eq!(unplugged, json!([[0.0, 0.0], true]), "a_pad_needs_polling");
    Ok(())
}

fn contexts() -> Result<(), SmokeFailure> {
    let (mut world, [_, _, jump, accepting]) = rig();
    let mut keys = DeviceGatherer::default();
    keys.key("Enter", true);
    let read = |input: &Input| {
        json!([
            input.held(jump, ONE),
            input.held(accepting, ONE),
            input.uses_key("Enter")
        ])
    };
    let unasked = read(frame(&mut world, &mut keys));
    smoke_eq!(unasked, json!([false, false, false]), "unasked_contexts_stay_off");
    let game = ask(&mut world, "test", false);
    let menu = ask(&mut world, "menu", true);
    let opened = read(frame(&mut world, &mut keys));
    smoke_eq!(opened, json!([false, true, true]), "a_blocking_context_takes_the_keys");
    drop(menu);
    let closed = read(frame(&mut world, &mut keys));
    smoke_eq!(closed, json!([true, false, true]), "closing_hands_the_keys_back");
    let shared = ask(&mut world, "menu", false);
    let newest = read(frame(&mut world, &mut keys));
    smoke_eq!(newest, json!([false, true, true]), "the_newest_context_wins_a_key");
    drop(shared);
    world.insert_resource(InputSettings {
        players: Vec::new(),
        bindings: vec![Binding::new("test.look", BindingSource::key("Space"))],
    });
    let conflicts: Vec<Vec<&str>> = frame(&mut world, &mut keys)
        .conflicts()
        .iter()
        .map(|conflict| conflict.verbs.clone())
        .collect();
    smoke_eq!(
        json!(conflicts),
        json!([["test.look", "test.jump"]]),
        "conflicts_name_both_verbs"
    );
    drop(game);
    Ok(())
}

fn players() -> Result<(), SmokeFailure> {
    let (mut world, [moving, ..]) = rig();
    world.insert_resource(InputSettings {
        players: vec![
            PlayerDevices {
                keyboard_mouse: true,
                pads: Vec::new(),
            },
            PlayerDevices {
                keyboard_mouse: false,
                pads: vec![3],
            },
        ],
        bindings: Vec::new(),
    });
    let _asked = ask(&mut world, "test", false);
    let mut keys = DeviceGatherer::default();
    keys.key("KeyD", true);
    keys.pad(3, 0, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
    let input = frame(&mut world, &mut keys);
    smoke_eq!(
        json!([
            input.players(),
            input.value2(moving, Player(1)).to_array(),
            input.value2(moving, Player(2)).to_array()
        ]),
        json!([2, [1.0, 0.0], [0.0, 1.0]]),
        "each_player_reads_its_own_devices"
    );
    Ok(())
}

fn touches() -> Result<(), SmokeFailure> {
    let mut keys = DeviceGatherer::default();
    keys.touch(7, 10.0, 20.0, TouchPhase::Began);
    keys.touch(7, 12.0, 20.0, TouchPhase::Ended);
    let phases = [keys.take(), keys.take(), keys.take()].map(|devices| {
        devices
            .touches
            .iter()
            .map(|touch| touch.phase)
            .collect::<Vec<_>>()
    });
    smoke_eq!(
        json!(phases),
        json!([["began"], ["ended"], []]),
        "a_tap_inside_one_frame_still_ends"
    );
    Ok(())
}

fn edges() -> Result<(), SmokeFailure> {
    let (mut world, [_, _, jump, _]) = rig();
    let _asked = ask(&mut world, "test", false);
    let mut keys = DeviceGatherer::default();
    let shown = |world: &World| {
        let input = world.resource::<Input>();
        json!([input.pressed(jump, ONE), input.released(jump, ONE)])
    };
    let run = |world: &mut World, change: fn(&mut Input)| {
        let mut input = world.resource_mut::<Input>();
        input.hide_edges();
        change(&mut input);
    };
    keys.key("Space", true);
    accept(&mut world, &mut keys);
    smoke_eq!(
        json!(world.resource::<Input>().pending_tick()),
        json!(1),
        "a_press_is_stamped_with_its_tick"
    );
    run(&mut world, |input| input.begin_tick(1));
    smoke_eq!(shown(&world), json!([true, false]), "first_tick_sees_a_press");
    run(&mut world, |input| input.begin_tick(2));
    smoke_eq!(shown(&world), json!([false, false]), "second_tick_does_not");
    run(&mut world, Input::begin_frame);
    smoke_eq!(shown(&world), json!([true, false]), "the_frame_sees_the_press");
    run(&mut world, Input::begin_frame);
    smoke_eq!(shown(&world), json!([false, false]), "the_next_frame_does_not");
    keys.key("Space", false);
    accept(&mut world, &mut keys);
    run(&mut world, Input::begin_frame);
    smoke_eq!(
        shown(&world),
        json!([false, true]),
        "a_frame_without_ticks_sees_a_release"
    );
    run(&mut world, |input| input.begin_tick(3));
    smoke_eq!(shown(&world), json!([false, true]), "the_tick_after_still_sees_it");
    keys.key("Space", true);
    accept(&mut world, &mut keys);
    run(&mut world, Input::drop_tick_edges);
    run(&mut world, |input| input.begin_tick(4));
    smoke_eq!(
        shown(&world),
        json!([false, false]),
        "a_press_dropped_by_a_pause_never_fires"
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
        "textures": [{ "id": "alpha", "uri": ALPHA_URI }],
        "materials": [{ "id": "alpha", "color": [1, 1, 1, 1], "slots": {
            "base_color": { "module": "mapped", "maps": { "base_color": "alpha" } },
            "opacity": "mapped",
        } }],
        "meshes": [{ "id": "panel", "source": { "kind": "cube" } }],
        "entities": [{ "id": "panel", "drawn": {"mesh": "panel", "material": "alpha"}, "transform": { "translation": [2, 0, 0] } },
            { "id": "camera", "transform": { "translation": [0, 2, 4], "rotation_degrees": [-26.5651, 0, 0] }, "node": { "clock": "real", "mobility": "moving" }, "camera": { "near": 0.1, "far": 100, "projection": { "Perspective": { "vertical_fov_radians": std::f64::consts::FRAC_PI_3 } } }, "view": {} }],
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
