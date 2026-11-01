use engine_core::{Runs, Smoke, SmokeFailure, SmokeSession, smoke_eq};
use serde_json::json;

use crate::host::{InputGatherer, SceneLoad};

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

fn input_keys(_: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    let built = InputGatherer::new(&[("Space", 0)]);
    smoke_eq!(built.is_ok(), true, "gatherer_builds");
    let Ok(mut keys) = built else {
        return Ok(());
    };
    smoke_eq!(keys.key("KeyW", true), true, "movement_key_used");
    let held = [keys.take(), keys.take()].map(|input| input.state.move_forward);
    smoke_eq!(json!(held), json!([1.0, 1.0]), "key_moves");
    keys.key("KeyW", false);
    smoke_eq!(
        json!(keys.take().state.move_forward),
        json!(0.0),
        "release_stops"
    );
    keys.look(120.0, 40.0);
    let turned = keys.take().state;
    smoke_eq!(
        json!([turned.look_yaw, turned.look_pitch]),
        json!([120.0, 40.0]),
        "drag_turns"
    );
    let after = keys.take().state;
    smoke_eq!(
        json!([after.look_yaw, after.look_pitch]),
        json!([0.0, 0.0]),
        "turns_apply_once"
    );
    smoke_eq!(
        json!([keys.key("KeyW", true), keys.key("KeyF", true)]),
        json!([true, false]),
        "unused_keys_stay"
    );
    keys.key("KeyW", false);
    keys.key("Space", true);
    let pressed = keys.take();
    smoke_eq!(
        json!([pressed.held & 1, pressed.pressed & 1]),
        json!([1, 1]),
        "actions_press"
    );
    keys.key("KeyW", true);
    keys.clear();
    let blurred = keys.take();
    smoke_eq!(
        json!([blurred.state.move_forward, blurred.held]),
        json!([0.0, 0]),
        "blur_releases"
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
