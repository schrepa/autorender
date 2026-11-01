use engine_core::{SmokeFailure, smoke_eq};
use engine_standard_smokes::requests::seen;
use serde_json::{Value, json};

const PNG: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];

pub(crate) struct Pixels {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) rgba: Vec<u8>,
}

#[cfg(not(target_arch = "wasm32"))]
fn saved(path: &str) -> Option<Vec<u8>> {
    Some(std::fs::read(path).unwrap_or_default())
}

#[cfg(target_arch = "wasm32")]
fn saved(_: &str) -> Option<Vec<u8>> {
    None
}

pub(crate) fn pixels(reply: &Value) -> Result<Option<Pixels>, SmokeFailure> {
    let artifact = &reply["result"]["artifact"];
    let path = artifact["path"].as_str().unwrap_or_default();
    let Some(bytes) = saved(path) else {
        return Ok(None);
    };
    let wanted = json!({ "mime_type": "image/png" });
    smoke_eq!(
        seen(bytes.starts_with(&PNG), artifact.clone(), &wanted),
        wanted,
        "capture_is_png"
    );
    smoke_eq!(
        json!(bytes.len()),
        artifact["bytes"].clone(),
        "capture_bytes_match"
    );
    let image = engine::host::decode_image(path, &bytes);
    smoke_eq!(
        json!(image.as_ref().err()),
        json!(null),
        "capture_decodes"
    );
    Ok(image.ok().map(|image| Pixels {
        width: image.width,
        height: image.height,
        rgba: image.pixels,
    }))
}

pub(crate) fn differing(a: &Pixels, b: &Pixels, tolerance: u8) -> Result<usize, SmokeFailure> {
    smoke_eq!(
        json!([b.width, b.height]),
        json!([a.width, a.height]),
        "capture_size_kept"
    );
    Ok(a.rgba
        .as_chunks::<4>()
        .0
        .iter()
        .zip(b.rgba.as_chunks::<4>().0)
        .filter(|(a, b)| (0..3).any(|channel| a[channel].abs_diff(b[channel]) > tolerance))
        .count())
}
