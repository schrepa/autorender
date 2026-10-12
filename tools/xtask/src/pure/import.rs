//! The import corpus: `terrain_import` over the same heights in its three
//! heightmap formats: the terrain fixture's 65 by 65 dome
//! (web/scripts/fixtures/terrain-smoke/dome.r16) as that little-endian
//! 16-bit RAW file, as a 16-bit grayscale PNG, and as a projected GeoTIFF
//! of point samples in meters (32-bit floats, the heights the RAW and PNG
//! samples map to over 0 to 200 m), each 128 m across. The three must write
//! byte-identical `height.r16`, which is held to the goldens with the
//! import's numbers.

use std::io::Cursor;
use std::path::Path;
use std::process::Command;

use serde_json::{Value, json};
use tiff::encoder::{TiffEncoder, colortype};
use tiff::tags::Tag;

use super::Observations;

const DOME: &str = "web/scripts/fixtures/terrain-smoke/dome.r16";
/// The folder the inputs and imports go to.
const FOLDER: &str = "target/pure/import";
const SIDE: u32 = 65;
/// Meters across the dome, and the heights its samples 0 and 65535 stand
/// for.
const EXTENT: f64 = 128.0;
const RANGE: [f64; 2] = [0.0, 200.0];

pub fn run(root: &Path, observed: &mut Observations) -> Result<(), String> {
    let raw =
        std::fs::read(root.join(DOME)).map_err(|error| format!("cannot read {DOME}: {error}"))?;
    let samples: Vec<u16> = raw
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes(*pair))
        .collect();
    if samples.len() != (SIDE * SIDE) as usize {
        return Err(format!(
            "{DOME} holds {} samples, not {SIDE} by {SIDE}",
            samples.len()
        ));
    }
    let folder = root.join(FOLDER);
    if folder.exists() {
        std::fs::remove_dir_all(&folder)
            .map_err(|error| format!("cannot clear {FOLDER}: {error}"))?;
    }
    std::fs::create_dir_all(&folder).map_err(|error| format!("cannot create {FOLDER}: {error}"))?;
    let write = |name: &str, bytes: &[u8]| {
        std::fs::write(folder.join(name), bytes)
            .map_err(|error| format!("cannot write {FOLDER}/{name}: {error}"))
    };
    write("dome.r16", &raw)?;
    write("dome.png", &png(&samples)?)?;
    write("dome.tif", &geotiff(&samples)?)?;
    let extent = format!("{EXTENT},{EXTENT}");
    let range = format!("{},{}", RANGE[0], RANGE[1]);
    let mut heights = Vec::new();
    for (id, input, georeferenced) in [
        ("raw", "dome.r16", false),
        ("png", "dome.png", false),
        ("geotiff", "dome.tif", true),
    ] {
        let mut arguments = vec![
            "run",
            "-q",
            "--locked",
            "-p",
            "engine_assetc",
            "--bin",
            "terrain_import",
            "--",
        ];
        let out = folder.display().to_string();
        let path = folder.join(input).display().to_string();
        arguments.extend(["--out", &out, "--id", id, "--size", "1024", "--detail", "0"]);
        if !georeferenced {
            arguments.extend(["--extent", &extent, "--range", &range]);
        }
        arguments.push(&path);
        let output = Command::new("cargo")
            .args(&arguments)
            .current_dir(root)
            .output()
            .map_err(|error| format!("cannot run cargo: {error}"))?;
        if !output.status.success() {
            return Err(format!(
                "terrain_import of {input} failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        let height = std::fs::read(folder.join(id).join("height.r16"))
            .map_err(|error| format!("terrain_import of {input} wrote no height.r16: {error}"))?;
        let report: Value = std::fs::read_to_string(folder.join(id).join("import.json"))
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .ok_or_else(|| format!("terrain_import of {input} wrote no report"))?;
        heights.push((id, height, report));
    }
    let (_, first, report) = &heights[0];
    for (id, height, _) in &heights[1..] {
        observed.check(height == first, || {
            format!("the {id} import's height.r16 differs from the RAW import's")
        });
    }
    observed.hash("import height.r16", first);
    observed.record(
        "import",
        json!({
            "samples": report["samples"],
            "lowest": report["lowest"],
            "highest": report["highest"],
            "filled gaps": report["filled_gaps"],
        }),
    );
    Ok(())
}

/// `samples` as a 16-bit grayscale PNG.
fn png(samples: &[u16]) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    let mut encoder = png::Encoder::new(&mut bytes, SIDE, SIDE);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Sixteen);
    let mut writer = encoder.write_header().map_err(|error| error.to_string())?;
    let data: Vec<u8> = samples
        .iter()
        .flat_map(|sample| sample.to_be_bytes())
        .collect();
    writer
        .write_image_data(&data)
        .map_err(|error| error.to_string())?;
    writer.finish().map_err(|error| error.to_string())?;
    Ok(bytes)
}

/// `samples` as a projected GeoTIFF of point samples: each one's height in
/// meters as the RAW and PNG imports read it, a 32-bit float, two meters
/// apart, the first at model (0, 128).
fn geotiff(samples: &[u16]) -> Result<Vec<u8>, String> {
    let heights: Vec<f32> = samples
        .iter()
        .map(|&value| (RANGE[0] + (RANGE[1] - RANGE[0]) * f64::from(value) / 65_535.0) as f32)
        .collect();
    let fail = |error: tiff::TiffError| error.to_string();
    let mut bytes = Cursor::new(Vec::new());
    let mut encoder = TiffEncoder::new(&mut bytes).map_err(fail)?;
    let mut image = encoder
        .new_image::<colortype::Gray32Float>(SIDE, SIDE)
        .map_err(fail)?;
    let spacing = EXTENT / f64::from(SIDE - 1);
    image
        .encoder()
        .write_tag(Tag::ModelPixelScaleTag, &[spacing, spacing, 0.0][..])
        .map_err(fail)?;
    image
        .encoder()
        .write_tag(
            Tag::ModelTiepointTag,
            &[0.0, 0.0, 0.0, 0.0, EXTENT, 0.0][..],
        )
        .map_err(fail)?;
    // GeoKeys: version 1.1.0, two keys: a projected model (1024 = 1) of
    // point samples (1025 = 2).
    image
        .encoder()
        .write_tag(
            Tag::GeoKeyDirectoryTag,
            &[1u16, 1, 0, 2, 1024, 0, 1, 1, 1025, 0, 1, 2][..],
        )
        .map_err(fail)?;
    image.write_data(&heights).map_err(fail)?;
    Ok(bytes.into_inner())
}
