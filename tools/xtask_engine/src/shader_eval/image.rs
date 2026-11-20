use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use serde::{Deserialize, Serialize};

pub struct Picture {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VisualDiff {
    pub passed: bool,
    pub exact: bool,
    pub max_channel_diff: u8,
    pub mean_abs_diff: f64,
    pub psnr_db: Option<f64>,
    pub changed_pixels: u64,
    pub exceeded_pixels: u64,
    pub changed_pct: f64,
    pub max_channel_limit: u8,
    pub max_mean_limit: f64,
    pub diff_picture: Option<String>,
}

pub fn read_png(path: &Path) -> Result<Picture, String> {
    let file = File::open(path).map_err(|error| format!("cannot open {}: {error}", path.display()))?;
    let mut decoder = png::Decoder::new(BufReader::new(file));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::ALPHA);
    let mut reader = decoder
        .read_info()
        .map_err(|error| format!("cannot decode {}: {error}", path.display()))?;
    let mut raw = vec![0_u8; reader.output_buffer_size()];
    let info = reader
        .next_frame(&mut raw)
        .map_err(|error| format!("cannot read frame from {}: {error}", path.display()))?;
    let bytes = &raw[..info.buffer_size()];
    let rgba = match info.color_type {
        png::ColorType::Rgba => bytes.to_vec(),
        png::ColorType::Rgb => {
            let mut expanded = Vec::with_capacity(bytes.len() / 3 * 4);
            for pixel in bytes.as_chunks::<3>().0 {
                expanded.extend_from_slice(&[pixel[0], pixel[1], pixel[2], 255]);
            }
            expanded
        }
        other => {
            return Err(format!(
                "{} has unsupported PNG color type {other:?}",
                path.display()
            ));
        }
    };
    Ok(Picture {
        width: info.width,
        height: info.height,
        rgba,
    })
}

pub fn write_png(path: &Path, picture: &Picture) -> Result<(), String> {
    let file =
        File::create(path).map_err(|error| format!("cannot create {}: {error}", path.display()))?;
    let mut encoder = png::Encoder::new(BufWriter::new(file), picture.width, picture.height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder
        .write_header()
        .map_err(|error| format!("cannot write PNG header to {}: {error}", path.display()))?;
    writer
        .write_image_data(&picture.rgba)
        .map_err(|error| format!("cannot write PNG pixels to {}: {error}", path.display()))?;
    writer
        .finish()
        .map_err(|error| format!("cannot finish PNG {}: {error}", path.display()))
}

pub fn compare(
    baseline: &Picture,
    candidate: &Picture,
    max_channel_limit: u8,
    max_mean_limit: f64,
) -> Result<(VisualDiff, Option<Picture>), String> {
    if (baseline.width, baseline.height) != (candidate.width, candidate.height) {
        return Err(format!(
            "image size changed from {}x{} to {}x{}",
            baseline.width, baseline.height, candidate.width, candidate.height
        ));
    }
    if baseline.rgba.len() != candidate.rgba.len() {
        return Err(format!(
            "image byte length changed from {} to {}",
            baseline.rgba.len(),
            candidate.rgba.len()
        ));
    }
    let pixels = u64::from(baseline.width) * u64::from(baseline.height);
    let channels = (pixels * 4).max(1);
    let mut max_channel_diff = 0_u8;
    let mut sum_abs = 0_u64;
    let mut sum_sq = 0_u64;
    let mut changed_pixels = 0_u64;
    let mut exceeded_pixels = 0_u64;
    let mut diff_rgba = Vec::with_capacity(baseline.rgba.len());
    for (before, after) in baseline
        .rgba
        .as_chunks::<4>()
        .0
        .iter()
        .zip(candidate.rgba.as_chunks::<4>().0)
    {
        let mut pixel_max = 0_u8;
        for channel in 0..4 {
            let delta = before[channel].abs_diff(after[channel]);
            if delta > pixel_max {
                pixel_max = delta;
            }
            if delta > max_channel_diff {
                max_channel_diff = delta;
            }
            sum_abs += u64::from(delta);
            sum_sq += u64::from(delta) * u64::from(delta);
        }
        if pixel_max > 0 {
            changed_pixels += 1;
        }
        if pixel_max > max_channel_limit {
            exceeded_pixels += 1;
        }
        for channel in 0..3 {
            let delta = before[channel].abs_diff(after[channel]);
            diff_rgba.push(delta.saturating_mul(16));
        }
        diff_rgba.push(255);
    }
    let mean_abs_diff = sum_abs as f64 / channels as f64;
    let mse = sum_sq as f64 / channels as f64;
    let psnr_db = if mse == 0.0 {
        None
    } else {
        Some(10.0 * (255.0_f64 * 255.0_f64 / mse).log10())
    };
    let changed_pct = if pixels == 0 {
        0.0
    } else {
        100.0 * changed_pixels as f64 / pixels as f64
    };
    let passed = max_channel_diff <= max_channel_limit && mean_abs_diff <= max_mean_limit;
    let diff = VisualDiff {
        passed,
        exact: changed_pixels == 0,
        max_channel_diff,
        mean_abs_diff,
        psnr_db,
        changed_pixels,
        exceeded_pixels,
        changed_pct,
        max_channel_limit,
        max_mean_limit,
        diff_picture: None,
    };
    let diff_image = (changed_pixels > 0).then_some(Picture {
        width: baseline.width,
        height: baseline.height,
        rgba: diff_rgba,
    });
    Ok((diff, diff_image))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(width: u32, height: u32, rgba: [u8; 4]) -> Picture {
        let mut pixels = Vec::with_capacity((width * height * 4) as usize);
        for _ in 0..(width * height) {
            pixels.extend_from_slice(&rgba);
        }
        Picture {
            width,
            height,
            rgba: pixels,
        }
    }

    #[test]
    fn exact_match_passes_without_diff_image() -> Result<(), String> {
        let a = solid(4, 4, [10, 20, 30, 255]);
        let b = solid(4, 4, [10, 20, 30, 255]);
        let (diff, image) = compare(&a, &b, 1, 0.05)?;
        assert!(diff.passed, "identical images should pass");
        assert!(diff.exact, "identical images should be exact");
        assert_eq!(diff.max_channel_diff, 0, "max diff should be zero");
        assert_eq!(diff.changed_pixels, 0, "no pixels should change");
        assert!(diff.psnr_db.is_none(), "exact match has no finite PSNR");
        assert!(image.is_none(), "exact match should emit no diff image");
        Ok(())
    }

    #[test]
    fn one_lsb_within_limits_passes() -> Result<(), String> {
        let a = solid(4, 4, [100, 100, 100, 255]);
        let mut b = solid(4, 4, [100, 100, 100, 255]);
        b.rgba[0] = 101;
        let (diff, image) = compare(&a, &b, 1, 0.05)?;
        assert!(diff.passed, "single 1-LSB pixel should pass default limits");
        assert!(!diff.exact, "changed pixel should not be marked exact");
        assert_eq!(diff.max_channel_diff, 1, "max channel diff should be 1");
        assert_eq!(diff.changed_pixels, 1, "one pixel changed");
        assert_eq!(diff.exceeded_pixels, 0, "no pixel exceeded channel limit");
        assert!(image.is_some(), "changed image should produce a diff picture");
        Ok(())
    }

    #[test]
    fn channel_excess_fails_and_roundtrips_diff_png() -> Result<(), String> {
        let a = solid(2, 2, [50, 60, 70, 255]);
        let mut b = solid(2, 2, [50, 60, 70, 255]);
        b.rgba[1] = 55;
        let (diff, image) = compare(&a, &b, 1, 0.5)?;
        assert!(!diff.passed, "5-LSB delta should fail channel limit 1");
        assert_eq!(diff.max_channel_diff, 5, "max diff should be 5");
        assert_eq!(diff.exceeded_pixels, 1, "one pixel exceeded channel limit");
        let diff_picture = image.ok_or_else(|| "expected diff picture".to_owned())?;
        let path = std::env::temp_dir().join(format!(
            "xtask-shader-eval-test-{}.png",
            std::process::id()
        ));
        write_png(&path, &diff_picture)?;
        let decoded = read_png(&path)?;
        let _ = std::fs::remove_file(&path);
        assert_eq!(
            (decoded.width, decoded.height),
            (2, 2),
            "roundtrip PNG dimensions should match"
        );
        assert_eq!(decoded.rgba[1], 80, "amplified green channel should be 5 * 16");
        Ok(())
    }

    #[test]
    fn mismatched_dimensions_error() {
        let a = solid(2, 2, [0, 0, 0, 255]);
        let b = solid(4, 2, [0, 0, 0, 255]);
        let outcome = compare(&a, &b, 1, 0.05);
        assert!(outcome.is_err(), "mismatched sizes must return an error");
    }
}
