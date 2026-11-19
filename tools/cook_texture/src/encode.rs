use std::time::Instant;

use engine_assets::ColorSpace;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::image::Image;
use crate::ktx2::write_ktx2;
use crate::mips::Texels;
use crate::{bc5, bc7};

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Encoding {
    Rgba8Srgb,
    Rgba8,
    Bc7Srgb,
    Bc7,
    Bc5,
}

impl Encoding {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Rgba8Srgb => "rgba8_srgb",
            Self::Rgba8 => "rgba8",
            Self::Bc7Srgb => "bc7_srgb",
            Self::Bc7 => "bc7",
            Self::Bc5 => "bc5",
        }
    }

    pub const fn srgb(self) -> bool {
        matches!(self, Self::Rgba8Srgb | Self::Bc7Srgb)
    }
}

pub struct Encoded {
    pub file: Vec<u8>,
    pub gpu_bytes: usize,
    pub seconds: f64,
    pub quality: Value,
}

fn blocks(image: &Image, encode: fn(&[[u8; 4]; 16]) -> [u8; 16]) -> Vec<u8> {
    let (width, height) = (image.width as usize, image.height as usize);
    let (columns, rows) = (width.div_ceil(4), height.div_ceil(4));
    let block_row = |row: usize| -> Vec<u8> {
        let mut out = Vec::with_capacity(columns * 16);
        for column in 0..columns {
            let texels: [[u8; 4]; 16] = std::array::from_fn(|index| {
                let x = (column * 4 + index % 4).min(width - 1);
                let y = (row * 4 + index / 4).min(height - 1);
                let at = (y * width + x) * 4;
                [
                    image.pixels[at],
                    image.pixels[at + 1],
                    image.pixels[at + 2],
                    image.pixels[at + 3],
                ]
            });
            out.extend_from_slice(&encode(&texels));
        }
        out
    };
    rows_in_parallel(rows, block_row).concat()
}

fn rows_in_parallel<T: Send>(rows: usize, make: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let threads = std::thread::available_parallelism().map_or(1, usize::from);
    if threads < 2 || rows < 2 {
        return (0..rows).map(make).collect();
    }
    let per = rows.div_ceil(threads);
    let make = &make;
    std::thread::scope(|scope| {
        let parts: Vec<_> = (0..rows)
            .step_by(per)
            .map(|first| {
                scope.spawn(move || {
                    (first..(first + per).min(rows))
                        .map(make)
                        .collect::<Vec<T>>()
                })
            })
            .collect();
        parts
            .into_iter()
            .flat_map(|part| {
                part.join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
            })
            .collect()
    })
}

fn blocks_decoded(
    width: u32,
    height: u32,
    level: &[u8],
    decode: fn(&[u8; 16]) -> [[u8; 4]; 16],
) -> Vec<u8> {
    let (width, height) = (width as usize, height as usize);
    let columns = width.div_ceil(4);
    let mut pixels = vec![0_u8; width * height * 4];
    for (index, block) in level.as_chunks::<16>().0.iter().enumerate() {
        let (column, row) = (index % columns, index / columns);
        for (texel, color) in decode(block).iter().enumerate() {
            let (x, y) = (column * 4 + texel % 4, row * 4 + texel / 4);
            if x < width && y < height {
                let at = (y * width + x) * 4;
                pixels[at..at + 4].copy_from_slice(color);
            }
        }
    }
    pixels
}

fn decoded(encoding: Encoding, width: u32, height: u32, level: &[u8]) -> Vec<u8> {
    match encoding {
        Encoding::Rgba8 | Encoding::Rgba8Srgb => level.to_vec(),
        Encoding::Bc5 => blocks_decoded(width, height, level, bc5::decode_block),
        Encoding::Bc7 | Encoding::Bc7Srgb => {
            blocks_decoded(width, height, level, bc7::decode_block)
        }
    }
}

fn psnr(source: &[u8], made: &[u8], channels: usize) -> f64 {
    let mut sum = 0.0;
    let mut count = 0.0;
    for (a, b) in source
        .as_chunks::<4>()
        .0
        .iter()
        .zip(made.as_chunks::<4>().0)
    {
        for channel in 0..channels {
            let miss = f64::from(a[channel]) - f64::from(b[channel]);
            sum += miss * miss;
            count += 1.0;
        }
    }
    if sum == 0.0 {
        return 99.0;
    }
    10.0 * (255.0 * 255.0 / (sum / count)).log10()
}

fn angles(source: &[u8], made: &[u8]) -> (f64, f64) {
    let unit = |texel: &[u8]| {
        let vector = [0, 1, 2].map(|channel| f64::from(texel[channel]) / 255.0 * 2.0 - 1.0);
        let length = (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt();
        vector.map(|value| value / length)
    };
    let mut total = 0.0;
    let mut widest: f64 = 0.0;
    let mut count = 0.0;
    for (a, b) in source
        .as_chunks::<4>()
        .0
        .iter()
        .zip(made.as_chunks::<4>().0)
    {
        let (a, b) = (unit(a), unit(b));
        let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let angle = dot.clamp(-1.0, 1.0).acos().to_degrees();
        total += angle;
        if angle > widest {
            widest = angle;
        }
        count += 1.0;
    }
    (total / count, widest)
}

fn quality(texels: Texels, source: &[u8], pixels: &[u8]) -> Value {
    match texels {
        Texels::Normal => {
            let (mean, widest) = angles(source, pixels);
            json!({"normal_mean_degrees": mean, "normal_widest_degrees": widest})
        }
        Texels::Srgb => {
            json!({"psnr_rgb_db": psnr(source, pixels, 3), "psnr_rgba_db": psnr(source, pixels, 4)})
        }
        Texels::Linear => {
            let alpha_differs = source
                .as_chunks::<4>()
                .0
                .iter()
                .zip(pixels.as_chunks::<4>().0)
                .filter(|(held, read)| held[3] != read[3])
                .count();
            json!({"psnr_rgb_db": psnr(source, pixels, 3), "psnr_rgba_db": psnr(source, pixels, 4), "alpha_texels_differing": alpha_differs})
        }
    }
}

pub fn encode(
    texels: Texels,
    encoding: Encoding,
    chain: &[Image],
    exact_alpha: bool,
) -> Result<Encoded, String> {
    let started = Instant::now();
    let levels: Vec<Vec<u8>> = chain
        .iter()
        .map(|image| match encoding {
            Encoding::Rgba8 | Encoding::Rgba8Srgb => image.pixels.clone(),
            Encoding::Bc5 => blocks(image, bc5::encode_block),
            Encoding::Bc7 | Encoding::Bc7Srgb if exact_alpha => {
                blocks(image, bc7::encode_block_exact_alpha)
            }
            Encoding::Bc7 | Encoding::Bc7Srgb => blocks(image, bc7::encode_block),
        })
        .collect();
    let seconds = started.elapsed().as_secs_f64();
    let (width, height) = (chain[0].width, chain[0].height);
    let file = write_ktx2(encoding, width, height, &levels);
    let engine =
        engine_textures::decode_texture("ktx2", &file).map_err(|error| error.to_string())?;
    let space = if encoding.srgb() {
        ColorSpace::Srgb
    } else {
        ColorSpace::Linear
    };
    let same_levels = engine.levels().len() == levels.len()
        && engine
            .levels()
            .iter()
            .zip(&levels)
            .all(|(read, written)| **read == written[..]);
    if !same_levels || engine.color_space() != space {
        return Err("the engine's KTX2 reader reads the file differently".into());
    }
    let pixels = decoded(encoding, width, height, &levels[0]);
    Ok(Encoded {
        file,
        gpu_bytes: levels.iter().map(Vec::len).sum(),
        seconds,
        quality: quality(texels, &chain[0].pixels, &pixels),
    })
}
