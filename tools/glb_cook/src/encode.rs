use std::time::Instant;

use engine_assets::{ColorSpace, Texture, TextureFormat};
use engine_textures::Image;
use serde_json::{Value, json};

use crate::config::Encoding;
use crate::maps::MapKind;
use crate::{bc5, bc7, ktx2};

pub struct Encoded {
    pub file: Vec<u8>,
    pub level_zero: Image,
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

fn bc5_decoded(width: u32, height: u32, level: &[u8]) -> Vec<u8> {
    let (width, height) = (width as usize, height as usize);
    let columns = width.div_ceil(4);
    let mut pixels = vec![0_u8; width * height * 4];
    for (index, block) in level.as_chunks::<16>().0.iter().enumerate() {
        let (column, row) = (index % columns, index / columns);
        for (texel, color) in bc5::decode_block(block).iter().enumerate() {
            let (x, y) = (column * 4 + texel % 4, row * 4 + texel / 4);
            if x < width && y < height {
                let at = (y * width + x) * 4;
                pixels[at..at + 4].copy_from_slice(color);
            }
        }
    }
    pixels
}

fn decoded(encoding: Encoding, width: u32, height: u32, level: &[u8]) -> Result<Vec<u8>, String> {
    match encoding {
        Encoding::Rgba8 | Encoding::Rgba8Srgb => Ok(level.to_vec()),
        Encoding::Bc5 => Ok(bc5_decoded(width, height, level)),
        Encoding::Bc7 | Encoding::Bc7Srgb => {
            let texture =
                Texture::from_levels(TextureFormat::Bc7, width, height, vec![level.to_vec()])
                    .map_err(|error| error.to_string())?;
            let plain = engine_textures::decompress(&texture).map_err(|error| error.to_string())?;
            Ok(plain.levels()[0].to_vec())
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

pub fn encode(kind: MapKind, encoding: Encoding, chain: &[Image]) -> Result<Encoded, String> {
    let started = Instant::now();
    let levels: Vec<Vec<u8>> = chain
        .iter()
        .map(|image| match encoding {
            Encoding::Rgba8 | Encoding::Rgba8Srgb => image.pixels.clone(),
            Encoding::Bc5 => blocks(image, bc5::encode_block),
            Encoding::Bc7 | Encoding::Bc7Srgb => blocks(image, bc7::encode_block),
        })
        .collect();
    let seconds = started.elapsed().as_secs_f64();
    let (width, height) = (chain[0].width, chain[0].height);
    let file = ktx2::write(encoding, width, height, &levels);
    let read = ktx2::read(&file)?;
    let expected: Vec<usize> = chain
        .iter()
        .map(|image| ktx2::level_bytes(encoding, image.width, image.height))
        .collect();
    let lengths: Vec<usize> = read.levels.iter().map(Vec::len).collect();
    if read.vk_format != ktx2::layout(encoding).vk_format
        || (read.width, read.height) != (width, height)
        || lengths != expected
        || read.levels != levels
    {
        return Err("the written KTX2 file reads back differently".into());
    }
    let pixels = decoded(encoding, width, height, &read.levels[0])?;
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
    let source = &chain[0].pixels;
    let quality = match kind {
        MapKind::Normal => {
            let (mean, widest) = angles(source, &pixels);
            json!({"normal_mean_degrees": mean, "normal_widest_degrees": widest})
        }
        MapKind::BaseColor | MapKind::Orm => {
            json!({"psnr_rgb_db": psnr(source, &pixels, 3), "psnr_rgba_db": psnr(source, &pixels, 4)})
        }
    };
    Ok(Encoded {
        file,
        level_zero: Image {
            width,
            height,
            pixels,
        },
        gpu_bytes: expected.iter().sum(),
        seconds,
        quality,
    })
}
