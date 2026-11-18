use engine_textures::Image;
use gltf::texture::WrappingMode;

use crate::maps::MapKind;

const TAPS: [f32; 4] = [0.125, 0.375, 0.375, 0.125];

fn address(at: isize, size: usize, wrap: WrappingMode) -> usize {
    let size = size as isize;
    let held = match wrap {
        WrappingMode::Repeat => at.rem_euclid(size),
        WrappingMode::ClampToEdge => at.clamp(0, size - 1),
        WrappingMode::MirroredRepeat => {
            let period = at.rem_euclid(2 * size);
            if period < size {
                period
            } else {
                2 * size - 1 - period
            }
        }
    };
    held as usize
}

pub fn srgb_to_linear(value: u8) -> f32 {
    let value = f32::from(value) / 255.0;
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

pub fn linear_to_srgb(value: f32) -> u8 {
    let encoded = if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    };
    byte(encoded)
}

pub fn byte(value: f32) -> u8 {
    let scaled = (value * 255.0).round();
    if scaled <= 0.0 {
        0
    } else if scaled >= 255.0 {
        255
    } else {
        scaled as u8
    }
}

fn unpacked(kind: MapKind, linear: &[f32; 256], texel: &[u8]) -> [f32; 4] {
    match kind {
        MapKind::BaseColor => [
            linear[usize::from(texel[0])],
            linear[usize::from(texel[1])],
            linear[usize::from(texel[2])],
            f32::from(texel[3]) / 255.0,
        ],
        MapKind::Normal => [
            f32::from(texel[0]) / 255.0 * 2.0 - 1.0,
            f32::from(texel[1]) / 255.0 * 2.0 - 1.0,
            f32::from(texel[2]) / 255.0 * 2.0 - 1.0,
            f32::from(texel[3]) / 255.0,
        ],
        MapKind::Orm => texel
            .iter()
            .map(|&value| f32::from(value) / 255.0)
            .collect::<Vec<f32>>()
            .try_into()
            .expect("four_channels"),
    }
}

fn packed(kind: MapKind, value: [f32; 4]) -> [u8; 4] {
    match kind {
        MapKind::BaseColor => [
            linear_to_srgb(value[0]),
            linear_to_srgb(value[1]),
            linear_to_srgb(value[2]),
            byte(value[3]),
        ],
        MapKind::Normal => {
            let length = (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt();
            let unit = if length > 0.0 && length.is_finite() {
                [value[0] / length, value[1] / length, value[2] / length]
            } else {
                [0.0, 0.0, 1.0]
            };
            [
                byte(unit[0] * 0.5 + 0.5),
                byte(unit[1] * 0.5 + 0.5),
                byte(unit[2] * 0.5 + 0.5),
                byte(value[3]),
            ]
        }
        MapKind::Orm => value.map(byte),
    }
}

fn halved(kind: MapKind, image: &Image, wrap: [WrappingMode; 2], linear: &[f32; 256]) -> Image {
    let (width, height) = (image.width as usize, image.height as usize);
    let (next_width, next_height) = ((width / 2).max(1), (height / 2).max(1));
    let values: Vec<[f32; 4]> = image
        .pixels
        .as_chunks::<4>()
        .0
        .iter()
        .map(|texel| unpacked(kind, linear, texel))
        .collect();
    let (step_x, step_y) = (width / next_width, height / next_height);
    let mut pixels = Vec::with_capacity(next_width * next_height * 4);
    for y in 0..next_height {
        for x in 0..next_width {
            let mut sum = [0.0f32; 4];
            let mut total = 0.0;
            for (row_tap, row_weight) in
                TAPS.iter()
                    .enumerate()
                    .take(if step_y == 1 { 1 } else { 4 })
            {
                let row_weight = if step_y == 1 { 1.0 } else { *row_weight };
                let row_at = if step_y == 1 {
                    y
                } else {
                    address((2 * y + row_tap) as isize - 1, height, wrap[1])
                };
                for (column_tap, column_weight) in
                    TAPS.iter()
                        .enumerate()
                        .take(if step_x == 1 { 1 } else { 4 })
                {
                    let column_weight = if step_x == 1 { 1.0 } else { *column_weight };
                    let column_at = if step_x == 1 {
                        x
                    } else {
                        address((2 * x + column_tap) as isize - 1, width, wrap[0])
                    };
                    let weight = row_weight * column_weight;
                    let value = values[row_at * width + column_at];
                    for (held, channel) in sum.iter_mut().zip(value) {
                        *held += channel * weight;
                    }
                    total += weight;
                }
            }
            pixels.extend_from_slice(&packed(kind, sum.map(|value| value / total)));
        }
    }
    Image {
        width: next_width as u32,
        height: next_height as u32,
        pixels,
    }
}

pub fn chain(kind: MapKind, top: &Image, wrap: [WrappingMode; 2]) -> Vec<Image> {
    let linear: [f32; 256] = std::array::from_fn(|value| srgb_to_linear(value as u8));
    let mut levels = vec![top.clone()];
    while let Some(last) = levels.last() {
        if last.width == 1 && last.height == 1 {
            break;
        }
        let next = halved(kind, last, wrap, &linear);
        levels.push(next);
    }
    levels
}
