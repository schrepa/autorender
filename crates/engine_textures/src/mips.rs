use std::sync::OnceLock;

use engine_assets::srgb_decoded;

use crate::Usage;

fn srgb_thresholds() -> &'static [f32; 255] {
    static TABLE: OnceLock<[f32; 255]> = OnceLock::new();
    TABLE.get_or_init(|| {
        std::array::from_fn(|index| {
            let byte = index as u32 + 1;
            let (mut low, mut high) = (0_u32, 1.0_f32.to_bits());
            while low < high {
                let middle = low + (high - low) / 2;
                if u32::from(encode_exactly(f32::from_bits(middle))) >= byte {
                    high = middle;
                } else {
                    low = middle + 1;
                }
            }
            f32::from_bits(low)
        })
    })
}

fn encode_exactly(value: f32) -> u8 {
    let value = engine_math::clamp(value, 0.0, 1.0);
    let encoded = if value <= 0.003_130_8 {
        12.92 * value
    } else {
        1.055 * libm::powf(value, 1.0 / 2.4) - 0.055
    };
    (encoded * 255.0 + 0.5) as u8
}

pub(crate) fn srgb_byte(value: f32) -> u8 {
    srgb_thresholds().partition_point(|&threshold| threshold <= value) as u8
}

fn unorm_byte(value: f32) -> u8 {
    (engine_math::clamp(value, 0.0, 1.0) * 255.0 + 0.5) as u8
}

pub(crate) fn chain(width: u32, height: u32, base: &[u8], usage: Usage) -> Vec<Vec<u8>> {
    let mut levels = vec![base.to_vec()];
    let (mut width, mut height) = (width as usize, height as usize);
    while width > 1 || height > 1 {
        let (next_width, next_height) = ((width / 2).max(1), (height / 2).max(1));
        let above = levels.last().map(Vec::as_slice).unwrap_or_default();
        let mut next = Vec::with_capacity(next_width * next_height * 4);
        for y in 0..next_height {
            for x in 0..next_width {
                let mut texels = [[0_u8; 4]; 4];
                let mut count = 0;
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let (sx, sy) = (2 * x + dx, 2 * y + dy);
                    if sx < width && sy < height {
                        let at = (sy * width + sx) * 4;
                        texels[count].copy_from_slice(&above[at..at + 4]);
                        count += 1;
                    }
                }
                next.extend_from_slice(&average(&texels[..count], usage));
            }
        }
        levels.push(next);
        (width, height) = (next_width, next_height);
    }
    levels
}

fn average(texels: &[[u8; 4]], usage: Usage) -> [u8; 4] {
    let count = texels.len() as u32;
    let alpha = (texels.iter().map(|texel| u32::from(texel[3])).sum::<u32>() + count / 2) / count;
    match usage {
        Usage::Color => {
            let decode = srgb_decoded();
            let mut out = [0, 0, 0, alpha as u8];
            for (channel, value) in out.iter_mut().take(3).enumerate() {
                let sum: f32 = texels
                    .iter()
                    .map(|texel| decode[usize::from(texel[channel])])
                    .sum();
                *value = srgb_byte(sum / count as f32);
            }
            out
        }
        Usage::Data => {
            let mut out = [0_u8; 4];
            for (channel, value) in out.iter_mut().enumerate() {
                let sum: u32 = texels.iter().map(|texel| u32::from(texel[channel])).sum();
                *value = ((sum + count / 2) / count) as u8;
            }
            out
        }
        Usage::Normal => {
            let mut normal = [0.0_f32; 3];
            for texel in texels {
                for (sum, &value) in normal.iter_mut().zip(texel) {
                    *sum += f32::from(value) / 127.5 - 1.0;
                }
            }
            let normal = normal.map(|sum| sum / count as f32);
            [
                unorm_byte(normal[0] * 0.5 + 0.5),
                unorm_byte(normal[1] * 0.5 + 0.5),
                unorm_byte(normal[2] * 0.5 + 0.5),
                alpha as u8,
            ]
        }
    }
}
