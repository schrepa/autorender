//! A small BC7 encoder for the metro's generated textures:
//! every 4 × 4 block in mode 6, one RGBA line of 16 colors between two
//! endpoints with per-endpoint parity bits. Stylized textures of flat
//! regions and gentle gradients keep their look; hosts transcoding real
//! assets use the Basis Universal transcoder instead.

use std::collections::HashMap;

use engine::prelude::*;

/// Mode 6 weights of its 16 palette entries, out of 64.
const WEIGHTS: [u32; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

/// The first palette entry whose weight reaches each integer from 0 to 64:
/// for a weight `w`, the entry at `ceil(w)`, as weights are integers.
const FIRST_REACHING: [usize; 65] = {
    let mut table = [0; 65];
    let mut value = 0;
    while value < 65 {
        let mut entry = 0;
        while WEIGHTS[entry] < value as u32 {
            entry += 1;
        }
        table[value] = entry;
        value += 1;
    }
    table
};

/// BC7 blocks of an RGBA8 level whose dimensions are multiples of 4. Blocks
/// of one color encode once per level.
pub(crate) fn encode(width: u32, height: u32, pixels: &[u8]) -> Vec<u8> {
    debug_assert!(width.is_multiple_of(4) && height.is_multiple_of(4));
    let mut blocks = Vec::with_capacity((width * height) as usize);
    let mut uniform: HashMap<[u8; 4], [u8; 16]> = HashMap::new();
    for by in 0..height / 4 {
        for bx in 0..width / 4 {
            let texel_at = |index: u32| {
                let (x, y) = (bx * 4 + index % 4, by * 4 + index / 4);
                let at = ((y * width + x) * 4) as usize;
                [pixels[at], pixels[at + 1], pixels[at + 2], pixels[at + 3]]
            };
            let first = texel_at(0);
            if (1..16).all(|index| texel_at(index) == first) {
                let encoded = uniform
                    .entry(first)
                    .or_insert_with(|| block(&[first.map(f32::from); 16]));
                blocks.extend_from_slice(encoded);
                continue;
            }
            let texels: [[f32; 4]; 16] =
                std::array::from_fn(|index| texel_at(index as u32).map(f32::from));
            blocks.extend_from_slice(&block(&texels));
        }
    }
    blocks
}

/// Endpoint `color` in 7 bits per channel plus a shared parity bit, the
/// parity chosen to fit best.
fn quantize(color: [f32; 4]) -> ([u32; 4], u32) {
    let mut best = ([0; 4], 0, f32::INFINITY);
    for parity in 0..2 {
        let mut error = 0.0;
        let mut values = [0; 4];
        for channel in 0..4 {
            let value = math::clamp(((color[channel] - parity as f32) / 2.0).round(), 0.0, 127.0);
            let decoded = value * 2.0 + parity as f32;
            error += (decoded - color[channel]) * (decoded - color[channel]);
            values[channel] = value as u32;
        }
        if error < best.2 {
            best = (values, parity, error);
        }
    }
    (best.0, best.1)
}

fn expand(values: [u32; 4], parity: u32) -> [f32; 4] {
    values.map(|value| ((value << 1) | parity) as f32)
}

/// The 16 palette colors between two endpoints, as a decoder computes them.
fn palette(a: [f32; 4], b: [f32; 4]) -> [[f32; 4]; 16] {
    WEIGHTS.map(|weight| {
        [0, 1, 2, 3].map(|channel| {
            (((64 - weight) * a[channel] as u32 + weight * b[channel] as u32 + 32) >> 6) as f32
        })
    })
}

fn distance(a: [f32; 4], b: [f32; 4]) -> f32 {
    (0..4)
        .map(|channel| (a[channel] - b[channel]).powi(2))
        .sum()
}

/// Nearest palette entries and the total squared error: each texel's
/// projection on the endpoint line picks an entry, and its neighbors are
/// checked against the palette's rounding.
fn fit(texels: &[[f32; 4]; 16], colors: &[[f32; 4]; 16]) -> ([u32; 16], f32) {
    let (first, last) = (colors[0], colors[15]);
    let axis = [0, 1, 2, 3].map(|channel| last[channel] - first[channel]);
    let length = axis.iter().map(|value| value * value).sum::<f32>();
    let mut indices = [0; 16];
    let mut total = 0.0;
    for (texel, index) in texels.iter().zip(&mut indices) {
        let guess = if length > 0.0 {
            let along: f32 = (0..4)
                .map(|channel| (texel[channel] - first[channel]) * axis[channel])
                .sum();
            let weight = math::clamp(along / length, 0.0, 1.0) * 64.0;
            // The first weight reaching it; NaN reaches none.
            if weight >= 0.0 {
                FIRST_REACHING[weight.ceil() as usize]
            } else {
                15
            }
        } else {
            0
        };
        let (mut best, mut error) = (0, f32::INFINITY);
        let low = guess.saturating_sub(1);
        for (offset, color) in colors[low..(guess + 2).min(16)].iter().enumerate() {
            let candidate_error = distance(*texel, *color);
            if candidate_error < error {
                (best, error) = ((low + offset) as u32, candidate_error);
            }
        }
        *index = best;
        total += error;
    }
    (indices, total)
}

/// Endpoints along the block's principal axis through its mean.
fn endpoints(texels: &[[f32; 4]; 16]) -> ([f32; 4], [f32; 4]) {
    let mut mean = [0.0f32; 4];
    for texel in texels {
        for channel in 0..4 {
            mean[channel] += texel[channel] / 16.0;
        }
    }
    let mut covariance = [[0.0f32; 4]; 4];
    for texel in texels {
        let d = [0, 1, 2, 3].map(|channel| texel[channel] - mean[channel]);
        for i in 0..4 {
            for j in 0..4 {
                covariance[i][j] += d[i] * d[j];
            }
        }
    }
    // Power iteration from the widest channel.
    let mut axis = [1.0f32, 1.0, 1.0, 1.0];
    for _ in 0..8 {
        let next = [0, 1, 2, 3].map(|i| (0..4).map(|j| covariance[i][j] * axis[j]).sum::<f32>());
        let length = next.iter().map(|value| value * value).sum::<f32>().sqrt();
        if length < 1.0e-6 {
            return (mean, mean);
        }
        axis = next.map(|value| value / length);
    }
    let (mut low, mut high) = (f32::INFINITY, f32::NEG_INFINITY);
    for texel in texels {
        let along: f32 = (0..4)
            .map(|channel| (texel[channel] - mean[channel]) * axis[channel])
            .sum();
        low = math::min(low, along);
        high = math::max(high, along);
    }
    let point = |t: f32| {
        [0, 1, 2, 3].map(|channel| math::clamp(mean[channel] + axis[channel] * t, 0.0, 255.0))
    };
    (point(low), point(high))
}

/// Least-squares endpoints for fixed palette indices.
fn refine(texels: &[[f32; 4]; 16], indices: &[u32; 16]) -> Option<([f32; 4], [f32; 4])> {
    let (mut aa, mut ab, mut bb) = (0.0f32, 0.0f32, 0.0f32);
    let (mut ax, mut bx) = ([0.0f32; 4], [0.0f32; 4]);
    for (texel, index) in texels.iter().zip(indices) {
        let weight = WEIGHTS[*index as usize] as f32 / 64.0;
        let (alpha, beta) = (1.0 - weight, weight);
        aa += alpha * alpha;
        ab += alpha * beta;
        bb += beta * beta;
        for channel in 0..4 {
            ax[channel] += alpha * texel[channel];
            bx[channel] += beta * texel[channel];
        }
    }
    let determinant = aa * bb - ab * ab;
    if determinant.abs() < 1.0e-6 {
        return None;
    }
    let solve = |x: [f32; 4], y: [f32; 4], first: bool| {
        [0, 1, 2, 3].map(|channel| {
            let value = if first {
                (bb * x[channel] - ab * y[channel]) / determinant
            } else {
                (aa * y[channel] - ab * x[channel]) / determinant
            };
            math::clamp(value, 0.0, 255.0)
        })
    };
    Some((solve(ax, bx, true), solve(ax, bx, false)))
}

/// A quantized endpoint: 7 bits per channel and its parity bit.
type Endpoint = ([u32; 4], u32);

/// One 16-byte mode 6 block.
fn block(texels: &[[f32; 4]; 16]) -> [u8; 16] {
    let (first, second) = endpoints(texels);
    let mut best: Option<(Endpoint, Endpoint, [u32; 16], f32)> = None;
    let mut candidate = (first, second);
    for _ in 0..2 {
        let (a, b) = (quantize(candidate.0), quantize(candidate.1));
        let colors = palette(expand(a.0, a.1), expand(b.0, b.1));
        let (indices, error) = fit(texels, &colors);
        if best.as_ref().is_none_or(|best| error < best.3) {
            best = Some((a, b, indices, error));
        }
        // Nothing fits better than exactly.
        if error == 0.0 {
            break;
        }
        match refine(texels, &indices) {
            Some(refined) => candidate = refined,
            None => break,
        }
    }
    let (mut a, mut b, mut indices, _) = best.expect("one candidate was fitted");
    // The first texel's index drops its top bit: keep it below 8.
    if indices[0] >= 8 {
        std::mem::swap(&mut a, &mut b);
        for index in &mut indices {
            *index = 15 - *index;
        }
    }
    let mut bits = Bits::default();
    bits.push(1 << 6, 7);
    for channel in 0..4 {
        bits.push(a.0[channel], 7);
        bits.push(b.0[channel], 7);
    }
    bits.push(a.1, 1);
    bits.push(b.1, 1);
    bits.push(indices[0], 3);
    for index in &indices[1..] {
        bits.push(*index, 4);
    }
    bits.value.to_le_bytes()
}

/// A block's bits, first pushed in the least significant.
#[derive(Default)]
struct Bits {
    value: u128,
    at: u32,
}

impl Bits {
    /// Appends the `count` (below 32) low bits of `value`.
    fn push(&mut self, value: u32, count: u32) {
        self.value |= u128::from(value & ((1 << count) - 1)) << self.at;
        self.at += count;
    }
}
