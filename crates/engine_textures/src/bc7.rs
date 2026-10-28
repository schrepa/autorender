use engine_math::{max, min};

const WEIGHTS: [&[u32]; 3] = [
    &[0, 21, 43, 64],
    &[0, 9, 18, 27, 37, 46, 55, 64],
    &[0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64],
];

fn weights(bits: u32) -> &'static [u32] {
    WEIGHTS[(bits - 2) as usize]
}

const NEAREST_ENTRY: [[u8; 65]; 3] = {
    let mut table = [[0_u8; 65]; 3];
    let mut precision = 0;
    while precision < 3 {
        let entries = WEIGHTS[precision];
        let mut weight = 0;
        while weight <= 64 {
            let (mut best, mut distance) = (0, u32::MAX);
            let mut index = 0;
            while index < entries.len() {
                if entries[index].abs_diff(weight) < distance {
                    distance = entries[index].abs_diff(weight);
                    best = index;
                }
                index += 1;
            }
            table[precision][weight as usize] = best as u8;
            weight += 1;
        }
        precision += 1;
    }
    table
};

#[derive(Clone, Copy)]
struct Mode {
    subsets: usize,
    partition_bits: u32,
    rotation_bits: u32,
    selection_bits: u32,
    color_bits: u32,
    alpha_bits: u32,
    parity: Parity,
    index_bits: [u32; 2],
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Parity {
    None,
    Endpoint,
    Shared,
}

const MODES: [Mode; 8] = [
    Mode {
        subsets: 3,
        partition_bits: 4,
        rotation_bits: 0,
        selection_bits: 0,
        color_bits: 4,
        alpha_bits: 0,
        parity: Parity::Endpoint,
        index_bits: [3, 0],
    },
    Mode {
        subsets: 2,
        partition_bits: 6,
        rotation_bits: 0,
        selection_bits: 0,
        color_bits: 6,
        alpha_bits: 0,
        parity: Parity::Shared,
        index_bits: [3, 0],
    },
    Mode {
        subsets: 3,
        partition_bits: 6,
        rotation_bits: 0,
        selection_bits: 0,
        color_bits: 5,
        alpha_bits: 0,
        parity: Parity::None,
        index_bits: [2, 0],
    },
    Mode {
        subsets: 2,
        partition_bits: 6,
        rotation_bits: 0,
        selection_bits: 0,
        color_bits: 7,
        alpha_bits: 0,
        parity: Parity::Endpoint,
        index_bits: [2, 0],
    },
    Mode {
        subsets: 1,
        partition_bits: 0,
        rotation_bits: 2,
        selection_bits: 1,
        color_bits: 5,
        alpha_bits: 6,
        parity: Parity::None,
        index_bits: [2, 3],
    },
    Mode {
        subsets: 1,
        partition_bits: 0,
        rotation_bits: 2,
        selection_bits: 0,
        color_bits: 7,
        alpha_bits: 8,
        parity: Parity::None,
        index_bits: [2, 2],
    },
    Mode {
        subsets: 1,
        partition_bits: 0,
        rotation_bits: 0,
        selection_bits: 0,
        color_bits: 7,
        alpha_bits: 7,
        parity: Parity::Endpoint,
        index_bits: [4, 0],
    },
    Mode {
        subsets: 2,
        partition_bits: 6,
        rotation_bits: 0,
        selection_bits: 0,
        color_bits: 5,
        alpha_bits: 5,
        parity: Parity::Endpoint,
        index_bits: [2, 0],
    },
];

const PARTITIONS2: [u16; 64] = [
    0xcccc, 0x8888, 0xeeee, 0xecc8, 0xc880, 0xfeec, 0xfec8, 0xec80, 0xc800, 0xffec, 0xfe80, 0xe800,
    0xffe8, 0xff00, 0xfff0, 0xf000, 0xf710, 0x008e, 0x7100, 0x08ce, 0x008c, 0x7310, 0x3100, 0x8cce,
    0x088c, 0x3110, 0x6666, 0x366c, 0x17e8, 0x0ff0, 0x718e, 0x399c, 0xaaaa, 0xf0f0, 0x5a5a, 0x33cc,
    0x3c3c, 0x55aa, 0x9696, 0xa55a, 0x73ce, 0x13c8, 0x324c, 0x3bdc, 0x6996, 0xc33c, 0x9966, 0x0660,
    0x0272, 0x04e4, 0x4e40, 0x2720, 0xc936, 0x936c, 0x39c6, 0x639c, 0x9336, 0x9cc6, 0x817e, 0xe718,
    0xccf0, 0x0fcc, 0x7744, 0xee22,
];

const PARTITIONS3: [u32; 64] = [
    0xaa68_5050,
    0x6a5a_5040,
    0x5a5a_4200,
    0x5450_a0a8,
    0xa5a5_0000,
    0xa0a0_5050,
    0x5555_a0a0,
    0x5a5a_5050,
    0xaa55_0000,
    0xaa55_5500,
    0xaaaa_5500,
    0x9090_9090,
    0x9494_9494,
    0xa4a4_a4a4,
    0xa9a5_9450,
    0x2a0a_4250,
    0xa594_5040,
    0x0a42_5054,
    0xa5a5_a500,
    0x55a0_a0a0,
    0xa8a8_5454,
    0x6a6a_4040,
    0xa4a4_5000,
    0x1a1a_0500,
    0x0050_a4a4,
    0xaaa5_9090,
    0x1469_6914,
    0x6969_1400,
    0xa085_85a0,
    0xaa82_1414,
    0x50a4_a450,
    0x6a5a_0200,
    0xa9a5_8000,
    0x5090_a0a8,
    0xa8a0_9050,
    0x2424_2424,
    0x00aa_5500,
    0x2492_4924,
    0x2449_9224,
    0x50a5_0a50,
    0x500a_a550,
    0xaaaa_4444,
    0x6666_0000,
    0xa5a0_a5a0,
    0x50a0_50a0,
    0x6928_6928,
    0x44aa_aa44,
    0x6666_6600,
    0xaa44_4444,
    0x54a8_54a8,
    0x9580_9580,
    0x9696_9600,
    0xa854_54a8,
    0x8095_9580,
    0xaa14_1414,
    0x9696_0000,
    0xaaaa_1414,
    0xa050_50a0,
    0xa0a5_a5a0,
    0x9600_0000,
    0x4080_4080,
    0xa9a8_a9a8,
    0xaaaa_aa44,
    0x2a4a_5254,
];

const ANCHORS2: [u8; 64] = [
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 2, 8, 2, 2, 8, 8, 15, 2, 8,
    2, 2, 8, 8, 2, 2, 15, 15, 6, 8, 2, 8, 15, 15, 2, 8, 2, 2, 2, 15, 15, 6, 6, 2, 6, 8, 15, 15, 2,
    2, 15, 15, 15, 15, 15, 2, 2, 15,
];

const ANCHORS3: [[u8; 64]; 2] = [
    [
        3, 3, 15, 15, 8, 3, 15, 15, 8, 8, 6, 6, 6, 5, 3, 3, 3, 3, 8, 15, 3, 3, 6, 10, 5, 8, 8, 6,
        8, 5, 15, 15, 8, 15, 3, 5, 6, 10, 8, 15, 15, 3, 15, 5, 15, 15, 15, 15, 3, 15, 5, 5, 5, 8,
        5, 10, 5, 10, 8, 13, 15, 12, 3, 3,
    ],
    [
        15, 8, 8, 3, 15, 15, 3, 8, 15, 15, 15, 15, 15, 15, 15, 8, 15, 8, 15, 3, 15, 8, 15, 8, 3,
        15, 6, 10, 15, 15, 10, 8, 15, 3, 15, 10, 10, 8, 9, 10, 6, 15, 8, 15, 3, 6, 6, 8, 15, 3, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 3, 15, 15, 8,
    ],
];

fn subset_of(subsets: usize, partition: usize, texel: usize) -> usize {
    match subsets {
        2 => usize::from((PARTITIONS2[partition] >> texel) & 1 == 1),
        3 => ((PARTITIONS3[partition] >> (2 * texel)) & 3) as usize,
        _ => 0,
    }
}

fn anchor_of(subsets: usize, partition: usize, subset: usize) -> usize {
    match (subsets, subset) {
        (2, 1) => usize::from(ANCHORS2[partition]),
        (3, 1 | 2) => usize::from(ANCHORS3[subset - 1][partition]),
        _ => 0,
    }
}

fn expand(value: u32, bits: u32) -> u32 {
    (value << (8 - bits)) | (value >> (2 * bits - 8))
}

fn quantize(target: f32, bits: u32, parity: Option<u32>) -> (u32, u32) {
    let total = bits + u32::from(parity.is_some());
    let top = (1 << bits) - 1;
    let guess = (target * top as f32 / 255.0 + 0.5) as i64;
    let mut best = (u32::MAX, 0, 0);
    for candidate in guess - 1..=guess + 1 {
        let stored = candidate.clamp(0, i64::from(top)) as u32;
        let full = match parity {
            Some(bit) => (stored << 1) | bit,
            None => stored,
        };
        let widened = expand(full, total);
        let error = (widened as f32 - target).abs();
        let error = (error * 1024.0) as u32;
        if error < best.0 {
            best = (error, stored, widened);
        }
    }
    (best.1, best.2)
}

type Texels = [[f32; 4]; 16];

#[derive(Clone, Copy)]
struct Fit {
    stored: [[u32; 4]; 2],
    widened: [[u32; 4]; 2],
    parity: [u32; 2],
    indices: [u32; 16],
    error: u32,
}

#[derive(Clone, Copy)]
struct Span {
    first: usize,
    count: usize,
}

fn principal(
    texels: &Texels,
    members: &[usize],
    span: Span,
    iterations: u32,
) -> ([f32; 4], [f32; 4]) {
    let mut mean = [0.0_f32; 4];
    for &texel in members {
        for channel in 0..span.count {
            mean[channel] += texels[texel][span.first + channel];
        }
    }
    let count = members.len().max(1) as f32;
    for value in &mut mean {
        *value /= count;
    }
    let mut covariance = [[0.0_f32; 4]; 4];
    for &texel in members {
        let mut offset = [0.0_f32; 4];
        for channel in 0..span.count {
            offset[channel] = texels[texel][span.first + channel] - mean[channel];
        }
        for row in 0..span.count {
            for column in 0..span.count {
                covariance[row][column] += offset[row] * offset[column];
            }
        }
    }
    let mut axis = [0.0_f32; 4];
    let mut widest = 0;
    for channel in 1..span.count {
        if covariance[channel][channel] > covariance[widest][widest] {
            widest = channel;
        }
    }
    axis[widest] = 1.0;
    for _ in 0..iterations {
        let mut next = [0.0_f32; 4];
        for row in 0..span.count {
            for column in 0..span.count {
                next[row] += covariance[row][column] * axis[column];
            }
        }
        let length = next.iter().map(|value| value * value).sum::<f32>().sqrt();
        if length < 1.0e-12 {
            break;
        }
        axis = next.map(|value| value / length);
    }
    (mean, axis)
}

fn extremes(
    texels: &Texels,
    members: &[usize],
    span: Span,
    (mean, axis): ([f32; 4], [f32; 4]),
) -> [[f32; 4]; 2] {
    let (mut low, mut high) = (f32::MAX, f32::MIN);
    for &texel in members {
        let mut along = 0.0;
        for channel in 0..span.count {
            along += (texels[texel][span.first + channel] - mean[channel]) * axis[channel];
        }
        low = min(low, along);
        high = max(high, along);
    }
    let point = |t: f32| {
        let mut point = [0.0_f32; 4];
        for channel in 0..span.count {
            point[channel] = engine_math::clamp(mean[channel] + axis[channel] * t, 0.0, 255.0);
        }
        point
    };
    [point(low), point(high)]
}

fn assign(
    texels: &Texels,
    members: &[usize],
    span: Span,
    ends: [[f32; 4]; 2],
    (bits, parity, index_bits): (u32, Parity, u32),
    limit: u32,
) -> Fit {
    let missed = |end: usize, bit: u32| -> f32 {
        (0..span.count)
            .map(|channel| {
                let target = ends[end][channel];
                let miss = quantize(target, bits, Some(bit)).1 as f32 - target;
                miss * miss
            })
            .sum()
    };
    let pair = match parity {
        Parity::None => [0, 0],
        Parity::Shared => {
            let bit = u32::from(missed(0, 1) + missed(1, 1) < missed(0, 0) + missed(1, 0));
            [bit, bit]
        }
        Parity::Endpoint => [0, 1].map(|end| u32::from(missed(end, 1) < missed(end, 0))),
    };
    let choices = [pair];
    let palette_weights = weights(index_bits);
    let mut best = Fit {
        stored: [[0; 4]; 2],
        widened: [[0; 4]; 2],
        parity: [0; 2],
        indices: [0; 16],
        error: u32::MAX,
    };
    for pair in choices {
        let mut fit = Fit {
            parity: pair,
            error: 0,
            ..best
        };
        for (end, end_values) in ends.iter().enumerate() {
            let bit = (parity != Parity::None).then_some(pair[end]);
            for (channel, &target) in end_values.iter().enumerate().take(span.count) {
                let (stored, widened) = quantize(target, bits, bit);
                fit.stored[end][channel] = stored;
                fit.widened[end][channel] = widened;
            }
        }
        let mut palette = [[0_i32; 4]; 16];
        for (entry, &weight) in palette.iter_mut().zip(palette_weights) {
            for (channel, value) in entry.iter_mut().enumerate().take(span.count) {
                *value = ((64 - weight) * fit.widened[0][channel]
                    + weight * fit.widened[1][channel]
                    + 32) as i32
                    >> 6;
            }
        }
        let mut along = [0_i32; 4];
        let mut length = 0;
        for (channel, step) in along.iter_mut().enumerate().take(span.count) {
            *step = fit.widened[1][channel] as i32 - fit.widened[0][channel] as i32;
            length += *step * *step;
        }
        let last = palette_weights.len() - 1;
        let scale = if length > 0 {
            64.0 / length as f32
        } else {
            0.0
        };
        for &texel in members {
            let color = &texels[texel];
            let mut guess = 0;
            if length > 0 {
                let mut dot = 0;
                for channel in 0..span.count {
                    dot += (color[span.first + channel] as i32 - fit.widened[0][channel] as i32)
                        * along[channel];
                }
                let weight = engine_math::clamp(dot as f32 * scale + 0.5, 0.0, 64.0) as usize;
                guess = NEAREST_ENTRY[(index_bits - 2) as usize][weight] as usize;
            }
            let mut nearest = (u32::MAX, 0);
            let (low, high) = (guess.saturating_sub(1), (guess + 1).min(last));
            for (index, entry) in palette.iter().enumerate().take(high + 1).skip(low) {
                let mut error = 0;
                for channel in 0..span.count {
                    let difference = entry[channel] - color[span.first + channel] as i32;
                    error += (difference * difference) as u32;
                }
                if error < nearest.0 {
                    nearest = (error, index as u32);
                }
            }
            fit.indices[texel] = nearest.1;
            fit.error += nearest.0;
            if fit.error >= best.error || fit.error >= limit {
                break;
            }
        }
        if fit.error < best.error {
            best = fit;
        }
    }
    best
}

fn least_squares(
    texels: &Texels,
    members: &[usize],
    span: Span,
    fit: &Fit,
    index_bits: u32,
) -> Option<[[f32; 4]; 2]> {
    let palette_weights = weights(index_bits);
    let (mut aa, mut ab, mut bb) = (0.0_f32, 0.0_f32, 0.0_f32);
    let mut along_a = [0.0_f32; 4];
    let mut along_b = [0.0_f32; 4];
    for &texel in members {
        let weight = palette_weights[fit.indices[texel] as usize] as f32 / 64.0;
        let rest = 1.0 - weight;
        aa += rest * rest;
        ab += rest * weight;
        bb += weight * weight;
        for channel in 0..span.count {
            let value = texels[texel][span.first + channel];
            along_a[channel] += rest * value;
            along_b[channel] += weight * value;
        }
    }
    let determinant = aa * bb - ab * ab;
    if determinant.abs() < 1.0e-6 {
        return None;
    }
    let mut ends = [[0.0_f32; 4]; 2];
    for channel in 0..span.count {
        let a = (bb * along_a[channel] - ab * along_b[channel]) / determinant;
        let b = (aa * along_b[channel] - ab * along_a[channel]) / determinant;
        ends[0][channel] = engine_math::clamp(a, 0.0, 255.0);
        ends[1][channel] = engine_math::clamp(b, 0.0, 255.0);
    }
    Some(ends)
}

fn fit(
    texels: &Texels,
    members: &[usize],
    span: Span,
    layout: (u32, Parity, u32),
    limit: u32,
) -> Fit {
    let line = principal(texels, members, span, FITTING_ITERATIONS);
    let ends = extremes(texels, members, span, line);
    let mut best = assign(texels, members, span, ends, layout, limit.saturating_mul(2));
    for _ in 0..2 {
        if best.error == 0 || best.error >= limit.saturating_mul(2) {
            break;
        }
        let Some(ends) = least_squares(texels, members, span, &best, layout.2) else {
            break;
        };
        let refined = assign(texels, members, span, ends, layout, best.error);
        if refined.error >= best.error {
            break;
        }
        best = refined;
    }
    best
}

struct Writer {
    bits: u128,
    at: u32,
}

impl Writer {
    fn put(&mut self, value: u32, count: u32) {
        if count > 0 {
            self.bits |= (u128::from(value) & ((1_u128 << count) - 1)) << self.at;
            self.at += count;
        }
    }
}

#[derive(Clone, Copy)]
struct Candidate {
    number: usize,
    choice: (usize, u32, u32),
    fits: [Fit; 3],
    alpha: Option<Fit>,
    error: u32,
}

fn encode_mode(
    texels: &Texels,
    number: usize,
    (partition, rotation, selection): (usize, u32, u32),
    bound: u32,
) -> Option<Candidate> {
    let mode = MODES[number];
    let mut rotated = *texels;
    if rotation > 0 {
        for texel in &mut rotated {
            texel.swap(3, rotation as usize - 1);
        }
    }
    let mut members = [[0_usize; 16]; 3];
    let mut counts = [0_usize; 3];
    for texel in 0..16 {
        let subset = subset_of(mode.subsets, partition, texel);
        members[subset][counts[subset]] = texel;
        counts[subset] += 1;
    }
    let separate = mode.rotation_bits > 0;
    let joint = Span {
        first: 0,
        count: if separate || mode.alpha_bits == 0 {
            3
        } else {
            4
        },
    };
    let (color_index, alpha_index) = if selection == 1 {
        (mode.index_bits[1], mode.index_bits[0])
    } else {
        (mode.index_bits[0], mode.index_bits[1])
    };
    let mut error = if mode.alpha_bits == 0 {
        texels
            .iter()
            .map(|texel| {
                let difference = 255 - texel[3] as i32;
                (difference * difference) as u32
            })
            .sum::<u32>()
    } else {
        0
    };
    let mut fits = [None; 3];
    for (subset, fitted) in fits.iter_mut().enumerate().take(mode.subsets) {
        let members = &members[subset][..counts[subset]];
        let subset_fit = fit(
            &rotated,
            members,
            joint,
            (mode.color_bits, mode.parity, color_index),
            bound.saturating_sub(error),
        );
        error += subset_fit.error;
        if error >= bound {
            return None;
        }
        *fitted = Some(subset_fit);
    }
    let alpha = separate.then(|| {
        fit(
            &rotated,
            &members[0][..16],
            Span { first: 3, count: 1 },
            (mode.alpha_bits, Parity::None, alpha_index),
            bound.saturating_sub(error),
        )
    });
    error += alpha.map_or(0, |alpha| alpha.error);
    let first = fits[0]?;
    (error < bound).then(|| Candidate {
        number,
        choice: (partition, rotation, selection),
        fits: fits.map(|fit| fit.unwrap_or(first)),
        alpha,
        error,
    })
}

fn anchor(fit: &mut Fit, members: impl Iterator<Item = usize> + Clone, at: usize, bits: u32) {
    let top = (1 << bits) - 1;
    if fit.indices[at] >> (bits - 1) == 0 {
        return;
    }
    fit.stored.swap(0, 1);
    fit.widened.swap(0, 1);
    fit.parity.swap(0, 1);
    for texel in members {
        fit.indices[texel] = top - fit.indices[texel];
    }
}

fn pack(candidate: Candidate) -> [u8; 16] {
    let Candidate {
        number,
        choice: (partition, rotation, selection),
        mut fits,
        mut alpha,
        ..
    } = candidate;
    let mode = MODES[number];
    let fits = &mut fits[..mode.subsets];
    let (color_index, alpha_index) = if selection == 1 {
        (mode.index_bits[1], mode.index_bits[0])
    } else {
        (mode.index_bits[0], mode.index_bits[1])
    };
    for (subset, fit) in fits.iter_mut().enumerate() {
        let members = (0..16).filter(|&texel| subset_of(mode.subsets, partition, texel) == subset);
        anchor(
            fit,
            members,
            anchor_of(mode.subsets, partition, subset),
            color_index,
        );
    }
    if let Some(alpha) = &mut alpha {
        anchor(alpha, 0..16, 0, alpha_index);
    }
    let mut out = Writer { bits: 0, at: 0 };
    out.put(1 << number, number as u32 + 1);
    out.put(partition as u32, mode.partition_bits);
    out.put(rotation, mode.rotation_bits);
    out.put(selection, mode.selection_bits);
    for channel in 0..3 {
        for fit in fits.iter() {
            out.put(fit.stored[0][channel], mode.color_bits);
            out.put(fit.stored[1][channel], mode.color_bits);
        }
    }
    match &alpha {
        Some(alpha) => {
            out.put(alpha.stored[0][0], mode.alpha_bits);
            out.put(alpha.stored[1][0], mode.alpha_bits);
        }
        None if mode.alpha_bits > 0 => {
            for fit in fits.iter() {
                out.put(fit.stored[0][3], mode.alpha_bits);
                out.put(fit.stored[1][3], mode.alpha_bits);
            }
        }
        None => {}
    }
    for fit in fits.iter() {
        match mode.parity {
            Parity::Endpoint => {
                out.put(fit.parity[0], 1);
                out.put(fit.parity[1], 1);
            }
            Parity::Shared => out.put(fit.parity[0], 1),
            Parity::None => {}
        }
    }
    let (first, second) = match &alpha {
        Some(alpha) if selection == 1 => (alpha.indices, fits[0].indices),
        Some(alpha) => (fits[0].indices, alpha.indices),
        None => {
            let mut joined = [0; 16];
            for (subset, fit) in fits.iter().enumerate() {
                for (texel, index) in joined.iter_mut().enumerate() {
                    if subset_of(mode.subsets, partition, texel) == subset {
                        *index = fit.indices[texel];
                    }
                }
            }
            (joined, [0; 16])
        }
    };
    let anchors: Vec<usize> = (0..mode.subsets)
        .map(|subset| anchor_of(mode.subsets, partition, subset))
        .collect();
    for (texel, &index) in first.iter().enumerate() {
        let bits = mode.index_bits[0] - u32::from(anchors.contains(&texel));
        out.put(index, bits);
    }
    if mode.index_bits[1] > 0 {
        for (texel, &index) in second.iter().enumerate() {
            out.put(index, mode.index_bits[1] - u32::from(texel == 0));
        }
    }
    out.bits.to_le_bytes()
}

fn ranked(texels: &Texels, subsets: usize, limit: usize, span: Span, keep: usize) -> Vec<usize> {
    let mut scored: Vec<(f32, usize)> = (0..limit)
        .map(|partition| {
            let mut members = [[0_usize; 16]; 3];
            let mut counts = [0_usize; 3];
            for texel in 0..16 {
                let subset = subset_of(subsets, partition, texel);
                members[subset][counts[subset]] = texel;
                counts[subset] += 1;
            }
            let mut spread = 0.0;
            for (members, &count) in members.iter().zip(&counts).take(subsets) {
                let members = &members[..count];
                let (mean, axis) = principal(texels, members, span, RANKING_ITERATIONS);
                for &texel in members {
                    let mut along = 0.0;
                    let mut total = 0.0;
                    for channel in 0..span.count {
                        let offset = texels[texel][span.first + channel] - mean[channel];
                        along += offset * axis[channel];
                        total += offset * offset;
                    }
                    spread += total - along * along;
                }
            }
            (spread, partition)
        })
        .collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    scored
        .into_iter()
        .take(keep)
        .map(|(_, partition)| partition)
        .collect()
}

const GOOD_ENOUGH: u32 = 16 * 4;
const RANKING_ITERATIONS: u32 = 2;
const FITTING_ITERATIONS: u32 = 4;
const TRIED2: usize = 4;
const TRIED3: usize = 2;

pub(crate) fn encode_block(texels: &[[u8; 4]; 16]) -> [u8; 16] {
    let colors: Texels = texels.map(|texel| texel.map(f32::from));
    let opaque = texels.iter().all(|texel| texel[3] == 255);
    let Some(mut best) = encode_mode(&colors, 6, (0, 0, 0), u32::MAX) else {
        return [0; 16];
    };
    let consider = |best: &mut Candidate, number: usize, choice: (usize, u32, u32)| {
        if let Some(candidate) = encode_mode(&colors, number, choice, best.error) {
            *best = candidate;
        }
    };
    if best.error <= GOOD_ENOUGH {
        return pack(best);
    }
    for rotation in u32::from(opaque)..4 {
        consider(&mut best, 5, (0, rotation, 0));
        for selection in 0..2 {
            consider(&mut best, 4, (0, rotation, selection));
        }
    }
    if best.error <= GOOD_ENOUGH {
        return pack(best);
    }
    if opaque {
        let rgb = Span { first: 0, count: 3 };
        for partition in ranked(&colors, 2, 64, rgb, TRIED2) {
            consider(&mut best, 1, (partition, 0, 0));
            consider(&mut best, 3, (partition, 0, 0));
        }
        let three = ranked(&colors, 3, 64, rgb, 64);
        for &partition in three
            .iter()
            .filter(|&&partition| partition < 16)
            .take(TRIED3)
        {
            consider(&mut best, 0, (partition, 0, 0));
        }
        for &partition in three.iter().take(TRIED3) {
            consider(&mut best, 2, (partition, 0, 0));
        }
    } else {
        let rgba = Span { first: 0, count: 4 };
        for partition in ranked(&colors, 2, 64, rgba, TRIED2) {
            consider(&mut best, 7, (partition, 0, 0));
        }
    }
    pack(best)
}

struct Reader {
    bits: u128,
    at: u32,
}

impl Reader {
    fn take(&mut self, count: u32) -> u32 {
        let value = ((self.bits >> self.at) & ((1_u128 << count) - 1)) as u32;
        self.at += count;
        value
    }
}

pub(crate) fn decode_block(block: &[u8; 16]) -> [[u8; 4]; 16] {
    let bits = u128::from_le_bytes(*block);
    let number = bits.trailing_zeros() as usize;
    if number >= 8 {
        return [[0; 4]; 16];
    }
    let mode = MODES[number];
    let mut read = Reader {
        bits,
        at: number as u32 + 1,
    };
    let partition = read.take(mode.partition_bits) as usize;
    let rotation = read.take(mode.rotation_bits);
    let selection = read.take(mode.selection_bits);
    let mut ends = [[[0_u32; 4]; 2]; 3];
    for channel in 0..4 {
        let bits = if channel < 3 {
            mode.color_bits
        } else {
            mode.alpha_bits
        };
        for subset_ends in ends.iter_mut().take(mode.subsets) {
            for end in subset_ends.iter_mut() {
                end[channel] = read.take(bits);
            }
        }
    }
    let parity_bits = u32::from(mode.parity != Parity::None);
    for subset_ends in ends.iter_mut().take(mode.subsets) {
        let parity = match mode.parity {
            Parity::Endpoint => [read.take(1), read.take(1)],
            Parity::Shared => {
                let bit = read.take(1);
                [bit, bit]
            }
            Parity::None => [0, 0],
        };
        for (end, bit) in subset_ends.iter_mut().zip(parity) {
            for (channel, value) in end.iter_mut().enumerate() {
                let bits = if channel < 3 {
                    mode.color_bits
                } else {
                    mode.alpha_bits
                };
                *value = if bits == 0 {
                    255
                } else {
                    expand((*value << parity_bits) | bit, bits + parity_bits)
                };
            }
        }
    }
    let anchors: Vec<usize> = (0..mode.subsets)
        .map(|subset| anchor_of(mode.subsets, partition, subset))
        .collect();
    let mut first = [0_u32; 16];
    for (texel, index) in first.iter_mut().enumerate() {
        *index = read.take(mode.index_bits[0] - u32::from(anchors.contains(&texel)));
    }
    let mut second = first;
    if mode.index_bits[1] > 0 {
        for (texel, index) in second.iter_mut().enumerate() {
            *index = read.take(mode.index_bits[1] - u32::from(texel == 0));
        }
    }
    let (color_bits, alpha_bits) = if mode.index_bits[1] == 0 {
        (mode.index_bits[0], mode.index_bits[0])
    } else if selection == 1 {
        (mode.index_bits[1], mode.index_bits[0])
    } else {
        (mode.index_bits[0], mode.index_bits[1])
    };
    let (color_indices, alpha_indices) = if selection == 1 {
        (second, first)
    } else {
        (first, second)
    };
    std::array::from_fn(|texel| {
        let subset = subset_of(mode.subsets, partition, texel);
        let [low, high] = ends[subset];
        let color = weights(color_bits)[color_indices[texel] as usize];
        let alpha = weights(alpha_bits)[alpha_indices[texel] as usize];
        let mix = |channel: usize, weight: u32| {
            (((64 - weight) * low[channel] + weight * high[channel] + 32) >> 6) as u8
        };
        let mut out = [mix(0, color), mix(1, color), mix(2, color), mix(3, alpha)];
        if rotation > 0 {
            out.swap(3, rotation as usize - 1);
        }
        out
    })
}
