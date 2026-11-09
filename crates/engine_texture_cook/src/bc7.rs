use engine_core::textures::bc7::{MODES, Parity, WEIGHTS, anchor_of, expand, subset_of, weights};
use engine_math::{max, min};

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
