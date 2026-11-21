use engine_math::{clamp, max, min};

use crate::astc_bits::{
    COLOR_RANGES, SEEDS_2, WEIGHT_QUANT, WEIGHT_RANGES, WEIGHT_SORTED, block_mode_4x4,
    color_quant, color_range_for, pack_ise, unquantize_weight,
};

pub(crate) use crate::astc_bits::{decode_block_linear, decode_block_srgb};

type Texels = [[f32; 4]; 16];

fn encode_void_extent(color: [u8; 4]) -> [u8; 16] {
    let mut bits: u128 = 0xFFFF_FFFF_FFFF_FDFC;
    for (channel, &value) in color.iter().enumerate() {
        let unorm16 = (u16::from(value) << 8) | u16::from(value);
        bits |= u128::from(unorm16) << (64 + channel * 16);
    }
    bits.to_le_bytes()
}

fn principal_channels(
    texels: &Texels,
    members: &[usize],
    channels: &[usize],
    iterations: u32,
) -> ([f32; 4], [f32; 4]) {
    let mut mean = [0.0_f32; 4];
    for &texel in members {
        for &ch in channels {
            mean[ch] += texels[texel][ch];
        }
    }
    let count = members.len().max(1) as f32;
    for &ch in channels {
        mean[ch] /= count;
    }
    let mut cov = [[0.0_f32; 4]; 4];
    for &texel in members {
        for &r in channels {
            let dr = texels[texel][r] - mean[r];
            for &c in channels {
                cov[r][c] += dr * (texels[texel][c] - mean[c]);
            }
        }
    }
    let mut axis = [0.0_f32; 4];
    let mut widest = channels[0];
    for &ch in channels.iter().skip(1) {
        if cov[ch][ch] > cov[widest][widest] {
            widest = ch;
        }
    }
    axis[widest] = 1.0;
    for _ in 0..iterations {
        let mut next = [0.0_f32; 4];
        for &r in channels {
            for &c in channels {
                next[r] += cov[r][c] * axis[c];
            }
        }
        let len = channels.iter().map(|&ch| next[ch] * next[ch]).sum::<f32>().sqrt();
        if len < 1.0e-12 {
            break;
        }
        for &ch in channels {
            axis[ch] = next[ch] / len;
        }
    }
    (mean, axis)
}

fn endpoints_from_line(
    texels: &Texels,
    members: &[usize],
    channels: &[usize],
    (mean, axis): ([f32; 4], [f32; 4]),
) -> [[f32; 4]; 2] {
    let (mut low, mut high) = (f32::MAX, f32::MIN);
    for &texel in members {
        let mut along = 0.0_f32;
        for &ch in channels {
            along += (texels[texel][ch] - mean[ch]) * axis[ch];
        }
        low = min(low, along);
        high = max(high, along);
    }
    let mut ends = [[0.0_f32; 4]; 2];
    for &ch in channels {
        ends[0][ch] = clamp(mean[ch] + axis[ch] * low, 0.0, 255.0);
        ends[1][ch] = clamp(mean[ch] + axis[ch] * high, 0.0, 255.0);
    }
    ends
}

#[derive(Clone, Copy)]
struct PlaneFit {
    stored_ends: [[u8; 4]; 2],
    unq_ends: [[u8; 4]; 2],
    stored_weights: [u8; 16],
    unq_weights: [u8; 16],
    error: u32,
}

fn assign_plane(
    texels: &Texels,
    members: &[usize],
    channels: &[usize],
    mut ends: [[f32; 4]; 2],
    (color_range, weight_range): (usize, usize),
    order_rgb: bool,
    limit: u32,
) -> PlaneFit {
    let mut fit = PlaneFit {
        stored_ends: [[0; 4]; 2],
        unq_ends: [[0; 4]; 2],
        stored_weights: [0; 16],
        unq_weights: [0; 16],
        error: 0,
    };
    if order_rgb {
        let sum0 = ends[0][0] + ends[0][1] + ends[0][2];
        let sum1 = ends[1][0] + ends[1][1] + ends[1][2];
        if sum1 < sum0 {
            ends.swap(0, 1);
        }
    }
    let quant = &color_quant()[color_range];
    for (end, end_values) in ends.iter().enumerate() {
        for &ch in channels {
            let target = clamp(end_values[ch] + 0.5, 0.0, 255.0) as usize;
            let (stored, unq) = quant[target];
            fit.stored_ends[end][ch] = stored;
            fit.unq_ends[end][ch] = unq;
        }
    }
    if order_rgb {
        let s0 = u32::from(fit.unq_ends[0][0])
            + u32::from(fit.unq_ends[0][1])
            + u32::from(fit.unq_ends[0][2]);
        let s1 = u32::from(fit.unq_ends[1][0])
            + u32::from(fit.unq_ends[1][1])
            + u32::from(fit.unq_ends[1][2]);
        if s1 < s0 {
            fit.stored_ends.swap(0, 1);
            fit.unq_ends.swap(0, 1);
        }
    }
    let (items, count) = &WEIGHT_SORTED[weight_range];
    let mut along = [0_i32; 4];
    let mut length = 0_i32;
    for &ch in channels {
        let diff = i32::from(fit.unq_ends[1][ch]) - i32::from(fit.unq_ends[0][ch]);
        along[ch] = diff;
        length += diff * diff;
    }
    let scale = if length > 0 {
        64.0 / length as f32
    } else {
        0.0
    };
    for &texel in members {
        let color = &texels[texel];
        let guess_w = if length > 0 {
            let mut dot = 0_i32;
            for &ch in channels {
                dot += (color[ch] as i32 - i32::from(fit.unq_ends[0][ch])) * along[ch];
            }
            clamp(dot as f32 * scale + 0.5, 0.0, 64.0) as usize
        } else {
            0
        };
        let (guess_stored, guess_unq) = WEIGHT_QUANT[weight_range][guess_w];
        let pos = items[..*count]
            .iter()
            .position(|&(sv, _)| sv == guess_stored)
            .unwrap_or(0);
        let lo = pos.saturating_sub(1);
        let hi = (pos + 1).min(*count - 1);
        let mut best = (u32::MAX, guess_stored, guess_unq);
        for &(sv, uw) in &items[lo..=hi] {
            let w = u32::from(uw);
            let mut err = 0_u32;
            for &ch in channels {
                let e0 = u32::from(fit.unq_ends[0][ch]);
                let e1 = u32::from(fit.unq_ends[1][ch]);
                let interp = ((e0 * (64 - w) + e1 * w + 32) >> 6) as i32;
                let diff = interp - color[ch] as i32;
                err += (diff * diff) as u32;
            }
            if err < best.0 {
                best = (err, sv, uw);
            }
        }
        fit.stored_weights[texel] = best.1;
        fit.unq_weights[texel] = best.2;
        fit.error += best.0;
        if fit.error >= limit {
            break;
        }
    }
    fit
}

fn refine_ends(
    texels: &Texels,
    members: &[usize],
    channels: &[usize],
    fit: &PlaneFit,
) -> Option<[[f32; 4]; 2]> {
    let (mut aa, mut ab, mut bb) = (0.0_f32, 0.0_f32, 0.0_f32);
    let mut sum_a = [0.0_f32; 4];
    let mut sum_b = [0.0_f32; 4];
    for &texel in members {
        let w = f32::from(fit.unq_weights[texel]) / 64.0;
        let rest = 1.0 - w;
        aa += rest * rest;
        ab += rest * w;
        bb += w * w;
        for &ch in channels {
            let value = texels[texel][ch];
            sum_a[ch] += rest * value;
            sum_b[ch] += w * value;
        }
    }
    let det = aa * bb - ab * ab;
    if det.abs() < 1.0e-6 {
        return None;
    }
    let mut ends = [[0.0_f32; 4]; 2];
    for &ch in channels {
        ends[0][ch] = clamp((bb * sum_a[ch] - ab * sum_b[ch]) / det, 0.0, 255.0);
        ends[1][ch] = clamp((aa * sum_b[ch] - ab * sum_a[ch]) / det, 0.0, 255.0);
    }
    Some(ends)
}

fn fit_plane(
    texels: &Texels,
    members: &[usize],
    channels: &[usize],
    ranges: (usize, usize),
    order_rgb: bool,
    limit: u32,
) -> PlaneFit {
    let ends = if channels.len() == 1 {
        let ch = channels[0];
        let (mut lo, mut hi) = (255.0_f32, 0.0_f32);
        for &texel in members {
            lo = min(lo, texels[texel][ch]);
            hi = max(hi, texels[texel][ch]);
        }
        let mut pair = [[0.0_f32; 4]; 2];
        pair[0][ch] = lo;
        pair[1][ch] = hi;
        pair
    } else {
        let line = principal_channels(texels, members, channels, 4);
        endpoints_from_line(texels, members, channels, line)
    };
    let mut best = assign_plane(texels, members, channels, ends, ranges, order_rgb, limit);
    for _ in 0..2 {
        if best.error == 0 || best.error >= limit {
            break;
        }
        let Some(next_ends) = refine_ends(texels, members, channels, &best) else {
            break;
        };
        let refined = assign_plane(
            texels,
            members,
            channels,
            next_ends,
            ranges,
            order_rgb,
            best.error,
        );
        if refined.error >= best.error {
            break;
        }
        best = refined;
    }
    best
}

fn pack_single_plane(
    weight_range: usize,
    cem: u32,
    color_range: usize,
    fit: &PlaneFit,
) -> [u8; 16] {
    let count = if cem == 8 { 6 } else { 8 };
    let mut endpoints = [0_u8; 8];
    for ch in 0..(count / 2) {
        endpoints[ch * 2] = fit.stored_ends[0][ch];
        endpoints[ch * 2 + 1] = fit.stored_ends[1][ch];
    }
    let mut block: u128 = u128::from(block_mode_4x4(false, weight_range));
    block |= u128::from(cem) << 13;
    let ep_bits = pack_ise(COLOR_RANGES[color_range], &endpoints[..count]);
    block |= ep_bits << 17;
    let w_bits = pack_ise(WEIGHT_RANGES[weight_range], &fit.stored_weights);
    block |= w_bits.reverse_bits();
    block.to_le_bytes()
}

fn invert_weights(weights: &mut [u8; 16], weight_range: usize) {
    for stored in weights.iter_mut() {
        let unq = unquantize_weight(weight_range, *stored);
        *stored = WEIGHT_QUANT[weight_range][(64 - unq) as usize].0;
    }
}

fn pack_dual_plane(
    weight_range: usize,
    cem: u32,
    ccs: usize,
    color_range: usize,
    plane0: &PlaneFit,
    plane1: &PlaneFit,
) -> [u8; 16] {
    let mut stored_ends = plane0.stored_ends;
    let mut unq_ends = plane0.unq_ends;
    stored_ends[0][ccs] = plane1.stored_ends[0][ccs];
    stored_ends[1][ccs] = plane1.stored_ends[1][ccs];
    unq_ends[0][ccs] = plane1.unq_ends[0][ccs];
    unq_ends[1][ccs] = plane1.unq_ends[1][ccs];
    let mut w0 = plane0.stored_weights;
    let mut w1 = plane1.stored_weights;
    let s0 = u32::from(unq_ends[0][0]) + u32::from(unq_ends[0][1]) + u32::from(unq_ends[0][2]);
    let s1 = u32::from(unq_ends[1][0]) + u32::from(unq_ends[1][1]) + u32::from(unq_ends[1][2]);
    if s1 < s0 {
        stored_ends.swap(0, 1);
        invert_weights(&mut w0, weight_range);
        invert_weights(&mut w1, weight_range);
    }
    let count = if cem == 8 { 6 } else { 8 };
    let mut endpoints = [0_u8; 8];
    for ch in 0..(count / 2) {
        endpoints[ch * 2] = stored_ends[0][ch];
        endpoints[ch * 2 + 1] = stored_ends[1][ch];
    }
    let mut weights = [0_u8; 32];
    for texel in 0..16 {
        weights[texel * 2] = w0[texel];
        weights[texel * 2 + 1] = w1[texel];
    }
    let weight_bits = WEIGHT_RANGES[weight_range].stream_bits(32);
    let mut block: u128 = u128::from(block_mode_4x4(true, weight_range));
    block |= u128::from(cem) << 13;
    block |= pack_ise(COLOR_RANGES[color_range], &endpoints[..count]) << 17;
    block |= u128::from((ccs & 3) as u32) << (128 - weight_bits - 2);
    block |= pack_ise(WEIGHT_RANGES[weight_range], &weights).reverse_bits();
    block.to_le_bytes()
}

fn pack_two_partitions(
    weight_range: usize,
    cem: u32,
    seed: u16,
    mask: u16,
    color_range: usize,
    fits: &[PlaneFit; 2],
) -> [u8; 16] {
    let per = if cem == 8 { 6 } else { 8 };
    let mut endpoints = [0_u8; 16];
    for (part, fit) in fits.iter().enumerate() {
        for ch in 0..(per / 2) {
            endpoints[part * per + ch * 2] = fit.stored_ends[0][ch];
            endpoints[part * per + ch * 2 + 1] = fit.stored_ends[1][ch];
        }
    }
    let mut weights = [0_u8; 16];
    for (texel, slot) in weights.iter_mut().enumerate() {
        let part = usize::from(((mask >> texel) & 1) != 0);
        *slot = fits[part].stored_weights[texel];
    }
    let mut block: u128 = u128::from(block_mode_4x4(false, weight_range));
    block |= 1_u128 << 11;
    block |= u128::from(seed) << 13;
    block |= u128::from(cem) << 25;
    block |= pack_ise(COLOR_RANGES[color_range], &endpoints[..per * 2]) << 29;
    block |= pack_ise(WEIGHT_RANGES[weight_range], &weights).reverse_bits();
    block.to_le_bytes()
}

fn try_dual_plane(
    colors: &Texels,
    opaque: bool,
    ccs: usize,
    weight_range: usize,
    bound: u32,
) -> Option<(u32, [u8; 16])> {
    let cem = if opaque { 8 } else { 12 };
    let value_count = if opaque { 6 } else { 8 };
    let weight_bits = WEIGHT_RANGES[weight_range].stream_bits(32);
    let color_range = color_range_for(128 - 19 - weight_bits, value_count)?;
    let all: [usize; 16] = std::array::from_fn(|i| i);
    let mut primary = [0_usize; 3];
    let mut primary_len = 0;
    let total_channels = if opaque { 3 } else { 4 };
    for ch in 0..total_channels {
        if ch != ccs {
            primary[primary_len] = ch;
            primary_len += 1;
        }
    }
    let plane0 = fit_plane(
        colors,
        &all,
        &primary[..primary_len],
        (color_range, weight_range),
        false,
        bound,
    );
    if plane0.error >= bound {
        return None;
    }
    let plane1 = fit_plane(
        colors,
        &all,
        &[ccs],
        (color_range, weight_range),
        false,
        bound - plane0.error,
    );
    let total = plane0.error + plane1.error;
    if total >= bound {
        return None;
    }
    let packed = pack_dual_plane(weight_range, cem, ccs, color_range, &plane0, &plane1);
    Some((total, packed))
}

fn rank_seeds(colors: &Texels, channels: &[usize], keep: usize) -> [(u16, u16); 4] {
    let mut scored = [(f32::MAX, 0_u16, 0_u16); 64];
    for (index, &(seed, mask)) in SEEDS_2.iter().enumerate() {
        let mut members = [[0_usize; 16]; 2];
        let mut counts = [0_usize; 2];
        for texel in 0..16 {
            let part = usize::from(((mask >> texel) & 1) != 0);
            members[part][counts[part]] = texel;
            counts[part] += 1;
        }
        let mut spread = 0.0_f32;
        for part in 0..2 {
            let slice = &members[part][..counts[part]];
            let (mean, axis) = principal_channels(colors, slice, channels, 2);
            for &texel in slice {
                let mut along = 0.0_f32;
                let mut total = 0.0_f32;
                for &ch in channels {
                    let delta = colors[texel][ch] - mean[ch];
                    along += delta * axis[ch];
                    total += delta * delta;
                }
                spread += total - along * along;
            }
        }
        scored[index] = (spread, seed, mask);
    }
    scored.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    std::array::from_fn(|i| {
        if i < keep {
            (scored[i].1, scored[i].2)
        } else {
            (0, 0)
        }
    })
}

fn try_two_partitions(
    colors: &Texels,
    opaque: bool,
    (seed, mask): (u16, u16),
    weight_range: usize,
    bound: u32,
) -> Option<(u32, [u8; 16])> {
    let cem = if opaque { 8 } else { 12 };
    let total_values = if opaque { 12 } else { 16 };
    let weight_bits = WEIGHT_RANGES[weight_range].stream_bits(16);
    let color_range = color_range_for(128 - 29 - weight_bits, total_values)?;
    let channels: &[usize] = if opaque { &[0, 1, 2] } else { &[0, 1, 2, 3] };
    let mut members = [[0_usize; 16]; 2];
    let mut counts = [0_usize; 2];
    for texel in 0..16 {
        let part = usize::from(((mask >> texel) & 1) != 0);
        members[part][counts[part]] = texel;
        counts[part] += 1;
    }
    let fit0 = fit_plane(
        colors,
        &members[0][..counts[0]],
        channels,
        (color_range, weight_range),
        true,
        bound,
    );
    if fit0.error >= bound {
        return None;
    }
    let fit1 = fit_plane(
        colors,
        &members[1][..counts[1]],
        channels,
        (color_range, weight_range),
        true,
        bound - fit0.error,
    );
    let total = fit0.error + fit1.error;
    if total >= bound {
        return None;
    }
    let packed = pack_two_partitions(weight_range, cem, seed, mask, color_range, &[fit0, fit1]);
    Some((total, packed))
}

fn search_block(texels: &[[u8; 4]; 16], good_enough: u32) -> [u8; 16] {
    if texels.iter().all(|&color| color == texels[0]) {
        return encode_void_extent(texels[0]);
    }
    let colors: Texels = texels.map(|texel| texel.map(f32::from));
    let opaque = texels.iter().all(|texel| texel[3] == 255);
    let cem = if opaque { 8 } else { 12 };
    let channels: &[usize] = if opaque { &[0, 1, 2] } else { &[0, 1, 2, 3] };
    let all: [usize; 16] = std::array::from_fn(|i| i);
    let mut best = (u32::MAX, [0_u8; 16]);
    let single_ranges: &[usize] = if opaque { &[8, 7, 5] } else { &[8, 5, 4] };
    let value_count = if opaque { 6 } else { 8 };
    for &wr in single_ranges {
        let weight_bits = WEIGHT_RANGES[wr].stream_bits(16);
        let Some(cr) = color_range_for(128 - 17 - weight_bits, value_count) else {
            continue;
        };
        let fit = fit_plane(&colors, &all, channels, (cr, wr), true, best.0);
        if fit.error < best.0 {
            best = (fit.error, pack_single_plane(wr, cem, cr, &fit));
        }
    }
    if best.0 <= good_enough {
        return best.1;
    }
    let dual_channels: &[usize] = if opaque { &[0, 1, 2] } else { &[3, 0, 1, 2] };
    let dual_ranges: &[usize] = if opaque { &[3, 2, 4] } else { &[2, 1] };
    for &ccs in dual_channels {
        for &wr in dual_ranges {
            if let Some(candidate) = try_dual_plane(&colors, opaque, ccs, wr, best.0) {
                best = candidate;
            }
        }
    }
    if best.0 <= good_enough {
        return best.1;
    }
    let top_seeds = rank_seeds(&colors, channels, 4);
    let part_ranges: &[usize] = if opaque { &[3, 2, 5] } else { &[2] };
    for &(seed, mask) in &top_seeds {
        if mask == 0 {
            continue;
        }
        for &wr in part_ranges {
            if let Some(candidate) = try_two_partitions(&colors, opaque, (seed, mask), wr, best.0) {
                best = candidate;
            }
        }
    }
    best.1
}

pub(crate) fn encode_block(texels: &[[u8; 4]; 16]) -> [u8; 16] {
    search_block(texels, 64)
}

pub(crate) fn encode_block_normal(texels: &[[u8; 4]; 16]) -> [u8; 16] {
    search_block(texels, 16)
}

fn alpha_kept(block: &[u8; 16], texels: &[[u8; 4]; 16]) -> bool {
    decode_block_linear(block)
        .iter()
        .zip(texels)
        .all(|(read, held)| read[3] == held[3])
}

pub(crate) fn encode_block_exact_alpha(texels: &[[u8; 4]; 16]) -> [u8; 16] {
    let best = encode_block(texels);
    if texels.iter().all(|texel| texel[3] == 255) || alpha_kept(&best, texels) {
        return best;
    }
    let colors: Texels = texels.map(|texel| texel.map(f32::from));
    let mut exact: Option<(u32, [u8; 16])> = None;
    for wr in [0_usize, 1, 2] {
        let bound = exact.map_or(u32::MAX, |(err, _)| err);
        if let Some((err, candidate)) = try_dual_plane(&colors, false, 3, wr, bound)
            && alpha_kept(&candidate, texels)
        {
            exact = Some((err, candidate));
        }
    }
    exact.map_or(best, |(_, block)| block)
}
