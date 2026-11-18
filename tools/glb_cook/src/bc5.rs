fn palette(first: u8, second: u8) -> [u8; 8] {
    let (a, b) = (u32::from(first), u32::from(second));
    let mut out = [first, second, 0, 0, 0, 0, 0, 0];
    if first > second {
        for step in 1..7_u32 {
            out[step as usize + 1] = ((a * (7 - step) + b * step + 3) / 7) as u8;
        }
    } else {
        for step in 1..5_u32 {
            out[step as usize + 1] = ((a * (5 - step) + b * step + 2) / 5) as u8;
        }
        out[6] = 0;
        out[7] = 255;
    }
    out
}

fn fitted(values: &[u8; 16], first: u8, second: u8) -> (u32, [u8; 16]) {
    let colors = palette(first, second);
    let mut error = 0;
    let mut indices = [0_u8; 16];
    for (index, &value) in values.iter().enumerate() {
        let (best, miss) = colors
            .iter()
            .enumerate()
            .map(|(at, &color)| (at, u32::from(color.abs_diff(value))))
            .min_by_key(|&(_, miss)| miss)
            .expect("palette_has_entries");
        indices[index] = best as u8;
        error += miss * miss;
    }
    (error, indices)
}

pub fn encode_channel(values: &[u8; 16]) -> [u8; 8] {
    let low = *values.iter().min().expect("sixteen_values");
    let high = *values.iter().max().expect("sixteen_values");
    let mut best = (u32::MAX, high, low, [0_u8; 16]);
    if low == high {
        best = (0, high, low, [0; 16]);
    } else {
        for inner_high in 0..=3_u8 {
            for inner_low in 0..=3_u8 {
                let first = high.saturating_sub(inner_high);
                let second = low.saturating_add(inner_low);
                if first <= second {
                    continue;
                }
                let (error, indices) = fitted(values, first, second);
                if error < best.0 {
                    best = (error, first, second, indices);
                }
            }
        }
    }
    let (_, first, second, indices) = best;
    let mut bits: u64 = 0;
    for (index, &value) in indices.iter().enumerate() {
        bits |= u64::from(value) << (3 * index);
    }
    let mut out = [0_u8; 8];
    out[0] = first;
    out[1] = second;
    out[2..].copy_from_slice(&bits.to_le_bytes()[..6]);
    out
}

pub fn decode_channel(block: &[u8]) -> [u8; 16] {
    let colors = palette(block[0], block[1]);
    let mut word = [0_u8; 8];
    word[..6].copy_from_slice(&block[2..8]);
    let bits = u64::from_le_bytes(word);
    std::array::from_fn(|index| colors[((bits >> (3 * index)) & 7) as usize])
}

pub fn encode_block(texels: &[[u8; 4]; 16]) -> [u8; 16] {
    let mut out = [0_u8; 16];
    for channel in 0..2 {
        let values: [u8; 16] = std::array::from_fn(|index| texels[index][channel]);
        out[channel * 8..channel * 8 + 8].copy_from_slice(&encode_channel(&values));
    }
    out
}

pub fn decode_block(block: &[u8]) -> [[u8; 4]; 16] {
    let red = decode_channel(&block[..8]);
    let green = decode_channel(&block[8..16]);
    std::array::from_fn(|index| {
        let x = f32::from(red[index]) / 255.0 * 2.0 - 1.0;
        let y = f32::from(green[index]) / 255.0 * 2.0 - 1.0;
        let z = engine_math::max(1.0 - x * x - y * y, 0.0).sqrt();
        [
            red[index],
            green[index],
            crate::mips::byte(z * 0.5 + 0.5),
            255,
        ]
    })
}
