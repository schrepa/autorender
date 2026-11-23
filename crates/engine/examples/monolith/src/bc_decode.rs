use engine_math::{clamp, max};

const WEIGHTS: [&[u32]; 3] = [
    &[0, 21, 43, 64],
    &[0, 9, 18, 27, 37, 46, 55, 64],
    &[0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64],
];

#[derive(Clone, Copy)]
struct Mode {
    subsets: usize,
    partition_bits: u32,
    rotation_bits: u32,
    selection_bits: u32,
    color_bits: u32,
    alpha_bits: u32,
    parity: u8,
    index_bits: [u32; 2],
}

const MODES: [Mode; 8] = [
    Mode { subsets: 3, partition_bits: 4, rotation_bits: 0, selection_bits: 0, color_bits: 4, alpha_bits: 0, parity: 1, index_bits: [3, 0] },
    Mode { subsets: 2, partition_bits: 6, rotation_bits: 0, selection_bits: 0, color_bits: 6, alpha_bits: 0, parity: 2, index_bits: [3, 0] },
    Mode { subsets: 3, partition_bits: 6, rotation_bits: 0, selection_bits: 0, color_bits: 5, alpha_bits: 0, parity: 0, index_bits: [2, 0] },
    Mode { subsets: 2, partition_bits: 6, rotation_bits: 0, selection_bits: 0, color_bits: 7, alpha_bits: 0, parity: 1, index_bits: [2, 0] },
    Mode { subsets: 1, partition_bits: 0, rotation_bits: 2, selection_bits: 1, color_bits: 5, alpha_bits: 6, parity: 0, index_bits: [2, 3] },
    Mode { subsets: 1, partition_bits: 0, rotation_bits: 2, selection_bits: 0, color_bits: 7, alpha_bits: 8, parity: 0, index_bits: [2, 2] },
    Mode { subsets: 1, partition_bits: 0, rotation_bits: 0, selection_bits: 0, color_bits: 7, alpha_bits: 7, parity: 1, index_bits: [4, 0] },
    Mode { subsets: 2, partition_bits: 6, rotation_bits: 0, selection_bits: 0, color_bits: 5, alpha_bits: 5, parity: 1, index_bits: [2, 0] },
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
    0xaa68_5050, 0x6a5a_5040, 0x5a5a_4200, 0x5450_a0a8, 0xa5a5_0000, 0xa0a0_5050, 0x5555_a0a0,
    0x5a5a_5050, 0xaa55_0000, 0xaa55_5500, 0xaaaa_5500, 0x9090_9090, 0x9494_9494, 0xa4a4_a4a4,
    0xa9a5_9450, 0x2a0a_4250, 0xa594_5040, 0x0a42_5054, 0xa5a5_a500, 0x55a0_a0a0, 0xa8a8_5454,
    0x6a6a_4040, 0xa4a4_5000, 0x1a1a_0500, 0x0050_a4a4, 0xaaa5_9090, 0x1469_6914, 0x6969_1400,
    0xa085_85a0, 0xaa82_1414, 0x50a4_a450, 0x6a5a_0200, 0xa9a5_8000, 0x5090_a0a8, 0xa8a0_9050,
    0x2424_2424, 0x00aa_5500, 0x2492_4924, 0x2449_9224, 0x50a5_0a50, 0x500a_a550, 0xaaaa_4444,
    0x6666_0000, 0xa5a0_a5a0, 0x50a0_50a0, 0x6928_6928, 0x44aa_aa44, 0x6666_6600, 0xaa44_4444,
    0x54a8_54a8, 0x9580_9580, 0x9696_9600, 0xa854_54a8, 0x8095_9580, 0xaa14_1414, 0x9696_0000,
    0xaaaa_1414, 0xa050_50a0, 0xa0a5_a5a0, 0x9600_0000, 0x4080_4080, 0xa9a8_a9a8, 0xaaaa_aa44,
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

struct BitReader {
    bits: u128,
    at: u32,
}

impl BitReader {
    fn take(&mut self, count: u32) -> u32 {
        let value = ((self.bits >> self.at) & ((1_u128 << count) - 1)) as u32;
        self.at += count;
        value
    }
}

fn decode_bc7_block(block: &[u8; 16]) -> [[u8; 4]; 16] {
    let bits = u128::from_le_bytes(*block);
    let number = bits.trailing_zeros() as usize;
    if number >= 8 {
        return [[0; 4]; 16];
    }
    let mode = MODES[number];
    let mut read = BitReader {
        bits,
        at: number as u32 + 1,
    };
    let partition = read.take(mode.partition_bits) as usize;
    let rotation = read.take(mode.rotation_bits);
    let selection = read.take(mode.selection_bits);
    let mut ends = [[[0_u32; 4]; 2]; 3];
    for channel in 0..4 {
        let cb = if channel < 3 {
            mode.color_bits
        } else {
            mode.alpha_bits
        };
        for subset_ends in ends.iter_mut().take(mode.subsets) {
            for end in subset_ends.iter_mut() {
                end[channel] = read.take(cb);
            }
        }
    }
    let parity_bits = u32::from(mode.parity != 0);
    for subset_ends in ends.iter_mut().take(mode.subsets) {
        let parity = match mode.parity {
            1 => [read.take(1), read.take(1)],
            2 => {
                let bit = read.take(1);
                [bit, bit]
            }
            _ => [0, 0],
        };
        for (end, bit) in subset_ends.iter_mut().zip(parity) {
            for (channel, value) in end.iter_mut().enumerate() {
                let cb = if channel < 3 {
                    mode.color_bits
                } else {
                    mode.alpha_bits
                };
                *value = if cb == 0 {
                    255
                } else {
                    expand((*value << parity_bits) | bit, cb + parity_bits)
                };
            }
        }
    }
    let anchors: [usize; 3] = std::array::from_fn(|s| {
        if s < mode.subsets {
            anchor_of(mode.subsets, partition, s)
        } else {
            usize::MAX
        }
    });
    let mut first = [0_u32; 16];
    for (texel, index) in first.iter_mut().enumerate() {
        let is_anchor = anchors[..mode.subsets].contains(&texel);
        *index = read.take(mode.index_bits[0] - u32::from(is_anchor));
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
    let c_weights = WEIGHTS[(color_bits - 2) as usize];
    let a_weights = WEIGHTS[(alpha_bits - 2) as usize];
    std::array::from_fn(|texel| {
        let subset = subset_of(mode.subsets, partition, texel);
        let [low, high] = ends[subset];
        let cw = c_weights[color_indices[texel] as usize];
        let aw = a_weights[alpha_indices[texel] as usize];
        let mix = |ch: usize, w: u32| (((64 - w) * low[ch] + w * high[ch] + 32) >> 6) as u8;
        let mut out = [mix(0, cw), mix(1, cw), mix(2, cw), mix(3, aw)];
        if rotation > 0 {
            out.swap(3, rotation as usize - 1);
        }
        out
    })
}

fn bc5_palette(first: u8, second: u8) -> [u8; 8] {
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

fn decode_bc5_channel(block: &[u8]) -> [u8; 16] {
    let colors = bc5_palette(block[0], block[1]);
    let mut word = [0_u8; 8];
    word[..6].copy_from_slice(&block[2..8]);
    let bits = u64::from_le_bytes(word);
    std::array::from_fn(|index| colors[((bits >> (3 * index)) & 7) as usize])
}

fn decode_bc5_block(block: &[u8; 16]) -> [[u8; 4]; 16] {
    let red = decode_bc5_channel(&block[..8]);
    let green = decode_bc5_channel(&block[8..16]);
    std::array::from_fn(|index| {
        let x = f32::from(red[index]) / 255.0 * 2.0 - 1.0;
        let y = f32::from(green[index]) / 255.0 * 2.0 - 1.0;
        let z = max(1.0 - x * x - y * y, 0.0).sqrt();
        let zb = (clamp(z * 0.5 + 0.5, 0.0, 1.0) * 255.0).round() as u8;
        [red[index], green[index], zb, 255]
    })
}

fn decode_blocks(dim: u32, data: &[u8], decode: fn(&[u8; 16]) -> [[u8; 4]; 16]) -> Vec<u8> {
    let d = dim.max(1) as usize;
    let cols = d.div_ceil(4);
    let mut pixels = vec![0_u8; d * d * 4];
    for (idx, block) in data.as_chunks::<16>().0.iter().enumerate() {
        let (col, row) = (idx % cols, idx / cols);
        for (texel, rgba) in decode(block).iter().enumerate() {
            let (x, y) = (col * 4 + texel % 4, row * 4 + texel / 4);
            if x < d && y < d {
                let at = (y * d + x) * 4;
                pixels[at..at + 4].copy_from_slice(rgba);
            }
        }
    }
    pixels
}

#[must_use]
pub fn decode_bc7_level(dim: u32, data: &[u8]) -> Vec<u8> {
    decode_blocks(dim, data, decode_bc7_block)
}

#[must_use]
pub fn decode_bc5_level(dim: u32, data: &[u8]) -> Vec<u8> {
    decode_blocks(dim, data, decode_bc5_block)
}
