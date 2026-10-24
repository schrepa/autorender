//! ASTC 4 × 4, low dynamic range: blocks of 4 × 4 texels in 128 bits (the
//! Khronos Data Format Specification's ASTC). The encoder writes blocks of
//! one partition over a 4 × 4 grid of weights, one per texel: colors on a
//! line between two endpoints (RGB direct, or RGBA direct where alpha
//! varies), or with one channel on a second plane of weights of its own;
//! and blocks of one color as void extents, exactly. It tries each layout
//! of weight and endpoint precision a block's colors suit, endpoints from
//! the principal axis refined by least squares, and keeps the least
//! squared error as decoders reconstruct it. Integer and plain `f32`
//! arithmetic only: the same blocks on every host.

use std::sync::OnceLock;

/// How a range's values are stored: bits alone, or a trit or a quint above
/// them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Bits,
    Trits,
    Quints,
}

/// A range of values the block stores in integer sequences.
#[derive(Clone, Copy, Debug)]
struct Range {
    kind: Kind,
    bits: u32,
}

impl Range {
    const fn new(kind: Kind, bits: u32) -> Self {
        Self { kind, bits }
    }

    fn levels(self) -> u32 {
        let digit = match self.kind {
            Kind::Bits => 1,
            Kind::Trits => 3,
            Kind::Quints => 5,
        };
        digit << self.bits
    }

    /// Bits `count` values take.
    fn sequence_bits(self, count: u32) -> u32 {
        count * self.bits
            + match self.kind {
                Kind::Bits => 0,
                Kind::Trits => (count * 8).div_ceil(5),
                Kind::Quints => (count * 7).div_ceil(3),
            }
    }

    /// Weight `symbol`'s value, 0 to 64.
    fn weight(self, symbol: u32) -> u32 {
        let digit = symbol >> self.bits;
        let low = symbol & ((1 << self.bits) - 1);
        let value = match (self.kind, self.bits) {
            (Kind::Bits, 1) => low * 63,
            (Kind::Bits, 2) => (low << 4) | (low << 2) | low,
            (Kind::Bits, 3) => (low << 3) | low,
            (Kind::Bits, 4) => (low << 2) | (low >> 2),
            (Kind::Bits, _) => (low << 1) | (low >> 4),
            (Kind::Trits, 0) => return digit * 32,
            (Kind::Quints, 0) => return digit * 16,
            (Kind::Trits | Kind::Quints, _) => {
                let scrambled = match (self.kind, self.bits) {
                    (Kind::Trits, 1) => digit * 50,
                    (Kind::Trits, 2) => digit * 23 + if low & 2 != 0 { 0b100_0101 } else { 0 },
                    (Kind::Trits, _) => digit * 11 + (((low << 4) | (low >> 1)) & 0b110_0011),
                    (Kind::Quints, 1) => digit * 28,
                    (_, _) => digit * 13 + if low & 2 != 0 { 0b100_0010 } else { 0 },
                };
                let a = (low & 1) * 0x7f;
                (a & 0x20) | ((scrambled ^ a) >> 2)
            }
        };
        if value > 32 { value + 1 } else { value }
    }

    /// Endpoint `symbol`'s value, 0 to 255.
    fn endpoint(self, symbol: u32) -> u32 {
        let digit = symbol >> self.bits;
        let low = symbol & ((1 << self.bits) - 1);
        match self.kind {
            Kind::Bits => match self.bits {
                1 => low * 0xff,
                2 => low * 0x55,
                3 => (low << 5) | (low << 2) | (low >> 1),
                4 => (low << 4) | low,
                5 => (low << 3) | (low >> 2),
                6 => (low << 2) | (low >> 4),
                7 => (low << 1) | (low >> 6),
                _ => low,
            },
            Kind::Trits | Kind::Quints => {
                let a = (low & 1) * 0x1ff;
                let x = low >> 1;
                let (b, c) = match (self.kind, self.bits) {
                    (Kind::Trits, 1) => (0, 204),
                    (Kind::Trits, 2) => (0b1_0001_0110 * x, 93),
                    (Kind::Trits, 3) => ((x << 7) | (x << 2) | x, 44),
                    (Kind::Trits, 4) => ((x << 6) | x, 22),
                    (Kind::Trits, 5) => ((x << 5) | (x >> 2), 11),
                    (Kind::Trits, _) => ((x << 4) | (x >> 4), 5),
                    (_, 1) => (0, 113),
                    (_, 2) => (0b1_0000_1100 * x, 54),
                    (_, 3) => ((x << 7) | (x << 1) | (x >> 1), 26),
                    (_, 4) => ((x << 6) | (x >> 1), 13),
                    (_, _) => ((x << 5) | (x >> 3), 6),
                };
                (a & 0x80) | (((digit * c + b) ^ a) >> 2)
            }
        }
    }
}

/// Weight ranges by their index in block modes (invalid indices hold the
/// smallest).
const WEIGHT_RANGES: [Range; 16] = [
    Range::new(Kind::Bits, 1),
    Range::new(Kind::Bits, 1),
    Range::new(Kind::Bits, 1),
    Range::new(Kind::Trits, 0),
    Range::new(Kind::Bits, 2),
    Range::new(Kind::Quints, 0),
    Range::new(Kind::Trits, 1),
    Range::new(Kind::Bits, 3),
    Range::new(Kind::Bits, 1),
    Range::new(Kind::Bits, 1),
    Range::new(Kind::Quints, 1),
    Range::new(Kind::Trits, 2),
    Range::new(Kind::Bits, 4),
    Range::new(Kind::Quints, 2),
    Range::new(Kind::Trits, 3),
    Range::new(Kind::Bits, 5),
];

/// Endpoint ranges, most levels first: a block's is the first whose values
/// fit the bits its weights and layout leave.
const ENDPOINT_RANGES: [Range; 19] = [
    Range::new(Kind::Bits, 8),
    Range::new(Kind::Trits, 6),
    Range::new(Kind::Quints, 5),
    Range::new(Kind::Bits, 7),
    Range::new(Kind::Trits, 5),
    Range::new(Kind::Quints, 4),
    Range::new(Kind::Bits, 6),
    Range::new(Kind::Trits, 4),
    Range::new(Kind::Quints, 3),
    Range::new(Kind::Bits, 5),
    Range::new(Kind::Trits, 3),
    Range::new(Kind::Quints, 2),
    Range::new(Kind::Bits, 4),
    Range::new(Kind::Trits, 2),
    Range::new(Kind::Quints, 1),
    Range::new(Kind::Bits, 3),
    Range::new(Kind::Trits, 1),
    Range::new(Kind::Bits, 2),
    Range::new(Kind::Bits, 1),
];

/// For each endpoint range, the symbol whose value lies nearest each byte.
fn nearest_endpoints() -> &'static [[u8; 256]; 19] {
    static TABLE: OnceLock<[[u8; 256]; 19]> = OnceLock::new();
    TABLE.get_or_init(|| {
        std::array::from_fn(|range| {
            let range = ENDPOINT_RANGES[range];
            std::array::from_fn(|target| {
                let mut best = (u32::MAX, 0);
                for symbol in 0..range.levels() {
                    let distance = range.endpoint(symbol).abs_diff(target as u32);
                    if distance < best.0 {
                        best = (distance, symbol);
                    }
                }
                best.1 as u8
            })
        })
    })
}

/// The trits of an 8-bit packed group, as the specification unpacks them.
fn trits_of(packed: u32) -> [u32; 5] {
    let bits = |high: u32, low: u32, of: u32| (of >> low) & ((1 << (high - low + 1)) - 1);
    let (c, t4, t3) = if bits(4, 2, packed) == 0b111 {
        ((bits(7, 5, packed) << 2) | bits(1, 0, packed), 2, 2)
    } else if bits(6, 5, packed) == 0b11 {
        (bits(4, 0, packed), 2, bits(7, 7, packed))
    } else {
        (bits(4, 0, packed), bits(7, 7, packed), bits(6, 5, packed))
    };
    let (t2, t1, t0) = if bits(1, 0, c) == 0b11 {
        let c3 = bits(3, 3, c);
        (2, bits(4, 4, c), (c3 << 1) | (bits(2, 2, c) & !c3 & 1))
    } else if bits(3, 2, c) == 0b11 {
        (2, 2, bits(1, 0, c))
    } else {
        let c1 = bits(1, 1, c);
        (
            bits(4, 4, c),
            bits(3, 2, c),
            (c1 << 1) | (bits(0, 0, c) & !c1 & 1),
        )
    };
    [t0, t1, t2, t3, t4]
}

/// The quints of a 7-bit packed group, as the specification unpacks them.
fn quints_of(packed: u32) -> [u32; 3] {
    let bits = |high: u32, low: u32, of: u32| (of >> low) & ((1 << (high - low + 1)) - 1);
    if bits(2, 1, packed) == 0b11 && bits(6, 5, packed) == 0 {
        let q0 = bits(0, 0, packed);
        let high =
            (q0 << 2) | ((bits(4, 4, packed) & !q0 & 1) << 1) | (bits(3, 3, packed) & !q0 & 1);
        return [4, 4, high];
    }
    let (q2, c) = if bits(2, 1, packed) == 0b11 {
        let flipped = !bits(6, 5, packed) & 0b11;
        (
            4,
            (bits(4, 3, packed) << 3) | (flipped << 1) | bits(0, 0, packed),
        )
    } else {
        (bits(6, 5, packed), bits(4, 0, packed))
    };
    if bits(2, 0, c) == 0b101 {
        [bits(4, 3, c), 4, q2]
    } else {
        [bits(2, 0, c), bits(4, 3, c), q2]
    }
}

/// The packed group of each group of five trits (`t0 + 3 t1 + ... + 81
/// t4`) and of three quints (`q0 + 5 q1 + 25 q2`): the first packing
/// unpacking to it.
fn packings() -> &'static ([u8; 243], [u8; 125]) {
    static TABLES: OnceLock<([u8; 243], [u8; 125])> = OnceLock::new();
    TABLES.get_or_init(|| {
        let mut trits = [0_u8; 243];
        let mut seen = [false; 243];
        for packed in 0..256 {
            let [t0, t1, t2, t3, t4] = trits_of(packed);
            let index = (t0 + 3 * t1 + 9 * t2 + 27 * t3 + 81 * t4) as usize;
            if !seen[index] {
                seen[index] = true;
                trits[index] = packed as u8;
            }
        }
        let mut quints = [0_u8; 125];
        let mut seen = [false; 125];
        for packed in 0..128 {
            let [q0, q1, q2] = quints_of(packed);
            let index = (q0 + 5 * q1 + 25 * q2) as usize;
            if !seen[index] {
                seen[index] = true;
                quints[index] = packed as u8;
            }
        }
        (trits, quints)
    })
}

/// Bits written from the least significant up.
#[derive(Default)]
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

    /// `symbols` of `range` as an integer sequence: groups of five values
    /// sharing a packed trit group, or of three sharing a quint group, the
    /// last group cut to the bits its values take.
    fn sequence(&mut self, symbols: &[u32], range: Range) {
        let low = |symbol: u32| symbol & ((1 << range.bits) - 1);
        let digit = |symbol: u32| symbol >> range.bits;
        let (trits, quints) = packings();
        match range.kind {
            Kind::Bits => {
                for &symbol in symbols {
                    self.put(symbol, range.bits);
                }
            }
            Kind::Trits => {
                for group in symbols.chunks(5) {
                    let mut values = [0_u32; 5];
                    values[..group.len()].copy_from_slice(group);
                    let index = values
                        .iter()
                        .rev()
                        .fold(0, |index, &value| index * 3 + digit(value));
                    let packed = u32::from(trits[index as usize]);
                    let mut part = Writer::default();
                    let b = range.bits;
                    part.put(low(values[0]), b);
                    part.put(packed, 2);
                    part.put(low(values[1]), b);
                    part.put(packed >> 2, 2);
                    part.put(low(values[2]), b);
                    part.put(packed >> 4, 1);
                    part.put(low(values[3]), b);
                    part.put(packed >> 5, 2);
                    part.put(low(values[4]), b);
                    part.put(packed >> 7, 1);
                    let count = group.len() as u32;
                    let kept = count * b + (count * 8).div_ceil(5);
                    self.put_wide(part.bits, kept);
                }
            }
            Kind::Quints => {
                for group in symbols.chunks(3) {
                    let mut values = [0_u32; 3];
                    values[..group.len()].copy_from_slice(group);
                    let index = values
                        .iter()
                        .rev()
                        .fold(0, |index, &value| index * 5 + digit(value));
                    let packed = u32::from(quints[index as usize]);
                    let mut part = Writer::default();
                    let b = range.bits;
                    part.put(low(values[0]), b);
                    part.put(packed, 3);
                    part.put(low(values[1]), b);
                    part.put(packed >> 3, 2);
                    part.put(low(values[2]), b);
                    part.put(packed >> 5, 2);
                    let count = group.len() as u32;
                    let kept = count * b + (count * 7).div_ceil(3);
                    self.put_wide(part.bits, kept);
                }
            }
        }
    }

    fn put_wide(&mut self, value: u128, count: u32) {
        if count > 0 {
            self.bits |= (value & ((1_u128 << count) - 1)) << self.at;
            self.at += count;
        }
    }
}

/// A layout a block may take: its color endpoint mode (8, RGB direct; 12,
/// RGBA direct), its weights' range by index, and the channel on a second
/// plane of weights, if any.
#[derive(Clone, Copy)]
struct Layout {
    mode: u32,
    weights: u32,
    plane: Option<usize>,
}

impl Layout {
    fn weight_range(self) -> Range {
        WEIGHT_RANGES[self.weights as usize]
    }

    fn weight_count(self) -> u32 {
        if self.plane.is_some() { 32 } else { 16 }
    }

    fn weight_bits(self) -> u32 {
        self.weight_range().sequence_bits(self.weight_count())
    }

    fn values(self) -> u32 {
        if self.mode == 12 { 8 } else { 6 }
    }

    /// The endpoint range the bits left after the weights hold, by its
    /// index.
    fn endpoint_range(self) -> usize {
        let header = 17 + if self.plane.is_some() { 2 } else { 0 };
        let left = 128 - header - self.weight_bits();
        ENDPOINT_RANGES
            .iter()
            .position(|range| range.sequence_bits(self.values()) <= left)
            .unwrap_or(ENDPOINT_RANGES.len() - 1)
    }

    /// The 11 bits of its block mode: a 4 × 4 grid of weights in the
    /// layout `D H B B A A R0 0 0 R2 R1` (width B + 4, height A + 2).
    fn block_mode(self) -> u32 {
        let range = self.weights;
        let (r0, r1, r2, high) = (range & 1, (range >> 1) & 1, (range >> 2) & 1, range >> 3);
        let dual = u32::from(self.plane.is_some());
        r1 | (r2 << 1) | (r0 << 4) | (2 << 5) | (high << 9) | (dual << 10)
    }
}

/// Layouts tried for opaque blocks and for blocks whose alpha varies.
const OPAQUE: [Layout; 6] = [
    Layout {
        mode: 8,
        weights: 12,
        plane: None,
    },
    Layout {
        mode: 8,
        weights: 11,
        plane: None,
    },
    Layout {
        mode: 8,
        weights: 7,
        plane: None,
    },
    Layout {
        mode: 8,
        weights: 4,
        plane: Some(0),
    },
    Layout {
        mode: 8,
        weights: 4,
        plane: Some(1),
    },
    Layout {
        mode: 8,
        weights: 4,
        plane: Some(2),
    },
];
const ALPHA: [Layout; 7] = [
    Layout {
        mode: 12,
        weights: 7,
        plane: None,
    },
    Layout {
        mode: 12,
        weights: 11,
        plane: None,
    },
    Layout {
        mode: 12,
        weights: 12,
        plane: None,
    },
    Layout {
        mode: 12,
        weights: 4,
        plane: Some(0),
    },
    Layout {
        mode: 12,
        weights: 4,
        plane: Some(1),
    },
    Layout {
        mode: 12,
        weights: 4,
        plane: Some(2),
    },
    Layout {
        mode: 12,
        weights: 4,
        plane: Some(3),
    },
];

/// A layout's fit: each channel's two endpoint symbols and their values,
/// each texel's weight symbols (the second plane's after the first's) and
/// the squared error decoders reconstruct.
#[derive(Clone, Copy)]
struct Fit {
    symbols: [[u32; 2]; 4],
    values: [[u32; 2]; 4],
    weights: [[u32; 2]; 16],
    error: u32,
}

/// A channel of a decoded texel: the decoders' 16-bit interpolation of
/// endpoint values `low` and `high` at weight `weight` (0 to 64), read as a
/// byte.
fn decoded(low: u32, high: u32, weight: u32) -> u32 {
    let interpolated =
        (((low << 8) | low) * (64 - weight) + ((high << 8) | high) * weight + 32) >> 6;
    (interpolated * 255 + 32_768) / 65_536
}

/// The best fit of `texels` in `layout` from endpoints `ends` (per
/// channel): endpoints quantized and ordered as the mode requires, then
/// each texel's weights on each plane picked by the error they leave.
fn assign(texels: &[[f32; 4]; 16], layout: Layout, ends: [[f32; 2]; 4]) -> Fit {
    let channels = if layout.mode == 12 { 4 } else { 3 };
    let range_index = layout.endpoint_range();
    let nearest = &nearest_endpoints()[range_index];
    let range = ENDPOINT_RANGES[range_index];
    let mut fit = Fit {
        symbols: [[0; 2]; 4],
        values: [[255; 2]; 4],
        weights: [[0; 2]; 16],
        error: 0,
    };
    for (channel, channel_ends) in ends.iter().enumerate().take(channels) {
        for (end, &target) in channel_ends.iter().enumerate() {
            let target = engine_math::clamp(target + 0.5, 0.0, 255.0) as usize;
            let symbol = u32::from(nearest[target]);
            fit.symbols[channel][end] = symbol;
            fit.values[channel][end] = range.endpoint(symbol);
        }
    }
    // Decoders swap endpoints whose second sums less (RGB), contracting
    // blue: the second end sums at least as much.
    let sum = |end: usize| (0..3).map(|channel| fit.values[channel][end]).sum::<u32>();
    if sum(0) > sum(1) {
        for channel in 0..4 {
            fit.symbols[channel].swap(0, 1);
            fit.values[channel].swap(0, 1);
        }
    }
    let weights = layout.weight_range();
    let levels: Vec<(u32, u32)> = (0..weights.levels())
        .map(|symbol| (symbol, weights.weight(symbol)))
        .collect();
    for (texel, color) in texels.iter().enumerate() {
        for plane in 0..1 + usize::from(layout.plane.is_some()) {
            let members = |channel: usize| match layout.plane {
                Some(apart) => (channel == apart) == (plane == 1),
                None => true,
            };
            let mut best = (u32::MAX, 0);
            for &(symbol, weight) in &levels {
                let mut error = 0;
                for channel in (0..channels).filter(|&channel| members(channel)) {
                    let [low, high] = fit.values[channel];
                    let difference = decoded(low, high, weight) as i32 - color[channel] as i32;
                    error += (difference * difference) as u32;
                }
                if error < best.0 {
                    best = (error, symbol);
                }
            }
            fit.weights[texel][plane] = best.1;
            fit.error += best.0;
        }
        if channels == 3 {
            let difference = 255 - color[3] as i32;
            fit.error += (difference * difference) as u32;
        }
    }
    fit
}

/// The endpoints least squares find for each channel given the weights
/// `fit` picks on that channel's plane; a channel whose texels all take one
/// weight keeps `ends`.
fn least_squares(
    texels: &[[f32; 4]; 16],
    layout: Layout,
    fit: &Fit,
    ends: [[f32; 2]; 4],
) -> [[f32; 2]; 4] {
    let weights = layout.weight_range();
    let mut refined = ends;
    let channels = if layout.mode == 12 { 4 } else { 3 };
    for (channel, refined) in refined.iter_mut().enumerate().take(channels) {
        let plane = usize::from(layout.plane == Some(channel));
        let (mut aa, mut ab, mut bb, mut along_a, mut along_b) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for (texel, color) in texels.iter().enumerate() {
            let weight = weights.weight(fit.weights[texel][plane]) as f32 / 64.0;
            let rest = 1.0 - weight;
            aa += rest * rest;
            ab += rest * weight;
            bb += weight * weight;
            along_a += rest * color[channel];
            along_b += weight * color[channel];
        }
        let determinant = aa * bb - ab * ab;
        if determinant.abs() < 1.0e-6 {
            continue;
        }
        *refined = [
            engine_math::clamp((bb * along_a - ab * along_b) / determinant, 0.0, 255.0),
            engine_math::clamp((aa * along_b - ab * along_a) / determinant, 0.0, 255.0),
        ];
    }
    refined
}

/// The ends of `texels`' colors in `layout`: the line through the first
/// plane's channels along their principal axis, and the second plane's
/// channel from its least to its most.
fn starting_ends(texels: &[[f32; 4]; 16], layout: Layout) -> [[f32; 2]; 4] {
    let channels = if layout.mode == 12 { 4 } else { 3 };
    let on_line: Vec<usize> = (0..channels)
        .filter(|&channel| layout.plane != Some(channel))
        .collect();
    let mut mean = [0.0_f32; 4];
    for color in texels {
        for &channel in &on_line {
            mean[channel] += color[channel] / 16.0;
        }
    }
    let mut covariance = [[0.0_f32; 4]; 4];
    for color in texels {
        for &row in &on_line {
            for &column in &on_line {
                covariance[row][column] +=
                    (color[row] - mean[row]) * (color[column] - mean[column]);
            }
        }
    }
    let mut axis = [0.0_f32; 4];
    let widest = on_line.iter().copied().fold(on_line[0], |widest, channel| {
        if covariance[channel][channel] > covariance[widest][widest] {
            channel
        } else {
            widest
        }
    });
    axis[widest] = 1.0;
    for _ in 0..8 {
        let mut next = [0.0_f32; 4];
        for &row in &on_line {
            for &column in &on_line {
                next[row] += covariance[row][column] * axis[column];
            }
        }
        let length = next.iter().map(|value| value * value).sum::<f32>().sqrt();
        if length < 1.0e-12 {
            break;
        }
        axis = next.map(|value| value / length);
    }
    // The line ascends in red, green and blue, as the mode orders it.
    if axis[0] + axis[1] + axis[2] < 0.0 {
        axis = axis.map(|value| -value);
    }
    let (mut low, mut high) = (f32::MAX, f32::MIN);
    for color in texels {
        let along: f32 = on_line
            .iter()
            .map(|&channel| (color[channel] - mean[channel]) * axis[channel])
            .sum();
        low = engine_math::min(low, along);
        high = engine_math::max(high, along);
    }
    let mut ends = [[255.0_f32; 2]; 4];
    for &channel in &on_line {
        ends[channel] = [
            mean[channel] + axis[channel] * low,
            mean[channel] + axis[channel] * high,
        ];
    }
    if let Some(apart) = layout.plane {
        let (mut least, mut most) = (f32::MAX, f32::MIN);
        for color in texels {
            least = engine_math::min(least, color[apart]);
            most = engine_math::max(most, color[apart]);
        }
        ends[apart] = [least, most];
    }
    ends
}

/// The best fit of `texels` in `layout`: from its starting ends, refined
/// twice by least squares while that helps.
fn fit(texels: &[[f32; 4]; 16], layout: Layout) -> Fit {
    let mut ends = starting_ends(texels, layout);
    let mut best = assign(texels, layout, ends);
    for _ in 0..2 {
        if best.error == 0 {
            break;
        }
        ends = least_squares(texels, layout, &best, ends);
        let refined = assign(texels, layout, ends);
        if refined.error >= best.error {
            break;
        }
        best = refined;
    }
    best
}

/// `fit`'s block in `layout`: block mode, one partition, its color
/// endpoint mode, the endpoints' integer sequence, then from the top down
/// the weights' (bit-reversed), the second plane's channel just below
/// them.
fn pack(layout: Layout, fit: &Fit) -> [u8; 16] {
    let mut out = Writer::default();
    out.put(layout.block_mode(), 11);
    out.put(0, 2);
    out.put(layout.mode, 4);
    let channels = if layout.mode == 12 { 4 } else { 3 };
    let endpoints: Vec<u32> = (0..channels)
        .flat_map(|channel| fit.symbols[channel])
        .collect();
    out.sequence(&endpoints, ENDPOINT_RANGES[layout.endpoint_range()]);
    let planes = 1 + usize::from(layout.plane.is_some());
    let weights: Vec<u32> = fit
        .weights
        .iter()
        .flat_map(|texel| texel.iter().take(planes).copied())
        .collect();
    let mut sequence = Writer::default();
    sequence.sequence(&weights, layout.weight_range());
    let weight_bits = layout.weight_bits();
    let mut bits = out.bits | sequence.bits.reverse_bits();
    if let Some(apart) = layout.plane {
        bits |= (apart as u128) << (128 - weight_bits - 2);
    }
    bits.to_le_bytes()
}

/// A void-extent block: every texel `color`.
fn solid(color: [u8; 4]) -> [u8; 16] {
    let mut block = [0xff_u8; 16];
    block[0] = 0xfc;
    block[1] = 0xfd;
    for (channel, &value) in color.iter().enumerate() {
        let wide = u16::from(value) * 257;
        block[8 + 2 * channel..10 + 2 * channel].copy_from_slice(&wide.to_le_bytes());
    }
    block
}

/// The block of `texels` (RGBA8, rows from the top) with the least squared
/// error over the layouts tried.
pub(crate) fn encode_block(texels: &[[u8; 4]; 16]) -> [u8; 16] {
    if texels.iter().all(|texel| *texel == texels[0]) {
        return solid(texels[0]);
    }
    let colors = texels.map(|texel| texel.map(f32::from));
    let layouts: &[Layout] = if texels.iter().all(|texel| texel[3] == 255) {
        &OPAQUE
    } else {
        &ALPHA
    };
    let mut best: Option<(Layout, Fit)> = None;
    for &layout in layouts {
        let fitted = fit(&colors, layout);
        if best.is_none_or(|(_, best)| fitted.error < best.error) {
            best = Some((layout, fitted));
        }
    }
    match best {
        Some((layout, fitted)) => pack(layout, &fitted),
        None => solid(texels[0]),
    }
}
