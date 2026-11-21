use std::sync::OnceLock;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Range {
    pub(crate) max: u8,
    pub(crate) trits: bool,
    pub(crate) quints: bool,
    pub(crate) bits: u8,
}

impl Range {
    const fn new(max: u8, trits: bool, quints: bool, bits: u8) -> Self {
        Self {
            max,
            trits,
            quints,
            bits,
        }
    }

    pub(crate) const fn stream_bits(self, count: usize) -> usize {
        let extra = if self.trits {
            (count * 8).div_ceil(5)
        } else if self.quints {
            (count * 7).div_ceil(3)
        } else {
            0
        };
        count * self.bits as usize + extra
    }
}

pub(crate) const WEIGHT_RANGES: [Range; 12] = [
    Range::new(1, false, false, 1),
    Range::new(2, true, false, 0),
    Range::new(3, false, false, 2),
    Range::new(4, false, true, 0),
    Range::new(5, true, false, 1),
    Range::new(7, false, false, 3),
    Range::new(9, false, true, 1),
    Range::new(11, true, false, 2),
    Range::new(15, false, false, 4),
    Range::new(19, false, true, 2),
    Range::new(23, true, false, 3),
    Range::new(31, false, false, 5),
];

pub(crate) const COLOR_RANGES: [Range; 17] = [
    Range::new(5, true, false, 1),
    Range::new(7, false, false, 3),
    Range::new(9, false, true, 1),
    Range::new(11, true, false, 2),
    Range::new(15, false, false, 4),
    Range::new(19, false, true, 2),
    Range::new(23, true, false, 3),
    Range::new(31, false, false, 5),
    Range::new(39, false, true, 3),
    Range::new(47, true, false, 4),
    Range::new(63, false, false, 6),
    Range::new(79, false, true, 4),
    Range::new(95, true, false, 5),
    Range::new(127, false, false, 7),
    Range::new(159, false, true, 5),
    Range::new(191, true, false, 6),
    Range::new(255, false, false, 8),
];

pub(crate) const fn decode_trits(byte: u8) -> [u8; 5] {
    let (c, t3, t4) = if (byte >> 2) & 7 == 7 {
        (((byte >> 3) & 0x1C) | (byte & 3), 2, 2)
    } else {
        let c = byte & 0x1F;
        if (byte >> 5) & 3 == 3 {
            (c, (byte >> 7) & 1, 2)
        } else {
            (c, (byte >> 5) & 3, (byte >> 7) & 1)
        }
    };
    let (t0, t1, t2) = if c & 3 == 3 {
        let b3 = (c >> 3) & 1;
        let b2 = (c >> 2) & 1;
        ((b3 << 1) | (b2 & (1 ^ b3)), (c >> 4) & 1, 2)
    } else if (c >> 2) & 3 == 3 {
        (c & 3, 2, 2)
    } else {
        let b1 = (c >> 1) & 1;
        let b0 = c & 1;
        ((b1 << 1) | (b0 & (1 ^ b1)), (c >> 2) & 3, (c >> 4) & 1)
    };
    [t0, t1, t2, t3, t4]
}

pub(crate) const fn decode_quints(byte: u8) -> [u8; 3] {
    if ((byte >> 1) & 3) == 3 && ((byte >> 5) & 3) == 0 {
        let b0 = byte & 1;
        let b3 = (byte >> 3) & 1;
        let b4 = (byte >> 4) & 1;
        let q2 = (b0 << 2) | ((b4 & (1 ^ b0)) << 1) | (b3 & (1 ^ b0));
        return [4, 4, q2];
    }
    let (c, q2) = if ((byte >> 1) & 3) == 3 {
        let hi = ((!byte) >> 5) & 3;
        ((byte & 0x18) | (hi << 1) | (byte & 1), 4)
    } else {
        (byte & 0x1F, (byte >> 5) & 3)
    };
    if (c & 7) == 5 {
        [(c >> 3) & 3, 4, q2]
    } else {
        [c & 7, (c >> 3) & 3, q2]
    }
}

pub(crate) const TRIT_ENCODE: [u8; 243] = {
    let mut table = [0_u8; 243];
    let mut byte = 256_usize;
    while byte > 0 {
        byte -= 1;
        let t = decode_trits(byte as u8);
        if t[0] < 3 && t[1] < 3 && t[2] < 3 && t[3] < 3 && t[4] < 3 {
            let key = t[0] as usize
                + 3 * (t[1] as usize
                    + 3 * (t[2] as usize + 3 * (t[3] as usize + 3 * t[4] as usize)));
            table[key] = byte as u8;
        }
    }
    table
};

pub(crate) const QUINT_ENCODE: [u8; 125] = {
    let mut table = [0_u8; 125];
    let mut byte = 128_usize;
    while byte > 0 {
        byte -= 1;
        let q = decode_quints(byte as u8);
        if q[0] < 5 && q[1] < 5 && q[2] < 5 {
            let key = q[0] as usize + 5 * (q[1] as usize + 5 * q[2] as usize);
            table[key] = byte as u8;
        }
    }
    table
};

pub(crate) const fn unquantize_color(range_index: usize, value: u8) -> u8 {
    let range = COLOR_RANGES[range_index];
    let v = value as u32;
    if !range.trits && !range.quints {
        return match range.bits {
            8 => v as u8,
            7 => ((v << 1) | (v >> 6)) as u8,
            6 => ((v << 2) | (v >> 4)) as u8,
            5 => ((v << 3) | (v >> 2)) as u8,
            4 => ((v << 4) | v) as u8,
            3 => ((v << 5) | (v << 2) | (v >> 1)) as u8,
            _ => 0,
        };
    }
    let bits = v & ((1 << range.bits) - 1);
    let d = v >> range.bits;
    let a = if (bits & 1) != 0 { 0x1FF } else { 0 };
    let b = (bits >> 1) & 1;
    let c = (bits >> 2) & 1;
    let bd = (bits >> 3) & 1;
    let e = (bits >> 4) & 1;
    let f = (bits >> 5) & 1;
    let (big_b, factor) = match (range.trits, range.bits) {
        (true, 1) => (0, 204),
        (false, 1) => (0, 113),
        (true, 2) => ((b << 8) | (b << 4) | (b << 2) | (b << 1), 93),
        (false, 2) => ((b << 8) | (b << 3) | (b << 2), 54),
        (true, 3) => ((c << 8) | (b << 7) | (c << 3) | (b << 2) | (c << 1) | b, 44),
        (false, 3) => ((c << 8) | (b << 7) | (c << 2) | (b << 1) | c, 26),
        (true, 4) => ((bd << 8) | (c << 7) | (b << 6) | (bd << 2) | (c << 1) | b, 22),
        (false, 4) => ((bd << 8) | (c << 7) | (b << 6) | (bd << 1) | c, 13),
        (true, 5) => ((e << 8) | (bd << 7) | (c << 6) | (b << 5) | (e << 1) | bd, 11),
        (false, 5) => ((e << 8) | (bd << 7) | (c << 6) | (b << 5) | e, 6),
        (true, 6) => ((f << 8) | (e << 7) | (bd << 6) | (c << 5) | (b << 4) | f, 5),
        _ => (0, 0),
    };
    let mut t = d * factor + big_b;
    t ^= a;
    t = (a & 0x80) | (t >> 2);
    t as u8
}

pub(crate) const fn unquantize_weight(range_index: usize, value: u8) -> u8 {
    let range = WEIGHT_RANGES[range_index];
    let v = value as u32;
    let mut t = if !range.trits && !range.quints {
        match range.bits {
            1 => v * 63,
            2 => (v << 4) | (v << 2) | v,
            3 => (v << 3) | v,
            4 => (v << 2) | (v >> 2),
            5 => (v << 1) | (v >> 4),
            _ => 0,
        }
    } else if range.bits == 0 {
        if range.trits {
            match v {
                0 => 0,
                1 => 32,
                _ => 63,
            }
        } else {
            match v {
                0 => 0,
                1 => 16,
                2 => 32,
                3 => 47,
                _ => 63,
            }
        }
    } else {
        let bits = v & ((1 << range.bits) - 1);
        let d = v >> range.bits;
        let a = if (bits & 1) != 0 { 0x7F } else { 0 };
        let b = (bits >> 1) & 1;
        let c = (bits >> 2) & 1;
        let (big_b, factor) = match (range.trits, range.bits) {
            (true, 1) => (0, 50),
            (false, 1) => (0, 28),
            (true, 2) => ((b << 6) | (b << 2) | b, 23),
            (false, 2) => ((b << 6) | (b << 1), 13),
            (true, 3) => ((c << 6) | (b << 5) | (c << 1) | b, 11),
            _ => (0, 0),
        };
        let mut mixed = d * factor + big_b;
        mixed ^= a;
        (a & 0x20) | (mixed >> 2)
    };
    if t > 32 {
        t += 1;
    }
    t as u8
}

pub(crate) type ColorQuantTable = [[(u8, u8); 256]; 17];

pub(crate) fn color_quant() -> &'static ColorQuantTable {
    static TABLE: OnceLock<Box<ColorQuantTable>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = Box::new([[(0_u8, 0_u8); 256]; 17]);
        for (r, row) in table.iter_mut().enumerate() {
            let top = COLOR_RANGES[r].max;
            for (target, slot) in row.iter_mut().enumerate() {
                let (mut best_err, mut best_v, mut best_u) = (u32::MAX, 0_u8, 0_u8);
                for v in 0..=top {
                    let u = unquantize_color(r, v);
                    let err = u32::from(u.abs_diff(target as u8));
                    if err < best_err {
                        best_err = err;
                        best_v = v;
                        best_u = u;
                    }
                }
                *slot = (best_v, best_u);
            }
        }
        table
    })
}

pub(crate) const WEIGHT_QUANT: [[(u8, u8); 65]; 12] = {
    let mut table = [[(0_u8, 0_u8); 65]; 12];
    let mut r = 0;
    while r < 12 {
        let top = WEIGHT_RANGES[r].max as usize;
        let mut target = 0;
        while target <= 64 {
            let (mut best_err, mut best_v, mut best_u) = (u32::MAX, 0_u8, 0_u8);
            let mut v = 0;
            while v <= top {
                let u = unquantize_weight(r, v as u8);
                let err = u.abs_diff(target as u8) as u32;
                if err < best_err {
                    best_err = err;
                    best_v = v as u8;
                    best_u = u;
                }
                v += 1;
            }
            table[r][target] = (best_v, best_u);
            target += 1;
        }
        r += 1;
    }
    table
};

pub(crate) const WEIGHT_SORTED: [([(u8, u8); 32], usize); 12] = {
    let mut table = [([(0_u8, 0_u8); 32], 0_usize); 12];
    let mut r = 0;
    while r < 12 {
        let count = WEIGHT_RANGES[r].max as usize + 1;
        let mut items = [(0_u8, 0_u8); 32];
        let mut v = 0;
        while v < count {
            items[v] = (v as u8, unquantize_weight(r, v as u8));
            v += 1;
        }
        let mut i = 0;
        while i < count {
            let mut j = i + 1;
            while j < count {
                if items[j].1 < items[i].1 {
                    let held = items[i];
                    items[i] = items[j];
                    items[j] = held;
                }
                j += 1;
            }
            i += 1;
        }
        table[r] = (items, count);
        r += 1;
    }
    table
};

pub(crate) const fn color_range_for(remaining_bits: usize, value_count: usize) -> Option<usize> {
    let mut index = COLOR_RANGES.len();
    while index > 0 {
        index -= 1;
        if COLOR_RANGES[index].stream_bits(value_count) <= remaining_bits {
            return Some(index);
        }
    }
    None
}

const fn hash52(mut p: u32) -> u32 {
    p ^= p >> 15;
    p = p.wrapping_sub(p << 17);
    p = p.wrapping_add(p << 7);
    p = p.wrapping_add(p << 4);
    p ^= p >> 5;
    p = p.wrapping_add(p << 16);
    p ^= p >> 7;
    p ^= p >> 3;
    p ^= p << 6;
    p ^= p >> 17;
    p
}

const fn select_partition_4x4(seed: u32, x: u32, y: u32, partitions: u32) -> u32 {
    let x = x << 1;
    let y = y << 1;
    let seed = seed + (partitions - 1) * 1024;
    let rnum = hash52(seed);
    let mut s1 = rnum & 0xF;
    let mut s2 = (rnum >> 4) & 0xF;
    let mut s3 = (rnum >> 8) & 0xF;
    let mut s4 = (rnum >> 12) & 0xF;
    let mut s5 = (rnum >> 16) & 0xF;
    let mut s6 = (rnum >> 20) & 0xF;
    let mut s7 = (rnum >> 24) & 0xF;
    let mut s8 = (rnum >> 28) & 0xF;
    s1 *= s1;
    s2 *= s2;
    s3 *= s3;
    s4 *= s4;
    s5 *= s5;
    s6 *= s6;
    s7 *= s7;
    s8 *= s8;
    let (sh1, sh2) = if (seed & 1) != 0 {
        (
            if (seed & 2) != 0 { 4 } else { 5 },
            if partitions == 3 { 6 } else { 5 },
        )
    } else {
        (
            if partitions == 3 { 6 } else { 5 },
            if (seed & 2) != 0 { 4 } else { 5 },
        )
    };
    s1 >>= sh1;
    s2 >>= sh2;
    s3 >>= sh1;
    s4 >>= sh2;
    s5 >>= sh1;
    s6 >>= sh2;
    s7 >>= sh1;
    s8 >>= sh2;
    let a = (s1 * x + s2 * y + (rnum >> 14)) & 0x3F;
    let b = (s3 * x + s4 * y + (rnum >> 10)) & 0x3F;
    let c = if partitions >= 3 {
        (s5 * x + s6 * y + (rnum >> 6)) & 0x3F
    } else {
        0
    };
    let d = if partitions >= 4 {
        (s7 * x + s8 * y + (rnum >> 2)) & 0x3F
    } else {
        0
    };
    if a >= b && a >= c && a >= d {
        0
    } else if b >= c && b >= d {
        1
    } else if c >= d {
        2
    } else {
        3
    }
}

const fn partition_mask_2(seed: u32) -> u16 {
    let mut mask = 0_u16;
    let mut texel = 0_u32;
    while texel < 16 {
        let part = select_partition_4x4(seed, texel % 4, texel / 4, 2);
        if part == 1 {
            mask |= 1 << texel;
        }
        texel += 1;
    }
    mask
}

pub(crate) const SEEDS_2: [(u16, u16); 64] = {
    let mut out = [(0_u16, 0_u16); 64];
    let mut found = 0;
    let mut seed = 0_u32;
    while seed < 1024 && found < 64 {
        let mask = partition_mask_2(seed);
        let ones = mask.count_ones();
        if ones >= 2 && ones <= 14 {
            let canonical = if (mask & 1) != 0 { !mask } else { mask };
            let mut seen = false;
            let mut i = 0;
            while i < found {
                let held = out[i].1;
                let held_canonical = if (held & 1) != 0 { !held } else { held };
                if held_canonical == canonical {
                    seen = true;
                    break;
                }
                i += 1;
            }
            if !seen {
                out[found] = (seed as u16, mask);
                found += 1;
            }
        }
        seed += 1;
    }
    out
};

struct BitWriter {
    bits: u128,
    at: u32,
}

impl BitWriter {
    fn put(&mut self, value: u32, count: u32) {
        if count > 0 {
            self.bits |= (u128::from(value) & ((1_u128 << count) - 1)) << self.at;
            self.at += count;
        }
    }
}

struct BitReader {
    bits: u128,
    at: u32,
}

impl BitReader {
    fn take(&mut self, count: u32) -> u32 {
        if count == 0 {
            return 0;
        }
        let value = ((self.bits >> self.at) & ((1_u128 << count) - 1)) as u32;
        self.at += count;
        value
    }
}

pub(crate) fn pack_ise(range: Range, values: &[u8]) -> u128 {
    let mut out = BitWriter { bits: 0, at: 0 };
    let n = u32::from(range.bits);
    let mask = if n == 0 { 0 } else { (1_u8 << n) - 1 };
    if !range.trits && !range.quints {
        for &value in values {
            out.put(u32::from(value), n);
        }
        return out.bits;
    }
    if range.trits {
        for chunk in values.chunks(5) {
            let mut m = [0_u32; 5];
            let mut t = [0_usize; 5];
            for (i, &value) in chunk.iter().enumerate() {
                m[i] = u32::from(value & mask);
                t[i] = usize::from(value >> n);
            }
            let key = t[0] + 3 * (t[1] + 3 * (t[2] + 3 * (t[3] + 3 * t[4])));
            let byte = u32::from(TRIT_ENCODE[key]);
            let slots = [
                (m[0], byte & 3, 2),
                (m[1], (byte >> 2) & 3, 2),
                (m[2], (byte >> 4) & 1, 1),
                (m[3], (byte >> 5) & 3, 2),
                (m[4], (byte >> 7) & 1, 1),
            ];
            for &(low, high, high_bits) in &slots[..chunk.len()] {
                out.put(low, n);
                out.put(high, high_bits);
            }
        }
        return out.bits;
    }
    for chunk in values.chunks(3) {
        let mut m = [0_u32; 3];
        let mut q = [0_usize; 3];
        for (i, &value) in chunk.iter().enumerate() {
            m[i] = u32::from(value & mask);
            q[i] = usize::from(value >> n);
        }
        let key = q[0] + 5 * (q[1] + 5 * q[2]);
        let byte = u32::from(QUINT_ENCODE[key]);
        let slots = [
            (m[0], byte & 7, 3),
            (m[1], (byte >> 3) & 3, 2),
            (m[2], (byte >> 5) & 3, 2),
        ];
        for &(low, high, high_bits) in &slots[..chunk.len()] {
            out.put(low, n);
            out.put(high, high_bits);
        }
    }
    out.bits
}

pub(crate) fn unpack_ise(range: Range, stream: u128, count: usize, out: &mut [u8]) {
    let total = range.stream_bits(count) as u32;
    let masked = if total >= 128 {
        stream
    } else {
        stream & ((1_u128 << total) - 1)
    };
    let mut reader = BitReader {
        bits: masked,
        at: 0,
    };
    let n = u32::from(range.bits);
    if !range.trits && !range.quints {
        for slot in out.iter_mut().take(count) {
            *slot = reader.take(n) as u8;
        }
        return;
    }
    if range.trits {
        let mut index = 0;
        while index < count {
            let chunk = (count - index).min(5);
            let mut m = [0_u8; 5];
            let mut byte = 0_u32;
            let specs: [(u32, u32); 5] = [(0, 2), (2, 2), (4, 1), (5, 2), (7, 1)];
            for (i, &(shift, bits)) in specs.iter().enumerate().take(chunk) {
                m[i] = reader.take(n) as u8;
                byte |= reader.take(bits) << shift;
            }
            let t = decode_trits(byte as u8);
            for i in 0..chunk {
                out[index + i] = (t[i] << n) | m[i];
            }
            index += chunk;
        }
        return;
    }
    let mut index = 0;
    while index < count {
        let chunk = (count - index).min(3);
        let mut m = [0_u8; 3];
        let mut byte = 0_u32;
        let specs: [(u32, u32); 3] = [(0, 3), (3, 2), (5, 2)];
        for (i, &(shift, bits)) in specs.iter().enumerate().take(chunk) {
            m[i] = reader.take(n) as u8;
            byte |= reader.take(bits) << shift;
        }
        let q = decode_quints(byte as u8);
        for i in 0..chunk {
            out[index + i] = (q[i] << n) | m[i];
        }
        index += chunk;
    }
}

pub(crate) const fn block_mode_4x4(dual_plane: bool, weight_range: usize) -> u32 {
    let high = (weight_range >= 6) as u32;
    let r = (weight_range % 6) as u32 + 2;
    let r0 = r & 1;
    let r1 = (r >> 1) & 1;
    let r2 = (r >> 2) & 1;
    let d = dual_plane as u32;
    (d << 10) | (high << 9) | (2 << 5) | (r0 << 4) | (r2 << 1) | r1
}

struct BlockMode {
    width: usize,
    height: usize,
    dual_plane: bool,
    weight_range: usize,
}

fn decode_block_mode(mode: u32) -> Option<BlockMode> {
    let (width, height, dual_plane, high, r) = if (mode & 3) != 0 {
        let r = (((mode >> 1) & 1) << 2) | ((mode & 1) << 1) | ((mode >> 4) & 1);
        let a = ((mode >> 5) & 3) as usize;
        let b = ((mode >> 7) & 3) as usize;
        let d = (mode >> 10) & 1 != 0;
        let h = (mode >> 9) & 1 != 0;
        let (w, ht) = match (mode >> 2) & 3 {
            0 => (b + 4, a + 2),
            1 => (b + 8, a + 2),
            2 => (a + 2, b + 8),
            _ if (mode >> 8) & 1 == 0 => (a + 2, (((mode >> 7) & 1) as usize) + 6),
            _ => ((((mode >> 7) & 1) as usize) + 2, a + 2),
        };
        (w, ht, d, h, r)
    } else {
        if (mode & 0xF) == 0 {
            return None;
        }
        let r = (((mode >> 3) & 1) << 2) | (((mode >> 2) & 1) << 1) | ((mode >> 4) & 1);
        let a = ((mode >> 5) & 3) as usize;
        let d = (mode >> 10) & 1 != 0;
        let h = (mode >> 9) & 1 != 0;
        match (mode >> 7) & 3 {
            0 => (12, a + 2, d, h, r),
            1 => (a + 2, 12, d, h, r),
            2 => (a + 6, (((mode >> 9) & 3) as usize) + 6, false, false, r),
            _ => match a {
                0 => (6, 10, d, h, r),
                1 => (10, 6, d, h, r),
                _ => return None,
            },
        }
    };
    if r < 2 || width > 4 || height > 4 {
        return None;
    }
    let weight_range = (if high { 6 } else { 0 }) + (r - 2) as usize;
    Some(BlockMode {
        width,
        height,
        dual_plane,
        weight_range,
    })
}

fn blue_contract(r: u32, g: u32, b: u32, a: u32) -> [u32; 4] {
    [(r + b) >> 1, (g + b) >> 1, b, a]
}

fn decode_endpoints(cem: u32, values: &[u8]) -> Option<[[u32; 4]; 2]> {
    let v = |i: usize| u32::from(values[i]);
    match cem {
        8 => {
            let (s0, s1) = (v(0) + v(2) + v(4), v(1) + v(3) + v(5));
            if s1 >= s0 {
                Some([[v(0), v(2), v(4), 255], [v(1), v(3), v(5), 255]])
            } else {
                Some([
                    blue_contract(v(1), v(3), v(5), 255),
                    blue_contract(v(0), v(2), v(4), 255),
                ])
            }
        }
        12 => {
            let (s0, s1) = (v(0) + v(2) + v(4), v(1) + v(3) + v(5));
            if s1 >= s0 {
                Some([[v(0), v(2), v(4), v(6)], [v(1), v(3), v(5), v(7)]])
            } else {
                Some([
                    blue_contract(v(1), v(3), v(5), v(7)),
                    blue_contract(v(0), v(2), v(4), v(6)),
                ])
            }
        }
        _ => None,
    }
}

fn interpolate_channel(e0: u32, e1: u32, weight: u32, srgb: bool) -> u8 {
    let (c0, c1) = if srgb {
        ((e0 << 8) | 0x80, (e1 << 8) | 0x80)
    } else {
        ((e0 << 8) | e0, (e1 << 8) | e1)
    };
    let c16 = (c0 * (64 - weight) + c1 * weight + 32) >> 6;
    if srgb {
        (c16 >> 8) as u8
    } else {
        ((c16 * 255 + 32768) / 65535) as u8
    }
}

fn decode_void_extent(bits: u128, srgb: bool) -> [[u8; 4]; 16] {
    let error = [[255, 0, 255, 255]; 16];
    if ((bits >> 9) & 1) != 0 || ((bits >> 10) & 3) != 3 {
        return error;
    }
    let min_s = ((bits >> 12) & 0x1FFF) as u32;
    let max_s = ((bits >> 25) & 0x1FFF) as u32;
    let min_t = ((bits >> 38) & 0x1FFF) as u32;
    let max_t = ((bits >> 51) & 0x1FFF) as u32;
    let all_ones = min_s == 0x1FFF && max_s == 0x1FFF && min_t == 0x1FFF && max_t == 0x1FFF;
    if !all_ones && (min_s >= max_s || min_t >= max_t) {
        return error;
    }
    let color: [u8; 4] = std::array::from_fn(|channel| {
        let value = ((bits >> (64 + channel * 16)) & 0xFFFF) as u32;
        if srgb && channel < 3 {
            (value >> 8) as u8
        } else {
            ((value * 255 + 32768) / 65535) as u8
        }
    });
    [color; 16]
}

fn decode_block(block: &[u8; 16], srgb: bool) -> [[u8; 4]; 16] {
    let error = [[255, 0, 255, 255]; 16];
    let bits = u128::from_le_bytes(*block);
    if (bits & 0x1FF) == 0x1FC {
        return decode_void_extent(bits, srgb);
    }
    let Some(mode) = decode_block_mode((bits & 0x7FF) as u32) else {
        return error;
    };
    let partitions = (((bits >> 11) & 3) + 1) as usize;
    if partitions > 2 || (mode.dual_plane && partitions > 1) || mode.width != 4 || mode.height != 4 {
        return error;
    }
    let planes = if mode.dual_plane { 2 } else { 1 };
    let weight_count = 16 * planes;
    let weight_bits = WEIGHT_RANGES[mode.weight_range].stream_bits(weight_count);
    if !(24..=96).contains(&weight_bits) {
        return error;
    }
    let (cem, seed, config_bits, endpoint_start) = if partitions == 1 {
        let cem = ((bits >> 13) & 0xF) as u32;
        (cem, 0_u32, 17 + if mode.dual_plane { 2 } else { 0 }, 17_u32)
    } else {
        let seed = ((bits >> 13) & 0x3FF) as u32;
        let selector = ((bits >> 23) & 3) as u32;
        if selector != 0 {
            return error;
        }
        let cem = ((bits >> 25) & 0xF) as u32;
        (cem, seed, 29, 29_u32)
    };
    let values_per_part = match cem {
        8 => 6,
        12 => 8,
        _ => return error,
    };
    let total_values = values_per_part * partitions;
    let remaining = 128_usize.saturating_sub(config_bits + weight_bits);
    let Some(color_range) = color_range_for(remaining, total_values) else {
        return error;
    };
    let mut raw_endpoints = [0_u8; 16];
    unpack_ise(
        COLOR_RANGES[color_range],
        bits >> endpoint_start,
        total_values,
        &mut raw_endpoints,
    );
    for value in raw_endpoints.iter_mut().take(total_values) {
        *value = unquantize_color(color_range, *value);
    }
    let mut endpoints = [[[0_u32; 4]; 2]; 2];
    for part in 0..partitions {
        let slice = &raw_endpoints[part * values_per_part..(part + 1) * values_per_part];
        let Some(pair) = decode_endpoints(cem, slice) else {
            return error;
        };
        endpoints[part] = pair;
    }
    let mut raw_weights = [0_u8; 32];
    unpack_ise(
        WEIGHT_RANGES[mode.weight_range],
        bits.reverse_bits(),
        weight_count,
        &mut raw_weights,
    );
    for weight in raw_weights.iter_mut().take(weight_count) {
        *weight = unquantize_weight(mode.weight_range, *weight);
    }
    let ccs = if mode.dual_plane {
        ((bits >> (128 - weight_bits - 2)) & 3) as usize
    } else {
        4
    };
    std::array::from_fn(|texel| {
        let part = if partitions == 1 {
            0
        } else {
            select_partition_4x4(seed, (texel % 4) as u32, (texel / 4) as u32, 2) as usize
        };
        let [e0, e1] = endpoints[part];
        let w0 = u32::from(raw_weights[texel * planes]);
        let w1 = if mode.dual_plane {
            u32::from(raw_weights[texel * planes + 1])
        } else {
            w0
        };
        std::array::from_fn(|channel| {
            let weight = if channel == ccs { w1 } else { w0 };
            interpolate_channel(e0[channel], e1[channel], weight, srgb && channel < 3)
        })
    })
}

pub(crate) fn decode_block_linear(block: &[u8; 16]) -> [[u8; 4]; 16] {
    decode_block(block, false)
}

pub(crate) fn decode_block_srgb(block: &[u8; 16]) -> [[u8; 4]; 16] {
    decode_block(block, true)
}
