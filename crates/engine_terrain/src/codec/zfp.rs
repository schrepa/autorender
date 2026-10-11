//! Reversible float-1 blocks from LLNL zfp 1.0.1, commit
//! f40868a6a1c190c802e7d8b5987064f044bf7812: template/rev{en,de}code*.c,
//! encode.c, decode.c, codecf.c and traitsf.h. See zfp.LICENSE.
//!
//! Only complete, unconstrained bit planes are emitted. The bounded writer
//! abandons blocks exceeding the terrain format's 120-bit raw-escape limit.

use super::CodecError;

const NEGABINARY: u32 = 0xaaaa_aaaa;
const SIGN_MAGNITUDE: u32 = 0x7fff_ffff;

#[derive(Default)]
pub(super) struct Block {
    pub bytes: [u8; 15],
    pub bits: u8,
}

impl Block {
    fn put(&mut self, value: u32, count: u8) -> Option<()> {
        if usize::from(self.bits) + usize::from(count) > 120 {
            return None;
        }
        for i in 0..count {
            self.bytes[usize::from(self.bits / 8)] |= (((value >> i) & 1) as u8) << (self.bits % 8);
            self.bits += 1;
        }
        Some(())
    }

    pub fn payload(&self) -> &[u8] {
        &self.bytes[..usize::from(self.bits).div_ceil(8)]
    }
}

struct Bits<'a> {
    bytes: &'a [u8],
    count: u8,
    at: u8,
}

impl Bits<'_> {
    fn get(&mut self, count: u8) -> Result<u32, CodecError> {
        if usize::from(self.at) + usize::from(count) > usize::from(self.count) {
            return Err(CodecError("unfinished float block"));
        }
        let mut value = 0;
        for i in 0..count {
            value |= u32::from((self.bytes[usize::from(self.at / 8)] >> (self.at % 8)) & 1) << i;
            self.at += 1;
        }
        Ok(value)
    }
}

fn inverse_cast(values: [u32; 4], exponent: i32) -> [f32; 4] {
    if exponent == -127 {
        [0.0; 4]
    } else {
        let scale = libm::ldexpf(1.0, exponent - 30);
        values.map(|value| scale * (value as i32 as f32))
    }
}

fn cast(values: &[f32; 4]) -> Option<([u32; 4], i32)> {
    // Positive IEEE bits have the same order as their finite values.
    // Do not evaluate an overflowing scale or cast NaN/infinity to an int.
    if values.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let maximum = values
        .iter()
        .map(|value| value.to_bits() & SIGN_MAGNITUDE)
        .max()?;
    let exponent = if maximum == 0 {
        -127
    } else {
        ((maximum >> 23) as i32 - 126).max(-126)
    };
    let integers = if exponent == -127 {
        [0; 4]
    } else {
        let scale = libm::ldexpf(1.0, 30 - exponent);
        if !scale.is_finite() {
            return None;
        }
        values.map(|value| (scale * value) as i32 as u32)
    };
    let restored = inverse_cast(integers, exponent);
    values
        .iter()
        .zip(restored)
        .all(|(value, restored)| value.to_bits() == restored.to_bits())
        .then_some((integers, exponent))
}

pub(super) fn encode(values: &[f32; 4]) -> Option<Block> {
    let mut out = Block::default();
    let [x, mut y, mut z, mut w] = if let Some((integers, exponent)) = cast(values) {
        if exponent == -127 {
            out.put(0, 1)?;
            return Some(out);
        }
        out.put(1, 2)?;
        out.put((exponent + 127) as u32, 8)?;
        integers
    } else {
        out.put(3, 2)?;
        values.map(|value| {
            let bits = value.to_bits();
            if bits >> 31 != 0 {
                bits ^ SIGN_MAGNITUDE
            } else {
                bits
            }
        })
    };
    // The reversible Lorenzo lifting transform, with modulo-2^32 sums.
    w = w.wrapping_sub(z);
    z = z.wrapping_sub(y);
    y = y.wrapping_sub(x);
    w = w.wrapping_sub(z);
    z = z.wrapping_sub(y);
    w = w.wrapping_sub(z);
    // In one dimension the upstream coefficient permutation is identity.
    let data = [x, y, z, w].map(|value| value.wrapping_add(NEGABINARY) ^ NEGABINARY);
    let precision = (32
        - data
            .iter()
            .fold(0, |bits, value| bits | value)
            .trailing_zeros())
    .max(1);
    out.put(precision - 1, 5)?;
    let mut n = 0;
    for k in (32 - precision..32).rev() {
        let mut plane = 0;
        for (i, value) in data.iter().enumerate() {
            plane |= ((value >> k) & 1) << i;
        }
        out.put(plane, n)?;
        plane >>= n;
        while n < 4 {
            out.put(u32::from(plane != 0), 1)?;
            if plane == 0 {
                break;
            }
            while n < 3 {
                out.put(plane & 1, 1)?;
                if plane & 1 != 0 {
                    break;
                }
                plane >>= 1;
                n += 1;
            }
            plane >>= 1;
            n += 1;
        }
    }
    Some(out)
}

pub(super) fn decode(bytes: &[u8], count: u8) -> Result<[f32; 4], CodecError> {
    if !(1..=120).contains(&count) || bytes.len() != usize::from(count).div_ceil(8) {
        return Err(CodecError("invalid float block size"));
    }
    if !count.is_multiple_of(8) && bytes[bytes.len() - 1] >> (count % 8) != 0 {
        return Err(CodecError("nonzero float block padding"));
    }
    let mut bits = Bits {
        bytes,
        count,
        at: 0,
    };
    let values = if bits.get(1)? == 0 {
        [0.0; 4]
    } else {
        let reinterpret = bits.get(1)? != 0;
        let exponent = if reinterpret {
            0
        } else {
            bits.get(8)? as i32 - 127
        };
        let precision = bits.get(5)? + 1;
        let mut data = [0_u32; 4];
        let mut n = 0;
        for k in (32 - precision..32).rev() {
            let mut plane = bits.get(n)?;
            while n < 4 && bits.get(1)? != 0 {
                while n < 3 && bits.get(1)? == 0 {
                    n += 1;
                }
                plane |= 1 << n;
                n += 1;
            }
            for (i, value) in data.iter_mut().enumerate() {
                *value |= ((plane >> i) & 1) << k;
            }
        }
        let [x, mut y, mut z, mut w] =
            data.map(|value| (value ^ NEGABINARY).wrapping_sub(NEGABINARY));
        w = w.wrapping_add(z);
        z = z.wrapping_add(y);
        w = w.wrapping_add(z);
        y = y.wrapping_add(x);
        z = z.wrapping_add(y);
        w = w.wrapping_add(z);
        let integers = [x, y, z, w];
        if reinterpret {
            integers.map(|value| {
                f32::from_bits(if value >> 31 != 0 {
                    value ^ SIGN_MAGNITUDE
                } else {
                    value
                })
            })
        } else {
            inverse_cast(integers, exponent)
        }
    };
    if bits.at != count {
        return Err(CodecError("extra float block bits"));
    }
    Ok(values)
}
