//! What records are written and read with: little-endian numbers after a
//! tag of four bytes naming what a record holds, and half-precision floats
//! as GPUs store them in 16-bit float textures.

/// A record being written.
pub(crate) struct Writer {
    bytes: Vec<u8>,
}

impl Writer {
    /// A record tagged `tag`, about `capacity` bytes long.
    pub(crate) fn new(tag: &[u8; 4], capacity: usize) -> Self {
        let mut bytes = Vec::with_capacity(capacity + 4);
        bytes.extend_from_slice(tag);
        Self { bytes }
    }

    pub(crate) fn u32(&mut self, value: u32) -> &mut Self {
        self.bytes.extend_from_slice(&value.to_le_bytes());
        self
    }

    pub(crate) fn f32(&mut self, value: f32) -> &mut Self {
        self.u32(value.to_bits())
    }

    pub(crate) fn u16(&mut self, value: u16) -> &mut Self {
        self.bytes.extend_from_slice(&value.to_le_bytes());
        self
    }

    pub(crate) fn bytes(&mut self, bytes: &[u8]) -> &mut Self {
        self.bytes.extend_from_slice(bytes);
        self
    }

    pub(crate) fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

/// A record being read.
pub(crate) struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    /// `bytes`, which must be a record tagged `tag`.
    pub(crate) fn new(bytes: &'a [u8], tag: &[u8; 4]) -> Result<Self, String> {
        match bytes.split_first_chunk::<4>() {
            Some((found, rest)) if found == tag => Ok(Self { bytes: rest }),
            _ => Err(format!(
                "not a {} record",
                String::from_utf8_lossy(tag).to_lowercase()
            )),
        }
    }

    fn take<const N: usize>(&mut self) -> Result<[u8; N], String> {
        let (taken, rest) = self
            .bytes
            .split_first_chunk::<N>()
            .ok_or("the record ends early")?;
        self.bytes = rest;
        Ok(*taken)
    }

    pub(crate) fn u32(&mut self) -> Result<u32, String> {
        self.take::<4>().map(u32::from_le_bytes)
    }

    pub(crate) fn f32(&mut self) -> Result<f32, String> {
        self.u32().map(f32::from_bits)
    }

    /// The next `count` bytes.
    pub(crate) fn bytes(&mut self, count: usize) -> Result<&'a [u8], String> {
        if self.bytes.len() < count {
            return Err("the record ends early".into());
        }
        let (taken, rest) = self.bytes.split_at(count);
        self.bytes = rest;
        Ok(taken)
    }

    /// The end of the record: nothing may follow.
    pub(crate) fn finish(self) -> Result<(), String> {
        if self.bytes.is_empty() {
            Ok(())
        } else {
            Err(format!("{} bytes follow the record", self.bytes.len()))
        }
    }
}

/// The half-precision bits nearest `value` (ties to even; beyond the
/// largest half, infinity), as GPUs store floats written to 16-bit float
/// textures.
pub fn f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        // Infinity, or a quiet NaN.
        return sign | 0x7c00 | if mantissa == 0 { 0 } else { 0x200 };
    }
    let unbiased = exponent - 127;
    if unbiased > 15 {
        return sign | 0x7c00;
    }
    if unbiased < -25 {
        return sign;
    }
    // Normal halves keep the mantissa's top ten bits under their exponent;
    // subnormal ones (below 2^-14) shift the whole mantissa, its implicit
    // bit with it, into units of 2^-24.
    let (full, shift, base) = if unbiased >= -14 {
        (mantissa, 13, ((unbiased + 15) as u32) << 10)
    } else {
        (mantissa | 0x80_0000, (-1 - unbiased) as u32, 0)
    };
    let kept = base | (full >> shift);
    let rest = full & ((1 << shift) - 1);
    let half = 1 << (shift - 1);
    // A carry out of the mantissa raises the exponent, as it should.
    let rounded = kept + u32::from(rest > half || (rest == half && (kept & 1) == 1));
    sign | rounded as u16
}

/// The value of half-precision `bits`.
pub fn f16_value(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from(bits >> 10) & 0x1f;
    let mantissa = u32::from(bits & 0x3ff);
    match exponent {
        // Subnormal: the mantissa's units are 2^-24, exactly.
        0 => f32::from_bits(sign | (mantissa as f32 * (1.0 / 16_777_216.0)).to_bits()),
        0x1f => f32::from_bits(sign | 0x7f80_0000 | (mantissa << 13)),
        _ => f32::from_bits(sign | ((exponent + 112) << 23) | (mantissa << 13)),
    }
}
