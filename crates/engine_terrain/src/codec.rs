//! A compact binary encoding of terrain data: what engines
//! hand workers and workers hand back. Floats travel as their bits, so a
//! tile a worker generates is the tile the engine would have generated.
//! Lengths are checked against the bytes left before anything allocates,
//! and readers refuse bytes left over.
// Bytes come back from workers and caches: every narrowing cast is
// checked.
#![deny(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]

use std::sync::Arc;

use glam::{Vec2, Vec3};

/// Why bytes could not be read back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CodecError(pub &'static str);

impl std::fmt::Display for CodecError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "invalid terrain data: {}", self.0)
    }
}

impl std::error::Error for CodecError {}

/// Bytes being written.
#[derive(Default)]
pub struct Writer {
    bytes: Vec<u8>,
}

impl Writer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Writes `value`.
    pub fn put<T: Codec>(&mut self, value: &T) -> &mut Self {
        value.write(self);
        self
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    fn raw(&mut self, bytes: &[u8]) {
        self.bytes.extend_from_slice(bytes);
    }
}

/// Bytes being read.
pub struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes }
    }

    /// Reads a value.
    pub fn get<T: Codec>(&mut self) -> Result<T, CodecError> {
        T::read(self)
    }

    /// Whether every byte was read.
    pub fn finish(self) -> Result<(), CodecError> {
        if self.bytes.is_empty() {
            Ok(())
        } else {
            Err(CodecError("bytes left over"))
        }
    }

    fn take<const N: usize>(&mut self) -> Result<[u8; N], CodecError> {
        let (head, rest) = self
            .bytes
            .split_first_chunk::<N>()
            .ok_or(CodecError("ends early"))?;
        self.bytes = rest;
        Ok(*head)
    }

    /// A count of items that take at least a byte each: no more than the
    /// bytes left.
    fn count(&mut self) -> Result<usize, CodecError> {
        let count = usize::try_from(u64::from_le_bytes(self.take()?))
            .map_err(|_| CodecError("count out of range"))?;
        if count > self.bytes.len() {
            return Err(CodecError("count beyond the bytes left"));
        }
        Ok(count)
    }
}

/// Meters between the heights tiles travel at: a tile's heights and water
/// surfaces are written as whole steps of this grid, which `f32` holds
/// exactly within 65,536 m of the terrain's origin, so what reads back is
/// each value at its nearest step, and writing that again gives the same
/// bytes.
pub const HEIGHT_STEP: f32 = 1.0 / STEPS_PER_METER;
const STEPS_PER_METER: f32 = 256.0;

/// Writes `values` at their nearest steps of the height grid, rows of
/// `stride`: each value's steps from what its written neighbours predict
/// (left plus up less their corner, or the one there is), zigzagged into
/// variable-length bytes one more than the step, a zero byte for NaN.
/// Smooth ground costs about a byte a sample.
pub(crate) fn put_grid(out: &mut Writer, values: &[f32], stride: usize) {
    out.put(&(values.len() as u64));
    for at in 0..values.len() {
        let Some(value) = steps(values[at]) else {
            out.raw(&[0]);
            continue;
        };
        let predicted = predict(at, stride, |at| steps(values[at]));
        let residual = value - predicted;
        let mut code = ((residual << 1) ^ (residual >> 63)).cast_unsigned() + 1;
        loop {
            let byte = (code & 0x7f) as u8;
            code >>= 7;
            if code == 0 {
                out.raw(&[byte]);
                break;
            }
            out.raw(&[byte | 0x80]);
        }
    }
}

/// Reads a grid [`put_grid`] wrote, rows of `stride`.
pub(crate) fn get_grid(input: &mut Reader<'_>, stride: usize) -> Result<Box<[f32]>, CodecError> {
    let count = input.count()?;
    let mut steps: Vec<Option<i64>> = Vec::with_capacity(count);
    for at in 0..count {
        let mut code = 0_u64;
        let mut shift = 0;
        loop {
            let [byte] = input.take::<1>()?;
            if shift > 63 {
                return Err(CodecError("grid step out of range"));
            }
            code |= u64::from(byte & 0x7f) << shift;
            shift += 7;
            if byte & 0x80 == 0 {
                break;
            }
        }
        let Some(code) = code.checked_sub(1) else {
            steps.push(None);
            continue;
        };
        let residual = (code >> 1).cast_signed() ^ -(code & 1).cast_signed();
        let predicted = predict(at, stride, |at| steps[at]);
        let value = predicted
            .checked_add(residual)
            .ok_or(CodecError("grid step out of range"))?;
        steps.push(Some(value));
    }
    Ok(steps
        .into_iter()
        .map(|steps| steps.map_or(f32::NAN, |steps| steps as f32 / STEPS_PER_METER))
        .collect())
}

/// The nearest whole step of the height grid to `value`; none for NaN.
#[expect(
    clippy::cast_possible_truncation,
    reason = "a tile's heights are whole steps far within i64"
)]
fn steps(value: f32) -> Option<i64> {
    (!value.is_nan()).then_some((value * STEPS_PER_METER).round() as i64)
}

/// What sample `at`'s written neighbours (rows of `stride`) predict it:
/// left plus up less their corner where all three are numbers, else left,
/// else up, else nothing.
fn predict(at: usize, stride: usize, steps: impl Fn(usize) -> Option<i64>) -> i64 {
    let (x, z) = (at % stride, at / stride);
    let left = (x > 0).then(|| steps(at - 1)).flatten();
    let up = (z > 0).then(|| steps(at - stride)).flatten();
    let corner = (x > 0 && z > 0).then(|| steps(at - stride - 1)).flatten();
    match (left, up, corner) {
        (Some(left), Some(up), Some(corner)) => left + up - corner,
        (Some(left), _, _) => left,
        (None, Some(up), _) => up,
        (None, None, _) => 0,
    }
}

/// Reads one value from all of `bytes`.
pub fn decode<T: Codec>(bytes: &[u8]) -> Result<T, CodecError> {
    let mut reader = Reader::new(bytes);
    let value = reader.get()?;
    reader.finish()?;
    Ok(value)
}

/// Writes `value` alone.
pub fn encode<T: Codec>(value: &T) -> Vec<u8> {
    let mut writer = Writer::new();
    writer.put(value);
    writer.into_bytes()
}

/// A value written as bytes and read back exactly.
pub trait Codec: Sized {
    fn write(&self, out: &mut Writer);
    fn read(input: &mut Reader<'_>) -> Result<Self, CodecError>;
}

macro_rules! number {
    ($($type:ty),*) => {$(
        impl Codec for $type {
            fn write(&self, out: &mut Writer) {
                out.raw(&self.to_le_bytes());
            }

            fn read(input: &mut Reader<'_>) -> Result<Self, CodecError> {
                Ok(Self::from_le_bytes(input.take()?))
            }
        }
    )*};
}

number!(u8, u16, u32, u64, i64, f32);

impl Codec for usize {
    fn write(&self, out: &mut Writer) {
        (*self as u64).write(out);
    }

    fn read(input: &mut Reader<'_>) -> Result<Self, CodecError> {
        usize::try_from(u64::read(input)?).map_err(|_| CodecError("size out of range"))
    }
}

impl Codec for bool {
    fn write(&self, out: &mut Writer) {
        u8::from(*self).write(out);
    }

    fn read(input: &mut Reader<'_>) -> Result<Self, CodecError> {
        match u8::read(input)? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(CodecError("invalid flag")),
        }
    }
}

impl Codec for String {
    fn write(&self, out: &mut Writer) {
        (self.len() as u64).write(out);
        out.raw(self.as_bytes());
    }

    fn read(input: &mut Reader<'_>) -> Result<Self, CodecError> {
        let count = input.count()?;
        let (text, rest) = input.bytes.split_at(count);
        input.bytes = rest;
        String::from_utf8(text.to_vec()).map_err(|_| CodecError("invalid text"))
    }
}

impl<T: Codec> Codec for Vec<T> {
    fn write(&self, out: &mut Writer) {
        (self.len() as u64).write(out);
        for value in self {
            value.write(out);
        }
    }

    fn read(input: &mut Reader<'_>) -> Result<Self, CodecError> {
        let count = input.count()?;
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(T::read(input)?);
        }
        Ok(values)
    }
}

impl<T: Codec> Codec for Box<[T]> {
    fn write(&self, out: &mut Writer) {
        (self.len() as u64).write(out);
        for value in self.iter() {
            value.write(out);
        }
    }

    fn read(input: &mut Reader<'_>) -> Result<Self, CodecError> {
        Vec::<T>::read(input).map(Vec::into_boxed_slice)
    }
}

impl<T: Codec> Codec for Arc<[T]> {
    fn write(&self, out: &mut Writer) {
        (self.len() as u64).write(out);
        for value in self.iter() {
            value.write(out);
        }
    }

    fn read(input: &mut Reader<'_>) -> Result<Self, CodecError> {
        Vec::<T>::read(input).map(Arc::from)
    }
}

impl<T: Codec> Codec for Arc<T> {
    fn write(&self, out: &mut Writer) {
        self.as_ref().write(out);
    }

    fn read(input: &mut Reader<'_>) -> Result<Self, CodecError> {
        T::read(input).map(Arc::new)
    }
}

impl<T: Codec> Codec for Box<T> {
    fn write(&self, out: &mut Writer) {
        self.as_ref().write(out);
    }

    fn read(input: &mut Reader<'_>) -> Result<Self, CodecError> {
        T::read(input).map(Box::new)
    }
}

impl<T: Codec> Codec for Option<T> {
    fn write(&self, out: &mut Writer) {
        match self {
            Some(value) => {
                true.write(out);
                value.write(out);
            }
            None => false.write(out),
        }
    }

    fn read(input: &mut Reader<'_>) -> Result<Self, CodecError> {
        Ok(if bool::read(input)? {
            Some(T::read(input)?)
        } else {
            None
        })
    }
}

impl<T: Codec + Copy + Default, const N: usize> Codec for [T; N] {
    fn write(&self, out: &mut Writer) {
        for value in self {
            value.write(out);
        }
    }

    fn read(input: &mut Reader<'_>) -> Result<Self, CodecError> {
        let mut values = [T::default(); N];
        for value in &mut values {
            *value = T::read(input)?;
        }
        Ok(values)
    }
}

impl<A: Codec, B: Codec> Codec for (A, B) {
    fn write(&self, out: &mut Writer) {
        self.0.write(out);
        self.1.write(out);
    }

    fn read(input: &mut Reader<'_>) -> Result<Self, CodecError> {
        Ok((A::read(input)?, B::read(input)?))
    }
}

impl Codec for Vec2 {
    fn write(&self, out: &mut Writer) {
        self.to_array().write(out);
    }

    fn read(input: &mut Reader<'_>) -> Result<Self, CodecError> {
        <[f32; 2]>::read(input).map(Vec2::from_array)
    }
}

impl Codec for Vec3 {
    fn write(&self, out: &mut Writer) {
        self.to_array().write(out);
    }

    fn read(input: &mut Reader<'_>) -> Result<Self, CodecError> {
        <[f32; 3]>::read(input).map(Vec3::from_array)
    }
}

/// Implements [`Codec`] for a struct by its fields, in order: within the
/// struct's own module, so private fields travel too.
macro_rules! record {
    ($type:ident { $($field:ident),* $(,)? }) => {
        impl $crate::codec::Codec for $type {
            fn write(&self, out: &mut $crate::codec::Writer) {
                $( $crate::codec::Codec::write(&self.$field, out); )*
            }

            fn read(
                input: &mut $crate::codec::Reader<'_>,
            ) -> Result<Self, $crate::codec::CodecError> {
                Ok(Self { $( $field: $crate::codec::Codec::read(input)?, )* })
            }
        }
    };
}

pub(crate) use record;

/// Implements [`Codec`] for an enum as a tag byte then its variant's field,
/// from one table of each variant with its tag: writing matches every
/// variant, so one without a tag does not compile, and reading refuses an
/// unknown tag with `$invalid`.
macro_rules! tagged {
    ($type:ident, $invalid:literal { $($tag:literal => $variant:ident $(($field:ident))?,)* }) => {
        impl $crate::codec::Codec for $type {
            fn write(&self, out: &mut $crate::codec::Writer) {
                match self {
                    $(Self::$variant $(($field))? => {
                        out.put::<u8>(&$tag);
                        $(out.put($field);)?
                    })*
                }
            }

            fn read(
                input: &mut $crate::codec::Reader<'_>,
            ) -> Result<Self, $crate::codec::CodecError> {
                Ok(match input.get::<u8>()? {
                    $($tag => Self::$variant $(({
                        let $field = input.get()?;
                        $field
                    }))?,)*
                    _ => return Err($crate::codec::CodecError($invalid)),
                })
            }
        }
    };
}

pub(crate) use tagged;
