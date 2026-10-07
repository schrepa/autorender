//! A compact binary encoding of terrain data (decision 110): what engines
//! hand workers and workers hand back. Floats travel as their bits, so a
//! tile a worker generates is the tile the engine would have generated.
//! Lengths are checked against the bytes left before anything allocates,
//! and readers refuse bytes left over.

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
