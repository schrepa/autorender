#![deny(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]

mod codes;
mod error;
pub mod fields;

use std::sync::Arc;

use glam::{Vec2, Vec3};

pub use codes::ErrorCode;
pub use error::{Allowed, Detail, Error, Expected, MAX_GOT_BYTES, closest};
pub use fields::{
    ExtraRules, ID_PATTERN, InvalidSetting, MAX_ID_BYTES, Rule, Setting, is_id, state, stated,
    valid_uri,
};

pub mod rule {
    pub use crate::fields::{
        Bound, above, above_to, any, at_least, at_most, budget, count, finite, id, multiple,
        power_of_two, range, same,
    };
}

#[doc(hidden)]
pub mod __private {
    pub use ::schemars;
}

pub fn invalid(path: impl Into<String>) -> Error {
    Error::new(ErrorCode::InvalidRecord, path)
}

#[derive(Default)]
pub struct Writer {
    bytes: Vec<u8>,
}

impl Writer {
    pub fn new() -> Self {
        Self::default()
    }

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

pub struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes }
    }

    pub fn get<T: Codec>(&mut self) -> Result<T, Error> {
        T::read(self)
    }

    pub fn finish(self) -> Result<(), Error> {
        if self.bytes.is_empty() {
            Ok(())
        } else {
            Err(invalid("$")
                .expected(Expected::Absent)
                .got(self.bytes.len()))
        }
    }

    pub fn left(&self) -> usize {
        self.bytes.len()
    }

    fn take<const N: usize>(&mut self) -> Result<[u8; N], Error> {
        let Some((head, rest)) = self.bytes.split_first_chunk::<N>() else {
            return Err(invalid("$").at_least(N as u64).got(self.bytes.len()));
        };
        self.bytes = rest;
        Ok(*head)
    }

    fn count(&mut self) -> Result<usize, Error> {
        let count = size(u64::from_le_bytes(self.take()?), "$.count")?;
        if count > self.bytes.len() {
            return Err(invalid("$.count")
                .at_most(self.bytes.len() as u64)
                .got(count));
        }
        Ok(count)
    }
}

pub const HEIGHT_STEP: f32 = 1.0 / STEPS_PER_METER;
const STEPS_PER_METER: f32 = 256.0;

pub fn put_grid(out: &mut Writer, values: &[f32], stride: usize) {
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

fn size(count: u64, path: &str) -> Result<usize, Error> {
    usize::try_from(count).map_err(|_| invalid(path).at_most(usize::MAX as u64).got(count))
}

pub fn get_grid(input: &mut Reader<'_>, stride: usize) -> Result<Box<[f32]>, Error> {
    let count = input.count()?;
    let mut steps: Vec<Option<i64>> = Vec::with_capacity(count);
    for at in 0..count {
        let mut code = 0_u64;
        let mut shift = 0;
        loop {
            let [byte] = input.take::<1>()?;
            if shift > 63 {
                return Err(invalid(format!("$.grid[{at}]")));
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
            .ok_or_else(|| invalid(format!("$.grid[{at}]")))?;
        steps.push(Some(value));
    }
    Ok(steps
        .into_iter()
        .map(|steps| steps.map_or(f32::NAN, |steps| steps as f32 / STEPS_PER_METER))
        .collect())
}

pub fn put_planes(out: &mut Writer, planes: &[u8], side: usize) {
    out.put(&(planes.len() as u64));
    let area = side * side;
    let mut residuals = Vec::with_capacity(planes.len());
    for (at, &value) in planes.iter().enumerate() {
        let within = if area == 0 { 0 } else { at % area };
        let start = at - within;
        residuals
            .push(value.wrapping_sub(predict_byte(within, side, |within| planes[start + within])));
    }
    out.raw(&residuals);
}

pub fn get_planes(input: &mut Reader<'_>, side: usize) -> Result<Box<[u8]>, Error> {
    let count = input.count()?;
    let area = side * side;
    if area == 0 || count % area != 0 {
        return Err(invalid("$.planes").got(count));
    }
    let (residuals, rest) = input.bytes.split_at(count);
    input.bytes = rest;
    let mut planes = vec![0_u8; count];
    for at in 0..count {
        let within = at % area;
        let start = at - within;
        let predicted = predict_byte(within, side, |within| planes[start + within]);
        planes[at] = residuals[at].wrapping_add(predicted);
    }
    Ok(planes.into_boxed_slice())
}

pub fn put_wide_planes(out: &mut Writer, planes: &[u16], side: usize) {
    out.put(&(planes.len() as u64));
    let area = side * side;
    let mut residuals = Vec::with_capacity(planes.len() * 2);
    for (at, &value) in planes.iter().enumerate() {
        let within = if area == 0 { 0 } else { at % area };
        let start = at - within;
        let predicted = predict_wide(within, side, |within| planes[start + within]);
        residuals.extend_from_slice(&value.wrapping_sub(predicted).to_le_bytes());
    }
    out.raw(&residuals);
}

pub fn get_wide_planes(input: &mut Reader<'_>, side: usize) -> Result<Box<[u16]>, Error> {
    let count = size(u64::from_le_bytes(input.take()?), "$.count")?;
    let area = side * side;
    let bytes = count
        .checked_mul(2)
        .filter(|&bytes| bytes <= input.bytes.len())
        .ok_or_else(|| {
            invalid("$.count")
                .at_most((input.bytes.len() / 2) as u64)
                .got(count)
        })?;
    if area == 0 || count % area != 0 {
        return Err(invalid("$.planes").got(count));
    }
    let (residuals, rest) = input.bytes.split_at(bytes);
    input.bytes = rest;
    let mut planes = vec![0_u16; count];
    for at in 0..count {
        let within = at % area;
        let start = at - within;
        let predicted = predict_wide(within, side, |within| planes[start + within]);
        let residual = u16::from_le_bytes([residuals[2 * at], residuals[2 * at + 1]]);
        planes[at] = residual.wrapping_add(predicted);
    }
    Ok(planes.into_boxed_slice())
}

pub fn put_records(out: &mut Writer, values: &[u8], stride: usize) {
    out.put(&(values.len() as u64));
    let count = values.len().checked_div(stride).unwrap_or(0);
    let mut residuals = Vec::with_capacity(values.len());
    for place in 0..stride {
        let mut before = 0_u8;
        for record in 0..count {
            let value = values[record * stride + place];
            residuals.push(value.wrapping_sub(before));
            before = value;
        }
    }
    out.raw(&residuals);
}

pub fn get_records(input: &mut Reader<'_>, stride: usize) -> Result<Box<[u8]>, Error> {
    let count = input.count()?;
    if stride == 0 || !count.is_multiple_of(stride) {
        return Err(invalid("$.records").got(count));
    }
    let (residuals, rest) = input.bytes.split_at(count);
    input.bytes = rest;
    let records = count / stride;
    let mut values = vec![0_u8; count];
    let mut at = 0;
    for place in 0..stride {
        let mut before = 0_u8;
        for record in 0..records {
            before = before.wrapping_add(residuals[at]);
            values[record * stride + place] = before;
            at += 1;
        }
    }
    Ok(values.into_boxed_slice())
}

fn predict_wide(at: usize, side: usize, value: impl Fn(usize) -> u16) -> u16 {
    let (x, z) = (at % side, at / side);
    match (x > 0, z > 0) {
        (true, true) => value(at - 1)
            .wrapping_add(value(at - side))
            .wrapping_sub(value(at - side - 1)),
        (true, false) => value(at - 1),
        (false, true) => value(at - side),
        (false, false) => 0,
    }
}

fn predict_byte(at: usize, side: usize, byte: impl Fn(usize) -> u8) -> u8 {
    let (x, z) = (at % side, at / side);
    match (x > 0, z > 0) {
        (true, true) => byte(at - 1)
            .wrapping_add(byte(at - side))
            .wrapping_sub(byte(at - side - 1)),
        (true, false) => byte(at - 1),
        (false, true) => byte(at - side),
        (false, false) => 0,
    }
}

pub fn on_grid(value: f32) -> f32 {
    steps(value).map_or(f32::NAN, |steps| steps as f32 / STEPS_PER_METER)
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "a tile's heights are whole steps far within i64"
)]
fn steps(value: f32) -> Option<i64> {
    (!value.is_nan()).then_some((value * STEPS_PER_METER).round() as i64)
}

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

pub fn decode<T: Codec>(bytes: &[u8]) -> Result<T, Error> {
    let mut reader = Reader::new(bytes);
    let value = reader.get()?;
    reader.finish()?;
    Ok(value)
}

pub fn encode<T: Codec>(value: &T) -> Vec<u8> {
    let mut writer = Writer::new();
    writer.put(value);
    writer.into_bytes()
}

pub trait Codec: Sized {
    fn write(&self, out: &mut Writer);
    fn read(input: &mut Reader<'_>) -> Result<Self, Error>;
}

macro_rules! number {
    ($($type:ty),*) => {$(
        impl Codec for $type {
            fn write(&self, out: &mut Writer) {
                out.raw(&self.to_le_bytes());
            }

            fn read(input: &mut Reader<'_>) -> Result<Self, Error> {
                Ok(Self::from_le_bytes(input.take()?))
            }
        }
    )*};
}

number!(u8, u16, i16, u32, u64, i64, f32);

impl Codec for usize {
    fn write(&self, out: &mut Writer) {
        (*self as u64).write(out);
    }

    fn read(input: &mut Reader<'_>) -> Result<Self, Error> {
        size(u64::read(input)?, "$.size")
    }
}

impl Codec for bool {
    fn write(&self, out: &mut Writer) {
        u8::from(*self).write(out);
    }

    fn read(input: &mut Reader<'_>) -> Result<Self, Error> {
        match u8::read(input)? {
            0 => Ok(false),
            1 => Ok(true),
            flag => Err(invalid("$.flag").at_most(1).got(flag)),
        }
    }
}

impl Codec for String {
    fn write(&self, out: &mut Writer) {
        (self.len() as u64).write(out);
        out.raw(self.as_bytes());
    }

    fn read(input: &mut Reader<'_>) -> Result<Self, Error> {
        let count = input.count()?;
        let (text, rest) = input.bytes.split_at(count);
        input.bytes = rest;
        String::from_utf8(text.to_vec()).map_err(|_| Error::new(ErrorCode::InvalidUtf8, "$.text"))
    }
}

impl<T: Codec> Codec for Vec<T> {
    fn write(&self, out: &mut Writer) {
        (self.len() as u64).write(out);
        for value in self {
            value.write(out);
        }
    }

    fn read(input: &mut Reader<'_>) -> Result<Self, Error> {
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

    fn read(input: &mut Reader<'_>) -> Result<Self, Error> {
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

    fn read(input: &mut Reader<'_>) -> Result<Self, Error> {
        Vec::<T>::read(input).map(Arc::from)
    }
}

impl<T: Codec> Codec for Arc<T> {
    fn write(&self, out: &mut Writer) {
        self.as_ref().write(out);
    }

    fn read(input: &mut Reader<'_>) -> Result<Self, Error> {
        T::read(input).map(Arc::new)
    }
}

impl<T: Codec> Codec for Box<T> {
    fn write(&self, out: &mut Writer) {
        self.as_ref().write(out);
    }

    fn read(input: &mut Reader<'_>) -> Result<Self, Error> {
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

    fn read(input: &mut Reader<'_>) -> Result<Self, Error> {
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

    fn read(input: &mut Reader<'_>) -> Result<Self, Error> {
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

    fn read(input: &mut Reader<'_>) -> Result<Self, Error> {
        Ok((A::read(input)?, B::read(input)?))
    }
}

impl Codec for Vec2 {
    fn write(&self, out: &mut Writer) {
        self.to_array().write(out);
    }

    fn read(input: &mut Reader<'_>) -> Result<Self, Error> {
        <[f32; 2]>::read(input).map(Vec2::from_array)
    }
}

impl Codec for Vec3 {
    fn write(&self, out: &mut Writer) {
        self.to_array().write(out);
    }

    fn read(input: &mut Reader<'_>) -> Result<Self, Error> {
        <[f32; 3]>::read(input).map(Vec3::from_array)
    }
}

#[macro_export]
macro_rules! record {
    ($type:ident { $($field:ident),* $(,)? }) => {
        impl $crate::Codec for $type {
            fn write(&self, out: &mut $crate::Writer) {
                $( $crate::Codec::write(&self.$field, out); )*
            }

            fn read(
                input: &mut $crate::Reader<'_>,
            ) -> Result<Self, $crate::Error> {
                Ok(Self { $( $field: $crate::Codec::read(input)?, )* })
            }
        }
    };
}

#[macro_export]
macro_rules! tagged {
    ($type:ident { $($tag:literal => $variant:ident $(($field:ident))?,)* }) => {
        impl $crate::Codec for $type {
            fn write(&self, out: &mut $crate::Writer) {
                match self {
                    $(Self::$variant $(($field))? => {
                        out.put::<u8>(&$tag);
                        $(out.put($field);)?
                    })*
                }
            }

            fn read(
                input: &mut $crate::Reader<'_>,
            ) -> Result<Self, $crate::Error> {
                Ok(match input.get::<u8>()? {
                    $($tag => Self::$variant $(({
                        let $field = input.get()?;
                        $field
                    }))?,)*
                    tag => {
                        return Err($crate::invalid(concat!("$.", stringify!($type)))
                            .variants([$(stringify!($tag)),*])
                            .got(tag));
                    }
                })
            }
        }
    };
}

