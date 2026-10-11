//! TFZ1: independently decoded reversible four-float blocks, or exact raw
//! bits. This is the worker wire, not the ETR1 terrain file format.

use super::{Codec, CodecError, Reader, Writer, zfp};

pub(super) fn write(values: &[f32], out: &mut Writer) {
    (values.len() as u64).write(out);
    out.raw(b"TFZ1");
    let mode_at = out.bytes.len();
    1_u8.write(out);
    let length_at = out.bytes.len();
    0_u64.write(out);
    let body_at = out.bytes.len();
    let raw_length = std::mem::size_of_val(values);
    for values in values.chunks(4) {
        if let Ok(block) = <&[f32; 4]>::try_from(values)
            && let Some(block) = zfp::encode(block)
        {
            block.bits.write(out);
            out.raw(block.payload());
        } else {
            0_u8.write(out);
            for value in values {
                value.write(out);
            }
        }
        if out.bytes.len() - body_at >= raw_length {
            break;
        }
    }
    if values.is_empty() || out.bytes.len() - body_at >= raw_length {
        out.bytes[mode_at] = 0;
        out.bytes.truncate(body_at);
        for value in values {
            value.write(out);
        }
    }
    let length = (out.bytes.len() - body_at) as u64;
    out.bytes[length_at..length_at + 8].copy_from_slice(&length.to_le_bytes());
}

fn take<'a>(bytes: &mut &'a [u8], count: usize) -> Result<&'a [u8], CodecError> {
    let (head, tail) = bytes
        .split_at_checked(count)
        .ok_or(CodecError("float body ends early"))?;
    *bytes = tail;
    Ok(head)
}

fn block(bytes: &mut &[u8], count: usize) -> Result<[f32; 4], CodecError> {
    let descriptor = take(bytes, 1)?[0];
    if descriptor == 0 {
        let raw = take(bytes, count * 4)?;
        let mut values = [0.0; 4];
        for (value, raw) in values.iter_mut().zip(raw.as_chunks::<4>().0) {
            *value = f32::from_le_bytes(*raw);
        }
        Ok(values)
    } else {
        if descriptor > 120 || count != 4 {
            return Err(CodecError("invalid float block descriptor"));
        }
        zfp::decode(
            take(bytes, usize::from(descriptor).div_ceil(8))?,
            descriptor,
        )
    }
}

pub(super) fn read(input: &mut Reader<'_>) -> Result<Vec<f32>, CodecError> {
    let count =
        usize::try_from(u64::read(input)?).map_err(|_| CodecError("float count out of range"))?;
    let raw_length = count
        .checked_mul(4)
        .ok_or(CodecError("float byte count overflow"))?;
    if input.take::<4>()? != *b"TFZ1" {
        return Err(CodecError("unknown float sequence format"));
    }
    let mode = u8::read(input)?;
    let length =
        usize::try_from(u64::read(input)?).map_err(|_| CodecError("float length out of range"))?;
    let body = take(&mut input.bytes, length)?;
    match mode {
        0 if length == raw_length => {}
        1 if count != 0 && length < raw_length => {
            // Two bytes is the smallest complete four-value group. This
            // cheap bound avoids walking an untrusted huge declared count.
            if count.div_ceil(4) > length / 2 {
                return Err(CodecError("float count beyond its blocks"));
            }
            let mut rest = body;
            let mut remaining = count;
            while remaining != 0 {
                let group = remaining.min(4);
                block(&mut rest, group)?;
                remaining -= group;
            }
            if !rest.is_empty() {
                return Err(CodecError("extra float body bytes"));
            }
        }
        _ => return Err(CodecError("invalid float sequence mode or length")),
    }
    // No declared-size allocation occurs until the whole body is proved.
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| CodecError("float allocation failed"))?;
    if mode == 0 {
        for raw in body.as_chunks::<4>().0 {
            values.push(f32::from_le_bytes(*raw));
        }
    } else {
        let mut rest = body;
        while values.len() < count {
            let group = (count - values.len()).min(4);
            values.extend_from_slice(&block(&mut rest, group)?[..group]);
        }
    }
    Ok(values)
}
