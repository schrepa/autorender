use crate::RigError;

pub(crate) fn put_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

pub(crate) fn put_count(bytes: &mut Vec<u8>, count: usize) {
    put_u32(bytes, u32::try_from(count).expect("count_fits_u32"));
}

pub(crate) fn put_floats(bytes: &mut Vec<u8>, values: &[f32]) {
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}

pub(crate) fn put_name(bytes: &mut Vec<u8>, name: &str) {
    put_count(bytes, name.len());
    bytes.extend_from_slice(name.as_bytes());
    bytes.resize(bytes.len().next_multiple_of(4), 0);
}

pub(crate) struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
    item: &'static str,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(
        bytes: &'a [u8],
        item: &'static str,
        magic: [u8; 4],
    ) -> Result<Self, RigError> {
        let mut reader = Self { bytes, at: 0, item };
        let held = reader.take(4)?;
        if held != magic {
            return Err(RigError::Magic {
                item,
                got: held.to_vec(),
            });
        }
        Ok(reader)
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], RigError> {
        let end = self.at.saturating_add(length);
        let taken = self.bytes.get(self.at..end).ok_or(RigError::Truncated {
            item: self.item,
            needed: end,
            held: self.bytes.len(),
        })?;
        self.at = end;
        Ok(taken)
    }

    pub(crate) fn u32(&mut self) -> Result<u32, RigError> {
        let word = self.take(4)?;
        Ok(u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
    }

    pub(crate) fn count(&mut self) -> Result<usize, RigError> {
        self.u32().map(|count| count as usize)
    }

    pub(crate) fn f32(&mut self) -> Result<f32, RigError> {
        self.u32().map(f32::from_bits)
    }

    pub(crate) fn floats<const N: usize>(&mut self) -> Result<[f32; N], RigError> {
        let mut values = [0.0; N];
        for value in &mut values {
            *value = self.f32()?;
        }
        Ok(values)
    }

    pub(crate) fn name(&mut self) -> Result<String, RigError> {
        let length = self.count()?;
        let start = self.at;
        let raw = self.take(length)?;
        let name = std::str::from_utf8(raw)
            .map_err(|_| RigError::NameNotUtf8 {
                item: self.item,
                at: start,
            })?
            .to_owned();
        self.take(length.next_multiple_of(4) - length)?;
        Ok(name)
    }

    pub(crate) fn finish(self) -> Result<(), RigError> {
        match self.bytes.len() - self.at {
            0 => Ok(()),
            bytes => Err(RigError::TrailingBytes {
                item: self.item,
                bytes,
            }),
        }
    }
}
