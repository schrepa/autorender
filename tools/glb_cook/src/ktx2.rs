use crate::config::Encoding;

const IDENTIFIER: [u8; 12] = [
    0xab, 0x4b, 0x54, 0x58, 0x20, 0x32, 0x30, 0xbb, 0x0d, 0x0a, 0x1a, 0x0a,
];

pub struct Layout {
    pub vk_format: u32,
    pub block: u32,
    pub block_bytes: usize,
}

pub const fn layout(encoding: Encoding) -> Layout {
    match encoding {
        Encoding::Rgba8 => Layout {
            vk_format: 37,
            block: 1,
            block_bytes: 4,
        },
        Encoding::Rgba8Srgb => Layout {
            vk_format: 43,
            block: 1,
            block_bytes: 4,
        },
        Encoding::Bc5 => Layout {
            vk_format: 141,
            block: 4,
            block_bytes: 16,
        },
        Encoding::Bc7 => Layout {
            vk_format: 145,
            block: 4,
            block_bytes: 16,
        },
        Encoding::Bc7Srgb => Layout {
            vk_format: 146,
            block: 4,
            block_bytes: 16,
        },
    }
}

pub fn level_bytes(encoding: Encoding, width: u32, height: u32) -> usize {
    let held = layout(encoding);
    (width.div_ceil(held.block) * height.div_ceil(held.block)) as usize * held.block_bytes
}

fn sample(offset: u32, length: u32, channel: u32, lower: u32, upper: u32) -> [u32; 4] {
    [
        offset | ((length - 1) << 16) | (channel << 24),
        0,
        lower,
        upper,
    ]
}

fn descriptor(encoding: Encoding) -> Vec<u8> {
    let transfer = if encoding.srgb() { 2 } else { 1 };
    let (model, dimensions, plane, samples): (u32, u32, u32, Vec<[u32; 4]>) = match encoding {
        Encoding::Rgba8 | Encoding::Rgba8Srgb => {
            let alpha = if encoding.srgb() { 15 | 0x10 } else { 15 };
            (
                1,
                0,
                4,
                vec![
                    sample(0, 8, 0, 0, 255),
                    sample(8, 8, 1, 0, 255),
                    sample(16, 8, 2, 0, 255),
                    sample(24, 8, alpha, 0, 255),
                ],
            )
        }
        Encoding::Bc5 => (
            132,
            3 | (3 << 8),
            16,
            vec![
                sample(0, 64, 0, 0, u32::MAX),
                sample(64, 64, 1, 0, u32::MAX),
            ],
        ),
        Encoding::Bc7 | Encoding::Bc7Srgb => {
            (134, 3 | (3 << 8), 16, vec![sample(0, 128, 0, 0, u32::MAX)])
        }
    };
    let block = 24 + 16 * samples.len() as u32;
    let mut words = vec![
        4 + block,
        0,
        2 | (block << 16),
        model | (1 << 8) | (transfer << 16),
        dimensions,
        plane,
        0,
    ];
    for held in samples {
        words.extend_from_slice(&held);
    }
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn aligned(at: usize, to: usize) -> usize {
    at.div_ceil(to) * to
}

pub fn write(encoding: Encoding, width: u32, height: u32, levels: &[Vec<u8>]) -> Vec<u8> {
    let held = layout(encoding);
    let count = levels.len();
    let index_end = 80 + 24 * count;
    let dfd = descriptor(encoding);
    let mut at = index_end + dfd.len();
    let mut placed = vec![(0_usize, 0_usize); count];
    for level in (0..count).rev() {
        at = aligned(at, held.block_bytes.max(4));
        placed[level] = (at, levels[level].len());
        at += levels[level].len();
    }
    let mut out = vec![0_u8; at];
    out[..12].copy_from_slice(&IDENTIFIER);
    let words = [
        held.vk_format,
        1,
        width,
        height,
        0,
        0,
        1,
        count as u32,
        0,
        index_end as u32,
        dfd.len() as u32,
        0,
        0,
    ];
    for (index, word) in words.iter().enumerate() {
        out[12 + 4 * index..16 + 4 * index].copy_from_slice(&word.to_le_bytes());
    }
    for (level, (offset, length)) in placed.iter().enumerate() {
        let base = 80 + 24 * level;
        out[base..base + 8].copy_from_slice(&(*offset as u64).to_le_bytes());
        out[base + 8..base + 16].copy_from_slice(&(*length as u64).to_le_bytes());
        out[base + 16..base + 24].copy_from_slice(&(*length as u64).to_le_bytes());
        out[*offset..*offset + length].copy_from_slice(&levels[level]);
    }
    out[index_end..index_end + dfd.len()].copy_from_slice(&dfd);
    out
}

pub struct Read {
    pub vk_format: u32,
    pub width: u32,
    pub height: u32,
    pub levels: Vec<Vec<u8>>,
}

fn word(bytes: &[u8], at: usize) -> Result<u32, String> {
    bytes
        .get(at..at + 4)
        .map(|held| u32::from_le_bytes(held.try_into().expect("four_bytes")))
        .ok_or_else(|| format!("ends before byte {at}"))
}

pub fn read(bytes: &[u8]) -> Result<Read, String> {
    if bytes.get(..12) != Some(&IDENTIFIER[..]) {
        return Err("not KTX2".into());
    }
    let vk_format = word(bytes, 12)?;
    let (width, height) = (word(bytes, 20)?, word(bytes, 24)?);
    let count = word(bytes, 40)? as usize;
    if word(bytes, 28)? != 0
        || word(bytes, 32)? != 0
        || word(bytes, 36)? != 1
        || word(bytes, 44)? != 0
    {
        return Err("not a plain 2D texture".into());
    }
    let mut levels = Vec::with_capacity(count);
    for level in 0..count {
        let base = 80 + 24 * level;
        let offset = word(bytes, base)? as usize;
        let length = word(bytes, base + 8)? as usize;
        let held = bytes
            .get(offset..offset + length)
            .ok_or_else(|| format!("level {level} lies outside the file"))?;
        levels.push(held.to_vec());
    }
    Ok(Read {
        vk_format,
        width,
        height,
        levels,
    })
}
