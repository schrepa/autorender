//! The cook's textures: an RGBA8 image cooked into what devices sample.
//! Its mip chain averages as its usage reads it (colors in linear light,
//! normals as vectors, their shortening the slopes they average away, data
//! channel by channel); its summary is its
//! base level's ([`engine_assets::TextureSummary`]); every level is
//! compressed for a device family, BC7 for desktops and ASTC 4 × 4 for
//! phones and Apple's GPUs. A cooked texture travels as a record: a small
//! header (format, size, levels, whether alpha masks emission, the
//! summary), then its levels, which [`decode`] reads back into a texture.
//! Devices sampling neither family decompress the BC7 record into RGBA8
//! levels ([`decompress`]), the texels BC7 devices see. Every function
//! gives the same bits on every host: integer and plain `f32` arithmetic,
//! `libm` for powers, levels split among threads only by whole rows of
//! blocks.
//!
//! Cooking runs on the engine's workers too, as the job [`COOK_JOB`]: its
//! input an image and a family ([`cook_input`]), its output the texture's
//! record ([`cook_job`]), which hosts keep as engines key it. Workers run it
//! through [`TextureWorker`], whole at its first step, and hosts without
//! workers through [`run`], the same code.
#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

// `SOURCE`, the hash of the code cooking textures, which keys the records
// of the jobs cooking them.
include!(concat!(env!("OUT_DIR"), "/source.rs"));

mod astc;
mod bc7;
mod mips;

use engine_assets::{AssetError, Texture, TextureFormat, TextureSummary};

/// The job kind cooking an image for a family where the engine's workers
/// run it ([`cook_job`]), and where hosts without workers run it in place.
pub const COOK_JOB: &str = "texture_cook";

/// Bytes before a [`COOK_JOB`]'s texels: the image's width and height, its
/// usage and the family.
const COOK_HEADER: usize = 10;

/// The input of a [`COOK_JOB`] cooking `source` for `family`.
pub fn cook_input(source: &Source, family: Family) -> Vec<u8> {
    let mut input = Vec::with_capacity(COOK_HEADER + source.pixels.len());
    input.extend_from_slice(&source.width.to_le_bytes());
    input.extend_from_slice(&source.height.to_le_bytes());
    input.push(match source.usage {
        Usage::Color => 0,
        Usage::Data => 1,
        Usage::Normal => 2,
    });
    input.push(match family {
        Family::Bc7 => 0,
        Family::Astc => 1,
    });
    input.extend_from_slice(&source.pixels);
    input
}

/// A [`COOK_JOB`]'s output from its `input` ([`cook_input`]): the record
/// ([`record`]) of its image cooked for its family ([`cook`]).
pub fn cook_job(input: &[u8]) -> Result<Vec<u8>, String> {
    let header = input
        .get(..COOK_HEADER)
        .ok_or("a texture job is shorter than its header")?;
    let word = |at: usize| {
        u32::from_le_bytes([header[at], header[at + 1], header[at + 2], header[at + 3]])
    };
    let usage = match header[8] {
        0 => Usage::Color,
        1 => Usage::Data,
        2 => Usage::Normal,
        other => return Err(format!("a texture job names usage {other}")),
    };
    let family = match header[9] {
        0 => Family::Bc7,
        1 => Family::Astc,
        other => return Err(format!("a texture job names family {other}")),
    };
    let source = Source {
        width: word(0),
        height: word(4),
        pixels: input[COOK_HEADER..].to_vec(),
        usage,
    };
    cook(&source, family)
        .map(|texture| record(&texture))
        .map_err(|error| error.to_string())
}

/// The output of texture job `kind` on `input`, made whole; `None` where
/// `kind` names no texture job.
pub fn run(kind: &str, input: &[u8]) -> Option<Result<Vec<u8>, String>> {
    TextureWorker::runs(kind).then(|| cook_job(input))
}

/// The texture jobs a worker started, by id, with their inputs: each cooks
/// whole at its first step.
#[derive(Default)]
pub struct TextureWorker {
    running: Vec<(u64, Vec<u8>)>,
}

impl TextureWorker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether `kind` names a texture job.
    pub fn runs(kind: &str) -> bool {
        kind == COOK_JOB
    }

    /// Whether it started job `id`, which has yet to finish.
    pub fn holds(&self, id: u64) -> bool {
        self.running.iter().any(|(running, _)| *running == id)
    }

    /// Starts job `id` of `kind` on `input`, to [`Self::step`] until done;
    /// another of the same id stops.
    pub fn start(&mut self, id: u64, kind: &str, input: &[u8]) -> Result<(), String> {
        if !Self::runs(kind) {
            return Err(format!("unknown job kind {kind}"));
        }
        self.cancel(id);
        self.running.push((id, input.to_vec()));
        Ok(())
    }

    /// Works on job `id`: its record at its first step, whatever the
    /// budget, as a texture cooks whole. A job that failed or finished is
    /// gone.
    pub fn step(&mut self, id: u64, _budget: u64) -> Result<Option<Vec<u8>>, String> {
        let at = self
            .running
            .iter()
            .position(|(running, _)| *running == id)
            .ok_or("no such job")?;
        let (_, input) = self.running.remove(at);
        cook_job(&input).map(Some)
    }

    /// Stops job `id`, if it runs.
    pub fn cancel(&mut self, id: u64) {
        self.running.retain(|(running, _)| *running != id);
    }
}

/// How a texture's texels read, which its levels average by.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Usage {
    /// sRGB colors, alpha linear: base color and emissive maps.
    Color,
    /// Linear values, each channel apart: occlusion, roughness,
    /// metalness and height.
    Data,
    /// Tangent-space normals in red, green and blue (x along the image's
    /// right, y up it, z out), alpha linear.
    Normal,
}

/// The block-compressed formats devices sample: one family of records
/// each.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Family {
    /// BC7, which desktops sample (`texture-compression-bc`).
    Bc7,
    /// ASTC 4 × 4, which phones and Apple's GPUs sample
    /// (`texture-compression-astc`).
    Astc,
}

impl Family {
    /// Every family, in the order cooks make them.
    pub const ALL: [Family; 2] = [Family::Bc7, Family::Astc];

    /// The format of its textures.
    pub fn format(self) -> TextureFormat {
        match self {
            Self::Bc7 => TextureFormat::Bc7,
            Self::Astc => TextureFormat::Astc4x4,
        }
    }

    /// Its name, as record identities name it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Bc7 => "bc7",
            Self::Astc => "astc",
        }
    }
}

/// An image to cook: `width` × `height` RGBA8 texels, rows from the top,
/// read as `usage`. Compressed families take sides of multiples of 4.
#[derive(Clone, Debug)]
pub struct Source {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
    pub usage: Usage,
}

/// The RGBA8 levels of `source`'s mip chain, from its base down to 1 × 1:
/// what [`cook`] compresses.
pub fn levels(source: &Source) -> Vec<Vec<u8>> {
    mips::chain(source.width, source.height, &source.pixels, source.usage)
}

/// `source` cooked for `family`: every level of its mip chain compressed,
/// its summary, and whether any texel's alpha limits emission.
pub fn cook(source: &Source, family: Family) -> Result<Texture, AssetError> {
    let expected = TextureFormat::Rgba8.level_bytes(source.width, source.height);
    if source.pixels.len() != expected {
        return Err(AssetError::InvalidTextureLength {
            expected,
            actual: source.pixels.len(),
        });
    }
    let levels = levels(source);
    let compressed = levels
        .iter()
        .enumerate()
        .map(|(level, pixels)| {
            let width = (source.width >> level).max(1);
            let height = (source.height >> level).max(1);
            compress(width, height, pixels, family)
        })
        .collect();
    let masked = source
        .pixels
        .as_chunks::<4>()
        .0
        .iter()
        .any(|texel| texel[3] != 255);
    Ok(
        Texture::from_levels(family.format(), source.width, source.height, compressed)?
            .with_summary(TextureSummary::of_rgba8(&source.pixels))
            .with_emission_mask(masked),
    )
}

/// The blocks of an RGBA8 level `width` × `height` in `family`'s format,
/// row by row: texels past a small level's edge repeat its last row and
/// column. What [`cook`] makes of each level, and what levels made
/// elsewhere (impostors' layers, which the renderer bakes) compress by.
pub fn compress(width: u32, height: u32, pixels: &[u8], family: Family) -> Vec<u8> {
    let (columns, rows) = (width.div_ceil(4) as usize, height.div_ceil(4) as usize);
    let block_row = |row: usize| -> Vec<u8> {
        let mut out = Vec::with_capacity(columns * 16);
        for column in 0..columns {
            let texels: [[u8; 4]; 16] = std::array::from_fn(|index| {
                let x = (column * 4 + index % 4).min(width as usize - 1);
                let y = (row * 4 + index / 4).min(height as usize - 1);
                let at = (y * width as usize + x) * 4;
                [pixels[at], pixels[at + 1], pixels[at + 2], pixels[at + 3]]
            });
            out.extend_from_slice(&match family {
                Family::Bc7 => bc7::encode_block(&texels),
                Family::Astc => astc::encode_block(&texels),
            });
        }
        out
    };
    rows_in_parallel(rows, block_row).concat()
}

/// `make` of every row from 0 to `rows`, in order: split among the
/// machine's threads where it has them.
fn rows_in_parallel<T: Send>(rows: usize, make: impl Fn(usize) -> T + Sync) -> Vec<T> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let threads = std::thread::available_parallelism().map_or(1, usize::from);
        if threads > 1 && rows > 1 {
            let per = rows.div_ceil(threads);
            let make = &make;
            return std::thread::scope(|scope| {
                let parts: Vec<_> = (0..rows)
                    .step_by(per)
                    .map(|first| {
                        scope.spawn(move || {
                            (first..(first + per).min(rows))
                                .map(make)
                                .collect::<Vec<T>>()
                        })
                    })
                    .collect();
                parts
                    .into_iter()
                    .flat_map(|part| {
                        part.join()
                            .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
                    })
                    .collect()
            });
        }
    }
    (0..rows).map(make).collect()
}

/// A block of `format` every texel of which reads `color`, as nearly as
/// the format holds it (ASTC exactly): what an array's layers hold where a
/// material lacks a map. `None` for formats the cook does not write.
pub fn solid_block(format: TextureFormat, color: [u8; 4]) -> Option<[u8; 16]> {
    match format {
        TextureFormat::Bc7 => Some(bc7::encode_block(&[color; 16])),
        TextureFormat::Astc4x4 => Some(astc::encode_block(&[color; 16])),
        TextureFormat::Rgba8 | TextureFormat::Etc2Rgba8 => None,
    }
}

/// `texture`'s levels decoded to RGBA8, with its summary and emission
/// mask: what devices sampling no compressed format show of a BC7 texture.
/// Other formats are refused.
pub fn decompress(texture: &Texture) -> Result<Texture, AssetError> {
    if texture.format() != TextureFormat::Bc7 {
        return Err(AssetError::InvalidCompressedTexture {
            reason: "only BC7 textures decompress",
        });
    }
    let levels = texture
        .levels()
        .iter()
        .enumerate()
        .map(|(level, blocks)| {
            let width = (texture.width() >> level).max(1) as usize;
            let height = (texture.height() >> level).max(1) as usize;
            let columns = width.div_ceil(4);
            let mut pixels = vec![0_u8; width * height * 4];
            for (index, block) in blocks.as_chunks::<16>().0.iter().enumerate() {
                let (column, row) = (index % columns, index / columns);
                for (texel, color) in bc7::decode_block(block).iter().enumerate() {
                    let (x, y) = (column * 4 + texel % 4, row * 4 + texel / 4);
                    if x < width && y < height {
                        let at = (y * width + x) * 4;
                        pixels[at..at + 4].copy_from_slice(color);
                    }
                }
            }
            pixels
        })
        .collect();
    let decoded = Texture::from_levels(
        TextureFormat::Rgba8,
        texture.width(),
        texture.height(),
        levels,
    )?
    .with_emission_mask(texture.has_emission_mask());
    Ok(match texture.summary() {
        Some(summary) => decoded.with_summary(summary),
        None => decoded,
    })
}

/// A record's first bytes.
const MAGIC: &[u8; 4] = b"ETEX";
/// Bytes of a record's header: magic, format, emission mask, levels, a
/// spare zero, width, height and the summary's twelve values.
const HEADER: usize = 4 + 4 + 8 + 48;

/// `texture` as a record's bytes: its header, then its levels from the
/// base down. Textures without a summary write their base level's.
pub fn record(texture: &Texture) -> Vec<u8> {
    let summary = texture.summary().unwrap_or(TextureSummary {
        color: [1.0; 3],
        mean: [1.0; 4],
        slopes: 0.0,
        deviation: [0.0; 4],
    });
    let mut out = Vec::with_capacity(HEADER + texture.byte_len());
    out.extend_from_slice(MAGIC);
    out.push(match texture.format() {
        TextureFormat::Rgba8 => 0,
        TextureFormat::Bc7 => 1,
        TextureFormat::Etc2Rgba8 => 2,
        TextureFormat::Astc4x4 => 3,
    });
    out.push(u8::from(texture.has_emission_mask()));
    out.push(texture.levels().len() as u8);
    out.push(0);
    out.extend_from_slice(&texture.width().to_le_bytes());
    out.extend_from_slice(&texture.height().to_le_bytes());
    for value in summary
        .color
        .iter()
        .chain(&summary.mean)
        .chain([&summary.slopes])
        .chain(&summary.deviation)
    {
        out.extend_from_slice(&value.to_le_bytes());
    }
    for level in texture.levels() {
        out.extend_from_slice(level);
    }
    out
}

/// The texture a record's `bytes` hold, or why they hold none.
pub fn decode(bytes: &[u8]) -> Result<Texture, String> {
    let header = bytes
        .get(..HEADER)
        .ok_or("a texture record is shorter than its header")?;
    if &header[..4] != MAGIC {
        return Err("not a texture record".into());
    }
    let format = match header[4] {
        0 => TextureFormat::Rgba8,
        1 => TextureFormat::Bc7,
        2 => TextureFormat::Etc2Rgba8,
        3 => TextureFormat::Astc4x4,
        other => return Err(format!("a texture record names format {other}")),
    };
    let masked = header[5] != 0;
    let count = usize::from(header[6]);
    let word = |at: usize| {
        u32::from_le_bytes([header[at], header[at + 1], header[at + 2], header[at + 3]])
    };
    let (width, height) = (word(8), word(12));
    let value = |index: usize| f32::from_bits(word(16 + 4 * index));
    let summary = TextureSummary {
        color: [value(0), value(1), value(2)],
        mean: [value(3), value(4), value(5), value(6)],
        slopes: value(7),
        deviation: [value(8), value(9), value(10), value(11)],
    };
    let mut levels = Vec::with_capacity(count);
    let mut at = HEADER;
    for level in 0..count {
        let length = format.level_bytes((width >> level).max(1), (height >> level).max(1));
        let level_bytes = bytes
            .get(at..at + length)
            .ok_or("a texture record ends within its levels")?;
        levels.push(level_bytes.to_vec());
        at += length;
    }
    if at != bytes.len() {
        return Err("a texture record holds bytes past its levels".into());
    }
    Texture::from_levels(format, width, height, levels)
        .map(|texture| texture.with_summary(summary).with_emission_mask(masked))
        .map_err(|error| error.to_string())
}
