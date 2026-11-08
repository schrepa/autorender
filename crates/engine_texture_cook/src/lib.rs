#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod astc;
mod bc7;
mod brick;

pub use brick::{TEXTURE_COOK, TextureCook};
use engine_assets::Texture;
use engine_codec::Error;
use engine_textures::{Family, Source};

pub fn cook_job(input: &[u8]) -> Result<Vec<u8>, Error> {
    let (source, family) = engine_textures::cook_source(input)?;
    cook(&source, family).map(|texture| engine_textures::record(&texture))
}

pub fn cook(source: &Source, family: Family) -> Result<Texture, Error> {
    let plain = engine_textures::plain(source)?;
    let levels = plain
        .levels()
        .iter()
        .enumerate()
        .map(|(level, pixels)| {
            let width = (source.width >> level).max(1);
            let height = (source.height >> level).max(1);
            compress(width, height, pixels, family)
        })
        .collect();
    let texture = Texture::from_levels(family.format(), source.width, source.height, levels)?
        .with_color_space(plain.color_space());
    Ok(match plain.summary() {
        Some(summary) => texture.with_summary(summary),
        None => texture,
    })
}

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

fn rows_in_parallel<T: Send>(rows: usize, make: impl Fn(usize) -> T + Sync) -> Vec<T> {
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
    (0..rows).map(make).collect()
}
