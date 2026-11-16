use std::sync::Arc;

use engine::host::decode_image;
use engine::prelude::{GameFiles, Resource};
use engine_core::collections::HashMap;
use engine_core::{
    AssetLibrary, Error, ErrorCode, Expected, RecordExchange, TextureHandle, TextureSet,
    TextureSource, TextureUsage,
};
use serde::Deserialize;

use super::sprays::{self, Sheet, Source};
use super::sprigs::Atlas;

pub(crate) const FILES: [&str; 20] = [
    "highlands/plants/spruce_twigs_albedo.png",
    "highlands/plants/spruce_twigs_normal.png",
    "highlands/plants/spruce_twigs.json",
    "highlands/plants/pine_twigs_albedo.png",
    "highlands/plants/pine_twigs_normal.png",
    "highlands/plants/pine_twigs.json",
    "highlands/plants/birch_leaves_albedo.png",
    "highlands/plants/birch_leaves_normal.png",
    "highlands/plants/birch_leaves.json",
    "highlands/plants/undergrowth_albedo.png",
    "highlands/plants/undergrowth_normal.png",
    "highlands/bark/spruce_albedo.png",
    "highlands/bark/spruce_normal.png",
    "highlands/bark/spruce_orm.png",
    "highlands/bark/pine_albedo.png",
    "highlands/bark/pine_normal.png",
    "highlands/bark/pine_orm.png",
    "highlands/bark/birch_albedo.png",
    "highlands/bark/birch_normal.png",
    "highlands/bark/birch_orm.png",
];

pub(crate) const ATLAS_MAPS: [(&str, TextureUsage); 2] = [
    ("albedo", TextureUsage::Color),
    ("normal", TextureUsage::Normal),
];

pub(crate) const BARK_MAPS: [(&str, TextureUsage); 3] = [
    ("albedo", TextureUsage::Color),
    ("normal", TextureUsage::Normal),
    ("orm", TextureUsage::Data),
];

const PNG_HEADER: &[u8; 8] = b"\x89PNG\r\n\x1a\n";

const COVERED: u8 = 8;

#[derive(Resource, Default)]
pub struct Photographs {
    files: HashMap<&'static str, Arc<[u8]>>,
}

#[derive(Deserialize)]
struct Cell {
    #[serde(default)]
    name: String,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    root: [f32; 2],
}

impl Photographs {
    pub fn read(files: &GameFiles) -> Self {
        Self {
            files: FILES
                .iter()
                .filter_map(|&path| Some((path, Arc::from(files.bytes(path)?))))
                .collect(),
        }
    }

    fn file(&self, path: &str) -> Result<&Arc<[u8]>, Error> {
        self.files.get(path).ok_or_else(|| {
            Error::new(ErrorCode::MissingFile, "$.photographs")
                .expected(Expected::File)
                .got(path)
        })
    }

    fn cells(&self, name: &str) -> Result<Vec<Cell>, Error> {
        let invalid = |problem: String| {
            Error::new(ErrorCode::InvalidRecord, format!("$.photographs.{name}")).got(problem)
        };
        let cells = self.file(&format!("highlands/plants/{name}.json"))?;
        let cells: Vec<Cell> =
            serde_json::from_slice(cells).map_err(|error| invalid(error.to_string()))?;
        match cells.is_empty() {
            true => Err(invalid("no cells".to_owned())),
            false => Ok(cells),
        }
    }

    pub(crate) fn atlas(&self, name: &str) -> Result<Atlas, Error> {
        let image = self.file(&format!("highlands/plants/{name}_albedo.png"))?;
        let size = png_size(image).ok_or_else(|| {
            Error::new(ErrorCode::InvalidRecord, format!("$.photographs.{name}"))
                .expected(Expected::Image)
        })?;
        Ok(Atlas::new(
            size,
            self.cells(name)?
                .iter()
                .map(|cell| [cell.x, cell.y, cell.w, cell.h, cell.root[0], cell.root[1]]),
        ))
    }

    pub(crate) fn sprays(
        &self,
        (records, assets): (&mut RecordExchange, &mut AssetLibrary),
        name: &str,
    ) -> Result<Vec<TextureHandle>, Error> {
        let sources: Vec<Source> = self
            .cells(name)?
            .iter()
            .map(|cell| Source {
                rect: [cell.x, cell.y, cell.w, cell.h],
                root: cell.root,
                cluster: cell.name.starts_with("twig"),
            })
            .collect();
        let files = ["albedo", "normal", "json"].map(|part| match part {
            "json" => format!("highlands/plants/{name}.json"),
            map => format!("highlands/plants/{name}_{map}.png"),
        });
        let [leaves, relief, cells] = [
            self.file(&files[0])?,
            self.file(&files[1])?,
            self.file(&files[2])?,
        ]
        .map(Arc::clone);
        let digest = [&leaves, &relief, &cells]
            .iter()
            .fold(0xcbf2_9ce4_8422_2325, |hash, bytes| mixed(hash, bytes));
        let set = TextureSet {
            source: super::SOURCE.hash(),
            name: format!("highlands/plants/{name}/sprays/{digest:016x}"),
            textures: ATLAS_MAPS
                .iter()
                .map(|&(map, usage)| (map.to_owned(), usage))
                .collect(),
            make: Arc::new(move || {
                let sheet = |path: &str, bytes: &[u8]| {
                    let image = decode_image(path, bytes).map_err(|error| error.to_string())?;
                    Ok::<_, String>(Sheet {
                        width: image.width as usize,
                        height: image.height as usize,
                        pixels: image.pixels,
                    })
                };
                let [color, normal] = sprays::compose(
                    &sheet(&files[0], &leaves)?,
                    &sheet(&files[1], &relief)?,
                    &sources,
                );
                let size = sprays::SIZE as u32;
                Ok(vec![
                    TextureSource {
                        width: size,
                        height: size,
                        pixels: bled(color, sprays::SIZE, sprays::SIZE),
                        usage: TextureUsage::Color,
                    },
                    TextureSource {
                        width: size,
                        height: size,
                        pixels: normal,
                        usage: TextureUsage::Normal,
                    },
                ])
            }),
        };
        records.textures(assets, set)
    }

    pub(crate) fn textures(
        &self,
        (records, assets): (&mut RecordExchange, &mut AssetLibrary),
        (folder, name): (&str, &str),
        maps: &'static [(&'static str, TextureUsage)],
    ) -> Result<Vec<TextureHandle>, Error> {
        let files = maps
            .iter()
            .map(|(map, _)| {
                let path = format!("highlands/{folder}/{name}_{map}.png");
                self.file(&path).map(|bytes| (path, Arc::clone(bytes)))
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let digest = files
            .iter()
            .fold(0xcbf2_9ce4_8422_2325, |hash, (_, bytes)| mixed(hash, bytes));
        let set = TextureSet {
            source: super::SOURCE.hash(),
            name: format!("highlands/{folder}/{name}/{digest:016x}"),
            textures: maps
                .iter()
                .map(|&(map, usage)| (map.to_owned(), usage))
                .collect(),
            make: Arc::new(move || {
                files
                    .iter()
                    .zip(maps)
                    .map(|((path, bytes), &(_, usage))| source(path, bytes, usage))
                    .collect()
            }),
        };
        records.textures(assets, set)
    }
}

fn source(path: &str, bytes: &[u8], usage: TextureUsage) -> Result<TextureSource, String> {
    let image = decode_image(path, bytes).map_err(|error| error.to_string())?;
    let (width, height) = (image.width as usize, image.height as usize);
    let mut alphas = image.pixels.iter().skip(3).step_by(4);
    let cut_out = usage == TextureUsage::Color && alphas.any(|&alpha| alpha < COVERED);
    let pixels = if cut_out {
        bled(image.pixels, width, height)
    } else {
        image.pixels
    };
    Ok(TextureSource {
        width: image.width,
        height: image.height,
        pixels,
        usage,
    })
}

fn mixed(hash: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(hash, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn png_size(bytes: &[u8]) -> Option<[f32; 2]> {
    let header = bytes.get(..24)?;
    if &header[..8] != PNG_HEADER {
        return None;
    }
    let word = |at: usize| {
        u32::from_be_bytes([header[at], header[at + 1], header[at + 2], header[at + 3]]) as f32
    };
    Some([word(16), word(20)])
}

struct Level {
    width: usize,
    height: usize,
    texels: Vec<[u8; 4]>,
}

impl Level {
    fn halved(&self) -> Self {
        let (width, height) = (self.width.div_ceil(2), self.height.div_ceil(2));
        let mut texels = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                let mut sum = [0_u32; 3];
                let mut count = 0_u32;
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let (sx, sy) = (2 * x + dx, 2 * y + dy);
                    if sx >= self.width || sy >= self.height {
                        continue;
                    }
                    let texel = self.texels[sy * self.width + sx];
                    if texel[3] != 0 {
                        for (total, &channel) in sum.iter_mut().zip(&texel) {
                            *total += u32::from(channel);
                        }
                        count += 1;
                    }
                }
                texels.push(match count {
                    0 => [0; 4],
                    count => {
                        let mean = sum.map(|total| ((total + count / 2) / count) as u8);
                        [mean[0], mean[1], mean[2], 255]
                    }
                });
            }
        }
        Self {
            width,
            height,
            texels,
        }
    }

    fn fill_from(&mut self, coarser: &Self) {
        for y in 0..self.height {
            for x in 0..self.width {
                let texel = &mut self.texels[y * self.width + x];
                if texel[3] == 0 {
                    *texel = coarser.texels[(y / 2) * coarser.width + x / 2];
                }
            }
        }
    }
}

fn bled(mut pixels: Vec<u8>, width: usize, height: usize) -> Vec<u8> {
    let texels = pixels
        .as_chunks::<4>()
        .0
        .iter()
        .map(|texel| match texel[3] >= COVERED {
            true => [texel[0], texel[1], texel[2], 255],
            false => [0; 4],
        })
        .collect();
    let mut levels = vec![Level {
        width,
        height,
        texels,
    }];
    while levels
        .last()
        .is_some_and(|level| level.width > 1 || level.height > 1)
    {
        let next = levels.last().map(Level::halved);
        levels.extend(next);
    }
    for finer in (0..levels.len() - 1).rev() {
        let (fine, coarse) = levels.split_at_mut(finer + 1);
        fine[finer].fill_from(&coarse[0]);
    }
    for (texel, filled) in pixels.as_chunks_mut::<4>().0.iter_mut().zip(&levels[0].texels) {
        if texel[3] < COVERED {
            texel[..3].copy_from_slice(&filled[..3]);
        }
    }
    pixels
}
