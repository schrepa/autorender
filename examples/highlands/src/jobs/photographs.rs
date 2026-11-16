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
use crate::sources::sets::{ATLAS_PARTS, BARK_PARTS, cells_file, map_files, spray_files};
use crate::sources::{self, Sources};

pub(crate) const FILES: [&str; 3] = [
    "highlands/plants/spruce_twigs.json",
    "highlands/plants/pine_twigs.json",
    "highlands/plants/birch_leaves.json",
];

pub(crate) const ATLAS_MAPS: [(&str, TextureUsage); 2] = [
    (ATLAS_PARTS[0], TextureUsage::Color),
    (ATLAS_PARTS[1], TextureUsage::Normal),
];

pub(crate) const BARK_MAPS: [(&str, TextureUsage); 3] = [
    (BARK_PARTS[0], TextureUsage::Color),
    (BARK_PARTS[1], TextureUsage::Normal),
    (BARK_PARTS[2], TextureUsage::Data),
];

const COVERED: u8 = 8;

#[derive(Resource, Default)]
pub struct Photographs {
    cells: HashMap<&'static str, Arc<[u8]>>,
    sources: Sources,
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

fn missing(path: &str) -> Error {
    Error::new(ErrorCode::MissingFile, "$.photographs")
        .expected(Expected::File)
        .got(path)
}

fn set_digest(files: &[String]) -> Result<u64, Error> {
    sources::set_digest(files).ok_or_else(|| {
        let absent = files.iter().find(|path| sources::file(path).is_none());
        missing(absent.map_or("", String::as_str))
    })
}

fn read_all(sources: &Sources, files: &[String]) -> Result<Option<Vec<Arc<[u8]>>>, String> {
    let read = files
        .iter()
        .map(|path| sources.bytes(path))
        .collect::<Result<Vec<_>, String>>()?;
    Ok(read.into_iter().collect())
}

impl Photographs {
    pub(crate) fn read(files: &GameFiles, sources: &Sources) -> Self {
        Self {
            cells: FILES
                .iter()
                .filter_map(|&path| Some((path, Arc::from(files.bytes(path)?))))
                .collect(),
            sources: sources.clone(),
        }
    }

    fn cells(&self, name: &str) -> Result<Vec<Cell>, Error> {
        let invalid = |problem: String| {
            Error::new(ErrorCode::InvalidRecord, format!("$.photographs.{name}")).got(problem)
        };
        let path = cells_file(name);
        let cells = self
            .cells
            .get(path.as_str())
            .ok_or_else(|| missing(&path))?;
        let cells: Vec<Cell> =
            serde_json::from_slice(cells).map_err(|error| invalid(error.to_string()))?;
        match cells.is_empty() {
            true => Err(invalid("no cells".to_owned())),
            false => Ok(cells),
        }
    }

    pub(crate) fn atlas(&self, name: &str) -> Result<Atlas, Error> {
        let files = map_files("plants", name, &ATLAS_PARTS[..1]);
        let path = files.first().map_or("", String::as_str);
        let size = sources::file(path).ok_or_else(|| missing(path))?.size;
        if size == [0, 0] {
            return Err(
                Error::new(ErrorCode::InvalidRecord, format!("$.photographs.{name}"))
                    .expected(Expected::Image),
            );
        }
        Ok(Atlas::new(
            size.map(|side| side as f32),
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
        let files = spray_files(name);
        let digest = set_digest(&files)?;
        let photographs = self.sources.clone();
        let set = TextureSet {
            source: super::SOURCE.hash(),
            name: format!("highlands/plants/{name}/sprays/{digest:016x}"),
            textures: ATLAS_MAPS
                .iter()
                .map(|&(map, usage)| (map.to_owned(), usage))
                .collect(),
            make: Arc::new(move || {
                let Some(sheets) = read_all(&photographs, &files[..2])? else {
                    return Ok(None);
                };
                let sheet = |path: &str, bytes: &[u8]| {
                    let image = decode_image(path, bytes).map_err(|error| error.to_string())?;
                    Ok::<_, String>(Sheet {
                        width: image.width as usize,
                        height: image.height as usize,
                        pixels: image.pixels,
                    })
                };
                let [color, normal] = sprays::compose(
                    &sheet(&files[0], &sheets[0])?,
                    &sheet(&files[1], &sheets[1])?,
                    &sources,
                );
                let size = sprays::SIZE as u32;
                Ok(Some(vec![
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
                ]))
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
        let parts: Vec<&str> = maps.iter().map(|&(map, _)| map).collect();
        let files = map_files(folder, name, &parts);
        let digest = set_digest(&files)?;
        let photographs = self.sources.clone();
        let set = TextureSet {
            source: super::SOURCE.hash(),
            name: format!("highlands/{folder}/{name}/{digest:016x}"),
            textures: maps
                .iter()
                .map(|&(map, usage)| (map.to_owned(), usage))
                .collect(),
            make: Arc::new(move || {
                let Some(read) = read_all(&photographs, &files)? else {
                    return Ok(None);
                };
                files
                    .iter()
                    .zip(read)
                    .zip(maps)
                    .map(|((path, bytes), &(_, usage))| source(path, &bytes, usage))
                    .collect::<Result<Vec<_>, String>>()
                    .map(Some)
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
