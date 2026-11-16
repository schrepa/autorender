use std::sync::Arc;

use engine::prelude::{TextureSource, TextureUsage};

use crate::sources::{self, Sources};

const SIZE: u32 = 1024;

pub(crate) struct Map {
    pub(crate) path: &'static str,
    pub(crate) usage: TextureUsage,
}

pub(crate) fn sources(maps: &[Map], files: &Sources) -> Result<Option<Vec<TextureSource>>, String> {
    let read = maps
        .iter()
        .map(|map| match sources::file(map.path) {
            Some(_) => files.bytes(map.path),
            None => Err(format!("{} is missing", map.path)),
        })
        .collect::<Result<Vec<_>, String>>()?;
    let Some(read) = read.into_iter().collect::<Option<Vec<Arc<[u8]>>>>() else {
        return Ok(None);
    };
    maps.iter()
        .zip(read)
        .map(|(map, file)| source(map, &file))
        .collect::<Result<Vec<_>, String>>()
        .map(Some)
}

fn source(map: &Map, file: &[u8]) -> Result<TextureSource, String> {
    let image = engine::host::decode_image(map.path, file).map_err(|error| error.to_string())?;
    if (image.width, image.height) != (SIZE, SIZE) {
        return Err(format!(
            "{} is {}x{}, ground maps are {SIZE}x{SIZE}",
            map.path, image.width, image.height
        ));
    }
    Ok(TextureSource {
        width: image.width,
        height: image.height,
        pixels: image.pixels,
        usage: map.usage,
    })
}
