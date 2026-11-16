use std::sync::Arc;

use engine::prelude::{TextureSource, TextureUsage};

const SIZE: u32 = 1024;

pub(crate) struct Map {
    pub(crate) path: &'static str,
    pub(crate) file: Option<Arc<[u8]>>,
    pub(crate) usage: TextureUsage,
}

pub(crate) fn sources(maps: &[Map]) -> Result<Vec<TextureSource>, String> {
    maps.iter().map(source).collect()
}

fn source(map: &Map) -> Result<TextureSource, String> {
    let file = map
        .file
        .as_deref()
        .ok_or_else(|| format!("{} is missing", map.path))?;
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
