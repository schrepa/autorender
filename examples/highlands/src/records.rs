use std::sync::Arc;

use engine::prelude::*;
use engine_core::RecordExchange;

pub(crate) fn textures(
    (records, assets): (&mut RecordExchange, &mut AssetLibrary),
    source: [u8; 32],
    name: String,
    textures: &[(&str, TextureUsage)],
    make: impl Fn() -> Result<Option<Vec<TextureSource>>, String> + Send + Sync + 'static,
) -> engine::Result<Vec<TextureHandle>> {
    let set = TextureSet {
        source,
        name,
        textures: textures
            .iter()
            .map(|&(texture, usage)| (texture.to_owned(), usage))
            .collect(),
        make: Arc::new(make),
    };
    records.textures(assets, set)
}
