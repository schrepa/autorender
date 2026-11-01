use std::sync::Arc;

use engine::prelude::*;
use engine_core::RecordExchange;

pub(crate) fn textures(
    (records, assets): (&mut RecordExchange, &mut AssetLibrary),
    name: String,
    textures: &[(&str, TextureUsage)],
    make: impl Fn() -> Result<Vec<TextureSource>, String> + Send + Sync + 'static,
) -> engine::Result<Vec<TextureHandle>> {
    let set = TextureSet {
        source: crate::SOURCE.hash(),
        name,
        textures: textures
            .iter()
            .map(|&(texture, usage)| (texture.to_owned(), usage))
            .collect(),
        make: MakeSources::Here(Arc::new(make)),
    };
    records.textures(assets, set)
}
