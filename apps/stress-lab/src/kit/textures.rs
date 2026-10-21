//! The worlds' generated textures, loading as texture records
//! (`engine::TextureRecords`) keyed by the Stress Lab's code
//! ([`crate::SOURCE`]): a world names a set, its textures and how each
//! reads, and what generates their sources, and has their handles at once.
//! The engine fetches the records the device's family samples, and where no
//! pack or store holds one, generates the set and cooks it in place, as the
//! cook does.

use std::sync::Arc;

use engine::prelude::*;

/// Asks `records` for the textures `textures` of the set `name`, which
/// `make` generates (one source each, in order): their handles in `assets`.
pub(crate) fn request(
    (records, assets): (&mut TextureRecords, &mut AssetLibrary),
    name: String,
    textures: &[(&str, TextureUsage)],
    make: impl Fn() -> Result<Vec<TextureSource>, String> + Send + Sync + 'static,
) -> engine::Result<Vec<TextureHandle>> {
    let set = TextureSet {
        source: crate::SOURCE,
        name,
        textures: textures
            .iter()
            .map(|&(texture, usage)| (texture.to_owned(), usage))
            .collect(),
        make: Arc::new(make),
    };
    records.request(assets, set).map_err(engine::Error::from)
}
