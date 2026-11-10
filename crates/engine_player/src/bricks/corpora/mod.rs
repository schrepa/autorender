engine::brick!(corpora);

#[cfg(all(feature = "dev", not(target_arch = "wasm32")))]
use engine_core::Corpus;
use engine_core::{Brick, Bricks, Source};

pub struct PlayerCorporaBrick;

impl Brick for PlayerCorporaBrick {
    const SOURCE: &'static Source = &SOURCE;
    #[cfg(all(feature = "dev", not(target_arch = "wasm32")))]
    const CORPORA: &'static [Corpus] = &[
        Corpus {
            name: "terrain",
            replays: crate::pure::terrain::KINDS,
            run: crate::pure::terrain::run,
        },
        Corpus {
            name: "caches",
            replays: &[],
            run: crate::pure::caches::run,
        },
    ];

    fn register(_: &mut Bricks<'_>) {}
}
