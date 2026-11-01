#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod relief;

use engine_core::{Brick, Bricks, Error, ErrorCode, Job, JobCx, JobKind, JobRef, Source};
use engine_places::terrain::{self, TerrainLayers};

pub use relief::{HUMMOCKS, HummocksLayer, STEPS, StepsLayer};

pub struct LayerReliefBrick;

impl Brick for LayerReliefBrick {
    const SOURCE: &'static Source = &SOURCE;
    const JOBS: &'static [&'static JobKind] = &[&STEPS_JOB, &HUMMOCKS_JOB];

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .join::<TerrainLayers>(
                terrain::LayerKind {
                    layer: &STEPS,
                    job: JobRef::from(&STEPS_JOB),
                },
                None,
            )
            .join::<TerrainLayers>(
                terrain::LayerKind {
                    layer: &HUMMOCKS,
                    job: JobRef::from(&HUMMOCKS_JOB),
                },
                None,
            );
    }
}

pub static STEPS_JOB: JobKind = JobKind::new("layer_relief::steps", &SOURCE, refused);

pub static HUMMOCKS_JOB: JobKind = JobKind::new("layer_relief::hummocks", &SOURCE, refused);

fn refused(_input: &[u8], _cx: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    Err(Error::new(ErrorCode::JobUnsupported, "$.jobs"))
}
