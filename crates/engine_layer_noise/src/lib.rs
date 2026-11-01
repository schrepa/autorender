#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod layer;
mod tables;
mod tile;

pub use layer::{Noise, NoiseLayer};

use engine_core::{Brick, Bricks, Error, ErrorCode, Job, JobCx, JobKind, JobRef, Source};
use engine_heightfield::layer::{LayerKind, no_files, no_points};
use engine_places::terrain::{self, TerrainLayers};

pub struct LayerNoiseBrick;

impl Brick for LayerNoiseBrick {
    const SOURCE: &'static Source = &SOURCE;
    const JOBS: &'static [&'static JobKind] = &[&NOISE_JOB];

    fn register(bricks: &mut Bricks<'_>) {
        bricks.join::<TerrainLayers>(
            terrain::LayerKind {
                layer: &NOISE_KIND,
                job: JobRef::from(&NOISE_JOB),
            },
            None,
        );
    }
}

pub static NOISE_JOB: JobKind = JobKind::new("layer_noise::noise", &SOURCE, refused);

pub static NOISE_KIND: LayerKind = LayerKind {
    name: "noise",
    source: SOURCE.hash(),
    from_json: layer::from_json,
    read: layer::read,
    files: no_files,
    schema: engine_heightfield::schema::<NoiseLayer>,
    points: no_points,
};

fn refused(_input: &[u8], _cx: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    Err(Error::new(ErrorCode::JobUnsupported, "$.jobs"))
}
