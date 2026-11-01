#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

use std::any::Any;
use std::sync::Arc;

use engine_codec::{Reader, Writer};
use engine_core::{
    Brick, Bricks, Error, ErrorCode, Job, JobCx, JobKind, JobRef, Source, parse_value,
};
use engine_heightfield::layer::{
    Below, LayerFiles, LayerKind, Made, Row, TerrainLayer, no_files, no_points, same_as,
};
use engine_heightfield::layers::reaches;
use engine_heightfield::{Stamp, StampOp};
use engine_places::terrain::{self, TerrainLayers};
use glam::Vec2;
use serde_json::Value;

pub struct LayerStampsBrick;

impl Brick for LayerStampsBrick {
    const SOURCE: &'static Source = &SOURCE;
    const JOBS: &'static [&'static JobKind] = &[&STAMP_JOB];

    fn register(bricks: &mut Bricks<'_>) {
        bricks.join::<TerrainLayers>(
            terrain::LayerKind {
                layer: &STAMP,
                job: JobRef::from(&STAMP_JOB),
            },
            None,
        );
    }
}

pub static STAMP_JOB: JobKind = JobKind::new("layer_stamps::stamp", &SOURCE, refused);

pub static STAMP: LayerKind = LayerKind {
    name: "stamp",
    source: SOURCE.hash(),
    from_json,
    read,
    files: no_files,
    schema: engine_heightfield::schema::<Stamp>,
    points: no_points,
};

fn refused(_input: &[u8], _cx: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    Err(Error::new(ErrorCode::JobUnsupported, "$.jobs"))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StampLayer {
    pub stamp: Stamp,
}

fn made(stamp: Stamp) -> Made {
    stamp
        .validate()
        .map_err(|error| error.coded(ErrorCode::InvalidValue))?;
    Ok(Arc::new(StampLayer { stamp }))
}

fn from_json(settings: &Value, _files: &dyn LayerFiles) -> Made {
    made(parse_value(settings.clone())?)
}

fn read(input: &mut Reader<'_>) -> Made {
    made(input.get()?)
}

impl StampLayer {
    fn holes(&self) -> bool {
        matches!(self.stamp.op, StampOp::Hole)
    }
}

impl TerrainLayer for StampLayer {
    fn kind(&self) -> &'static LayerKind {
        &STAMP
    }

    fn write(&self, out: &mut Writer) {
        out.put(&self.stamp);
    }

    fn settings(&self) -> Value {
        serde_json::to_value(self.stamp).expect("stamp_settings_serialize")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn same(&self, other: &dyn TerrainLayer) -> bool {
        same_as(self, other)
    }

    fn apply_row(&self, below: Below<'_>, row: &Row<'_>, heights: &mut [f32]) {
        let stamp = &self.stamp;
        let z = row.z;
        if !reaches(stamp.center, stamp.reach(), row.low, row.high) {
            return;
        }
        for (height, &x) in heights.iter_mut().zip(row.xs) {
            let weight = stamp.weight(x, z) * below.shown(x, z);
            *height = match stamp.op {
                StampOp::Raise(amount) => *height + amount * weight,
                StampOp::Lower(amount) => *height - amount * weight,
                StampOp::Flatten(level) => *height + (level - *height) * weight,
                StampOp::Hole => *height,
            };
        }
    }

    fn may_hole(&self) -> bool {
        self.holes()
    }

    fn holes_reach(&self, low: Vec2, high: Vec2) -> bool {
        self.holes() && reaches(self.stamp.center, self.stamp.radius, low, high)
    }

    fn hole(&self, x: f32, z: f32) -> bool {
        if !self.holes() {
            return false;
        }
        let stamp = &self.stamp;
        let dx = x - stamp.center[0];
        let dz = z - stamp.center[1];
        dx * dx + dz * dz < stamp.radius * stamp.radius
    }
}
