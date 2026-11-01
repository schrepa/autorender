#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod bake;
mod erosion;

use std::any::Any;
use std::sync::Arc;

use engine_codec::{Reader, Writer};
use engine_core::{
    Brick, Bricks, Error, ErrorCode, Job, JobCx, JobKind, JobRef, Source, parse_value,
};
use engine_heightfield::layer::{
    Below, ChangeMap, LayerBake, LayerFiles, LayerKind, Made, Row, TerrainLayer, maps, no_files,
    no_points, same_as,
};
use engine_heightfield::{Pyramid, StartTile};
use engine_places::terrain::{self, TerrainLayers};
use serde_json::Value;

use crate::bake::ErosionWork;
pub use crate::erosion::{Erosion, ErosionLayer, ErosionStats};

pub struct LayerErosionBrick;

impl Brick for LayerErosionBrick {
    const SOURCE: &'static Source = &SOURCE;
    const JOBS: &'static [&'static JobKind] = &[&EROSION_JOB];

    fn register(bricks: &mut Bricks<'_>) {
        bricks.join::<TerrainLayers>(
            terrain::LayerKind {
                layer: &EROSION,
                job: JobRef::from(&EROSION_JOB),
            },
            None,
        );
    }
}

pub static EROSION_JOB: JobKind = JobKind::new("layer_erosion::erosion", &SOURCE, refused);

pub static EROSION: LayerKind = LayerKind {
    name: "erosion",
    source: SOURCE.hash(),
    from_json,
    read,
    files: no_files,
    schema: engine_heightfield::schema::<ErosionLayer>,
    points: no_points,
};

fn refused(_input: &[u8], _cx: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    Err(Error::new(ErrorCode::JobUnsupported, "$.jobs"))
}

#[derive(Clone, Debug, PartialEq)]
pub struct ErosionStage {
    pub layer: ErosionLayer,
    pub baked: Option<Arc<Erosion>>,
}

engine_codec::record!(ErosionStage { layer, baked });

fn made(stage: ErosionStage) -> Made {
    stage
        .layer
        .validate()
        .map_err(|invalid| invalid.error(ErrorCode::InvalidValue, "$"))?;
    Ok(Arc::new(stage))
}

fn from_json(value: &Value, _files: &dyn LayerFiles) -> Made {
    made(ErosionStage {
        layer: parse_value(value.clone())?,
        baked: None,
    })
}

fn read(input: &mut Reader<'_>) -> Made {
    made(input.get()?)
}

impl TerrainLayer for ErosionStage {
    fn kind(&self) -> &'static LayerKind {
        &EROSION
    }

    fn write(&self, out: &mut Writer) {
        out.put(self);
    }

    fn settings(&self) -> Value {
        serde_json::to_value(self.layer).expect("erosion_settings_serialize")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn same(&self, other: &dyn TerrainLayer) -> bool {
        same_as(self, other)
    }

    fn bakes(&self) -> bool {
        true
    }

    fn outputs(&self) -> Vec<String> {
        maps::CHANGE.map(str::to_owned).to_vec()
    }

    fn apply_row(&self, below: Below<'_>, row: &Row<'_>, heights: &mut [f32]) {
        let Some(baked) = &self.baked else {
            return;
        };
        let before = heights.to_vec();
        baked.wear_row(row.xs, row.z, row.spacing, &before, |index, worn| {
            heights[index] += worn * below.shown(row.xs[index], row.z);
        });
    }

    fn needs_bake(&self) -> bool {
        self.baked.is_none()
    }

    fn bake(&self, pyramid: &Pyramid, tiles: StartTile) -> Option<Box<dyn LayerBake>> {
        if !self.needs_bake() {
            return None;
        }
        Some(Box::new(ErosionWork::new(&self.layer, pyramid, tiles)))
    }

    fn unbaked(&self) -> Option<Arc<dyn TerrainLayer>> {
        Some(Arc::new(Self {
            layer: self.layer,
            baked: None,
        }))
    }

    fn change(&self) -> Option<ChangeMap<'_>> {
        self.baked.as_deref().map(|baked| ChangeMap {
            spacing: baked.spacing,
            image: &baked.change,
            range: baked.range,
        })
    }

    fn stats(&self) -> Option<Value> {
        self.baked
            .as_deref()
            .map(|baked| serde_json::to_value(baked.stats).expect("erosion_stats_serialize"))
    }
}
