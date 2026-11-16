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
use engine_core::{Brick, Bricks, Source, parse_value};
use engine_heightfield::layer::{
    Below, ChangeMap, LayerFiles, LayerKind, Made, Row, TerrainLayer, maps, no_files, no_points,
    same_as,
};
use engine_heightfield::{INVALID_TERRAIN, Layers, Pyramid, TileHeights};
use engine_open_world_kit::terrain::TerrainLayers;
use serde_json::Value;

pub use crate::erosion::{
    BRINK, Beds, CLIFF, Erosion, ErosionLayer, ErosionStats, FOOT, HARDNESS,
};

pub struct LayerErosionBrick;

impl Brick for LayerErosionBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .errors(&[INVALID_TERRAIN])
            .join::<TerrainLayers>(&EROSION, None);
    }
}

pub static EROSION: LayerKind = LayerKind {
    name: "erosion",
    source: SOURCE.hash(),
    from_json,
    read,
    files: no_files,
    schema: engine_heightfield::schema::<ErosionLayer>,
    points: no_points,
};

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
        .map_err(|invalid| invalid.error(INVALID_TERRAIN, "$"))?;
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
        (maps::CHANGE.iter().copied())
            .chain([CLIFF, BRINK, FOOT, HARDNESS])
            .map(str::to_owned)
            .collect()
    }

    fn apply_row(&self, below: Below<'_>, row: &Row<'_>, heights: &mut [f32]) {
        let Some(baked) = &self.baked else {
            return;
        };
        let before = heights.to_vec();
        let at = (row.xs, row.z, row.spacing);
        baked.wear_row(at, self.layer.mantle, &before, |index, worn| {
            heights[index] += worn * below.shown(row.xs[index], row.z);
        });
    }

    fn needs_bake(&self) -> bool {
        self.baked.is_none()
    }

    fn bake(
        &self,
        below: &Arc<Layers>,
        pyramid: &Pyramid,
        tiles: TileHeights,
    ) -> Option<Arc<dyn TerrainLayer>> {
        self.needs_bake()
            .then(|| bake::erosion_stage(&self.layer, below, pyramid, tiles))
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
            image: &baked.cover,
            range: baked.cover_range,
        })
    }

    fn map_row(
        &self,
        name: &str,
        below: Below<'_>,
        row: &Row<'_>,
        heights: &[f32],
        out: &mut [f32],
    ) -> bool {
        if name == HARDNESS {
            let beds = &self.layer.beds;
            for ((share, &x), &height) in out.iter_mut().zip(row.xs).zip(heights) {
                *share = beds.surface(x, row.z, height) * below.shown(x, row.z);
            }
            return true;
        }
        let Some(baked) = self.baked.as_deref() else {
            return false;
        };
        let count = out.len().min(row.xs.len());
        let at = (&row.xs[..count], row.z, row.spacing);
        baked.share_row(name, at, |index, share| {
            out[index] = share * below.shown(row.xs[index], row.z);
        })
    }

    fn stats(&self) -> Option<Value> {
        self.baked
            .as_deref()
            .map(|baked| serde_json::to_value(baked.stats).expect("erosion_stats_serialize"))
    }
}
