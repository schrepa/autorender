#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/source.rs"));

mod bake;
mod profile;

use std::any::Any;
use std::sync::Arc;

use engine_core::{Brick, Bricks, ErrorCode, Pyramid, parse_value};
use engine_terrain::codec::{Reader, Writer};
use engine_terrain::layer::{
    self, AddTerrainLayers, Below, LayerBake, LayerFiles, LayerKind, Made, Row, TerrainLayer, maps,
};
use engine_terrain::path::{Bridge, PathSample};
use glam::Vec2;
use serde_json::Value;

use crate::bake::PathWork;
pub use crate::profile::{
    MASK_MARGIN, MAX_PATH_POINTS, PORTAL_HEIGHT, PROFILE_STEP, PathLayer, PathProfile, PathStats,
};

pub struct LayerPathsBrick;

impl Brick for LayerPathsBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks.terrain_layers(KINDS);
    }
}

pub static KINDS: &[&LayerKind] = &[&PATH_KIND];

pub static PATH_KIND: LayerKind = LayerKind {
    name: "path",
    source: SOURCE,
    from_json,
    read,
    files: layer::no_files,
    schema: engine_terrain::layer_schema!(schema),
};

#[derive(Clone, Debug, PartialEq)]
pub struct TerrainPath {
    pub settings: Arc<PathLayer>,
    pub profile: Option<Arc<PathProfile>>,
}

fn from_json(value: &Value, _files: &dyn LayerFiles) -> Made {
    let settings: PathLayer = parse_value(value.clone())?;
    made(Arc::new(settings), None)
}

fn read(input: &mut Reader<'_>) -> Made {
    let settings = input.get()?;
    let profile = input.get()?;
    made(settings, profile)
}

fn made(settings: Arc<PathLayer>, profile: Option<Arc<PathProfile>>) -> Made {
    settings
        .validate()
        .map_err(|invalid| invalid.error(ErrorCode::InvalidTerrain, "$"))?;
    Ok(Arc::new(TerrainPath { settings, profile }))
}

#[cfg(feature = "schema")]
fn schema() -> schemars::Schema {
    schemars::schema_for!(PathLayer)
}

impl TerrainLayer for TerrainPath {
    fn kind(&self) -> &'static LayerKind {
        &PATH_KIND
    }

    fn write(&self, out: &mut Writer) {
        out.put(&self.settings).put(&self.profile);
    }

    fn settings(&self) -> Value {
        serde_json::to_value(&*self.settings).unwrap_or_default()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn same(&self, other: &dyn TerrainLayer) -> bool {
        layer::same_as(self, other)
    }

    fn bakes(&self) -> bool {
        true
    }

    fn outputs(&self) -> Vec<String> {
        vec![maps::PATH.to_owned()]
    }

    fn apply_row(&self, below: Below<'_>, row: &Row<'_>, heights: &mut [f32]) {
        let Some(profile) = &self.profile else {
            return;
        };
        let (min, max) = (row.low, row.high);
        let (low, high) = profile.bounds();
        if low.x > max.x || low.y > max.y || high.x < min.x || high.y < min.y {
            return;
        }
        let z = row.z;
        for (height, &x) in heights.iter_mut().zip(row.xs) {
            if let Some((distance, road, false)) = profile.nearest(Vec2::new(x, z)) {
                *height += (road - *height) * profile.weight(distance) * below.shown(x, z);
            }
        }
    }

    fn may_hole(&self) -> bool {
        self.settings.tunnel_cut > 0.0
    }

    fn holes_reach(&self, min: Vec2, max: Vec2) -> bool {
        self.profile.as_ref().is_some_and(|profile| {
            profile.portal_bounds().is_some_and(|(low, high)| {
                low.x <= max.x && low.y <= max.y && high.x >= min.x && high.y >= min.y
            })
        })
    }

    fn hole(&self, x: f32, z: f32) -> bool {
        self.profile
            .as_ref()
            .is_some_and(|profile| profile.portal_hole(Vec2::new(x, z)))
    }

    fn needs_bake(&self) -> bool {
        self.profile.is_none()
    }

    fn bake(&self, _pyramid: &Pyramid) -> Option<Box<dyn LayerBake>> {
        if !self.needs_bake() {
            return None;
        }
        Some(Box::new(PathWork::start(&self.settings)))
    }

    fn unbaked(&self) -> Option<Arc<dyn TerrainLayer>> {
        Some(Arc::new(Self {
            settings: self.settings.clone(),
            profile: None,
        }))
    }

    fn has_path(&self) -> bool {
        self.profile.is_some()
    }

    fn path(&self, point: Vec2) -> Option<PathSample> {
        self.profile.as_ref().map(|profile| profile.sample(point))
    }

    fn gpu_path(&self, words: &mut Vec<u32>) {
        if let Some(profile) = &self.profile {
            profile.gpu_words(words);
        }
    }

    fn bridges(&self) -> Vec<Bridge> {
        self.profile
            .as_ref()
            .map_or_else(Vec::new, |profile| profile.bridges())
    }

    fn changed_regions(&self, before: &dyn TerrainLayer) -> Option<Vec<(Vec2, Vec2)>> {
        let before = before.as_any().downcast_ref::<Self>()?;
        if self.settings != before.settings {
            return None;
        }
        let (Some(now), Some(then)) = (&self.profile, &before.profile) else {
            return None;
        };
        let ((low, high), (from, to)) = (now.bounds(), then.bounds());
        Some(vec![(low.min(from), high.max(to))])
    }

    fn stats(&self) -> Option<Value> {
        self.profile
            .as_ref()
            .and_then(|profile| serde_json::to_value(profile.stats).ok())
    }
}
