#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod bake;
mod given;
mod hydrology;
mod rivers;

use std::any::Any;
use std::sync::Arc;

use engine_codec::{Codec, Reader, Writer};
use engine_core::{Brick, Bricks, Error, ErrorCode, InvalidSetting, Source, parse_value};
use engine_heightfield::layer::{
    LayerFiles, LayerKind, Made, Place, TerrainLayer, maps, no_files, no_points, same_as,
};
use engine_heightfield::{GivenRiver, HydrologyBake, Layers, Pyramid, TileHeights};
use engine_places::terrain::TerrainLayers;
use glam::Vec2;
use serde::Serialize;
use serde_json::Value;

pub use given::RiversLayer;
pub use hydrology::HydrologyLayer;

pub struct LayerHydrologyBrick;

impl Brick for LayerHydrologyBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .join::<TerrainLayers>(&HYDROLOGY, None)
            .join::<TerrainLayers>(&RIVERS, None);
    }
}

pub static HYDROLOGY: LayerKind = LayerKind {
    name: "hydrology",
    source: SOURCE.hash(),
    from_json: hydrology_from_json,
    read: hydrology_read,
    files: no_files,
    schema: engine_heightfield::schema::<HydrologyLayer>,
    points: no_points,
};

pub static RIVERS: LayerKind = LayerKind {
    name: "rivers",
    source: SOURCE.hash(),
    from_json: rivers_from_json,
    read: rivers_read,
    files: no_files,
    schema: engine_heightfield::schema::<RiversLayer>,
    points: no_points,
};

#[derive(Clone, Debug, PartialEq)]
pub struct HydrologyStage {
    pub layer: HydrologyLayer,
    pub baked: Option<Arc<HydrologyBake>>,
}

engine_codec::record!(HydrologyStage { layer, baked });

impl TerrainLayer for HydrologyStage {
    fn kind(&self) -> &'static LayerKind {
        &HYDROLOGY
    }

    fn write(&self, out: &mut Writer) {
        Codec::write(self, out);
    }

    fn settings(&self) -> Value {
        settings_json(&self.layer)
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
        maps::WATER.iter().map(|name| (*name).to_owned()).collect()
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
            .then(|| bake::hydrology_stage(&self.layer, below, pyramid, tiles))
    }

    fn unbaked(&self) -> Option<Arc<dyn TerrainLayer>> {
        Some(Arc::new(Self {
            layer: self.layer,
            baked: None,
        }))
    }

    fn makes_water(&self) -> bool {
        true
    }

    fn water(&self) -> Option<&HydrologyBake> {
        self.baked.as_deref()
    }

    fn changed_regions(&self, before: &dyn TerrainLayer) -> Option<Vec<(Vec2, Vec2)>> {
        let then = before.as_any().downcast_ref::<Self>()?;
        if self.layer != then.layer {
            return None;
        }
        let (Some(now), Some(then)) = (&self.baked, &then.baked) else {
            return None;
        };
        if now.flow != then.flow
            || now.wetness != then.wetness
            || now.lakes != then.lakes
            || now.water != then.water
        {
            return None;
        }
        Some(now.rivers.changes_from(&then.rivers))
    }
}

impl TerrainLayer for RiversLayer {
    fn kind(&self) -> &'static LayerKind {
        &RIVERS
    }

    fn write(&self, out: &mut Writer) {
        Codec::write(self, out);
    }

    fn settings(&self) -> Value {
        settings_json(self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn same(&self, other: &dyn TerrainLayer) -> bool {
        same_as(self, other)
    }

    fn check(&self, place: Place<'_>) -> Result<(), Error> {
        if place.above().iter().any(|layer| layer.makes_water()) {
            Ok(())
        } else {
            Err(Error::new(ErrorCode::MissingLayer, "$").variants([HYDROLOGY.name]))
        }
    }

    fn given_rivers(&self) -> Option<&[GivenRiver]> {
        Some(&self.lines)
    }
}

fn settings_json(settings: &impl Serialize) -> Value {
    serde_json::to_value(settings).expect("layer_settings_serialize")
}

fn valid(checked: Result<(), InvalidSetting>) -> Result<(), Error> {
    checked.map_err(|invalid| Error::from(invalid).coded(ErrorCode::InvalidTerrain))
}

fn rivers_valid(layer: &RiversLayer) -> Result<(), Error> {
    layer
        .validate()
        .map_err(|error| error.coded(ErrorCode::InvalidTerrain))
}

fn hydrology_from_json(value: &Value, _files: &dyn LayerFiles) -> Made {
    let layer: HydrologyLayer = parse_value(value.clone())?;
    valid(layer.validate())?;
    Ok(Arc::new(HydrologyStage { layer, baked: None }))
}

fn hydrology_read(input: &mut Reader<'_>) -> Made {
    let stage: HydrologyStage = input.get()?;
    valid(stage.layer.validate())?;
    Ok(Arc::new(stage))
}

fn rivers_from_json(value: &Value, _files: &dyn LayerFiles) -> Made {
    let layer: RiversLayer = parse_value(value.clone())?;
    rivers_valid(&layer)?;
    Ok(Arc::new(layer))
}

fn rivers_read(input: &mut Reader<'_>) -> Made {
    let layer: RiversLayer = input.get()?;
    rivers_valid(&layer)?;
    Ok(Arc::new(layer))
}
