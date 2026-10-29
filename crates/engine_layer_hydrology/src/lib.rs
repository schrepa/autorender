#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/source.rs"));

mod bake;
mod given;
mod hydrology;
mod rivers;

use std::any::Any;
use std::sync::Arc;

use engine_core::{Brick, Bricks, Error, ErrorCode, parse_value};
use engine_settings::InvalidSetting;
use engine_terrain::codec::{Codec, Reader, Writer};
use engine_terrain::layer::{
    AddTerrainLayers, LayerBake, LayerFiles, LayerKind, Made, Place, TerrainLayer, maps,
    no_files, same_as,
};
use engine_terrain::{GivenRiver, Hydrology, Pyramid};
use glam::Vec2;
use serde::Serialize;
use serde_json::Value;

pub use bake::HydrologyWork;
pub use given::RiversLayer;
pub use hydrology::HydrologyLayer;

pub struct LayerHydrologyBrick;

impl Brick for LayerHydrologyBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks.terrain_layers(KINDS);
    }
}

pub static HYDROLOGY: LayerKind = LayerKind {
    name: "hydrology",
    source: crate::SOURCE,
    from_json: hydrology_from_json,
    read: hydrology_read,
    files: no_files,
    #[cfg(feature = "schema")]
    schema: hydrology_schema,
};

pub static RIVERS: LayerKind = LayerKind {
    name: "rivers",
    source: crate::SOURCE,
    from_json: rivers_from_json,
    read: rivers_read,
    files: no_files,
    #[cfg(feature = "schema")]
    schema: rivers_schema,
};

pub static KINDS: &[&LayerKind] = &[&HYDROLOGY, &RIVERS];

#[derive(Clone, Debug, PartialEq)]
pub struct HydrologyStage {
    pub layer: HydrologyLayer,
    pub baked: Option<Arc<Hydrology>>,
}

engine_terrain::codec::record!(HydrologyStage { layer, baked });

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

    fn bake(&self, pyramid: &Pyramid) -> Option<Box<dyn LayerBake>> {
        self.needs_bake()
            .then(|| Box::new(HydrologyWork::new(&self.layer, pyramid)) as Box<dyn LayerBake>)
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

    fn water(&self) -> Option<&Hydrology> {
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
    valid(layer.validate())?;
    Ok(Arc::new(layer))
}

fn rivers_read(input: &mut Reader<'_>) -> Made {
    let layer: RiversLayer = input.get()?;
    valid(layer.validate())?;
    Ok(Arc::new(layer))
}

#[cfg(feature = "schema")]
fn hydrology_schema() -> schemars::Schema {
    schemars::schema_for!(HydrologyLayer)
}

#[cfg(feature = "schema")]
fn rivers_schema() -> schemars::Schema {
    schemars::schema_for!(RiversLayer)
}
