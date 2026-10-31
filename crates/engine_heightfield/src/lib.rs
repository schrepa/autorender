use std::sync::Arc;

use glam::Vec2;

pub trait TerrainLayer: Send + Sync + 'static {
    fn name(&self) -> &str;
    fn apply(&self, below: &Layers, tile: &mut LayerTile);
}

#[derive(Clone, Default)]
pub struct Layers {
    pub layers: Vec<Arc<dyn TerrainLayer>>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct LayerTile {
    pub origin: Vec2,
    pub spacing: f32,
    pub heights: Vec<f32>,
}

pub fn schema<T: schemars::JsonSchema>() -> serde_json::Value {
    serde_json::to_value(schemars::schema_for!(T)).unwrap_or(serde_json::Value::Null)
}

#[derive(Clone, Debug, PartialEq)]
pub struct HeightImage {
    pub width: usize,
    pub height: usize,
    pub values: Vec<f32>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct WaterGrid {
    pub width: usize,
    pub height: usize,
    pub spacing: f32,
    pub surface: Vec<f32>,
    pub depth: Vec<f32>,
    pub flow: Vec<[f32; 2]>,
    pub lake: Vec<u16>,
    pub levels: Vec<f32>,
    pub areas: Vec<f32>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct River {
    pub points: Vec<[f32; 2]>,
    pub discharge: Vec<f32>,
    pub half: Vec<f32>,
    pub depth: Vec<f32>,
    pub bed: Vec<f32>,
    pub surface: Vec<f32>,
    pub speed: Vec<f32>,
    pub falls: Vec<bool>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Rivers {
    pub rivers: Vec<River>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Fall {
    pub top: [f32; 3],
    pub bottom: [f32; 3],
    pub path: Vec<[f32; 3]>,
    pub times: Vec<f32>,
    pub aloft: Vec<bool>,
    pub widths: Vec<f32>,
    pub sky: Vec<f32>,
    pub width: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HydrologyBake {
    pub spacing: f32,
    pub flow: HeightImage,
    pub wetness: HeightImage,
    pub lakes: HeightImage,
    pub water: Arc<WaterGrid>,
    pub rivers: Arc<Rivers>,
    pub falls: Arc<[Fall]>,
}

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GivenRiver {
    pub name: Option<String>,
    pub points: Vec<[f32; 2]>,
    pub width: Vec<f32>,
}
