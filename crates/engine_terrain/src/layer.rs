use std::any::Any;
use std::fmt::Debug;
use std::sync::{Arc, PoisonError, RwLock};

use bevy_ecs::prelude::Resource;
use engine_core::{Bricks, Error, ErrorCode, Expected, List};
use glam::Vec2;
use serde_json::Value;

use crate::codec::{Reader, Writer};
use crate::{Bridge, GivenRiver, HeightImage, Hydrology, Layers, PathSample, Pyramid, TileKey};

pub const TYPE_KEY: &str = "type";
pub const ID_KEY: &str = "id";

pub mod maps {
    pub const HEIGHT: &str = "height";
    pub const SLOPE: &str = "slope";
    pub const CURVATURE: &str = "curvature";
    pub const FLOW: &str = "flow";
    pub const WETNESS: &str = "wetness";
    pub const LAKE: &str = "lake";
    pub const RIVER: &str = "river";
    pub const GRAIN: &str = "grain";
    pub const POWER: &str = "power";
    pub const FLOOD: &str = "flood";
    pub const WORN: &str = "worn";
    pub const DEPOSIT: &str = "deposit";
    pub const PATH: &str = "path";
    pub const WATER: [&str; 7] = [FLOW, WETNESS, LAKE, RIVER, GRAIN, POWER, FLOOD];
    pub const CHANGE: [&str; 2] = [WORN, DEPOSIT];
    pub const SHAPE: [&str; 3] = [HEIGHT, SLOPE, CURVATURE];
    pub const GPU: [&str; 13] = [
        HEIGHT, SLOPE, CURVATURE, FLOW, WETNESS, LAKE, RIVER, GRAIN, POWER, FLOOD, WORN, DEPOSIT,
        PATH,
    ];
}

pub type Made = Result<Arc<dyn TerrainLayer>, Error>;

pub struct LayerKind {
    pub name: &'static str,
    pub source: [u8; 32],
    pub from_json: fn(&Value, &dyn LayerFiles) -> Made,
    pub read: fn(&mut Reader<'_>) -> Made,
    pub files: fn(&Value) -> Vec<String>,
    pub schema: fn() -> Value,
}

#[macro_export]
macro_rules! layer_schema {
    ($schema:path) => {{
        #[cfg(feature = "schema")]
        fn layer_schema() -> ::serde_json::Value {
            ::serde_json::to_value($schema()).unwrap_or(::serde_json::Value::Null)
        }
        #[cfg(not(feature = "schema"))]
        fn layer_schema() -> ::serde_json::Value {
            ::serde_json::Value::Null
        }
        layer_schema
    }};
}

impl Debug for LayerKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LayerKind")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

pub trait LayerFiles {
    fn buffer(&self, uri: &str, path: &str) -> Result<Option<Arc<[u8]>>, Error>;
}

pub struct NoFiles;

impl LayerFiles for NoFiles {
    fn buffer(&self, _uri: &str, _path: &str) -> Result<Option<Arc<[u8]>>, Error> {
        Ok(None)
    }
}

pub fn no_files(_settings: &Value) -> Vec<String> {
    Vec::new()
}

pub trait TerrainLayer: Send + Sync + Debug + 'static {
    fn kind(&self) -> &'static LayerKind;

    fn write(&self, out: &mut Writer);

    fn settings(&self) -> Value;

    fn as_any(&self) -> &dyn Any;

    fn same(&self, other: &dyn TerrainLayer) -> bool;

    fn bakes(&self) -> bool {
        false
    }

    fn outputs(&self) -> Vec<String> {
        Vec::new()
    }

    fn check(&self, _place: Place<'_>) -> Result<(), Error> {
        Ok(())
    }

    fn apply_row(&self, _below: Below<'_>, _row: &Row<'_>, _heights: &mut [f32]) {}

    fn apply_point(&self, below: Below<'_>, point: &SamplePoint, height: &mut f32) {
        let at = Vec2::new(point.x, point.z);
        let row = Row {
            xs: std::slice::from_ref(&point.x),
            z: point.z,
            spacing: point.spacing,
            low: at,
            high: at,
        };
        self.apply_row(below, &row, std::slice::from_mut(height));
    }

    fn tile(&self, _frame: &TileFrame<'_>) -> Option<Box<dyn LayerTile>> {
        None
    }

    fn row_cost(&self) -> u64 {
        1
    }

    fn point_cost(&self) -> u64 {
        1
    }

    fn may_hole(&self) -> bool {
        false
    }

    fn holes_reach(&self, _low: Vec2, _high: Vec2) -> bool {
        false
    }

    fn hole(&self, _x: f32, _z: f32) -> bool {
        false
    }

    fn needs_bake(&self) -> bool {
        false
    }

    fn bake(&self, _pyramid: &Pyramid) -> Option<Box<dyn LayerBake>> {
        None
    }

    fn unbaked(&self) -> Option<Arc<dyn TerrainLayer>> {
        None
    }

    fn makes_water(&self) -> bool {
        false
    }

    fn water(&self) -> Option<&Hydrology> {
        None
    }

    fn given_rivers(&self) -> Option<&[GivenRiver]> {
        None
    }

    fn change(&self) -> Option<ChangeMap<'_>> {
        None
    }

    fn has_path(&self) -> bool {
        false
    }

    fn path(&self, _point: Vec2) -> Option<PathSample> {
        None
    }

    fn gpu_path(&self, _words: &mut Vec<u32>) {}

    fn bridges(&self) -> Vec<Bridge> {
        Vec::new()
    }

    fn fine_region(&self) -> Option<FineRegion> {
        None
    }

    fn map_row(
        &self,
        _name: &str,
        _below: Below<'_>,
        _row: &Row<'_>,
        _heights: &[f32],
        _out: &mut [f32],
    ) -> bool {
        false
    }

    fn changed_regions(&self, _before: &dyn TerrainLayer) -> Option<Vec<(Vec2, Vec2)>> {
        None
    }

    fn stats(&self) -> Option<Value> {
        None
    }
}

impl PartialEq for dyn TerrainLayer {
    fn eq(&self, other: &Self) -> bool {
        self.same(other)
    }
}

pub fn same_as<T: PartialEq + 'static>(this: &T, other: &dyn TerrainLayer) -> bool {
    other
        .as_any()
        .downcast_ref::<T>()
        .is_some_and(|other| other == this)
}

pub trait LayerTile: Send {
    fn prepare(&mut self, budget: &mut u64) -> bool;

    fn apply_row(&mut self, below: Below<'_>, row: i32, at: &Row<'_>, heights: &mut [f32]);
}

pub trait LayerBake: Send {
    fn advance(
        &mut self,
        below: &Arc<Layers>,
        pyramid: &Pyramid,
        budget: &mut u64,
    ) -> Option<Arc<dyn TerrainLayer>>;
}

#[derive(Clone, Copy, Debug)]
pub struct Row<'a> {
    pub xs: &'a [f32],
    pub z: f32,
    pub spacing: f32,
    pub low: Vec2,
    pub high: Vec2,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplePoint {
    pub sample: [i64; 2],
    pub x: f32,
    pub z: f32,
    pub spacing: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct TileFrame<'a> {
    pub key: TileKey,
    pub first: [i64; 2],
    pub spacing: f32,
    pub xs: &'a [f32],
    pub zs: &'a [f32],
    pub low: Vec2,
    pub high: Vec2,
}

#[derive(Clone, Copy)]
pub struct Place<'a> {
    pub layers: &'a [Arc<dyn TerrainLayer>],
    pub index: usize,
}

impl Place<'_> {
    pub fn below(&self) -> &[Arc<dyn TerrainLayer>] {
        &self.layers[..self.index]
    }

    pub fn above(&self) -> &[Arc<dyn TerrainLayer>] {
        &self.layers[self.index + 1..]
    }
}

#[derive(Clone, Copy)]
pub struct Below<'a> {
    stack: &'a Layers,
    index: usize,
}

impl<'a> Below<'a> {
    pub fn new(stack: &'a Layers, index: usize) -> Self {
        Self { stack, index }
    }

    pub fn stack(&self) -> &'a Layers {
        self.stack
    }

    pub fn index(&self) -> usize {
        self.index
    }

    pub fn layers(&self) -> &'a [Arc<dyn TerrainLayer>] {
        self.stack.layers().get(..self.index).unwrap_or_default()
    }

    pub fn shown(&self, x: f32, z: f32) -> f32 {
        self.stack.shown(x, z)
    }

    pub fn water(&self) -> Option<&'a Hydrology> {
        self.layers().iter().find_map(|layer| layer.water())
    }

    pub fn changes(&self) -> impl Iterator<Item = ChangeMap<'a>> + 'a {
        self.layers().iter().filter_map(|layer| layer.change())
    }

    pub fn path_at(&self, x: f32, z: f32) -> PathSample {
        crate::layers::nearest_path(self.layers(), Vec2::new(x, z))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ChangeMap<'a> {
    pub spacing: f32,
    pub image: &'a HeightImage,
    pub range: [f32; 2],
}

impl ChangeMap<'_> {
    pub fn change_row(
        &self,
        xs: impl IntoIterator<Item = f32>,
        z: f32,
        spacing: f32,
        each: impl FnMut(usize, f32),
    ) {
        image_row(self.image, self.range, (self.spacing, spacing), xs, z, each);
    }
}

pub fn image_row(
    image: &HeightImage,
    [low, high]: [f32; 2],
    (own, spacing): (f32, f32),
    xs: impl IntoIterator<Item = f32>,
    z: f32,
    mut each: impl FnMut(usize, f32),
) {
    let level = crate::hydrology::image_level(own, image.levels(), spacing);
    let scale = 1.0 / own;
    let us = xs.into_iter().map(|x| x * scale);
    image.sample_smooth_row(us, z * scale, level, |index, sample| {
        each(index, low + (high - low) * sample * (1.0 / 65_535.0));
    });
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FineRegion {
    pub low: Vec2,
    pub high: Vec2,
    pub texel: f32,
}

static KINDS: RwLock<Vec<&'static LayerKind>> = RwLock::new(Vec::new());

pub fn register_kinds(kinds: &[&'static LayerKind]) {
    let mut held = KINDS.write().unwrap_or_else(PoisonError::into_inner);
    for &kind in kinds {
        match held.iter_mut().find(|known| known.name == kind.name) {
            Some(known) => *known = kind,
            None => held.push(kind),
        }
    }
}

pub fn layer_kind(name: &str) -> Option<&'static LayerKind> {
    KINDS
        .read()
        .unwrap_or_else(PoisonError::into_inner)
        .iter()
        .find(|kind| kind.name == name)
        .copied()
}

pub fn layer_kinds() -> Vec<&'static LayerKind> {
    KINDS.read().unwrap_or_else(PoisonError::into_inner).clone()
}

pub fn layer_names() -> Vec<&'static str> {
    layer_kinds().into_iter().map(|kind| kind.name).collect()
}

pub fn unknown_kind(path: &str, got: &str) -> Error {
    Error::new(ErrorCode::UnknownVariant, path)
        .got(got)
        .variants(layer_names())
}

pub fn layer_from_json(
    value: &Value,
    files: &dyn LayerFiles,
    path: &str,
) -> Result<Arc<dyn TerrainLayer>, Error> {
    let Some(object) = value.as_object() else {
        return Err(Error::new(ErrorCode::InvalidType, path).expected(Expected::Object));
    };
    let type_path = format!("{path}.{TYPE_KEY}");
    let Some(name) = object.get(TYPE_KEY) else {
        return Err(Error::new(ErrorCode::MissingField, type_path).variants(layer_names()));
    };
    let Some(name) = name.as_str() else {
        return Err(Error::new(ErrorCode::InvalidType, type_path).expected(Expected::String));
    };
    let kind = layer_kind(name).ok_or_else(|| unknown_kind(&type_path, name))?;
    let mut settings = object.clone();
    settings.remove(TYPE_KEY);
    settings.remove(ID_KEY);
    (kind.from_json)(&Value::Object(settings), files).map_err(|error| error.within(path))
}

pub fn layer_json(layer: &dyn TerrainLayer) -> Value {
    let mut settings = match layer.settings() {
        Value::Object(settings) => settings,
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) | Value::Array(_) => {
            serde_json::Map::new()
        }
    };
    settings.insert(TYPE_KEY.to_owned(), Value::from(layer.kind().name));
    Value::Object(settings)
}

pub fn write_entry(out: &mut Writer, layer: &dyn TerrainLayer) {
    let kind = layer.kind();
    out.put(&kind.name.to_owned()).put(&kind.source);
    layer.write(out);
}

pub fn read_entry(input: &mut Reader<'_>) -> Result<Arc<dyn TerrainLayer>, Error> {
    let name: String = input.get()?;
    let source: [u8; 32] = input.get()?;
    let kind = layer_kind(&name).ok_or_else(|| {
        crate::codec::invalid("$.layer")
            .got(&name)
            .variants(layer_names())
    })?;
    if kind.source != source {
        return Err(crate::codec::invalid("$.source")
            .expected(Expected::Matching)
            .got(&name));
    }
    (kind.read)(input)
}

pub struct TerrainLayersList;

impl List for TerrainLayersList {
    const NAME: &'static str = "terrain_layers";
}

#[derive(Resource, Clone, Debug, Default)]
pub struct TerrainLayerKinds {
    kinds: Vec<&'static LayerKind>,
}

impl TerrainLayerKinds {
    pub fn add(&mut self, kinds: &[&'static LayerKind]) {
        for &kind in kinds {
            if !self.kinds.iter().any(|known| known.name == kind.name) {
                self.kinds.push(kind);
            }
        }
    }

    pub fn get(&self, name: &str) -> Option<&'static LayerKind> {
        self.kinds.iter().find(|kind| kind.name == name).copied()
    }

    pub fn kinds(&self) -> &[&'static LayerKind] {
        &self.kinds
    }

    pub fn names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.kinds.iter().map(|kind| kind.name)
    }
}

pub trait AddTerrainLayers {
    fn terrain_layers(&mut self, kinds: &[&'static LayerKind]) -> &mut Self;
}

impl AddTerrainLayers for Bricks<'_> {
    fn terrain_layers(&mut self, kinds: &[&'static LayerKind]) -> &mut Self {
        register_kinds(kinds);
        let world = self.world_mut();
        if !world.contains_resource::<TerrainLayerKinds>() {
            world.init_resource::<TerrainLayerKinds>();
        }
        world.resource_mut::<TerrainLayerKinds>().add(kinds);
        self.list::<TerrainLayersList>()
    }
}
