use std::any::Any;
use std::collections::BTreeMap;
use std::sync::Arc;

use engine_core::{Error, ErrorCode, Expected};
use engine_math::clamp;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::bake::Gather;
use crate::codec::{Codec, Reader, Writer, get_grid, invalid, on_grid, put_grid};
use crate::expr::Expr;
use crate::layer::{
    Below, LayerBake, LayerFiles, LayerKind, Made, Place, Row, TerrainLayer, no_files, same_as,
};
use crate::surface::{
    Compiled, Input, Reads, RowInputs, RowScratch, SampleShape, Scope, terrain_error, unknown_map,
};
use crate::tile::RiverInputs;
use crate::{Blend, HydrologySample, Layers, MAX_LEVEL, PathSample, Pyramid};

pub static EXPRESSION: LayerKind = LayerKind {
    name: "expression",
    source: crate::SOURCE,
    from_json,
    read,
    files: no_files,
    schema: crate::layer_schema!(schema),
    points: crate::layer::no_points,
};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct ExpressionSettings {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<Expr>,
    #[serde(default, skip_serializing_if = "is_add")]
    pub blend: Blend,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub maps: BTreeMap<String, Expr>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub level: Option<u32>,
}

fn is_add(blend: &Blend) -> bool {
    *blend == Blend::Add
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExpressionLayer {
    settings: ExpressionSettings,
    height: Option<Compiled>,
    maps: Vec<(String, Compiled)>,
    users: Vec<(String, String)>,
    reads: Reads,
    shaped: bool,
    shapes: Option<Arc<ShapeGrid>>,
}

impl ExpressionLayer {
    pub fn new(settings: ExpressionSettings) -> Result<Self, Error> {
        let mut scope = Scope::cpu();
        let height = match &settings.height {
            Some(expr) => Some(Compiled::from(expr, "$.height", 0, &mut scope)?),
            None => None,
        };
        let mut maps = Vec::with_capacity(settings.maps.len());
        for (name, expr) in &settings.maps {
            let at = format!("$.maps.{name}");
            if !engine_settings::is_id(name) {
                return Err(Error::new(ErrorCode::InvalidId, at)
                    .expected(Expected::Id)
                    .got(name));
            }
            if Input::named(name).is_some() {
                return Err(Error::new(ErrorCode::DuplicateId, at)
                    .expected(Expected::Unique)
                    .got(name));
            }
            maps.push((name.clone(), Compiled::from(expr, &at, 0, &mut scope)?));
        }
        let programs = || (height.iter()).chain(maps.iter().map(|(_, program)| program));
        let shaped = programs().any(Compiled::shaped);
        let reads = programs().fold(Reads::default(), |reads, program| {
            reads.and(program.reads())
        });
        match settings.level {
            Some(level) if level > MAX_LEVEL => {
                return Err(terrain_error("$.level")
                    .range(0.0, f64::from(MAX_LEVEL))
                    .got(level));
            }
            None if shaped => {
                return Err(
                    Error::new(ErrorCode::MissingField, "$.level").expected(Expected::Present)
                );
            }
            Some(_) | None => {}
        }
        Ok(Self {
            settings,
            height,
            maps,
            users: scope.users(),
            reads,
            shaped,
            shapes: None,
        })
    }

    pub fn authored(&self) -> &ExpressionSettings {
        &self.settings
    }

    fn cost(&self) -> u64 {
        1 + self.height.as_ref().map_or(0, Compiled::cost)
    }

    fn ready(&self, program: &Compiled) -> bool {
        !program.shaped() || self.shapes.is_some()
    }

    fn evaluate(
        &self,
        program: &Compiled,
        (below, row): (Below<'_>, &Row<'_>),
        heights: &[f32],
        out: &mut [f32],
    ) {
        let len = heights.len().min(row.xs.len()).min(out.len());
        let row = &Row {
            xs: &row.xs[..len],
            ..*row
        };
        let (xs, heights, z) = (row.xs, &heights[..len], row.z);
        let zs = vec![z; len];
        let mut gradient = vec![[0.0_f32; 2]; len];
        let mut curvature = vec![0.0_f32; len];
        if let Some(shapes) = &self.shapes {
            for ((gradient, curvature), &x) in gradient.iter_mut().zip(&mut curvature).zip(xs) {
                (*gradient, *curvature) = shapes.at(x, z);
            }
        }
        let mut erosion = vec![0.0_f32; len];
        if self.reads.erosion {
            for change in below.changes() {
                change.change_row(xs.iter().copied(), z, row.spacing, |index, value| {
                    erosion[index] += value;
                });
            }
        }
        let water = below.water().filter(|_| self.reads.water);
        let mut waters = Vec::with_capacity(len);
        let mut paths = Vec::with_capacity(len);
        for ((&x, &height), erosion) in xs.iter().zip(heights).zip(&mut erosion) {
            let shown = below.shown(x, z);
            let sample = match water {
                Some(water) if shown > 0.0 => RiverInputs::at(
                    &water.rivers,
                    [x, z],
                    (height, row.spacing),
                )
                .with_maps(water, [x, z], row.spacing),
                Some(_) | None => HydrologySample::default(),
            };
            waters.push(HydrologySample {
                flow: sample.flow * shown,
                wetness: sample.wetness * shown,
                lake: sample.lake * shown,
                ..sample
            });
            *erosion *= shown;
            paths.push(if shown > 0.0 && self.reads.path {
                below.path_at(x, z)
            } else {
                PathSample::FAR
            });
        }
        let stack = below.stack();
        let under = &stack.layers()[..below.index()];
        let maps: Vec<Vec<f32>> = (self.users.iter())
            .map(|(name, _)| {
                let mut values = vec![0.0_f32; len];
                read_map(
                    stack,
                    producer(under, name),
                    name,
                    row,
                    heights,
                    &mut values,
                );
                values
            })
            .collect();
        let inputs = RowInputs {
            xs,
            zs: &zs,
            spacing: row.spacing,
            height: heights,
            gradient: &gradient,
            curvature: &curvature,
            water: &waters,
            erosion: &erosion,
            path: &paths,
        };
        program.evaluate_row(&inputs, &maps, &mut out[..len], &mut RowScratch::default());
    }
}

impl TerrainLayer for ExpressionLayer {
    fn kind(&self) -> &'static LayerKind {
        &EXPRESSION
    }

    fn write(&self, out: &mut Writer) {
        let maps: Vec<(String, Expr)> = (self.settings.maps.iter())
            .map(|(name, expr)| (name.clone(), expr.clone()))
            .collect();
        out.put(&self.settings.height)
            .put(&self.settings.blend)
            .put(&maps)
            .put(&self.settings.level)
            .put(&self.shapes);
    }

    fn settings(&self) -> Value {
        serde_json::to_value(&self.settings).unwrap_or(Value::Null)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn same(&self, other: &dyn TerrainLayer) -> bool {
        same_as(self, other)
    }

    fn bakes(&self) -> bool {
        self.shaped
    }

    fn outputs(&self) -> Vec<String> {
        self.maps.iter().map(|(name, _)| name.clone()).collect()
    }

    fn check(&self, place: Place<'_>) -> Result<(), Error> {
        let published: Vec<String> = (place.below().iter())
            .flat_map(|layer| layer.outputs())
            .collect();
        match (self.users.iter()).find(|(name, _)| !published.contains(name)) {
            Some((name, path)) => Err(unknown_map(path, name, published)),
            None => Ok(()),
        }
    }

    fn apply_row(&self, below: Below<'_>, row: &Row<'_>, heights: &mut [f32]) {
        let Some(program) = &self.height else {
            return;
        };
        if !self.ready(program) {
            return;
        }
        let mut values = vec![0.0_f32; heights.len()];
        self.evaluate(program, (below, row), heights, &mut values);
        let blend = self.settings.blend;
        for ((height, &value), &x) in heights.iter_mut().zip(&values).zip(row.xs) {
            *height = blend.apply(*height, value, below.shown(x, row.z));
        }
    }

    fn row_cost(&self) -> u64 {
        self.cost()
    }

    fn point_cost(&self) -> u64 {
        self.cost()
    }

    fn needs_bake(&self) -> bool {
        self.shaped && self.shapes.is_none()
    }

    fn bake(&self, pyramid: &Pyramid) -> Option<Box<dyn LayerBake>> {
        let level = self.settings.level.filter(|_| self.needs_bake())?;
        Some(Box::new(ShapeBake {
            layer: self.clone(),
            gather: Gather::new(level, pyramid),
        }))
    }

    fn unbaked(&self) -> Option<Arc<dyn TerrainLayer>> {
        self.shaped.then(|| {
            Arc::new(Self {
                shapes: None,
                ..self.clone()
            }) as Arc<dyn TerrainLayer>
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
        match self.maps.iter().find(|(known, _)| known == name) {
            Some((_, program)) if self.ready(program) => {
                self.evaluate(program, (below, row), heights, out);
                true
            }
            Some(_) | None => false,
        }
    }
}

struct ShapeBake {
    layer: ExpressionLayer,
    gather: Gather,
}

impl LayerBake for ShapeBake {
    fn advance(
        &mut self,
        below: &Arc<Layers>,
        pyramid: &Pyramid,
        budget: &mut u64,
    ) -> Option<Arc<dyn TerrainLayer>> {
        if !self.gather.advance(below, pyramid, budget) {
            return None;
        }
        let heights: Box<[f32]> = (self.gather.heights.iter())
            .map(|&height| on_grid(height))
            .collect();
        *budget = budget.saturating_sub(heights.len() as u64);
        let spacing = pyramid.spacing(self.gather.level);
        let shapes = ShapeGrid::new(spacing, self.gather.width, heights);
        Some(Arc::new(ExpressionLayer {
            shapes: Some(Arc::new(shapes)),
            ..self.layer.clone()
        }))
    }
}

#[derive(Debug, PartialEq)]
struct ShapeGrid {
    spacing: f32,
    width: usize,
    heights: Box<[f32]>,
    gradient: Box<[[f32; 2]]>,
    curvature: Box<[f32]>,
}

impl ShapeGrid {
    fn new(spacing: f32, width: usize, heights: Box<[f32]>) -> Self {
        let rows = heights.len() / width;
        let at = |column: usize, row: usize| heights[row * width + column];
        let mut gradient = Vec::with_capacity(heights.len());
        let mut curvature = Vec::with_capacity(heights.len());
        for row in 0..rows {
            let (up, down) = (row.saturating_sub(1), (row + 1).min(rows - 1));
            for column in 0..width {
                let (left, right) = (column.saturating_sub(1), (column + 1).min(width - 1));
                let shape = SampleShape::of(
                    [up, row, down].map(|row| [left, column, right].map(|column| at(column, row))),
                    spacing,
                );
                gradient.push(shape.gradient);
                curvature.push(shape.curvature);
            }
        }
        Self {
            spacing,
            width,
            heights,
            gradient: gradient.into_boxed_slice(),
            curvature: curvature.into_boxed_slice(),
        }
    }

    fn at(&self, x: f32, z: f32) -> ([f32; 2], f32) {
        let rows = self.heights.len() / self.width;
        let u = clamp(x / self.spacing, 0.0, (self.width - 1) as f32);
        let v = clamp(z / self.spacing, 0.0, (rows - 1) as f32);
        let (u0, v0) = (u.floor(), v.floor());
        let (fu, fv) = (u - u0, v - v0);
        let (column, row) = (u0 as usize, v0 as usize);
        let (right, down) = ((column + 1).min(self.width - 1), (row + 1).min(rows - 1));
        let mut gradient = [0.0_f32; 2];
        let mut curvature = 0.0_f32;
        for (at, weight) in [
            (row * self.width + column, (1.0 - fu) * (1.0 - fv)),
            (row * self.width + right, fu * (1.0 - fv)),
            (down * self.width + column, (1.0 - fu) * fv),
            (down * self.width + right, fu * fv),
        ] {
            gradient[0] += weight * self.gradient[at][0];
            gradient[1] += weight * self.gradient[at][1];
            curvature += weight * self.curvature[at];
        }
        (gradient, curvature)
    }
}

impl Codec for ShapeGrid {
    fn write(&self, out: &mut Writer) {
        out.put(&self.spacing).put(&self.width);
        put_grid(out, &self.heights, self.width);
    }

    fn read(input: &mut Reader<'_>) -> Result<Self, Error> {
        let spacing: f32 = input.get()?;
        let width: usize = input.get()?;
        if width < 2 || spacing.is_nan() || spacing <= 0.0 {
            return Err(invalid("$.shapes").got(width));
        }
        let heights = get_grid(input, width)?;
        if !heights.len().is_multiple_of(width)
            || heights.len() / width < 2
            || heights.iter().any(|height| height.is_nan())
        {
            return Err(invalid("$.shapes").got(heights.len()));
        }
        Ok(Self::new(spacing, width, heights))
    }
}

pub(crate) fn producer(layers: &[Arc<dyn TerrainLayer>], name: &str) -> Option<usize> {
    (layers.iter()).rposition(|layer| layer.outputs().iter().any(|output| output == name))
}

pub(crate) fn read_map(
    stack: &Layers,
    producer: Option<usize>,
    name: &str,
    row: &Row<'_>,
    heights: &[f32],
    out: &mut [f32],
) {
    let read = match producer {
        Some(index) => stack
            .layers()
            .get(index)
            .is_some_and(|layer| layer.map_row(name, Below::new(stack, index), row, heights, out)),
        None => false,
    };
    if !read {
        out.fill(0.0);
    }
}

fn from_json(value: &Value, _files: &dyn LayerFiles) -> Made {
    let settings: ExpressionSettings = engine_core::parse_value(value.clone())?;
    Ok(Arc::new(ExpressionLayer::new(settings)?))
}

fn read(input: &mut Reader<'_>) -> Made {
    let height = input.get()?;
    let blend = input.get()?;
    let maps: Vec<(String, Expr)> = input.get()?;
    let level = input.get()?;
    let shapes: Option<Arc<ShapeGrid>> = input.get()?;
    let layer = ExpressionLayer::new(ExpressionSettings {
        height,
        blend,
        maps: maps.into_iter().collect(),
        level,
    })?;
    let shaped = layer.shaped;
    Ok(Arc::new(ExpressionLayer {
        shapes: shapes.filter(|_| shaped),
        ..layer
    }))
}

#[cfg(feature = "schema")]
fn schema() -> schemars::Schema {
    schemars::schema_for!(ExpressionSettings)
}
