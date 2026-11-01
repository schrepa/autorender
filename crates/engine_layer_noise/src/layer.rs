use std::any::Any;
use std::sync::Arc;

use engine_codec::{Reader, Writer};
use engine_core::{Error, ErrorCode, InvalidSetting, Rule, Setting, fields, parse_value};
use engine_heightfield::Blend;
use engine_heightfield::layer::{
    Below, LayerFiles, LayerKind, LayerTile, Made, SamplePoint, TerrainLayer, TileFrame, same_as,
};
use engine_heightfield::layers::MAX_HEIGHT;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::tables::{NoiseTables, state_len};
use crate::tile::{
    COARSE_COST, GRID_STEP, NOISE_SAMPLE_COST, NoiseTile, QUARTER_WEIGHTS, interpolate,
};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[schemars(transform = state_noise_rules)]
#[serde(default, deny_unknown_fields)]
pub struct NoiseLayer {
    pub seed: u32,
    pub height: f32,
    pub offset: f32,
    pub wavelength: f32,
    pub ridges: f32,
    pub warp: f32,
    pub erosion: f32,
    pub smoothing: f32,
    pub valleys: f32,
    pub roughness: f32,
    pub blend: Blend,
}

impl Default for NoiseLayer {
    fn default() -> Self {
        Self {
            seed: 0,
            height: 900.0,
            offset: 0.0,
            wavelength: 3000.0,
            ridges: 0.3,
            warp: 0.35,
            erosion: 0.7,
            smoothing: 1.0,
            valleys: 0.7,
            roughness: 0.5,
            blend: Blend::Add,
        }
    }
}

impl NoiseLayer {
    fn rules(&self) -> [(&'static str, f32, Rule); 9] {
        [
            ("height", self.height, fields::range(0.0, MAX_HEIGHT)),
            ("offset", self.offset, fields::range(-MAX_HEIGHT, MAX_HEIGHT)),
            ("wavelength", self.wavelength, fields::range(16.0, 1.0e6)),
            ("ridges", self.ridges, fields::range(0.0, 1.0)),
            ("warp", self.warp, fields::range(0.0, 1.0)),
            ("erosion", self.erosion, fields::range(0.0, 1.0)),
            ("smoothing", self.smoothing, fields::range(0.0, 1.0)),
            ("valleys", self.valleys, fields::range(0.0, 1.0)),
            ("roughness", self.roughness, fields::range(0.0, 1.0)),
        ]
    }

    pub fn validate(&self) -> Result<(), InvalidSetting> {
        for (field, value, rule) in self.rules() {
            value.check(field, rule)?;
        }
        Ok(())
    }
}

fn state_noise_rules(schema: &mut schemars::Schema) {
    for (field, _, rule) in NoiseLayer::default().rules() {
        fields::state(schema, field, rule);
    }
}

engine_codec::record!(NoiseLayer {
    seed,
    height,
    offset,
    wavelength,
    ridges,
    warp,
    erosion,
    smoothing,
    valleys,
    roughness,
    blend,
});

#[derive(Clone, Debug, PartialEq)]
pub struct Noise {
    tables: Arc<NoiseTables>,
}

impl Noise {
    pub fn new(params: NoiseLayer) -> Self {
        Self {
            tables: Arc::new(NoiseTables::new(params)),
        }
    }

    pub fn params(&self) -> &NoiseLayer {
        &self.tables.params
    }
}

fn made(params: NoiseLayer) -> Made {
    params
        .validate()
        .map_err(|invalid| Error::from(invalid).coded(ErrorCode::InvalidTerrain))?;
    Ok(Arc::new(Noise::new(params)))
}

pub(crate) fn from_json(settings: &Value, _files: &dyn LayerFiles) -> Made {
    made(parse_value(settings.clone())?)
}

pub(crate) fn read(input: &mut Reader<'_>) -> Made {
    made(input.get()?)
}

impl TerrainLayer for Noise {
    fn kind(&self) -> &'static LayerKind {
        &crate::NOISE_KIND
    }

    fn write(&self, out: &mut Writer) {
        out.put(self.params());
    }

    fn settings(&self) -> Value {
        serde_json::to_value(self.params()).unwrap_or_default()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn same(&self, other: &dyn TerrainLayer) -> bool {
        same_as(self, other)
    }

    fn apply_point(&self, _below: Below<'_>, point: &SamplePoint, height: &mut f32) {
        let tables = &self.tables;
        let (spacing, sample) = (point.spacing, point.sample);
        let (x, z) = (point.x, point.z);
        let [cx, cz] = sample.map(|index| index.div_euclid(GRID_STEP));
        let horizontal = QUARTER_WEIGHTS[sample[0].rem_euclid(GRID_STEP) as usize];
        let vertical = QUARTER_WEIGHTS[sample[1].rem_euclid(GRID_STEP) as usize];
        let (octaves, fine, gullies) = tables.bands(spacing);
        let len = state_len(octaves);
        let mut points = vec![0.0_f32; 16 * len];
        for (index, state) in points.chunks_exact_mut(len).enumerate() {
            let column = (cx - 1 + (index / 4) as i64) * GRID_STEP;
            let row = (cz - 1 + (index % 4) as i64) * GRID_STEP;
            tables.coarse(
                column as f32 * spacing,
                row as f32 * spacing,
                octaves,
                gullies,
                state,
            );
        }
        let mut columns = vec![0.0_f32; 4 * len];
        for (column, value) in columns.chunks_exact_mut(len).enumerate() {
            let taps = [0, 1, 2, 3].map(|row| {
                let at = (column * 4 + row) * len;
                &points[at..at + len]
            });
            interpolate(value, vertical, taps);
        }
        let mut sum = vec![0.0_f32; len];
        let taps = [0, 1, 2, 3].map(|column| &columns[column * len..(column + 1) * len]);
        interpolate(&mut sum, horizontal, taps);
        let value = tables.finish(x, z, &sum, octaves, fine);
        *height = tables.params.blend.apply(*height, value, 1.0);
    }

    fn tile(&self, frame: &TileFrame<'_>) -> Option<Box<dyn LayerTile>> {
        Some(Box::new(NoiseTile::new(Arc::clone(&self.tables), frame)))
    }

    fn row_cost(&self) -> u64 {
        NOISE_SAMPLE_COST
    }

    fn point_cost(&self) -> u64 {
        16 * COARSE_COST + NOISE_SAMPLE_COST
    }
}
