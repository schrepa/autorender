#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

use std::any::Any;
use std::sync::Arc;

use engine_codec::{Reader, Writer};
use engine_core::{
    Brick, Bricks, Error, ErrorCode, InvalidSetting, Job, JobCx, JobKind, JobRef, Rule, Setting,
    Source, parse_value,
};
use engine_heightfield::given::MAX_CREDIT;
use engine_heightfield::layer::{
    Below, FineRegion, LayerFiles, LayerKind, Made, Row, TerrainLayer, no_points, same_as,
};
use engine_heightfield::layers::MAX_HEIGHT;
use engine_heightfield::noise::smooth;
use engine_heightfield::{Blend, HeightImage, Layers};
use engine_math::{max, min};
use engine_places::terrain::{self, TerrainLayers};
use glam::Vec2;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub struct LayerHeightmapBrick;

impl Brick for LayerHeightmapBrick {
    const SOURCE: &'static Source = &SOURCE;
    const JOBS: &'static [&'static JobKind] = &[&HEIGHTMAP_JOB];

    fn register(bricks: &mut Bricks<'_>) {
        bricks.join::<TerrainLayers>(
            terrain::LayerKind {
                layer: &HEIGHTMAP,
                job: JobRef::from(&HEIGHTMAP_JOB),
            },
            None,
        );
    }
}

pub static HEIGHTMAP_JOB: JobKind = JobKind::new("layer_heightmap::heightmap", &SOURCE, refused);

pub static HEIGHTMAP: LayerKind = LayerKind {
    name: "heightmap",
    source: SOURCE.hash(),
    from_json,
    read,
    files,
    schema: engine_heightfield::schema::<HeightmapSettings>,
    points: no_points,
};

fn refused(_input: &[u8], _cx: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    Err(Error::new(ErrorCode::JobUnsupported, "$.jobs"))
}

const MAX_HEIGHTMAP: u32 = 16_384;

const URI_PATH: &str = "$.uri";

const HEIGHTMAP_MIN: Rule = Rule::Range(
    -Layers::MAX_COORDINATE as f64,
    Layers::MAX_COORDINATE as f64,
);
const HEIGHTMAP_SIZE: Rule = Rule::Range(1.0e-3, Layers::MAX_COORDINATE as f64);
const HEIGHTMAP_HEIGHT: Rule = Rule::Range(-MAX_HEIGHT as f64, MAX_HEIGHT as f64);
const HEIGHTMAP_FALLOFF: Rule = Rule::Range(0.0, Layers::MAX_COORDINATE as f64);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[schemars(transform = state_heightmap_rules)]
#[serde(deny_unknown_fields)]
pub struct HeightmapSettings {
    pub uri: String,
    #[schemars(range(min = 2, max = MAX_HEIGHTMAP))]
    pub width: u32,
    #[schemars(range(min = 2, max = MAX_HEIGHTMAP))]
    pub height: u32,
    pub min: [f32; 2],
    pub size: [f32; 2],
    pub low: f32,
    pub high: f32,
    #[serde(default)]
    pub falloff: f32,
    #[serde(default)]
    pub blend: Blend,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(max = MAX_CREDIT))]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(max = MAX_CREDIT))]
    pub attribution: Option<String>,
}

fn state_heightmap_rules(schema: &mut schemars::Schema) {
    for field in ["min", "size", "low", "high", "falloff"] {
        let rule = HeightmapLayer::rule_of(field).expect("heightmap_rule_declared");
        engine_core::state(schema, field, rule);
    }
}

fn from_json(value: &Value, files: &dyn LayerFiles) -> Made {
    let settings: HeightmapSettings = parse_value(value.clone())?;
    Ok(Arc::new(settings.prepare(files)?))
}

fn read(input: &mut Reader<'_>) -> Made {
    let layer: HeightmapLayer = input.get()?;
    layer.validate().map_err(setting_error)?;
    Ok(Arc::new(layer))
}

fn files(settings: &Value) -> Vec<String> {
    settings
        .get("uri")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .into_iter()
        .collect()
}

fn setting_error(invalid: InvalidSetting) -> Error {
    invalid.error(ErrorCode::InvalidValue, "$")
}

impl HeightmapSettings {
    fn prepare(&self, files: &dyn LayerFiles) -> Result<HeightmapLayer, Error> {
        for (field, text) in [("source", &self.source), ("attribution", &self.attribution)] {
            let characters = text.as_ref().map_or(0, |text| text.chars().count());
            if characters > MAX_CREDIT {
                return Err(Error::new(ErrorCode::InvalidValue, format!("$.{field}"))
                    .at_most(MAX_CREDIT as u64)
                    .got(characters));
            }
        }
        for (field, samples) in [("width", self.width), ("height", self.height)] {
            if !(2..=MAX_HEIGHTMAP).contains(&samples) {
                return Err(Error::new(ErrorCode::InvalidValue, format!("$.{field}"))
                    .range(2.0, f64::from(MAX_HEIGHTMAP))
                    .got(samples));
            }
        }
        let image = match files.buffer(&self.uri, URI_PATH)? {
            Some(bytes) => {
                HeightImage::from_le_bytes(self.width, self.height, &bytes).ok_or_else(|| {
                    let wanted = f64::from(self.width) * f64::from(self.height) * 2.0;
                    Error::new(ErrorCode::InvalidValue, URI_PATH)
                        .range(wanted, wanted)
                        .got(bytes.len())
                })?
            }
            None => HeightImage::new(
                self.width,
                self.height,
                vec![0; self.width as usize * self.height as usize],
            )
            .expect("dimensions_checked"),
        };
        let layer = HeightmapLayer {
            image: Arc::new(image),
            min: Vec2::from_array(self.min),
            size: Vec2::from_array(self.size),
            low: self.low,
            high: self.high,
            falloff: self.falloff,
            blend: self.blend,
            uri: Some(self.uri.clone()),
            source: self.source.clone(),
            attribution: self.attribution.clone(),
        };
        layer.validate().map_err(setting_error)?;
        Ok(layer)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct HeightmapLayer {
    pub image: Arc<HeightImage>,
    pub min: Vec2,
    pub size: Vec2,
    pub low: f32,
    pub high: f32,
    pub falloff: f32,
    pub blend: Blend,
    pub uri: Option<String>,
    pub source: Option<String>,
    pub attribution: Option<String>,
}

impl HeightmapLayer {
    pub fn rule_of(field: &str) -> Option<Rule> {
        match field {
            "min" => Some(HEIGHTMAP_MIN),
            "size" => Some(HEIGHTMAP_SIZE),
            "low" | "high" => Some(HEIGHTMAP_HEIGHT),
            "falloff" => Some(HEIGHTMAP_FALLOFF),
            _ => None,
        }
    }

    fn validate(&self) -> Result<(), InvalidSetting> {
        self.min.to_array().check("min", HEIGHTMAP_MIN)?;
        self.size.to_array().check("size", HEIGHTMAP_SIZE)?;
        self.low.check("low", HEIGHTMAP_HEIGHT)?;
        self.high.check("high", HEIGHTMAP_HEIGHT)?;
        self.falloff.check("falloff", HEIGHTMAP_FALLOFF)
    }

    fn texel(&self) -> f32 {
        let image = &self.image;
        max(
            self.size.x / (image.width() - 1) as f32,
            self.size.y / (image.height() - 1) as f32,
        )
    }

    fn level_for(&self, spacing: f32) -> usize {
        let texel = self.texel();
        let mut level = 0;
        while level < self.image.levels() && texel * (2_u32 << level) as f32 <= spacing * 1.0001 {
            level += 1;
        }
        level
    }

    #[inline]
    fn at(&self, x: f32, z: f32, level: usize) -> (f32, f32) {
        let local = Vec2::new(x, z) - self.min;
        let inside = min(
            min(local.x, self.size.x - local.x),
            min(local.y, self.size.y - local.y),
        );
        if inside < 0.0 {
            return (0.0, 0.0);
        }
        let weight = if self.falloff > 0.0 {
            smooth(inside / self.falloff)
        } else {
            1.0
        };
        let u = local.x / self.size.x * (self.image.width() - 1) as f32;
        let v = local.y / self.size.y * (self.image.height() - 1) as f32;
        let value = self.image.sample(u, v, level) * (1.0 / 65_535.0);
        (self.low + (self.high - self.low) * value, weight)
    }
}

impl TerrainLayer for HeightmapLayer {
    fn kind(&self) -> &'static LayerKind {
        &HEIGHTMAP
    }

    fn write(&self, out: &mut Writer) {
        out.put(self);
    }

    fn settings(&self) -> Value {
        serde_json::to_value(HeightmapSettings {
            uri: self.uri.clone().unwrap_or_default(),
            width: self.image.width(),
            height: self.image.height(),
            min: self.min.to_array(),
            size: self.size.to_array(),
            low: self.low,
            high: self.high,
            falloff: self.falloff,
            blend: self.blend,
            source: self.source.clone(),
            attribution: self.attribution.clone(),
        })
        .expect("heightmap_settings_serialize")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn same(&self, other: &dyn TerrainLayer) -> bool {
        same_as(self, other)
    }

    fn apply_row(&self, below: Below<'_>, row: &Row<'_>, heights: &mut [f32]) {
        let (low, high) = (row.low, row.high);
        let map_max = self.min + self.size;
        if self.min.x > high.x || self.min.y > high.y || map_max.x < low.x || map_max.y < low.y {
            return;
        }
        let level = self.level_for(row.spacing);
        for (height, &x) in heights.iter_mut().zip(row.xs) {
            let (value, weight) = self.at(x, row.z, level);
            *height = self
                .blend
                .apply(*height, value, weight * below.shown(x, row.z));
        }
    }

    fn fine_region(&self) -> Option<FineRegion> {
        let inside = Vec2::splat(self.falloff);
        Some(FineRegion {
            low: self.min + inside,
            high: self.min + self.size - inside,
            texel: self.texel(),
        })
    }
}

engine_codec::record!(HeightmapLayer {
    image,
    min,
    size,
    low,
    high,
    falloff,
    blend,
    uri,
    source,
    attribution,
});
