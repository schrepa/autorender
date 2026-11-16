use std::any::Any;
use std::f32::consts::TAU;
use std::sync::Arc;

use engine_codec::{Expected, Reader, Writer};
use engine_core::{Error, fields, parse_value};
use engine_heightfield::layer::{
    Below, ChangeMap, LayerFiles, LayerKind, Made, Published, Rests, Row, SiteBytes, TerrainLayer,
    image_row, no_files, no_points, same_as,
};
use engine_heightfield::{Gather, HeightImage, INVALID_TERRAIN, Layers, Pyramid, TileHeights};
use engine_math::{clamp, max, min, sin_cos};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const SUN: &str = "sun";
pub const EXPOSURE: &str = "exposure";
pub const SHELTER: &str = "shelter";
pub const SOIL: &str = "soil";
pub const SNOW: &str = "snow";
const MAPS: [&str; 5] = [SUN, EXPOSURE, SHELTER, SOIL, SNOW];

const MAX_LEVEL: u32 = 12;
const AXIAL_TILT: f32 = 23.44;
const YEAR: f32 = 365.0;
const SOLSTICE: f32 = 10.0;
const DAYS: u32 = 5;
const HOURS: u32 = 48;
const AZIMUTHS: usize = 32;
const STRIDE: f32 = 1.3;
const PENUMBRA: f32 = 0.035;

fields! {
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
    #[serde(default, deny_unknown_fields)]
    pub struct SiteLayer {
        pub level: u32 = 4 => range(0, MAX_LEVEL),
        pub north: f32 = 0.0 => range(-360.0, 360.0),
        pub latitude: f32 = 45.0 => range(-90.0, 90.0),
        pub season: [f32; 2] = [1.0, 365.0] => range(1.0, 365.0),
        pub horizon: f32 = 2000.0 => range(1.0, 1.0e6),
        pub radius: f32 = 100.0 => range(1.0, 1.0e5),
        pub wind: f32 = 270.0 => range(-360.0, 360.0),
        pub fetch: f32 = 300.0 => range(1.0, 1.0e6),
        pub soil: f32 = 1.0 => range(0.0, 1000.0),
        pub soil_bare: f32 = 90.0 => range(1.0, 90.0),
        pub soil_gather: f32 = 0.0 => range(0.0, 1.0e4),
        pub soil_rest: [f32; 2] = [90.0, 90.0] => range(0.0, 90.0),
        pub snow: f32 = 0.0 => range(0.0, 1000.0),
        pub snow_line: [f32; 2] = [0.0, 1000.0] => finite,
        pub snow_sun: f32 = 0.0 => range(0.0, 1.0e5),
        pub snow_hollow: f32 = 0.0 => range(0.0, 1.0e4),
        pub snow_lee: f32 = 0.0 => range(0.0, 1.0e4),
        pub snow_gather: f32 = 0.0 => range(0.0, 1.0e4),
        pub snow_rest: [f32; 2] = [90.0, 90.0] => range(0.0, 90.0),
        pub plane_wetness: [f32; 2] = [-5.0, 25.0] => finite,
        pub plane_sun: [f32; 2] = [0.0, 2.0] => finite,
        pub plane_depth: [f32; 2] = [0.0, 2.0] => finite,
        pub plane_exposure: [f32; 2] = [-50.0, 50.0] => finite,
    }
}

engine_codec::record!(SiteLayer {
    level,
    north,
    latitude,
    season,
    horizon,
    radius,
    wind,
    fetch,
    soil,
    soil_bare,
    soil_gather,
    soil_rest,
    snow,
    snow_line,
    snow_sun,
    snow_hollow,
    snow_lee,
    snow_gather,
    snow_rest,
    plane_wetness,
    plane_sun,
    plane_depth,
    plane_exposure,
});

#[derive(Clone, Debug, PartialEq)]
pub struct Sheet {
    pub image: HeightImage,
    pub range: [f32; 2],
}

engine_codec::record!(Sheet { image, range });

#[derive(Clone, Debug, PartialEq)]
pub struct Site {
    pub spacing: f32,
    pub sun: Sheet,
    pub exposure: Sheet,
    pub shelter: Sheet,
    pub soil: Sheet,
    pub snow: Sheet,
}

engine_codec::record!(Site {
    spacing,
    sun,
    exposure,
    shelter,
    soil,
    snow,
});

#[derive(Clone, Debug, PartialEq)]
pub struct SiteStage {
    pub layer: SiteLayer,
    pub baked: Option<Arc<Site>>,
}

engine_codec::record!(SiteStage { layer, baked });

pub static SITE: LayerKind = LayerKind {
    name: "site",
    source: crate::SOURCE.hash(),
    from_json,
    read,
    files: no_files,
    schema: engine_heightfield::schema::<SiteLayer>,
    points: no_points,
};

fn made(stage: SiteStage) -> Made {
    let layer = &stage.layer;
    layer
        .validate()
        .map_err(|invalid| invalid.error(INVALID_TERRAIN, "$"))?;
    let sorted = [
        ("$.soil_rest", layer.soil_rest, false),
        ("$.snow_rest", layer.snow_rest, false),
        ("$.snow_line", layer.snow_line, false),
        ("$.plane_wetness", layer.plane_wetness, true),
        ("$.plane_sun", layer.plane_sun, true),
        ("$.plane_depth", layer.plane_depth, true),
        ("$.plane_exposure", layer.plane_exposure, true),
    ];
    for (path, [low, high], apart) in sorted {
        if low > high || (apart && low == high) {
            return Err(Error::new(INVALID_TERRAIN, path)
                .expected(Expected::Sorted)
                .got([low, high]));
        }
    }
    Ok(Arc::new(stage))
}

fn from_json(value: &Value, _files: &dyn LayerFiles) -> Made {
    made(SiteStage {
        layer: parse_value(value.clone())?,
        baked: None,
    })
}

fn read(input: &mut Reader<'_>) -> Made {
    made(input.get()?)
}

pub(crate) fn tangent(degrees: f32) -> f32 {
    libm::tanf(min(degrees, 89.9).to_radians())
}

impl SiteStage {
    fn sheet(&self, name: &str) -> Option<&Sheet> {
        let baked = self.baked.as_deref()?;
        match name {
            SUN => Some(&baked.sun),
            EXPOSURE => Some(&baked.exposure),
            SHELTER => Some(&baked.shelter),
            SOIL => Some(&baked.soil),
            SNOW => Some(&baked.snow),
            _ => None,
        }
    }

    fn resting(&self, name: &str) -> Option<Rests> {
        let layer = &self.layer;
        match name {
            SOIL => Some(Rests {
                slope: layer.soil_rest.map(tangent),
                gather: layer.soil_gather,
            }),
            SNOW => Some(Rests {
                slope: layer.snow_rest.map(tangent),
                gather: layer.snow_gather,
            }),
            _ => None,
        }
    }
}

impl TerrainLayer for SiteStage {
    fn kind(&self) -> &'static LayerKind {
        &SITE
    }

    fn write(&self, out: &mut Writer) {
        out.put(self);
    }

    fn settings(&self) -> Value {
        serde_json::to_value(self.layer).expect("site_settings_serialize")
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
        MAPS.map(str::to_owned).to_vec()
    }

    fn images(&self) -> Vec<String> {
        self.outputs()
    }

    fn rests(&self, name: &str) -> Option<Rests> {
        self.resting(name)
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
        self.needs_bake().then(|| {
            let gather = Gather::new(self.layer.level, below, pyramid, tiles);
            let grid = Grid {
                heights: &gather.heights,
                width: gather.width,
                rows: gather.heights.len() / gather.width,
                spacing: pyramid.spacing(gather.level),
            };
            Arc::new(Self {
                layer: self.layer,
                baked: Some(Arc::new(grid.site(&self.layer))),
            }) as Arc<dyn TerrainLayer>
        })
    }

    fn unbaked(&self) -> Option<Arc<dyn TerrainLayer>> {
        Some(Arc::new(Self {
            layer: self.layer,
            baked: None,
        }))
    }

    fn published(&self) -> Vec<Published<'_>> {
        let Some(baked) = self.baked.as_deref() else {
            return Vec::new();
        };
        MAPS.into_iter()
            .filter_map(|name| {
                let sheet = self.sheet(name)?;
                Some(Published {
                    name,
                    map: ChangeMap {
                        spacing: baked.spacing,
                        image: &sheet.image,
                        range: sheet.range,
                    },
                    rests: self.resting(name),
                })
            })
            .collect()
    }

    fn site(&self) -> Option<SiteBytes> {
        let layer = &self.layer;
        Some(SiteBytes {
            sun: SUN,
            soil: SOIL,
            exposure: EXPOSURE,
            ranges: [
                layer.plane_wetness,
                layer.plane_sun,
                layer.plane_depth,
                layer.plane_exposure,
            ],
        })
    }

    fn map_row(
        &self,
        name: &str,
        below: Below<'_>,
        row: &Row<'_>,
        _heights: &[f32],
        out: &mut [f32],
    ) -> bool {
        let (Some(baked), Some(sheet)) = (self.baked.as_deref(), self.sheet(name)) else {
            return false;
        };
        let count = out.len().min(row.xs.len());
        image_row(
            &sheet.image,
            sheet.range,
            (baked.spacing, row.spacing),
            row.xs[..count].iter().copied(),
            row.z,
            |index, value| out[index] = value * below.shown(row.xs[index], row.z),
        );
        true
    }
}

pub(crate) struct Grid<'a> {
    pub(crate) heights: &'a [f32],
    pub(crate) width: usize,
    pub(crate) rows: usize,
    pub(crate) spacing: f32,
}

impl Grid<'_> {
    fn at(&self, x: isize, z: isize) -> f32 {
        let x = x.clamp(0, self.width as isize - 1) as usize;
        let z = z.clamp(0, self.rows as isize - 1) as usize;
        self.heights[z * self.width + x]
    }

    pub(crate) fn sample(&self, values: &[f32], x: f32, z: f32) -> f32 {
        let x = clamp(x, 0.0, (self.width - 1) as f32);
        let z = clamp(z, 0.0, (self.rows - 1) as f32);
        let (left, up) = (x.floor(), z.floor());
        let (fx, fz) = (x - left, z - up);
        let (left, up) = (left as usize, up as usize);
        let (right, down) = ((left + 1).min(self.width - 1), (up + 1).min(self.rows - 1));
        let held = |x: usize, z: usize| values[z * self.width + x];
        let near = held(left, up) + (held(right, up) - held(left, up)) * fx;
        let far = held(left, down) + (held(right, down) - held(left, down)) * fx;
        near + (far - near) * fz
    }

    pub(crate) fn smooth(&self, x: f32, z: f32) -> f32 {
        self.sample(self.heights, x, z)
    }

    fn cells(&self) -> impl Iterator<Item = (isize, isize)> + '_ {
        (0..self.rows as isize).flat_map(|z| (0..self.width as isize).map(move |x| (x, z)))
    }

    pub(crate) fn gradient(&self, x: isize, z: isize) -> [f32; 2] {
        let across = 2.0 * self.spacing;
        [
            (self.at(x + 1, z) - self.at(x - 1, z)) / across,
            (self.at(x, z + 1) - self.at(x, z - 1)) / across,
        ]
    }

    fn curvature(&self, x: isize, z: isize) -> f32 {
        let mut around = -8.0 * self.at(x, z);
        for (dx, dz) in [
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        ] {
            around += self.at(x + dx, z + dz);
        }
        around / (3.0 * self.spacing * self.spacing)
    }

    fn horizon(&self, toward: [f32; 2], reach: f32) -> Vec<f32> {
        self.cells()
            .map(|(x, z)| {
                let own = self.at(x, z);
                let (mut steps, mut highest) = (1.0_f32, f32::NEG_INFINITY);
                loop {
                    let seen = self.smooth(x as f32 + toward[0] * steps, z as f32 + toward[1] * steps);
                    highest = max(highest, (seen - own) / (steps * self.spacing));
                    steps *= STRIDE;
                    if steps * self.spacing > reach {
                        break highest;
                    }
                }
            })
            .collect()
    }

    fn rays(layer: &SiteLayer) -> (Vec<Vec<[f32; 3]>>, f32) {
        let (north_sin, north_cos) = sin_cos(layer.north.to_radians());
        let (north, east) = ([-north_sin, -north_cos], [north_cos, -north_sin]);
        let (lat_sin, lat_cos) = sin_cos(layer.latitude.to_radians());
        let [first, last] = layer.season;
        let span = if last >= first {
            last - first
        } else {
            last + YEAR - first
        };
        let mut bins = vec![Vec::new(); AZIMUTHS];
        let mut possible = 0.0;
        for day in 0..DAYS {
            let date = first + span * (day as f32 + 0.5) / DAYS as f32;
            let (_, swing) = sin_cos(TAU * (date + SOLSTICE) / YEAR);
            let (tilt_sin, tilt_cos) = sin_cos(-AXIAL_TILT.to_radians() * swing);
            for step in 0..HOURS {
                let hour = TAU * ((step as f32 + 0.5) / HOURS as f32 - 0.5);
                let (hour_sin, hour_cos) = sin_cos(hour);
                let up = lat_sin * tilt_sin + lat_cos * tilt_cos * hour_cos;
                if up <= 0.0 {
                    continue;
                }
                let to_east = -tilt_cos * hour_sin;
                let to_north = lat_cos * tilt_sin - lat_sin * tilt_cos * hour_cos;
                let ray = [
                    to_north * north[0] + to_east * east[0],
                    up,
                    to_north * north[1] + to_east * east[1],
                ];
                let turn = (libm::atan2f(ray[2], ray[0]) + TAU * 0.5) / TAU;
                let bin = ((turn * AZIMUTHS as f32) as usize).min(AZIMUTHS - 1);
                bins[bin].push(ray);
                possible += up;
            }
        }
        (bins, possible)
    }

    fn sun(&self, layer: &SiteLayer) -> Vec<f32> {
        let normals: Vec<[f32; 3]> = self
            .cells()
            .map(|(x, z)| {
                let [gx, gz] = self.gradient(x, z);
                let length = (gx * gx + gz * gz + 1.0).sqrt();
                [-gx / length, 1.0 / length, -gz / length]
            })
            .collect();
        let (bins, possible) = Self::rays(layer);
        let mut received = vec![0.0_f32; self.heights.len()];
        for rays in bins.iter().filter(|rays| !rays.is_empty()) {
            let mut toward = [0.0_f32; 2];
            for ray in rays {
                toward = [toward[0] + ray[0], toward[1] + ray[2]];
            }
            let length = max((toward[0] * toward[0] + toward[1] * toward[1]).sqrt(), 1.0e-6);
            let horizon = self.horizon(toward.map(|part| part / length), layer.horizon);
            for ray in rays {
                let flat = max((ray[0] * ray[0] + ray[2] * ray[2]).sqrt(), 1.0e-6);
                let rise = ray[1] / flat;
                for ((sum, normal), &skyline) in received.iter_mut().zip(&normals).zip(&horizon) {
                    let lit = clamp((rise - skyline) / PENUMBRA + 0.5, 0.0, 1.0);
                    let facing = normal[0] * ray[0] + normal[1] * ray[1] + normal[2] * ray[2];
                    *sum += lit * max(facing, 0.0);
                }
            }
        }
        if possible > 0.0 {
            for sum in &mut received {
                *sum /= possible;
            }
        }
        received
    }

    fn exposure(&self, radius: f32) -> Vec<f32> {
        let (width, rows) = (self.width, self.rows);
        let across = width + 1;
        let mut sums = vec![0.0_f64; across * (rows + 1)];
        for z in 0..rows {
            let mut along = 0.0_f64;
            for x in 0..width {
                along += f64::from(self.heights[z * width + x]);
                sums[(z + 1) * across + x + 1] = sums[z * across + x + 1] + along;
            }
        }
        let reach = max(radius / self.spacing, 1.0) as usize;
        self.cells()
            .map(|(x, z)| {
                let (x, z) = (x as usize, z as usize);
                let (left, right) = (x.saturating_sub(reach), (x + reach + 1).min(width));
                let (up, down) = (z.saturating_sub(reach), (z + reach + 1).min(rows));
                let total = sums[down * across + right] - sums[up * across + right]
                    - sums[down * across + left]
                    + sums[up * across + left];
                let mean = total / ((right - left) * (down - up)) as f64;
                self.heights[z * width + x] - mean as f32
            })
            .collect()
    }

    fn site(&self, layer: &SiteLayer) -> Site {
        let sun = self.sun(layer);
        let exposure = self.exposure(layer.radius);
        let (wind_sin, wind_cos) = sin_cos(layer.wind.to_radians());
        let shelter: Vec<f32> = self
            .horizon([wind_sin, -wind_cos], layer.fetch)
            .into_iter()
            .map(|rise| libm::atanf(rise).to_degrees())
            .collect();
        let bare = tangent(layer.soil_bare);
        let soil: Vec<f32> = self
            .cells()
            .map(|(x, z)| {
                let [gx, gz] = self.gradient(x, z);
                let steep = (gx * gx + gz * gz).sqrt();
                layer.soil
                    * max(1.0 - steep / bare, 0.0)
                    * Rests::gathered(layer.soil_gather * self.curvature(x, z))
            })
            .collect();
        let [low, high] = layer.snow_line;
        let snow: Vec<f32> = (0..self.heights.len())
            .map(|cell| {
                let seen = self.heights[cell] + layer.snow_sun * (1.0 - sun[cell])
                    - layer.snow_hollow * exposure[cell]
                    + layer.snow_lee * shelter[cell];
                let share = if high > low {
                    clamp((seen - low) / (high - low), 0.0, 1.0)
                } else if seen >= low {
                    1.0
                } else {
                    0.0
                };
                layer.snow * share
            })
            .collect();
        let sheet = |values: &[f32]| sheet(values, self.width);
        Site {
            spacing: self.spacing,
            sun: sheet(&sun),
            exposure: sheet(&exposure),
            shelter: sheet(&shelter),
            soil: sheet(&soil),
            snow: sheet(&snow),
        }
    }
}

pub(crate) fn sheet(values: &[f32], width: usize) -> Sheet {
    let mut range = [f32::INFINITY, f32::NEG_INFINITY];
    for &value in values {
        range = [min(range[0], value), max(range[1], value)];
    }
    let span = max(range[1] - range[0], 1.0e-3);
    let samples = values
        .iter()
        .map(|&value| (clamp((value - range[0]) / span, 0.0, 1.0) * 65_535.0 + 0.5) as u16)
        .collect();
    let image = HeightImage::new(width as u32, (values.len() / width) as u32, samples)
        .expect("grid_two_samples_each_way");
    Sheet {
        image,
        range: [range[0], range[0] + span],
    }
}
