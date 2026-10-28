use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use engine_core::{GivenRiver, RiversLayer};
use serde_json::{Value, json};
use tiff::decoder::{Decoder, DecodingResult, Limits};
use tiff::tags::Tag;

fn main() {
    if let Err(error) = run() {
        eprintln!("terrain_import: {error}");
        std::process::exit(1);
    }
}

const USAGE: &str = "usage: terrain_import --out DIR --id NAME [--center A,B] [--size METERS] \
     [--spacing METERS] [--extent W,H] [--range LOW,HIGH] [--detail METERS] [--materials DIR] \
     [--attribution TEXT] [--sea LEVEL] [--rivers GEOJSON] [--river-precision METERS] \
     [--river-attribution TEXT] INPUT...";

struct Options {
    inputs: Vec<PathBuf>,
    out: PathBuf,
    id: String,
    center: Option<[f64; 2]>,
    size: f64,
    spacing: Option<f64>,
    extent: Option<[f64; 2]>,
    range: Option<[f64; 2]>,
    detail: f64,
    materials: Option<PathBuf>,
    attribution: Option<String>,
    sea: Option<f64>,
    rivers: Option<PathBuf>,
    river_precision: Option<f64>,
    river_attribution: Option<String>,
}

fn pair(text: &str, name: &str) -> Result<[f64; 2], String> {
    let values: Vec<f64> = text
        .split(',')
        .map(|part| part.trim().parse::<f64>())
        .collect::<Result<_, _>>()
        .map_err(|error| format!("{name} {text}: {error}"))?;
    match values.as_slice() {
        [a, b] if a.is_finite() && b.is_finite() => Ok([*a, *b]),
        _ => Err(format!("{name} takes two numbers, A,B")),
    }
}

fn options() -> Result<Options, String> {
    let mut arguments = std::env::args().skip(1);
    let mut options = Options {
        inputs: Vec::new(),
        out: PathBuf::new(),
        id: String::new(),
        center: None,
        size: 16_384.0,
        spacing: None,
        extent: None,
        range: None,
        detail: 8.0,
        materials: None,
        attribution: None,
        sea: None,
        rivers: None,
        river_precision: None,
        river_attribution: None,
    };
    while let Some(argument) = arguments.next() {
        let mut value = |name: &str| {
            arguments
                .next()
                .ok_or_else(|| format!("{name} needs a value"))
        };
        match argument.as_str() {
            "--out" => options.out = PathBuf::from(value("--out")?),
            "--id" => options.id = value("--id")?,
            "--center" => options.center = Some(pair(&value("--center")?, "--center")?),
            "--size" => {
                let text = value("--size")?;
                options.size = text
                    .parse()
                    .map_err(|error| format!("--size {text}: {error}"))?;
            }
            "--spacing" => {
                let text = value("--spacing")?;
                options.spacing = Some(
                    text.parse()
                        .map_err(|error| format!("--spacing {text}: {error}"))?,
                );
            }
            "--extent" => options.extent = Some(pair(&value("--extent")?, "--extent")?),
            "--range" => options.range = Some(pair(&value("--range")?, "--range")?),
            "--detail" => {
                let text = value("--detail")?;
                options.detail = text
                    .parse()
                    .map_err(|error| format!("--detail {text}: {error}"))?;
            }
            "--materials" => options.materials = Some(PathBuf::from(value("--materials")?)),
            "--attribution" => options.attribution = Some(value("--attribution")?),
            "--sea" => {
                let text = value("--sea")?;
                let level: f64 = text
                    .parse()
                    .map_err(|error| format!("--sea {text}: {error}"))?;
                if !(level.is_finite() && level.abs() <= 10_000.0) {
                    return Err("--sea must be within 10000 meters of zero".into());
                }
                options.sea = Some(level);
            }
            "--rivers" => options.rivers = Some(PathBuf::from(value("--rivers")?)),
            "--river-precision" => {
                let text = value("--river-precision")?;
                let precision: f64 = text
                    .parse()
                    .map_err(|error| format!("--river-precision {text}: {error}"))?;
                let most = f64::from(engine_core::terrain_data::MAX_PRECISION);
                if !(precision.is_finite() && (0.0..=most).contains(&precision)) {
                    return Err(format!("--river-precision must be 0 to {most} meters"));
                }
                options.river_precision = Some(precision);
            }
            "--river-attribution" => {
                options.river_attribution = Some(value("--river-attribution")?);
            }
            flag if flag.starts_with("--") => {
                return Err(format!("unknown option {flag}\n{USAGE}"));
            }
            input => options.inputs.push(PathBuf::from(input)),
        }
    }
    if options.inputs.is_empty() || options.id.is_empty() || options.out.as_os_str().is_empty() {
        return Err(USAGE.into());
    }
    let id_ok = options.id.len() <= 64
        && options
            .id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"_-".contains(&byte));
    if !id_ok {
        return Err("--id must be up to 64 letters, digits, '_' and '-'".into());
    }
    let size = options.size;
    if !((256.0..=262_144.0).contains(&size) && size % 256.0 == 0.0) {
        return Err("--size must be a multiple of 256 meters, 256 to 262144".into());
    }
    if !(options.detail >= 0.0 && options.detail <= 100.0) {
        return Err("--detail must be 0 to 100 meters".into());
    }
    if let Some(spacing) = options.spacing
        && !(spacing.is_finite() && spacing >= 0.25 && spacing <= size / 2.0)
    {
        return Err("--spacing must be at least 0.25 meters and at most half the size".into());
    }
    Ok(options)
}

struct Raster {
    width: usize,
    height: usize,
    samples: Vec<f32>,
    origin: [f64; 2],
    scale: [f64; 2],
    geographic: bool,
    description: String,
}

impl Raster {
    fn sample(&self, x: f64, y: f64) -> Option<f64> {
        let u = (x - self.origin[0]) / self.scale[0];
        let v = (self.origin[1] - y) / self.scale[1];
        let (last_u, last_v) = ((self.width - 1) as f64, (self.height - 1) as f64);
        if !(u >= 0.0 && v >= 0.0 && u <= last_u && v <= last_v) {
            return None;
        }
        let i = (u.floor() as usize).min(self.width - 2);
        let j = (v.floor() as usize).min(self.height - 2);
        let (fu, fv) = (u - i as f64, v - j as f64);
        let at = |i: usize, j: usize| f64::from(self.samples[j * self.width + i]);
        let corners = [
            (at(i, j), (1.0 - fu) * (1.0 - fv)),
            (at(i + 1, j), fu * (1.0 - fv)),
            (at(i, j + 1), (1.0 - fu) * fv),
            (at(i + 1, j + 1), fu * fv),
        ];
        let (mut sum, mut weight) = (0.0, 0.0);
        for (height, share) in corners {
            if height.is_finite() {
                sum += height * share;
                weight += share;
            }
        }
        (weight > 1.0e-9).then(|| sum / weight)
    }

    fn resolution(&self, latitude: f64) -> [f64; 2] {
        if self.geographic {
            [
                self.scale[0] * METERS_PER_DEGREE * latitude.to_radians().cos(),
                self.scale[1] * METERS_PER_DEGREE,
            ]
        } else {
            self.scale
        }
    }

    fn middle(&self) -> [f64; 2] {
        [
            self.origin[0] + 0.5 * (self.width - 1) as f64 * self.scale[0],
            self.origin[1] - 0.5 * (self.height - 1) as f64 * self.scale[1],
        ]
    }
}

const METERS_PER_DEGREE: f64 = 111_320.0;

fn read_geotiff(path: &Path) -> Result<Raster, String> {
    let name = path.display().to_string();
    let fail = |error: tiff::TiffError| format!("{name}: {error}");
    let file = File::open(path).map_err(|error| format!("{name}: {error}"))?;
    let mut decoder = Decoder::new(BufReader::new(file))
        .map_err(fail)?
        .with_limits(Limits::unlimited());
    let (width, height) = decoder.dimensions().map_err(fail)?;
    let color = decoder.colortype().map_err(fail)?;
    if !matches!(color, tiff::ColorType::Gray(_)) {
        return Err(format!(
            "{name}: elevation models hold one channel, not {color:?}"
        ));
    }
    let scale = decoder
        .get_tag_f64_vec(Tag::ModelPixelScaleTag)
        .map_err(|_| format!("{name}: no ModelPixelScale tag: not a georeferenced GeoTIFF"))?;
    let tie = decoder
        .get_tag_f64_vec(Tag::ModelTiepointTag)
        .map_err(|_| format!("{name}: no ModelTiepoint tag: not a georeferenced GeoTIFF"))?;
    if scale.len() < 2 || tie.len() < 6 || !(scale[0] > 0.0 && scale[1] > 0.0) {
        return Err(format!("{name}: unusable georeference"));
    }
    let keys = match decoder.find_tag(Tag::GeoKeyDirectoryTag).map_err(fail)? {
        Some(value) => value.into_u16_vec().map_err(fail)?,
        None => Vec::new(),
    };
    let key = |id: u16| {
        keys.get(4..)
            .unwrap_or_default()
            .as_chunks::<4>()
            .0
            .iter()
            .find(|entry| entry[0] == id && entry[1] == 0)
            .map(|entry| entry[3])
    };
    let geographic = match key(1024) {
        Some(1) => false,
        Some(2) => true,
        Some(other) => return Err(format!("{name}: unsupported model type {other}")),
        None => return Err(format!("{name}: no GeoKey model type")),
    };
    let point = key(1025) == Some(2);
    let crs = if geographic {
        key(2048).map_or("geographic".to_owned(), |code| format!("EPSG:{code}"))
    } else {
        key(3072).map_or("projected".to_owned(), |code| format!("EPSG:{code}"))
    };
    let nodata = match decoder.find_tag(Tag::GdalNodata).map_err(fail)? {
        Some(value) => value
            .into_string()
            .ok()
            .and_then(|text| text.trim_matches(char::from(0)).trim().parse::<f64>().ok()),
        None => None,
    };
    let samples: Vec<f32> = match decoder.read_image().map_err(fail)? {
        DecodingResult::F32(values) => values,
        DecodingResult::F64(values) => values.into_iter().map(|value| value as f32).collect(),
        DecodingResult::I16(values) => values.into_iter().map(f32::from).collect(),
        DecodingResult::U16(values) => values.into_iter().map(f32::from).collect(),
        DecodingResult::I32(values) => values.into_iter().map(|value| value as f32).collect(),
        DecodingResult::U8(values) => values.into_iter().map(f32::from).collect(),
        _ => return Err(format!("{name}: unsupported sample format")),
    };
    let samples = samples
        .into_iter()
        .map(|value| {
            let missing = !value.is_finite()
                || nodata.is_some_and(|nodata| f64::from(value) == nodata)
                || value < -12_000.0
                || value > 12_000.0;
            if missing { f32::NAN } else { value }
        })
        .collect();
    let half = if point { 0.0 } else { 0.5 };
    let origin = [
        tie[3] + (half - tie[0]) * scale[0],
        tie[4] - (half - tie[1]) * scale[1],
    ];
    Ok(Raster {
        width: width as usize,
        height: height as usize,
        samples,
        origin,
        scale: [scale[0], scale[1]],
        geographic,
        description: format!("{name} ({width}x{height}, {crs})"),
    })
}

fn read_heightmap(path: &Path, extent: [f64; 2], range: [f64; 2]) -> Result<Raster, String> {
    let name = path.display().to_string();
    let png = path
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("png"));
    let (width, height, values) = if png {
        let file = File::open(path).map_err(|error| format!("{name}: {error}"))?;
        let mut decoder = png::Decoder::new(BufReader::new(file));
        decoder.set_transformations(png::Transformations::IDENTITY);
        let mut reader = decoder
            .read_info()
            .map_err(|error| format!("{name}: {error}"))?;
        let mut buffer = vec![0; reader.output_buffer_size()];
        let info = reader
            .next_frame(&mut buffer)
            .map_err(|error| format!("{name}: {error}"))?;
        if info.color_type != png::ColorType::Grayscale || info.bit_depth != png::BitDepth::Sixteen
        {
            return Err(format!("{name}: heightmap PNGs are 16-bit grayscale"));
        }
        let values: Vec<u16> = buffer[..info.buffer_size()]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u16::from_be_bytes(*pair))
            .collect();
        (info.width as usize, info.height as usize, values)
    } else {
        let bytes = std::fs::read(path).map_err(|error| format!("{name}: {error}"))?;
        let count = bytes.len() / 2;
        let side = (count as f64).sqrt().round() as usize;
        if side * side != count || bytes.len() % 2 != 0 {
            return Err(format!("{name}: RAW heightmaps are square, 16-bit samples"));
        }
        let values = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u16::from_le_bytes(*pair))
            .collect();
        (side, side, values)
    };
    if width < 2 || height < 2 {
        return Err(format!("{name}: at least two samples each way"));
    }
    let samples = values
        .into_iter()
        .map(|value| (range[0] + (range[1] - range[0]) * f64::from(value) / 65_535.0) as f32)
        .collect();
    Ok(Raster {
        width,
        height,
        samples,
        origin: [0.0, extent[1]],
        scale: [
            extent[0] / (width - 1) as f64,
            extent[1] / (height - 1) as f64,
        ],
        geographic: false,
        description: format!(
            "{name} ({width}x{height}, {} by {} m)",
            extent[0], extent[1]
        ),
    })
}

const WGS84_A: f64 = 6_378_137.0;
const WGS84_F: f64 = 1.0 / 298.257_223_563;

struct TransverseMercator {
    latitude: f64,
    longitude: f64,
    e2: f64,
    e_prime2: f64,
    meridian_origin: f64,
}

impl TransverseMercator {
    fn new(latitude: f64, longitude: f64) -> Self {
        let e2 = WGS84_F * (2.0 - WGS84_F);
        let mut projection = Self {
            latitude: latitude.to_radians(),
            longitude: longitude.to_radians(),
            e2,
            e_prime2: e2 / (1.0 - e2),
            meridian_origin: 0.0,
        };
        projection.meridian_origin = projection.meridian(projection.latitude);
        projection
    }

    fn meridian(&self, phi: f64) -> f64 {
        let e2 = self.e2;
        let (e4, e6) = (e2 * e2, e2 * e2 * e2);
        WGS84_A
            * ((1.0 - e2 / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0) * phi
                - (3.0 * e2 / 8.0 + 3.0 * e4 / 32.0 + 45.0 * e6 / 1024.0) * (2.0 * phi).sin()
                + (15.0 * e4 / 256.0 + 45.0 * e6 / 1024.0) * (4.0 * phi).sin()
                - (35.0 * e6 / 3072.0) * (6.0 * phi).sin())
    }

    fn inverse(&self, east: f64, north: f64) -> [f64; 2] {
        let (e2, e_prime2) = (self.e2, self.e_prime2);
        let (e4, e6) = (e2 * e2, e2 * e2 * e2);
        let m = self.meridian_origin + north;
        let mu = m / (WGS84_A * (1.0 - e2 / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0));
        let root = (1.0 - e2).sqrt();
        let e1 = (1.0 - root) / (1.0 + root);
        let phi1 = mu
            + (3.0 * e1 / 2.0 - 27.0 * e1.powi(3) / 32.0) * (2.0 * mu).sin()
            + (21.0 * e1 * e1 / 16.0 - 55.0 * e1.powi(4) / 32.0) * (4.0 * mu).sin()
            + (151.0 * e1.powi(3) / 96.0) * (6.0 * mu).sin()
            + (1097.0 * e1.powi(4) / 512.0) * (8.0 * mu).sin();
        let (sin1, cos1) = phi1.sin_cos();
        let tan1 = sin1 / cos1;
        let c1 = e_prime2 * cos1 * cos1;
        let t1 = tan1 * tan1;
        let along = 1.0 - e2 * sin1 * sin1;
        let n1 = WGS84_A / along.sqrt();
        let r1 = WGS84_A * (1.0 - e2) / along.powf(1.5);
        let d = east / n1;
        let latitude = phi1
            - (n1 * tan1 / r1)
                * (d * d / 2.0
                    - (5.0 + 3.0 * t1 + 10.0 * c1 - 4.0 * c1 * c1 - 9.0 * e_prime2) * d.powi(4)
                        / 24.0
                    + (61.0 + 90.0 * t1 + 298.0 * c1 + 45.0 * t1 * t1
                        - 252.0 * e_prime2
                        - 3.0 * c1 * c1)
                        * d.powi(6)
                        / 720.0);
        let longitude = self.longitude
            + (d - (1.0 + 2.0 * t1 + c1) * d.powi(3) / 6.0
                + (5.0 - 2.0 * c1 + 28.0 * t1 - 3.0 * c1 * c1 + 8.0 * e_prime2 + 24.0 * t1 * t1)
                    * d.powi(5)
                    / 120.0)
                / cos1;
        [latitude.to_degrees(), longitude.to_degrees()]
    }

    fn forward(&self, latitude: f64, longitude: f64) -> [f64; 2] {
        let (e2, e_prime2) = (self.e2, self.e_prime2);
        let phi = latitude.to_radians();
        let (sin, cos) = phi.sin_cos();
        let tan = sin / cos;
        let n = WGS84_A / (1.0 - e2 * sin * sin).sqrt();
        let t = tan * tan;
        let c = e_prime2 * cos * cos;
        let a = (longitude.to_radians() - self.longitude) * cos;
        let east = n
            * (a + (1.0 - t + c) * a.powi(3) / 6.0
                + (5.0 - 18.0 * t + t * t + 72.0 * c - 58.0 * e_prime2) * a.powi(5) / 120.0);
        let north = self.meridian(phi) - self.meridian_origin
            + n * tan
                * (a * a / 2.0
                    + (5.0 - t + 9.0 * c + 4.0 * c * c) * a.powi(4) / 24.0
                    + (61.0 - 58.0 * t + t * t + 600.0 * c - 330.0 * e_prime2) * a.powi(6) / 720.0);
        [east, north]
    }
}

fn given_values(value: &Value, count: usize, most: f32) -> Vec<f32> {
    let number = |value: &Value| -> Option<f32> {
        let number: f64 = match value {
            Value::Number(number) => number.as_f64()?,
            Value::String(text) => {
                let text = text.trim();
                let end = text
                    .find(|letter: char| !(letter.is_ascii_digit() || letter == '.'))
                    .unwrap_or(text.len());
                text[..end].parse().ok()?
            }
            _ => return None,
        };
        let number = number as f32;
        (number > 0.0 && number <= most).then_some(number)
    };
    match value {
        Value::Array(values) if values.len() == count => values
            .iter()
            .map(number)
            .collect::<Option<Vec<f32>>>()
            .unwrap_or_default(),
        value => number(value).map_or_else(Vec::new, |number| vec![number]),
    }
}

fn read_rivers(
    path: &Path,
    place: &dyn Fn(f64, f64) -> [f64; 2],
    size: f64,
) -> Result<Vec<GivenRiver>, String> {
    use engine_core::terrain_data::{
        MAX_CREDIT, MAX_DEPTH, MAX_DISCHARGE, MAX_RIVER_POINTS, MAX_WIDTH,
    };
    let name = path.display().to_string();
    let text = std::fs::read_to_string(path).map_err(|error| format!("{name}: {error}"))?;
    let document: Value =
        serde_json::from_str(&text).map_err(|error| format!("{name}: {error}"))?;
    let features: Vec<&Value> = match document["type"].as_str() {
        Some("FeatureCollection") => document["features"]
            .as_array()
            .map(|features| features.iter().collect())
            .unwrap_or_default(),
        Some("Feature") => vec![&document],
        _ => return Err(format!("{name}: a GeoJSON FeatureCollection or Feature")),
    };
    let on =
        |point: &[f64; 2]| (0.0..=size).contains(&point[0]) && (0.0..=size).contains(&point[1]);
    let mut rivers = Vec::new();
    for feature in features {
        let (geometry, properties) = (&feature["geometry"], &feature["properties"]);
        let lines: Vec<&Value> = match geometry["type"].as_str() {
            Some("LineString") => vec![&geometry["coordinates"]],
            Some("MultiLineString") => geometry["coordinates"]
                .as_array()
                .map(|lines| lines.iter().collect())
                .unwrap_or_default(),
            _ => continue,
        };
        for line in lines {
            let Some(coordinates) = line.as_array() else {
                continue;
            };
            let points: Vec<[f64; 2]> = coordinates
                .iter()
                .filter_map(|point| {
                    let pair = point.as_array()?;
                    Some(place(pair.first()?.as_f64()?, pair.get(1)?.as_f64()?))
                })
                .collect();
            let count = points.len();
            if count < 2 || count != coordinates.len() || !points.iter().any(on) {
                continue;
            }
            let width = given_values(&properties["width"], count, MAX_WIDTH);
            let depth = given_values(&properties["depth"], count, MAX_DEPTH);
            let discharge = given_values(&properties["discharge"], count, MAX_DISCHARGE);
            let named: Option<String> = properties["name"]
                .as_str()
                .map(|name| name.chars().take(MAX_CREDIT).collect());
            let mut start = 0;
            while start + 1 < count {
                let end = (start + MAX_RIVER_POINTS - 1).min(count - 1);
                let part = |values: &[f32]| {
                    if values.len() == count {
                        values[start..=end].to_vec()
                    } else {
                        values.to_vec()
                    }
                };
                rivers.push(GivenRiver {
                    name: named.clone(),
                    points: points[start..=end]
                        .iter()
                        .map(|point| [point[0] as f32, point[1] as f32])
                        .collect(),
                    width: part(&width),
                    depth: part(&depth),
                    discharge: part(&discharge),
                });
                start = end;
            }
        }
    }
    Ok(rivers)
}

#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
fn run() -> Result<(), String> {
    let options = options()?;
    let rasters = options
        .inputs
        .iter()
        .map(|path| {
            let extension = path
                .extension()
                .and_then(|extension| extension.to_str())
                .unwrap_or_default()
                .to_ascii_lowercase();
            match extension.as_str() {
                "tif" | "tiff" => read_geotiff(path),
                "png" | "r16" | "raw" => {
                    let (Some(extent), Some(range)) = (options.extent, options.range) else {
                        return Err(format!(
                            "{}: PNG and RAW heightmaps need --extent and --range",
                            path.display()
                        ));
                    };
                    read_heightmap(path, extent, range)
                }
                _ => Err(format!(
                    "{}: inputs are GeoTIFF (.tif), PNG (.png) or RAW (.r16, .raw)",
                    path.display()
                )),
            }
        })
        .collect::<Result<Vec<_>, String>>()?;
    let geographic = rasters[0].geographic;
    if rasters.iter().any(|raster| raster.geographic != geographic) {
        return Err("inputs mix geographic and projected coordinates".into());
    }
    if rasters.len() > 1
        && !rasters
            .iter()
            .all(|raster| raster.description.contains("EPSG"))
    {
        return Err("only georeferenced inputs mosaic; import heightmaps one at a time".into());
    }
    let center = match options.center {
        Some(center) => center,
        None if geographic => {
            let [longitude, latitude] = rasters[0].middle();
            [latitude, longitude]
        }
        None => rasters[0].middle(),
    };
    let size = options.size;
    let projection = geographic.then(|| TransverseMercator::new(center[0], center[1]));
    let model = |east: f64, north: f64| match &projection {
        Some(projection) => {
            let [latitude, longitude] = projection.inverse(east, north);
            [longitude, latitude]
        }
        None => [center[0] + east, center[1] + north],
    };
    let resolution = rasters
        .iter()
        .flat_map(|raster| raster.resolution(if geographic { center[0] } else { 0.0 }))
        .fold(f64::INFINITY, f64::min);
    let wanted = options.spacing.unwrap_or(resolution);
    let samples = ((size / wanted).ceil() as usize + 1).clamp(2, 16_384);
    let spacing = size / (samples - 1) as f64;
    let sub = ((spacing / resolution).ceil() as usize).clamp(1, 16);
    println!(
        "{} input(s), {samples} by {samples} samples {spacing:.2} m apart, {sub} by {sub} reads each",
        rasters.len()
    );
    let mut heights = vec![f64::NAN; samples * samples];
    for row in 0..samples {
        for column in 0..samples {
            let (mut sum, mut count) = (0.0, 0);
            for sy in 0..sub {
                for sx in 0..sub {
                    let offset = |index: usize| ((index as f64 + 0.5) / sub as f64 - 0.5) * spacing;
                    let east = column as f64 * spacing - 0.5 * size + offset(sx);
                    let north = 0.5 * size - row as f64 * spacing - offset(sy);
                    let [x, y] = model(east, north);
                    if let Some(height) = rasters.iter().find_map(|raster| raster.sample(x, y)) {
                        sum += height;
                        count += 1;
                    }
                }
            }
            if count > 0 {
                heights[row * samples + column] = sum / f64::from(count);
            }
        }
    }
    let missing = heights.iter().filter(|height| height.is_nan()).count();
    if missing == heights.len() {
        return Err("no input covers the terrain: check --center and --size".into());
    }
    let mut remaining = missing;
    while remaining > 0 {
        let previous = heights.clone();
        for row in 0..samples {
            for column in 0..samples {
                if !previous[row * samples + column].is_nan() {
                    continue;
                }
                let (mut sum, mut count) = (0.0, 0);
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
                    let (x, z) = (column as i64 + dx, row as i64 + dz);
                    if x < 0 || z < 0 || x >= samples as i64 || z >= samples as i64 {
                        continue;
                    }
                    let neighbor = previous[z as usize * samples + x as usize];
                    if !neighbor.is_nan() {
                        sum += neighbor;
                        count += 1;
                    }
                }
                if count > 0 {
                    heights[row * samples + column] = sum / f64::from(count);
                }
            }
        }
        remaining = heights.iter().filter(|height| height.is_nan()).count();
    }
    let seabed = options
        .sea
        .map_or(0, |level| seabed(&mut heights, samples, spacing, level));
    let lowest = heights.iter().copied().fold(f64::INFINITY, f64::min);
    let highest = heights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let (low, high) = ((lowest - 1.0).floor(), (highest + 1.0).ceil());
    let mut bytes = Vec::with_capacity(heights.len() * 2);
    for height in &heights {
        let value = ((height - low) / (high - low) * 65_535.0)
            .round()
            .clamp(0.0, 65_535.0) as u16;
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    let folder = options.out.join(&options.id);
    std::fs::create_dir_all(&folder).map_err(|error| format!("{}: {error}", folder.display()))?;
    let height_file = folder.join("height.r16");
    std::fs::write(&height_file, &bytes)
        .map_err(|error| format!("{}: {error}", height_file.display()))?;
    let source = format!(
        "{}; {}",
        rasters
            .iter()
            .map(|raster| raster.description.as_str())
            .collect::<Vec<_>>()
            .join(", "),
        if geographic {
            format!(
                "transverse Mercator centered at {:.5}, {:.5}",
                center[0], center[1]
            )
        } else {
            format!("centered at {:.1}, {:.1}", center[0], center[1])
        }
    );
    if geographic && options.attribution.is_none() {
        eprintln!(
            "terrain_import: warning: no --attribution; elevation data licences usually require one"
        );
    }
    let rivers = match &options.rivers {
        Some(path) => {
            let place = |first: f64, second: f64| -> [f64; 2] {
                let [east, north] = match &projection {
                    Some(projection) => projection.forward(second, first),
                    None => [first - center[0], second - center[1]],
                };
                [east + 0.5 * size, 0.5 * size - north]
            };
            let mut lines = read_rivers(path, &place, size)?;
            let most = engine_core::terrain_data::MAX_GIVEN_RIVERS;
            if lines.len() > most {
                eprintln!(
                    "terrain_import: warning: {} rivers on the terrain; keeping the {most} longest",
                    lines.len()
                );
                let length = |river: &GivenRiver| -> f64 {
                    river
                        .points
                        .windows(2)
                        .map(|pair| {
                            f64::from(pair[1][0] - pair[0][0])
                                .hypot(f64::from(pair[1][1] - pair[0][1]))
                        })
                        .sum()
                };
                lines.sort_by(|a, b| length(b).total_cmp(&length(a)));
                lines.truncate(most);
            }
            if lines.is_empty() {
                eprintln!(
                    "terrain_import: warning: no river of {} lies on the terrain",
                    path.display()
                );
                None
            } else {
                if options.river_attribution.is_none() {
                    eprintln!(
                        "terrain_import: warning: no --river-attribution; waterway data licences usually require one"
                    );
                }
                let described = format!(
                    "{} ({} lines); {}",
                    path.display(),
                    lines.len(),
                    if geographic {
                        "longitude and latitude, as the heightmap's projection places them"
                    } else {
                        "the inputs' own x and y"
                    }
                );
                let layer = RiversLayer {
                    source: Some(
                        described
                            .chars()
                            .take(engine_core::terrain_data::MAX_CREDIT)
                            .collect(),
                    ),
                    lines,
                    precision: options
                        .river_precision
                        .unwrap_or(spacing)
                        .min(f64::from(engine_core::terrain_data::MAX_PRECISION))
                        as f32,
                    attribution: options.river_attribution.clone(),
                };
                layer
                    .validate()
                    .map_err(|invalid| format!("{}: {invalid}", path.display()))?;
                Some(layer)
            }
        }
        None => None,
    };
    let maps = match &options.materials {
        Some(directory) => {
            for look in LOOKS {
                for map in ["albedo", "normal", "orm"] {
                    let name = format!("{}_{map}.ktx2", look.maps);
                    std::fs::copy(directory.join(&name), folder.join(&name))
                        .map_err(|error| format!("{}: {error}", directory.join(&name).display()))?;
                }
            }
            true
        }
        None => false,
    };
    let scene = scene(
        &options,
        (samples, spacing),
        (low, high),
        &heights,
        &source,
        maps,
        rivers.as_ref(),
    );
    let scene_file = options.out.join(format!("{}.scene.json", options.id));
    std::fs::write(
        &scene_file,
        serde_json::to_string_pretty(&scene).expect("scenes serialize"),
    )
    .map_err(|error| format!("{}: {error}", scene_file.display()))?;
    let river_lines = rivers.as_ref().map(|layer| layer.lines.len());
    let river_points: Option<usize> = rivers
        .as_ref()
        .map(|layer| layer.lines.iter().map(|river| river.points.len()).sum());
    let report = json!({
        "id": options.id,
        "inputs": rasters.iter().map(|raster| raster.description.clone()).collect::<Vec<_>>(),
        "center": center,
        "size": size,
        "samples": samples,
        "spacing": spacing,
        "lowest": lowest,
        "highest": highest,
        "low": low,
        "high": high,
        "filled_gaps": missing,
        "sea": options.sea,
        "seabed_samples": seabed,
        "source": source,
        "attribution": options.attribution,
        "rivers": river_lines,
        "river_points": river_points,
        "river_precision": rivers.as_ref().map(|layer| layer.precision),
        "river_source": rivers.as_ref().and_then(|layer| layer.source.clone()),
        "river_attribution": options.river_attribution,
        "heightmap": format!("{}/height.r16", options.id),
        "scene": format!("{}.scene.json", options.id),
    });
    let report_file = folder.join("import.json");
    std::fs::write(
        &report_file,
        serde_json::to_string_pretty(&report).expect("reports serialize"),
    )
    .map_err(|error| format!("{}: {error}", report_file.display()))?;
    println!(
        "{}",
        serde_json::to_string(&report).expect("reports serialize")
    );
    Ok(())
}

const SHELF_DEPTH: f64 = 60.0;
const SHELF_REACH: f64 = 250.0;

fn seabed(heights: &mut [f64], samples: usize, spacing: f64, level: f64) -> usize {
    let wet = |height: f64| height <= level + 0.5;
    let mut sea = vec![false; heights.len()];
    let mut queue = std::collections::VecDeque::new();
    for index in 0..samples {
        for sample in [
            index,
            (samples - 1) * samples + index,
            index * samples,
            index * samples + samples - 1,
        ] {
            if !sea[sample] && wet(heights[sample]) {
                sea[sample] = true;
                queue.push_back(sample);
            }
        }
    }
    while let Some(sample) = queue.pop_front() {
        let (column, row) = (sample % samples, sample / samples);
        let neighbors = [
            (column > 0).then(|| sample - 1),
            (column + 1 < samples).then(|| sample + 1),
            (row > 0).then(|| sample - samples),
            (row + 1 < samples).then(|| sample + samples),
        ];
        for neighbor in neighbors.into_iter().flatten() {
            if !sea[neighbor] && wet(heights[neighbor]) {
                sea[neighbor] = true;
                queue.push_back(neighbor);
            }
        }
    }
    let mut distance: Vec<f64> = sea
        .iter()
        .map(|&sea| if sea { f64::INFINITY } else { 0.0 })
        .collect();
    let diagonal = std::f64::consts::SQRT_2;
    let at = |column: isize, row: isize| -> Option<usize> {
        (column >= 0 && row >= 0 && (column as usize) < samples && (row as usize) < samples)
            .then(|| row as usize * samples + column as usize)
    };
    for row in 0..samples as isize {
        for column in 0..samples as isize {
            let here = at(column, row).expect("inside");
            for (dx, dy, step) in [
                (-1, 0, 1.0),
                (0, -1, 1.0),
                (-1, -1, diagonal),
                (1, -1, diagonal),
            ] {
                if let Some(there) = at(column + dx, row + dy) {
                    distance[here] = distance[here].min(distance[there] + step);
                }
            }
        }
    }
    for row in (0..samples as isize).rev() {
        for column in (0..samples as isize).rev() {
            let here = at(column, row).expect("inside");
            for (dx, dy, step) in [
                (1, 0, 1.0),
                (0, 1, 1.0),
                (1, 1, diagonal),
                (-1, 1, diagonal),
            ] {
                if let Some(there) = at(column + dx, row + dy) {
                    distance[here] = distance[here].min(distance[there] + step);
                }
            }
        }
    }
    let mut count = 0;
    for (index, height) in heights.iter_mut().enumerate() {
        if sea[index] {
            let meters = distance[index] * spacing;
            *height = level - 0.5 - (SHELF_DEPTH - 0.5) * (1.0 - (-meters / SHELF_REACH).exp());
            count += 1;
        }
    }
    count
}

struct Look {
    name: &'static str,
    maps: &'static str,
    color: [f32; 3],
    tint: [f32; 3],
    roughness: f32,
    tiling: f32,
    height_contrast: f32,
    biplanar: bool,
    grass: Option<[f32; 3]>,
}

const LOOKS: [Look; 9] = [
    Look {
        name: "grass",
        maps: "grass",
        color: [0.105, 0.135, 0.045],
        tint: [1.0, 1.0, 1.0],
        roughness: 0.92,
        tiling: 3.0,
        height_contrast: 0.35,
        biplanar: false,
        grass: Some([0.16, 0.23, 0.06]),
    },
    Look {
        name: "dry_grass",
        maps: "dry_grass",
        color: [0.2, 0.16, 0.08],
        tint: [0.9, 0.9, 0.9],
        roughness: 0.92,
        tiling: 3.5,
        height_contrast: 0.35,
        biplanar: false,
        grass: Some([0.34, 0.27, 0.12]),
    },
    Look {
        name: "dirt",
        maps: "dirt",
        color: [0.21, 0.165, 0.11],
        tint: [1.0, 1.0, 1.0],
        roughness: 0.92,
        tiling: 3.0,
        height_contrast: 0.5,
        biplanar: false,
        grass: None,
    },
    Look {
        name: "wet_grass",
        maps: "grass",
        color: [0.07, 0.13, 0.04],
        tint: [0.72, 0.95, 0.68],
        roughness: 0.85,
        tiling: 3.0,
        height_contrast: 0.35,
        biplanar: false,
        grass: Some([0.1, 0.2, 0.05]),
    },
    Look {
        name: "scree",
        maps: "scree",
        color: [0.2, 0.19, 0.18],
        tint: [0.78, 0.74, 0.68],
        roughness: 0.82,
        tiling: 4.0,
        height_contrast: 0.8,
        biplanar: false,
        grass: None,
    },
    Look {
        name: "riverbed",
        maps: "scree",
        color: [0.13, 0.13, 0.12],
        tint: [0.55, 0.56, 0.54],
        roughness: 0.55,
        tiling: 3.0,
        height_contrast: 0.9,
        biplanar: false,
        grass: None,
    },
    Look {
        name: "rock",
        maps: "rock",
        color: [0.25, 0.235, 0.22],
        tint: [0.74, 0.7, 0.64],
        roughness: 0.82,
        tiling: 8.0,
        height_contrast: 0.7,
        biplanar: true,
        grass: None,
    },
    Look {
        name: "sand",
        maps: "snow",
        color: [0.62, 0.54, 0.4],
        tint: [0.8, 0.7, 0.53],
        roughness: 0.9,
        tiling: 4.0,
        height_contrast: 0.3,
        biplanar: false,
        grass: None,
    },
    Look {
        name: "snow",
        maps: "snow",
        color: [0.82, 0.84, 0.88],
        tint: [1.0, 1.0, 1.0],
        roughness: 0.55,
        tiling: 6.0,
        height_contrast: 0.4,
        biplanar: false,
        grass: None,
    },
];

#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
fn scene(
    options: &Options,
    (samples, spacing): (usize, f64),
    (low, high): (f64, f64),
    heights: &[f64],
    source: &str,
    maps: bool,
    rivers: Option<&RiversLayer>,
) -> Value {
    let size = options.size;
    let id = &options.id;
    let half = 0.5 * size;
    let (peak, &summit) = heights
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .expect("imports hold samples");
    let peak = [
        (peak % samples) as f64 * spacing - half,
        (peak / samples) as f64 * spacing - half,
    ];
    let toward = {
        let length = peak[0].hypot(peak[1]);
        if length > 1.0 {
            [-peak[0] / length, -peak[1] / length]
        } else {
            [0.0, 1.0]
        }
    };
    let reach = (0.3 * size).min(4500.0);
    let eye = [peak[0] + toward[0] * reach, peak[1] + toward[1] * reach];
    let ground = {
        let column = ((eye[0] + half) / spacing)
            .round()
            .clamp(0.0, (samples - 1) as f64) as usize;
        let row = ((eye[1] + half) / spacing)
            .round()
            .clamp(0.0, (samples - 1) as f64) as usize;
        heights[row * samples + column]
    };
    let camera_height = (ground + 350.0).max(summit - 1500.0);
    let snow = match options.sea {
        Some(_) => high + 1000.0,
        None => high.min(2900.0).max(low + 0.7 * (high - low)),
    };
    let shore = options.sea.map_or(low - 1000.0, |level| level + 3.0);
    let middle = heights[(samples / 2) * samples + samples / 2];
    let (mut textures, mut materials) = (
        Vec::new(),
        vec![json!({"id": "plain", "color": [0.8, 0.3, 0.2, 1.0]})],
    );
    let surface_materials: Vec<Value> = LOOKS
        .iter()
        .map(|look| {
            let mut material = json!({
                "id": look.name,
                "color": look.color,
                "roughness": look.roughness,
                "tiling": look.tiling,
                "height_contrast": look.height_contrast,
                "projection": if look.biplanar { "biplanar" } else { "top" },
            });
            if maps {
                for map in ["albedo", "normal", "orm"] {
                    let texture = format!("{}_{map}", look.maps);
                    if !textures.iter().any(|entry: &Value| entry["id"] == json!(texture)) {
                        textures.push(json!({"id": texture, "uri": format!("{id}/{texture}.ktx2")}));
                    }
                }
                let name = format!("{}_maps", look.name);
                materials.push(json!({
                    "id": name,
                    "color": [look.tint[0], look.tint[1], look.tint[2], 1.0],
                    "surface": {"roughness": 1.0},
                    "texture": format!("{}_albedo", look.maps),
                    "normal_texture": format!("{}_normal", look.maps),
                    "metallic_roughness_texture": format!("{}_orm", look.maps),
                    "occlusion_texture": format!("{}_orm", look.maps),
                }));
                material["material"] = json!(name);
            }
            if let Some(color) = look.grass {
                material["cover"] = json!({"density": 120.0, "height": [0.12, 0.4], "width": 0.03, "color": color, "sway": 0.6});
            }
            material
        })
        .collect();
    let noise = |wavelength: f64, seed: u32, octaves: u32| json!({"noise": {"wavelength": wavelength, "seed": seed, "octaves": octaves}});
    let scaled = |mask: Value, by: f64| json!({"multiply": [mask, {"constant": by}]});
    let remap =
        |mask: Value, from: f64, to: f64| json!({"remap": {"mask": mask, "from": from, "to": to}});
    let layer = |material: &str, mask: Value| json!({"material": material, "mask": mask});
    let biomes = json!([{
        "id": "alpine",
        "layers": [
            layer("grass", json!({"constant": 1.0})),
            layer("dry_grass", remap(json!({"add": [scaled(noise(160.0, 11, 3), 0.6), scaled(noise(28.0, 12, 2), 0.25), scaled(json!({"height": [snow - 900.0, snow - 300.0]}), 0.3)]}), 0.5, 0.72)),
            layer("dirt", remap(json!({"add": [scaled(noise(60.0, 13, 2), 0.45), scaled(json!({"slope": [18.0, 32.0]}), 0.5)]}), 0.6, 0.78)),
            layer("wet_grass", remap(json!({"add": [{"wetness": [10.0, 13.5]}, scaled(noise(24.0, 41, 2), 0.35), {"constant": -0.15}]}), 0.35, 0.6)),
            layer("scree", remap(json!({"add": [scaled(json!({"slope": [26.0, 36.0]}), 0.7), scaled(json!({"height": [snow - 700.0, snow - 200.0]}), 0.4), scaled(noise(40.0, 15, 2), 0.3)]}), 0.6, 0.8)),
            layer("riverbed", remap(json!({"add": [{"multiply": [{"flow": [5.5, 5.7]}, {"invert": {"lake": [0.5, 2.0]}}]}, scaled(noise(12.0, 42, 2), 0.3)]}), 0.55, 0.75)),
            layer("sand", json!({"multiply": [remap(json!({"add": [{"invert": {"height": [shore - 1.0, shore + 4.0]}}, scaled(noise(30.0, 51, 2), 0.3)]}), 0.5, 0.7), {"invert": {"slope": [16.0, 26.0]}}]})),
            layer("rock", remap(json!({"add": [{"slope": [34.0, 46.0]}, scaled(noise(70.0, 1, 2), 0.3)]}), 0.55, 0.75)),
            layer("snow", json!({"multiply": [remap(json!({"add": [{"height": [snow - 250.0, snow + 250.0]}, scaled(noise(80.0, 3, 2), 0.35), {"constant": -0.15}]}), 0.4, 0.6), {"invert": {"slope": [38.0, 52.0]}}]})),
        ],
    }]);
    let mut heightmap = json!({
        "uri": format!("{id}/height.r16"),
        "width": samples,
        "height": samples,
        "min": [0.0, 0.0],
        "size": [size, size],
        "low": low,
        "high": high,
        "blend": "replace",
        "source": source,
    });
    if let Some(attribution) = &options.attribution {
        heightmap["attribution"] = json!(attribution);
    }
    let mut environment = json!({
        "atmosphere": {"preset": "golden_hour"},
    });
    if let Some(level) = options.sea {
        environment["ocean"] = json!({"enabled": true, "level": level});
    }
    let tiles = (size / 256.0) as u32;
    let mut layers = vec![json!({"heightmap": heightmap})];
    if options.detail > 0.0 {
        layers.push(json!({"noise": {"seed": 7, "height": options.detail, "offset": -0.5 * options.detail, "wavelength": 260.0, "ridges": 0.35, "warp": 0.25, "erosion": 0.6, "smoothing": 0.7, "valleys": 0.0, "roughness": 0.55, "blend": "add"}}));
    }
    if let Some(rivers) = rivers {
        layers.push(json!({ "rivers": rivers }));
    }
    layers.push(json!({"hydrology": {"level": 4}}));
    json!({
        "id": id,
        "revision": 1,
        "camera": {
            "position": [eye[0], camera_height, eye[1]],
            "target": [peak[0], summit - 500.0, peak[1]],
            "fov_degrees": 55.0,
            "near": 0.5,
            "far": 60000.0,
        },
        "textures": textures,
        "materials": materials,
        "meshes": [{"id": "box", "source": {"kind": "cube"}}],
        "entities": [{
            "id": "player",
            "mesh": "box",
            "material": "plain",
            "tags": ["player"],
            "collider": {"shape": {"kind": "cuboid", "half_extents": [0.5, 0.5, 0.5]}, "memberships": 2, "filter": 1, "sensor": false},
            "transform": {"translation": [0.0, middle + 1.0, 0.0]},
        }, {
            "id": "ground",
            "terrain": {
                "origin": [-half, 0.0, -half],
                "tiles": [tiles, tiles],
                "spacing": 1.0,
                "layers": layers,
                "surface": {"materials": surface_materials, "biomes": biomes},
            },
        }],
        "environment": environment,
    })
}
