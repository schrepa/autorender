use std::path::Path;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use engine_assets::Texture;
use engine_core::math::sin;
use engine_textures::{COOK_JOB, Family, Source, TextureWorker, Usage};
use serde_json::{Map, json};

use super::Observations;

const SIDE: u32 = 64;
pub const JOBS: &str = "target/pure/textures-jobs.json";
const SLICE: u64 = 4096;

struct Image {
    name: &'static str,
    source: Source,
    floors: [f64; 2],
}

fn hash(x: u32, y: u32, salt: u32) -> u32 {
    let mut h =
        x.wrapping_mul(0x8da6_b343) ^ y.wrapping_mul(0xd816_3841) ^ salt.wrapping_mul(0xcb1a_b31f);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846c_a68b);
    h ^ (h >> 16)
}

fn texels(texel: impl Fn(u32, u32) -> [u8; 4]) -> Vec<u8> {
    (0..SIDE)
        .flat_map(|y| (0..SIDE).map(move |x| (x, y)))
        .flat_map(|(x, y)| texel(x, y))
        .collect()
}

fn images() -> Vec<Image> {
    let image = |name, usage, floors, pixels| Image {
        name,
        source: Source {
            width: SIDE,
            height: SIDE,
            pixels,
            usage,
        },
        floors,
    };
    let unit = |value: f32| (engine_core::math::clamp(value, 0.0, 1.0) * 255.0 + 0.5) as u8;
    let height = |x: u32, y: u32| {
        let (u, v) = (x as f32 / SIDE as f32, y as f32 / SIDE as f32);
        0.5 + 0.25 * sin(u * 25.1) * sin(v * 18.8) + 0.25 * sin((u + v) * 12.6)
    };
    vec![
        image(
            "smooth color",
            Usage::Color,
            [44.0, 42.0],
            texels(|x, y| [(x * 4) as u8, (y * 4) as u8, ((x + y) * 2) as u8, 255]),
        ),
        image(
            "noisy color",
            Usage::Color,
            [28.0, 26.0],
            texels(|x, y| {
                let random = hash(x, y, 1);
                [
                    random as u8,
                    (random >> 8) as u8 / 2 + 64,
                    (random >> 16) as u8 / 4,
                    255,
                ]
            }),
        ),
        image(
            "cutout",
            Usage::Color,
            [30.0, 28.0],
            texels(|x, y| {
                let inside = (x / 8 + y / 8) % 2 == 0;
                [90, (y * 3) as u8 + 40, 60, if inside { 255 } else { 0 }]
            }),
        ),
        image(
            "data with height",
            Usage::Data,
            [36.0, 34.0],
            texels(|x, y| {
                let ripple = height(x, y);
                [
                    unit(1.0 - 0.3 * ripple),
                    (128 + x) as u8,
                    0,
                    unit(height(y, x)),
                ]
            }),
        ),
        image(
            "normal map",
            Usage::Normal,
            [35.0, 31.0],
            texels(|x, y| {
                let dx = (height(x + 1, y) - height(x.saturating_sub(1), y)) * 8.0;
                let dy = (height(x, y + 1) - height(x, y.saturating_sub(1))) * 8.0;
                let length = (dx * dx + dy * dy + 1.0).sqrt();
                let channel = |value: f32| unit(value / length * 0.5 + 0.5);
                [channel(-dx), channel(dy), channel(1.0), 255]
            }),
        ),
    ]
}

fn psnr(expected: &[u8], decoded: &[u8]) -> f64 {
    let squared: f64 = expected
        .iter()
        .zip(decoded)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum();
    if squared == 0.0 {
        return 99.0;
    }
    10.0 * (255.0_f64 * 255.0 * expected.len() as f64 / squared).log10()
}

fn reference(texture: &Texture, level: usize) -> Result<Vec<u8>, String> {
    let width = (texture.width() >> level).max(1) as usize;
    let height = (texture.height() >> level).max(1) as usize;
    let blocks = &texture.levels()[level];
    let mut image = vec![0_u32; width * height];
    match texture.format() {
        engine_assets::TextureFormat::Bc7 => {
            texture2ddecoder::decode_bc7(blocks, width, height, &mut image)
        }
        engine_assets::TextureFormat::Astc4x4 => {
            texture2ddecoder::decode_astc(blocks, width, height, 4, 4, &mut image)
        }
        engine_assets::TextureFormat::Rgba8 | engine_assets::TextureFormat::Etc2Rgba8 => {
            Err("not a cooked family")
        }
    }
    .map_err(|error| format!("the reference decoder refused level {level}: {error}"))?;
    Ok(image
        .iter()
        .flat_map(|texel| {
            let [blue, green, red, alpha] = texel.to_le_bytes();
            [red, green, blue, alpha]
        })
        .collect())
}

pub fn run(root: &Path, observed: &mut Observations) -> Result<(), String> {
    let mut worker = Worker::default();
    for image in images() {
        let levels = engine_textures::levels(&image.source);
        let mut observation = Map::new();
        for (family, floor) in Family::ALL.into_iter().zip(image.floors) {
            let texture = engine_textures::cook(&image.source, family)
                .map_err(|error| format!("{} does not cook: {error}", image.name))?;
            let record = engine_textures::record(&texture);
            let back = engine_textures::decode(&record)?;
            observed.check(
                back.levels() == texture.levels() && back.summary() == texture.summary(),
                || {
                    format!(
                        "{} {}: its record reads back otherwise",
                        image.name,
                        family.name()
                    )
                },
            );
            let input = engine_textures::cook_input(&image.source, family);
            let (whole, sliced) = worker.both(&input)?;
            observed.check(whole == record && sliced == record, || {
                format!(
                    "{} {}: its cooking job makes another record",
                    image.name,
                    family.name()
                )
            });
            let mut errors = Vec::new();
            for (level, expected) in levels.iter().enumerate() {
                let decoded = reference(&texture, level)?;
                let ratio = psnr(expected, &decoded);
                errors.push((ratio * 10.0).round() / 10.0);
                if level == 0 {
                    observed.check(ratio >= floor, || {
                        format!(
                            "{} {}: its base level keeps {ratio:.2} dB, under {floor} dB",
                            image.name,
                            family.name()
                        )
                    });
                }
            }
            if family == Family::Bc7 {
                let own = engine_textures::decompress(&texture)
                    .map_err(|error| format!("{} does not decompress: {error}", image.name))?;
                for level in 0..levels.len() {
                    let same = own.levels()[level][..] == reference(&texture, level)?[..];
                    observed.check(same, || {
                        format!(
                            "{}: the engine's BC7 decoder reads level {level} otherwise",
                            image.name
                        )
                    });
                }
            }
            observation.insert(
                family.name().into(),
                json!({ "hash": super::hash(&record), "psnr by level": errors }),
            );
        }
        observed.record(format!("textures {}", image.name), observation);
    }
    worker.write(root)
}

#[derive(Default)]
struct Worker {
    worker: TextureWorker,
    next: u64,
    log: Vec<serde_json::Value>,
}

impl Worker {
    fn both(&mut self, input: &[u8]) -> Result<(Vec<u8>, Vec<u8>), String> {
        let whole =
            engine_textures::run(COOK_JOB, input).ok_or("the cooking job is no texture job")??;
        self.log.push(json!({
            "job": COOK_JOB,
            "input": STANDARD.encode(input),
            "output": super::hash(&whole),
        }));
        self.next += 1;
        self.worker.start(self.next, COOK_JOB, input)?;
        let sliced = loop {
            if let Some(output) = self.worker.step(self.next, SLICE)? {
                break output;
            }
        };
        self.log.push(json!({
            "job": COOK_JOB,
            "input": STANDARD.encode(input),
            "slice": SLICE,
            "output": super::hash(&sliced),
        }));
        Ok((whole, sliced))
    }

    fn write(&self, root: &Path) -> Result<(), String> {
        let path = root.join(JOBS);
        if let Some(folder) = path.parent() {
            std::fs::create_dir_all(folder)
                .map_err(|error| format!("cannot create {}: {error}", folder.display()))?;
        }
        std::fs::write(
            &path,
            serde_json::Value::Array(self.log.clone()).to_string(),
        )
        .map_err(|error| format!("cannot write {JOBS}: {error}"))
    }
}
