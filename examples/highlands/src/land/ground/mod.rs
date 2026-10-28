mod kinds;
mod noise;

use engine::prelude::math::{clamp, ln, max, pow};
use engine::prelude::{TextureSource, TextureUsage};

use noise::{low_pass, occlusion, saturate, scale};

pub(super) const SIZE: usize = 1024;
const SUPERSAMPLE: usize = 2;

pub(super) fn sources(name: &str) -> Result<Vec<TextureSource>, String> {
    let kind = Kind::named(name).ok_or_else(|| format!("no ground material is named {name}"))?;
    let mut maps = kind.generate(SIZE * SUPERSAMPLE).downsample();
    maps.equalize();
    let encoded = Encoded::new(&maps, kind.relief());
    let source = |pixels, usage| TextureSource {
        width: SIZE as u32,
        height: SIZE as u32,
        pixels,
        usage,
    };
    Ok(vec![
        source(encoded.albedo, TextureUsage::Color),
        source(encoded.normal, TextureUsage::Normal),
        source(encoded.orm, TextureUsage::Data),
    ])
}

#[derive(Clone, Copy)]
enum Kind {
    Grass,
    DryGrass,
    Dirt,
    Scree,
    Rock,
    Snow,
    Sand,
    Gravel,
    Cobble,
}

impl Kind {
    fn named(name: &str) -> Option<Self> {
        Some(match name {
            "grass" => Self::Grass,
            "dry_grass" => Self::DryGrass,
            "dirt" => Self::Dirt,
            "scree" => Self::Scree,
            "rock" => Self::Rock,
            "snow" => Self::Snow,
            "sand" => Self::Sand,
            "gravel" => Self::Gravel,
            "cobble" => Self::Cobble,
            _ => return None,
        })
    }

    fn relief(self) -> f32 {
        match self {
            Self::Grass => 4.0,
            Self::DryGrass => 4.0,
            Self::Dirt => 3.0,
            Self::Scree => 5.0,
            Self::Rock => 22.0,
            Self::Snow => 2.5,
            Self::Sand => 2.5,
            Self::Gravel => 6.0,
            Self::Cobble => 9.0,
        }
    }

    fn generate(self, size: usize) -> Maps {
        match self {
            Self::Grass => kinds::meadow(size, false),
            Self::DryGrass => kinds::meadow(size, true),
            Self::Dirt => kinds::dirt(size),
            Self::Scree => kinds::scree(size),
            Self::Rock => kinds::rock(size),
            Self::Snow => kinds::snow(size),
            Self::Sand => kinds::sand(size),
            Self::Gravel => kinds::gravel(size),
            Self::Cobble => kinds::cobble(size),
        }
    }
}

type Rgb = [f32; 3];

#[derive(Clone, Copy, Default)]
struct Texel {
    albedo: Rgb,
    height: f32,
    roughness: f32,
}

struct Maps {
    size: usize,
    albedo: Vec<Rgb>,
    height: Vec<f32>,
    occlusion: Vec<f32>,
    roughness: Vec<f32>,
}

impl Maps {
    fn new(size: usize, texels: Vec<Texel>, strength: f32) -> Self {
        let height: Vec<f32> = texels.iter().map(|texel| texel.height).collect();
        Self {
            size,
            occlusion: occlusion(&height, size, strength),
            albedo: texels.iter().map(|texel| texel.albedo).collect(),
            roughness: texels.iter().map(|texel| texel.roughness).collect(),
            height,
        }
    }

    fn equalize(&mut self) {
        let size = self.size;
        let radius = usize::max(1, size / 32);
        let flatten = |values: &mut [f32]| {
            let mean = values.iter().sum::<f32>() / values.len() as f32;
            let low = low_pass(values, size, radius);
            for (value, low) in values.iter_mut().zip(low) {
                *value += mean - low;
            }
        };
        flatten(&mut self.height);
        flatten(&mut self.occlusion);
        flatten(&mut self.roughness);
        let luminance: Vec<f32> = self
            .albedo
            .iter()
            .map(|color| 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2])
            .collect();
        let mean = luminance.iter().sum::<f32>() / luminance.len() as f32;
        let low = low_pass(&luminance, size, radius);
        for (color, low) in self.albedo.iter_mut().zip(low) {
            *color = scale(*color, mean / max(low, 1.0e-4));
        }
    }

    fn downsample(&self) -> Self {
        let half = self.size / 2;
        let mean = |values: &[f32], x: usize, y: usize| {
            let at = |dx: usize, dy: usize| values[(2 * y + dy) * self.size + 2 * x + dx];
            (at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1)) * 0.25
        };
        let mut maps = Self {
            size: half,
            albedo: Vec::with_capacity(half * half),
            height: Vec::with_capacity(half * half),
            occlusion: Vec::with_capacity(half * half),
            roughness: Vec::with_capacity(half * half),
        };
        let channels: [Vec<f32>; 3] =
            std::array::from_fn(|channel| self.albedo.iter().map(|rgb| rgb[channel]).collect());
        for y in 0..half {
            for x in 0..half {
                maps.albedo.push(std::array::from_fn(|channel| {
                    mean(&channels[channel], x, y)
                }));
                maps.height.push(mean(&self.height, x, y));
                maps.occlusion.push(mean(&self.occlusion, x, y));
                maps.roughness.push(mean(&self.roughness, x, y));
            }
        }
        maps
    }
}

struct Encoded {
    albedo: Vec<u8>,
    normal: Vec<u8>,
    orm: Vec<u8>,
}

impl Encoded {
    fn new(maps: &Maps, relief: f32) -> Self {
        let size = maps.size;
        let mut sorted = maps.height.clone();
        sorted.sort_by(f32::total_cmp);
        let tail = sorted.len() / 200;
        let (low, high) = (sorted[tail], sorted[sorted.len() - 1 - tail]);
        let span = max(high - low, 1.0e-6);
        let height: Vec<f32> = maps
            .height
            .iter()
            .map(|value| (value - low) / span)
            .collect();
        let mut alpha: Vec<f32> = height.iter().map(|&value| saturate(value)).collect();
        for _ in 0..4 {
            let mean = alpha.iter().sum::<f32>() / alpha.len() as f32;
            let power = clamp(ln(0.5) / ln(clamp(mean, 0.01, 0.99)), 0.25, 4.0);
            for value in &mut alpha {
                *value = pow(*value, power);
            }
        }
        let strength = relief * size as f32 / 512.0;
        let at = |x: usize, y: usize| height[(y % size) * size + x % size];
        let mut normal = Vec::with_capacity(size * size * 4);
        for y in 0..size {
            for x in 0..size {
                let du = (at(x + 1, y) - at(x + size - 1, y)) * 0.5;
                let dv = (at(x, y + 1) - at(x, y + size - 1)) * 0.5;
                let n = [-strength * du, strength * dv, 1.0];
                let length = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                for value in n {
                    normal.push(unorm(value / length * 0.5 + 0.5));
                }
                normal.push(255);
            }
        }
        let mut albedo = Vec::with_capacity(size * size * 4);
        let mut orm = Vec::with_capacity(size * size * 4);
        for (index, color) in maps.albedo.iter().enumerate() {
            albedo.extend(color.map(srgb));
            albedo.push(255);
            orm.extend([
                unorm(maps.occlusion[index]),
                unorm(maps.roughness[index]),
                0,
                unorm(alpha[index]),
            ]);
        }
        Self {
            albedo,
            normal,
            orm,
        }
    }
}

fn unorm(value: f32) -> u8 {
    (saturate(value) * 255.0 + 0.5) as u8
}

fn srgb(value: f32) -> u8 {
    let value = saturate(value);
    let encoded = if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * pow(value, 1.0 / 2.4) - 0.055
    };
    unorm(encoded)
}
