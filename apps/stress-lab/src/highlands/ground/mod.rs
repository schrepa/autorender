//! The highlands' ground materials: tileable procedural grass, dry grass,
//! dirt, scree, rock and snow, each three maps ([`sources`]): sRGB base
//! color; tangent-space normals in OpenGL's convention (+X the image's
//! right, +Y its up, world -z, +Z out of the ground); and ambient occlusion
//! in red, roughness in green, metalness in blue and height in alpha (0 to
//! 1, a half on average), which height-based blending reads. Maps lie
//! top-down: rows run from the texture's top, u along world x and v along
//! world z, and every map wraps seamlessly. Each generates at twice its
//! size and averages down. They load as texture records
//! (`engine::TextureRecords`): this code runs where no pack or store holds
//! them, as the cook runs it.
//!
//! Maps keep no structure broader than about a sixth of the texture: a
//! repeating texture's broad light and dark regions and tilts survive
//! mipmapping and show as a grid of blotches from afar. Engines vary the
//! ground broadly themselves.

mod kinds;
mod noise;

use engine::prelude::math::{clamp, ln, max, pow};
use engine::prelude::{TextureSource, TextureUsage};

use noise::{low_pass, occlusion, saturate, scale};

/// Texels along each side of every map.
pub(super) const SIZE: usize = 1024;
/// Materials generate at this many times their size, then average down.
const SUPERSAMPLE: usize = 2;

/// The maps of ground material `name` (`grass`, `dry_grass`, `dirt`,
/// `scree`, `rock` or `snow`) as sources the cook compresses: base color,
/// normal, and occlusion, roughness, metalness and height.
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

/// The generated materials.
#[derive(Clone, Copy)]
enum Kind {
    Grass,
    DryGrass,
    Dirt,
    Scree,
    Rock,
    Snow,
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
            _ => return None,
        })
    }

    /// How steeply height turns the normals: texels of rise per unit of
    /// height at 512 texels.
    fn relief(self) -> f32 {
        match self {
            Self::Grass => 4.0,
            Self::DryGrass => 4.0,
            Self::Dirt => 3.0,
            Self::Scree => 5.0,
            Self::Rock => 22.0,
            Self::Snow => 2.5,
        }
    }

    /// The material at `size` texels along each side.
    fn generate(self, size: usize) -> Maps {
        match self {
            Self::Grass => kinds::meadow(size, false),
            Self::DryGrass => kinds::meadow(size, true),
            Self::Dirt => kinds::dirt(size),
            Self::Scree => kinds::scree(size),
            Self::Rock => kinds::rock(size),
            Self::Snow => kinds::snow(size),
        }
    }
}

/// Linear RGB.
type Rgb = [f32; 3];

/// One texel of a material while it generates.
#[derive(Clone, Copy, Default)]
struct Texel {
    albedo: Rgb,
    height: f32,
    roughness: f32,
}

/// A generated material: `size` by `size` texels of linear albedo, height
/// (in its own units, about 0 to 1), ambient occlusion and roughness.
struct Maps {
    size: usize,
    albedo: Vec<Rgb>,
    height: Vec<f32>,
    occlusion: Vec<f32>,
    roughness: Vec<f32>,
}

impl Maps {
    /// Maps of `texels`, with occlusion of `strength` from their heights.
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

    /// Removes structure broader than about a sixth of the texture, which
    /// would repeat as a grid from afar: height and the occlusion and
    /// roughness maps lose their low-passed variation around their means
    /// (heights spread over 0 to 1 again when encoded, before their
    /// normals derive), and albedo is divided by its low-passed luminance
    /// and multiplied by its mean luminance, keeping hue and small-scale
    /// contrast.
    fn equalize(&mut self) {
        let size = self.size;
        // Three box passes of this radius: about a Gaussian whose low-pass
        // keeps half of structure a sixth of the texture across, nearly all
        // of anything broader.
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

    /// These maps at half the size, each texel the mean of four.
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

/// A material's three maps as RGBA8 texels.
struct Encoded {
    albedo: Vec<u8>,
    normal: Vec<u8>,
    orm: Vec<u8>,
}

impl Encoded {
    /// Encodes `maps`, its normals turned by `relief` texels of rise per
    /// unit of height at 512 texels.
    fn new(maps: &Maps, relief: f32) -> Self {
        let size = maps.size;
        // Heights spread over 0 to 1 between their extreme percentiles.
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
        // Alpha: clamped, with a power bringing its mean to about a half.
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

/// Linear to 8-bit sRGB.
fn srgb(value: f32) -> u8 {
    let value = saturate(value);
    let encoded = if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * pow(value, 1.0 / 2.4) - 0.055
    };
    unorm(encoded)
}
