//! Cutoff-specific coverage of an immutable RGBA8 texture's mip chain.
//! Only alpha statistics are retained; the shared color texture is untouched.

use engine_core::{AlphaMode, Filter, Material, Texture, TextureFormat};

/// One factor per admitted mip, packed as the material uniform's vec4 rows.
pub(crate) const FACTOR_ROWS: usize =
    (engine_core::MAX_TEXTURE_DIMENSION.ilog2() as usize + 1).div_ceil(4);

pub(crate) struct Curve {
    pub factors: [[f32; 4]; FACTOR_ROWS],
    // Enabled, first uniform level (legacy fallback), recorded level count.
    pub metadata: [f32; 4],
}

struct Level {
    histogram: [u32; 256],
    texels: u32,
}

/// Owned by the resident source texture, shared by all its material cutoffs.
pub(crate) struct Profile {
    levels: Box<[Level]>,
}

/// Constant material conditions. The fragment also checks the combined
/// material, instance and vertex opacity before using the curve.
pub(crate) fn cutoff(material: &Material) -> Option<f32> {
    let AlphaMode::Mask { cutoff } = material.alpha_mode() else {
        return None;
    };
    let map = material.base_color_texture()?;
    (cutoff.is_normal()
        && cutoff > 0.0
        && cutoff < 1.0
        && material.opacity() == 1.0
        && material.world_tiling().is_none()
        && map.sampler.mag_filter == Filter::Linear
        && map.sampler.min_filter == Filter::Linear
        && map.sampler.mipmap_filter == Some(Filter::Linear))
    .then_some(cutoff)
}

impl Profile {
    pub(crate) fn new(source: &Texture) -> Option<Self> {
        let (mut width, mut height) = (source.width(), source.height());
        if source.format() != TextureFormat::Rgba8
            || !width.is_power_of_two()
            || !height.is_power_of_two()
            || width.max(height) < 2
        {
            return None;
        }
        let mut pixels: Vec<u8> = source
            .pixels()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|rgba| rgba[3])
            .collect();
        let mut levels = Vec::with_capacity(width.max(height).ilog2() as usize + 1);
        loop {
            let mut histogram = [0; 256];
            for &alpha in &pixels {
                histogram[usize::from(alpha)] += 1;
            }
            levels.push(Level {
                histogram,
                texels: width * height,
            });
            if width == 1 && height == 1 {
                break;
            }
            let (next_width, next_height) = ((width / 2).max(1), (height / 2).max(1));
            let mut next = Vec::with_capacity((next_width * next_height) as usize);
            for y in 0..next_height {
                for x in 0..next_width {
                    let x0 = x * 2;
                    let y0 = y * 2;
                    let x1 = (x0 + 1).min(width - 1);
                    let y1 = (y0 + 1).min(height - 1);
                    let sum: u32 = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
                        .into_iter()
                        .map(|(x, y)| u32::from(pixels[(y * width + x) as usize]))
                        .sum();
                    let lower = sum >> 2;
                    let remainder = sum & 3;
                    let round_up = remainder > 2 || (remainder == 2 && lower & 1 != 0);
                    next.push((lower + u32::from(round_up)) as u8);
                }
            }
            pixels = next;
            (width, height) = (next_width, next_height);
        }
        Some(Self {
            levels: levels.into_boxed_slice(),
        })
    }

    pub(crate) fn curve(&self, cutoff: f32) -> Option<Curve> {
        // Callers admit normal cutoffs in (0, 1). Widen them exactly for
        // histogram comparisons; integer cross-products compare area errors
        // without depending on floating-point tie rounding.
        let cutoff = f64::from(cutoff);
        let base = &self.levels[0];
        let covered = count(base, cutoff);
        if covered == 0 || covered == base.texels {
            return None;
        }
        let mut curve = Curve {
            factors: [[1.0; 4]; FACTOR_ROWS],
            metadata: [
                1.0,
                (self.levels.len() - 1) as f32,
                self.levels.len() as f32,
                0.0,
            ],
        };
        for (index, level) in self.levels.iter().enumerate().skip(1) {
            let legacy = 1.0 + 0.25 * index as f32;
            if level.histogram.contains(&level.texels) {
                curve.factors[index >> 2][index & 3] = legacy;
                curve.metadata[1] = index as f32;
                break;
            }
            let old = cutoff / f64::from(legacy);
            let error = |count: u32| {
                (u64::from(count) * u64::from(base.texels))
                    .abs_diff(u64::from(covered) * u64::from(level.texels))
            };
            let mut selected = old;
            let mut best_error = error(count(level, old));
            let mut best_distance = 0.0;
            let mut remaining = level.texels;
            for (alpha, &count) in level.histogram.iter().enumerate() {
                // (k + 0.5)/255 lies strictly between bytes k and k+1.
                remaining -= count;
                let threshold = (alpha as f64 + 0.5) / 255.0;
                let candidate_error = error(remaining);
                let distance = (threshold - old).abs();
                if candidate_error < best_error
                    || (candidate_error == best_error
                        && (distance < best_distance
                            || (distance == best_distance && threshold > selected)))
                {
                    selected = threshold;
                    best_error = candidate_error;
                    best_distance = distance;
                }
            }
            let factor = (cutoff / selected) as f32;
            if !factor.is_normal() || factor <= 0.0 {
                return None;
            }
            curve.factors[index >> 2][index & 3] = factor;
        }
        Some(curve)
    }
}

fn count(level: &Level, threshold: f64) -> u32 {
    level
        .histogram
        .iter()
        .enumerate()
        .filter(|(alpha, _)| *alpha as f64 / 255.0 >= threshold)
        .map(|(_, count)| count)
        .sum()
}
