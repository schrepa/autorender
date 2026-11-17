use engine_core::math::min;
use engine_core::{AssetLibrary, ColorSpace, Error, TextureFormat, TextureHandle, Var};
use engine_render::{Allocation, Bind, Install, Lifetime, Prepare, Sampler, Texture, wgpu};

use crate::Water;
use crate::gpu::{RippleKept, WATER_RIPPLE_LEVELS};
use crate::shaders::water::{flow, foam};
use crate::water::WaterGpu;

const FLAT_FOAM: [u8; 4] = [128, 255, 128, 128];
const FLAT_RIPPLES: [u8; 4] = [128, 128, 128, 128];

#[derive(Clone, Copy, Default)]
pub(crate) struct RippleMeasure {
    pub kept: RippleKept,
    pub deviation: [f32; 2],
}

pub(crate) struct DetailGpu {
    foam: Detail,
    ripples: Detail,
    measured: RippleMeasure,
    sampler: Sampler<WaterGpu>,
}

struct Detail {
    texture: Texture<WaterGpu>,
    flat: [u8; 4],
    shown: Option<Option<(TextureHandle, u64)>>,
}

impl DetailGpu {
    pub(crate) fn install(cx: &mut Install<'_, WaterGpu>) -> Self {
        Self {
            foam: Detail::install(cx, "water_foam", FLAT_FOAM),
            ripples: Detail::install(cx, "water_ripples", FLAT_RIPPLES),
            measured: RippleMeasure::default(),
            sampler: cx.sampler(engine_core::Sampler::REPEAT),
        }
    }

    pub(crate) fn bound(&self) -> [(Var, Bind<WaterGpu>); 3] {
        [
            (foam::WATER_FOAM_TEXTURE, Bind::Texture(self.foam.texture)),
            (foam::WATER_FOAM_SAMPLER, Bind::Sampler(self.sampler)),
            (flow::WATER_RIPPLE_TEXTURE, Bind::Texture(self.ripples.texture)),
        ]
    }

    pub(crate) fn measured(&self) -> &RippleMeasure {
        &self.measured
    }

    pub(crate) fn prepare(
        &mut self,
        water: &Water,
        assets: &AssetLibrary,
        cx: &mut Prepare<'_, WaterGpu>,
    ) {
        let features = cx.features();
        if let Some(texture) = self.foam.changed(water.foam.texture, features, assets) {
            self.foam.show(texture, cx);
        }
        let Some(texture) = self
            .ripples
            .changed(water.ripples.texture, features, assets)
        else {
            return;
        };
        let measured = texture.map(|texture| measure(texture).map(|measured| (texture, measured)));
        let (texture, measured) = match measured {
            Some(Ok((texture, measured))) => (Some(texture), measured),
            Some(Err(error)) => {
                cx.refuse(error.within("$.environment.water.ripples.texture"));
                (None, RippleMeasure::default())
            }
            None => (None, RippleMeasure::default()),
        };
        self.measured = measured;
        self.ripples.show(texture, cx);
    }
}

impl Detail {
    fn install(cx: &mut Install<'_, WaterGpu>, label: &'static str, flat: [u8; 4]) -> Self {
        Self {
            texture: cx.texture(label, wgpu::TextureFormat::Rgba8Unorm, Lifetime::Kept),
            flat,
            shown: None,
        }
    }

    fn changed<'a>(
        &mut self,
        wanted: Option<TextureHandle>,
        features: wgpu::Features,
        assets: &'a AssetLibrary,
    ) -> Option<Option<&'a engine_core::Texture>> {
        let source = wanted.and_then(|handle| {
            let texture = assets.texture(handle)?;
            let revision = assets.revision_of(handle)?;
            let (_, feature) = engine_render::texture_format(texture.format(), ColorSpace::Linear);
            (!texture.levels().is_empty() && features.contains(feature))
                .then_some((handle, revision, &**texture))
        });
        let shown = source.map(|(handle, revision, _)| (handle, revision));
        if self.shown == Some(shown) {
            return None;
        }
        self.shown = Some(shown);
        Some(source.map(|(_, _, texture)| texture))
    }

    fn show(&self, texture: Option<&engine_core::Texture>, cx: &mut Prepare<'_, WaterGpu>) {
        let Some(texture) = texture else {
            let texel = wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            };
            let (Allocation::Fresh | Allocation::Kept) =
                cx.allocate_as(self.texture, wgpu::TextureFormat::Rgba8Unorm, texel, 1);
            cx.write_level(self.texture, (0, 0), (1, 1), TextureFormat::Rgba8, &self.flat);
            return;
        };
        let (format, _) = engine_render::texture_format(texture.format(), ColorSpace::Linear);
        let size = (texture.width(), texture.height());
        let levels = if generated(texture) {
            engine_render::mip_levels(size.0, size.1)
        } else {
            texture.levels().len() as u32
        };
        let extent = wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        };
        let (Allocation::Fresh | Allocation::Kept) =
            cx.allocate_as(self.texture, format, extent, levels);
        for (level, bytes) in texture.levels().iter().enumerate() {
            cx.write_level(
                self.texture,
                (level as u32, 0),
                size,
                texture.format(),
                bytes,
            );
        }
        if generated(texture) && levels > 1 {
            cx.mipmaps(self.texture, 0..1);
        }
    }
}

fn generated(texture: &engine_core::Texture) -> bool {
    !texture.format().is_compressed() && texture.levels().len() == 1
}

fn measure(texture: &engine_core::Texture) -> Result<RippleMeasure, Error> {
    let plain;
    let levels = if texture.format().is_compressed() {
        plain = engine_core::textures::decompress(texture)?;
        plain.levels()
    } else {
        texture.levels()
    };
    let mut spreads: Vec<[f64; 2]> = levels.iter().map(|level| spread(level)).collect();
    if generated(texture) {
        let mut size = (texture.width() as usize, texture.height() as usize);
        let mut pixels = texture.pixels().to_vec();
        while size != (1, 1) {
            (pixels, size) = halved(&pixels, size);
            spreads.push(spread(&pixels));
        }
    }
    let finest = spreads.first().copied().unwrap_or_default();
    let rows = WATER_RIPPLE_LEVELS as usize;
    let mut kept = RippleKept::default();
    for (level, spread) in spreads.iter().take(rows).enumerate() {
        for (field, (&whole, &left)) in finest.iter().zip(spread).enumerate() {
            let share = if whole > 0.0 { left / whole } else { 0.0 };
            let at = field * rows + level;
            kept[at / 4][at % 4] = min(share as f32, 1.0);
        }
    }
    Ok(RippleMeasure {
        kept,
        deviation: finest.map(|spread| ((0.5 * spread).sqrt() / 255.0) as f32),
    })
}

fn spread(pixels: &[u8]) -> [f64; 2] {
    let texels = pixels.as_chunks::<4>().0;
    let count = texels.len().max(1) as f64;
    let mut sums = [0_u64; 4];
    let mut squares = [0_u64; 4];
    for texel in texels {
        for ((sum, square), &value) in sums.iter_mut().zip(&mut squares).zip(texel) {
            *sum += u64::from(value);
            *square += u64::from(value) * u64::from(value);
        }
    }
    let spread = |channel: usize| {
        let mean = sums[channel] as f64 / count;
        squares[channel] as f64 / count - mean * mean
    };
    [spread(0) + spread(1), spread(2) + spread(3)]
}

fn halved(pixels: &[u8], (width, height): (usize, usize)) -> (Vec<u8>, (usize, usize)) {
    let (wide, high) = ((width / 2).max(1), (height / 2).max(1));
    let mut half = Vec::with_capacity(wide * high * 4);
    for y in 0..high {
        let (top, bottom, down) = taps(y, high, height);
        for x in 0..wide {
            let (left, right, along) = taps(x, wide, width);
            for channel in 0..4 {
                let at = |x: usize, y: usize| f32::from(pixels[(y * width + x) * 4 + channel]);
                let above = at(left, top) + (at(right, top) - at(left, top)) * along;
                let below = at(left, bottom) + (at(right, bottom) - at(left, bottom)) * along;
                half.push((above + (below - above) * down + 0.5) as u8);
            }
        }
    }
    (half, (wide, high))
}

fn taps(at: usize, count: usize, source: usize) -> (usize, usize, f32) {
    let center = (at as f32 + 0.5) * source as f32 / count as f32 - 0.5;
    let first = center as usize;
    (first, (first + 1).min(source - 1), center - first as f32)
}
