use gltf::image::Source as ImageSource;
use gltf::texture::{MagFilter, MinFilter, WrappingMode};

use crate::fault::Fault;
use crate::image::{self, Image};
use crate::mips::{byte, linear_to_srgb, srgb_to_linear};
use crate::source::Source;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MapKind {
    BaseColor,
    Normal,
    Orm,
}

impl MapKind {
    pub const ALL: [Self; 3] = [Self::BaseColor, Self::Normal, Self::Orm];

    pub const fn name(self) -> &'static str {
        match self {
            Self::BaseColor => "base_color",
            Self::Normal => "normal",
            Self::Orm => "orm",
        }
    }
}

#[derive(Clone, Copy)]
pub struct Sampling {
    pub wrap_u: WrappingMode,
    pub wrap_v: WrappingMode,
    pub mag: Option<MagFilter>,
    pub min: Option<MinFilter>,
    pub tex_coord: u32,
}

pub struct Map {
    pub kind: MapKind,
    pub material: String,
    pub stem: String,
    pub decoded: Image,
    pub sampling: Sampling,
}

pub fn image_bytes<'s>(source: &'s Source, image: &gltf::Image<'_>) -> Result<&'s [u8], Fault> {
    let item = format!("images[{}]", image.index());
    match image.source() {
        ImageSource::View { view, .. } => source.view(&view).ok_or_else(|| {
            source.fault(
                "ImageUnreadable",
                item,
                "buffer view outside the binary chunk",
            )
        }),
        ImageSource::Uri { uri, .. } => Err(source.fault("ExternalUri", item, uri)),
    }
}

pub fn decode(source: &Source, image: &gltf::Image<'_>) -> Result<Image, Fault> {
    let item = format!("images[{}]", image.index());
    image::decode(image_bytes(source, image)?)
        .map_err(|error| source.fault("ImageDecode", item, error))
}

fn block<const N: usize>(
    image: &Image,
    factor: usize,
    x: usize,
    y: usize,
    texel: impl Fn([u8; 4]) -> [f32; N],
) -> [f32; N] {
    let width = image.width as usize;
    let mut sum = [0.0f32; N];
    for row in y * factor..(y + 1) * factor {
        for column in x * factor..(x + 1) * factor {
            let at = (row * width + column) * 4;
            let held = &image.pixels[at..at + 4];
            let values = texel([held[0], held[1], held[2], held[3]]);
            for (total, value) in sum.iter_mut().zip(values) {
                *total += value;
            }
        }
    }
    let count = (factor * factor) as f32;
    sum.map(|total| total / count)
}

fn filtered(
    kind: MapKind,
    image: &Image,
    size: usize,
    factors: [f32; 2],
) -> Result<Vec<u8>, (usize, usize)> {
    let factor = image.width as usize / size;
    let mut out = Vec::with_capacity(size * size * 4);
    let srgb: Vec<f32> = (0..=255u8).map(srgb_to_linear).collect();
    for y in 0..size {
        for x in 0..size {
            match kind {
                MapKind::BaseColor => {
                    let [red, green, blue, alpha] = block(image, factor, x, y, |[r, g, b, a]| {
                        [
                            srgb[usize::from(r)],
                            srgb[usize::from(g)],
                            srgb[usize::from(b)],
                            f32::from(a) / 255.0,
                        ]
                    });
                    out.extend_from_slice(&[
                        linear_to_srgb(red),
                        linear_to_srgb(green),
                        linear_to_srgb(blue),
                        byte(alpha),
                    ]);
                }
                MapKind::Normal => {
                    let mean = block(image, factor, x, y, |[r, g, b, _]| {
                        [r, g, b].map(|value| f32::from(value) / 255.0 * 2.0 - 1.0)
                    });
                    let length = (mean[0] * mean[0] + mean[1] * mean[1] + mean[2] * mean[2]).sqrt();
                    if length == 0.0 || !length.is_finite() {
                        return Err((x, y));
                    }
                    let [red, green, blue] = mean.map(|value| byte((value / length) * 0.5 + 0.5));
                    out.extend_from_slice(&[red, green, blue, 255]);
                }
                MapKind::Orm => {
                    let [roughness, metallic, alpha] =
                        block(image, factor, x, y, |[_, g, b, a]| {
                            [
                                f32::from(g) / 255.0 * factors[0],
                                f32::from(b) / 255.0 * factors[1],
                                f32::from(a) / 255.0,
                            ]
                        });
                    out.extend_from_slice(&[255, byte(roughness), byte(metallic), byte(alpha)]);
                }
            }
        }
    }
    Ok(out)
}

pub fn texture_of<'a>(
    material: &gltf::Material<'a>,
    kind: MapKind,
) -> Option<(gltf::Texture<'a>, u32)> {
    let pbr = material.pbr_metallic_roughness();
    match kind {
        MapKind::BaseColor => pbr
            .base_color_texture()
            .map(|info| (info.texture(), info.tex_coord())),
        MapKind::Normal => material
            .normal_texture()
            .map(|info| (info.texture(), info.tex_coord())),
        MapKind::Orm => pbr
            .metallic_roughness_texture()
            .map(|info| (info.texture(), info.tex_coord())),
    }
}

pub fn sampling(texture: &gltf::Texture<'_>, tex_coord: u32) -> Sampling {
    let sampler = texture.sampler();
    Sampling {
        wrap_u: sampler.wrap_s(),
        wrap_v: sampler.wrap_t(),
        mag: sampler.mag_filter(),
        min: sampler.min_filter(),
        tex_coord,
    }
}

pub fn key(
    material: &gltf::Material<'_>,
    kind: MapKind,
    size: u32,
) -> Option<(usize, MapKind, u32, [u32; 2])> {
    let (texture, _) = texture_of(material, kind)?;
    let pbr = material.pbr_metallic_roughness();
    let factors = if kind == MapKind::Orm {
        [
            pbr.roughness_factor().to_bits(),
            pbr.metallic_factor().to_bits(),
        ]
    } else {
        [0, 0]
    };
    Some((texture.source()?.index(), kind, size, factors))
}

pub fn map(
    source: &Source,
    material: &gltf::Material<'_>,
    kind: MapKind,
    stem: &str,
    size: u32,
) -> Result<Map, Fault> {
    let item = format!(
        "materials[{}] {}",
        material.index().unwrap_or_default(),
        kind.name()
    );
    let (texture, tex_coord) = texture_of(material, kind).ok_or_else(|| {
        source.fault(
            "MissingMap",
            item.clone(),
            "the asset's material has no such texture (an entry may list `maps` to cook fewer)",
        )
    })?;
    let image = texture
        .source()
        .ok_or_else(|| source.fault("MissingMap", item.clone(), "the texture names no image"))?;
    let decoded = decode(source, &image)?;
    let fits =
        decoded.width == decoded.height && decoded.width >= size && decoded.width % size == 0;
    if !fits {
        return Err(source.fault(
            "ImageSize",
            format!("{item} images[{}]", image.index()),
            format!(
                "{}x{} cannot box-filter to {size}x{size}",
                decoded.width, decoded.height
            ),
        ));
    }
    let pbr = material.pbr_metallic_roughness();
    let pixels = filtered(
        kind,
        &decoded,
        size as usize,
        [pbr.roughness_factor(), pbr.metallic_factor()],
    )
    .map_err(|(x, y)| {
        source.fault(
            "NormalZero",
            item.clone(),
            format!("normals average to zero at {x},{y}"),
        )
    })?;
    Ok(Map {
        kind,
        material: material.name().unwrap_or_default().to_owned(),
        stem: format!("{stem}_{}", kind.name()),
        decoded: Image {
            width: size,
            height: size,
            pixels,
        },
        sampling: sampling(&texture, tex_coord),
    })
}
