use std::sync::Arc;

use engine_core::{AssetLibrary, Error, Texture, TextureFormat, TextureHandle};
use engine_places::ground::{SurfaceMaps, TerrainTiles};
use engine_render::{Install, Lifetime, Prepare, Sampler, SamplerDesc, wgpu};

use crate::terrain::TerrainGpu;

type LayerKey = [Option<TextureHandle>; 3];

const DEFAULTS: [[u8; 4]; 3] = [[255; 4], [128, 128, 255, 255], [255, 255, 255, 128]];

pub(crate) struct TerrainMaps {
    pub(crate) albedo: engine_render::Texture<TerrainGpu>,
    pub(crate) data: engine_render::Texture<TerrainGpu>,
    pub(crate) sampler: Sampler<TerrainGpu>,
    features: wgpu::Features,
    filled: bool,
    layers: Vec<LayerKey>,
    listed: Vec<LayerKey>,
    listing: Vec<LayerKey>,
    pub(crate) materials: u32,
    pub(crate) size: (u32, u32),
    pub(crate) dropped: u32,
    means: Vec<Means>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Means {
    pub color: [f32; 3],
    pub roughness: f32,
    pub occlusion: f32,
    pub slopes: f32,
    pub height: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MapMaterial {
    pub color: [f32; 3],
    pub roughness: f32,
    pub normal_scale: f32,
    pub packed: bool,
    pub occlusion_strength: f32,
    pub metalness: f32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Shape {
    size: (u32, u32),
    levels: usize,
    formats: [TextureFormat; 2],
}

impl TerrainMaps {
    pub(crate) fn install(cx: &mut Install<'_, TerrainGpu>) -> Result<Self, Error> {
        Ok(Self {
            albedo: cx.texture(
                "material_albedo",
                wgpu::TextureFormat::Rgba8UnormSrgb,
                Lifetime::Kept,
            ),
            data: cx.texture(
                "material_data",
                wgpu::TextureFormat::Rgba8Unorm,
                Lifetime::Kept,
            ),
            sampler: cx.sampler(SamplerDesc {
                filter: wgpu::FilterMode::Linear,
                mipmap: wgpu::MipmapFilterMode::Linear,
                address: [
                    wgpu::AddressMode::Repeat,
                    wgpu::AddressMode::Repeat,
                    wgpu::AddressMode::ClampToEdge,
                ],
                compare: None,
                anisotropy: engine_render::MATERIAL_ANISOTROPY,
            }),
            features: cx.features(),
            filled: false,
            layers: Vec::new(),
            listed: Vec::new(),
            listing: Vec::new(),
            materials: 1,
            size: (1, 1),
            dropped: 0,
            means: Vec::new(),
        })
    }

    pub(crate) fn layer(&self, maps: &SurfaceMaps, assets: &AssetLibrary) -> Option<u32> {
        let key = key(maps, assets);
        self.layers
            .iter()
            .position(|layer| *layer == key)
            .map(|layer| layer as u32)
    }

    pub(crate) fn means(&self, layer: u32) -> Means {
        self.means[layer as usize]
    }

    pub(crate) fn material(maps: &SurfaceMaps, assets: &AssetLibrary) -> Option<MapMaterial> {
        let material = assets.material(maps.material)?;
        let color = material.color();
        let surface = material.surface();
        let packed = material.occlusion_texture().map(|map| map.texture)
            == material.metallic_roughness_texture().map(|map| map.texture);
        Some(MapMaterial {
            color: [color[0], color[1], color[2]],
            roughness: surface.roughness,
            normal_scale: material.normal_scale(),
            packed,
            occlusion_strength: material.occlusion_strength(),
            metalness: surface.metalness,
        })
    }

    pub(crate) fn prepare(
        &mut self,
        cx: &mut Prepare<'_, TerrainGpu>,
        (tiles, assets): (&TerrainTiles, &AssetLibrary),
    ) {
        if !self.filled {
            self.defaults(cx);
            self.filled = true;
        }
        let every = || {
            tiles
                .terrains
                .iter()
                .flat_map(|terrain| terrain.maps.iter().flatten())
        };
        self.listing.clear();
        for maps in every() {
            let key = key(maps, assets);
            if key != [None; 3] && !self.listing.contains(&key) {
                self.listing.push(key);
            }
        }
        if self.listing == self.listed {
            return;
        }
        let listed: Vec<(LayerKey, &SurfaceMaps)> = self
            .listing
            .iter()
            .filter_map(|&wanted| Some((wanted, every().find(|maps| key(maps, assets) == wanted)?)))
            .collect();
        self.listed.clone_from(&self.listing);
        let features = self.features;
        let shape = listed
            .iter()
            .find_map(|(_, maps)| shape_of(maps, assets, features));
        let kept: Vec<&SurfaceMaps> = listed
            .iter()
            .filter(|(_, maps)| shape.is_some() && shape_of(maps, assets, features) == shape)
            .map(|(_, maps)| *maps)
            .collect();
        self.dropped = (listed.len() - kept.len()) as u32;
        self.layers = kept.iter().map(|maps| key(maps, assets)).collect();
        self.means = kept.iter().map(|maps| means(maps, assets)).collect();
        match shape.filter(|_| !kept.is_empty()) {
            Some(shape) => self.fill(cx, (&kept, assets), shape),
            None => self.defaults(cx),
        }
        self.materials = kept.len().max(1) as u32;
        self.size = shape.map_or((1, 1), |shape| shape.size);
    }

    fn fill(
        &self,
        cx: &mut Prepare<'_, TerrainGpu>,
        (kept, assets): (&[&SurfaceMaps], &AssetLibrary),
        shape: Shape,
    ) {
        let count = kept.len() as u32;
        let rendered = shape.levels == 1;
        let levels = if rendered {
            engine_render::mip_levels(shape.size.0, shape.size.1)
        } else {
            shape.levels as u32
        };
        let arrays = [self.albedo, self.data];
        for (array, texture) in arrays.into_iter().enumerate() {
            let format = engine_render::texture_format(shape.formats[array], array == 0).0;
            let layers = count * if array == 0 { 1 } else { 2 };
            cx.allocate_as(
                texture,
                format,
                wgpu::Extent3d {
                    width: shape.size.0,
                    height: shape.size.1,
                    depth_or_array_layers: layers,
                },
                levels,
            );
        }
        for (layer, maps) in kept.iter().enumerate() {
            let layer = layer as u32;
            let places = [(0, layer), (1, 2 * layer), (1, 2 * layer + 1)];
            for ((texture, (array, at)), default) in
                textures(maps, assets).iter().zip(places).zip(DEFAULTS)
            {
                let format = shape.formats[array];
                for level in 0..shape.levels {
                    let bytes = match texture {
                        Some(texture) => texture.levels()[level].to_vec(),
                        None => solid_level(format, shape.size, level as u32, default),
                    };
                    cx.write_level(
                        arrays[array],
                        (level as u32, at),
                        shape.size,
                        format,
                        &bytes,
                    );
                }
            }
        }
        if rendered && levels > 1 {
            for (array, layers) in arrays.into_iter().zip([count, 2 * count]) {
                cx.mipmaps(array, 0..layers);
            }
        }
    }

    fn defaults(&self, cx: &mut Prepare<'_, TerrainGpu>) {
        let arrays = [self.albedo, self.data];
        for (array, texture) in arrays.into_iter().enumerate() {
            let format = engine_render::texture_format(TextureFormat::Rgba8, array == 0).0;
            cx.allocate_as(
                texture,
                format,
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1 + array as u32,
                },
                1,
            );
        }
        for (default, (array, layer)) in DEFAULTS.iter().zip([(0, 0), (1, 0), (1, 1)]) {
            cx.write_level(
                arrays[array],
                (0, layer),
                (1, 1),
                TextureFormat::Rgba8,
                default,
            );
        }
    }
}

fn shape_of(maps: &SurfaceMaps, assets: &AssetLibrary, features: wgpu::Features) -> Option<Shape> {
    let textures = textures(maps, assets);
    let first = textures.iter().flatten().next()?;
    let size = (first.width(), first.height());
    let levels = first.levels().len();
    let mut formats = [None; 2];
    for (index, texture) in textures.iter().enumerate() {
        let Some(texture) = texture else {
            continue;
        };
        let array = usize::from(index > 0);
        let format = texture.format();
        let sampled = features.contains(engine_render::texture_format(format, array == 0).1);
        let chained = texture.has_mips() || (levels == 1 && !format.is_compressed());
        if (texture.width(), texture.height()) != size
            || texture.levels().len() != levels
            || !chained
            || !sampled
            || texture.summary().is_none()
            || formats[array].is_some_and(|known| known != format)
        {
            return None;
        }
        formats[array] = Some(format);
    }
    let fallback = formats.iter().flatten().next().copied()?;
    Some(Shape {
        size,
        levels,
        formats: formats.map(|format| format.unwrap_or(fallback)),
    })
}

fn solid_level(format: TextureFormat, size: (u32, u32), level: u32, color: [u8; 4]) -> Vec<u8> {
    let (width, height) = ((size.0 >> level).max(1), (size.1 >> level).max(1));
    match engine_core::solid_block(format, color) {
        Some(block) => block.repeat(format.level_bytes(width, height) / 16),
        None => color.repeat((width * height) as usize),
    }
}

fn means(maps: &SurfaceMaps, assets: &AssetLibrary) -> Means {
    let [base_color, normal, metallic_roughness] =
        textures(maps, assets).map(|texture| texture.and_then(|texture| texture.summary()));
    let middle = f32::from(DEFAULTS[2][3]) / 255.0;
    Means {
        color: base_color.map_or([1.0; 3], |summary| summary.color),
        roughness: metallic_roughness.map_or(1.0, |summary| summary.mean[1]),
        occlusion: metallic_roughness.map_or(1.0, |summary| summary.mean[0]),
        slopes: normal.map_or(0.0, |summary| averaged_slopes(summary.mean)),
        height: metallic_roughness.map_or(middle, |summary| summary.mean[3]),
    }
}

fn averaged_slopes(channels: [f32; 4]) -> f32 {
    let mean = [0, 1, 2].map(|channel| 2.0 * channels[channel] - 1.0);
    let length = (mean[0] * mean[0] + mean[1] * mean[1] + mean[2] * mean[2]).sqrt();
    let kept = engine_math::clamp(length, 1.0e-4, 1.0);
    2.0 * (1.0 - kept) / kept
}

fn key(maps: &SurfaceMaps, assets: &AssetLibrary) -> LayerKey {
    [maps.base_color, maps.normal, maps.metallic_roughness]
        .map(|handle| handle.filter(|&handle| assets.texture(handle).is_some()))
}

fn textures<'a>(maps: &SurfaceMaps, assets: &'a AssetLibrary) -> [Option<&'a Arc<Texture>>; 3] {
    [maps.base_color, maps.normal, maps.metallic_roughness]
        .map(|handle| handle.and_then(|handle| assets.texture(handle)))
}
