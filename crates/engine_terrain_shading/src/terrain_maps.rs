use std::sync::Arc;

use engine_core::{
    AssetLibrary, ColorSpace, Error, ProvidedAssets, Texture, TextureFormat, TextureHandle,
};
use engine_open_world_kit::ground::{SurfaceMaps, TerrainTiles};
use engine_places::surface::{METALLIC_ROUGHNESS_MAP, OCCLUSION_MAP, SurfaceValues};
use engine_render::{Allocation, Install, Lifetime, Prepare, Sampler, wgpu};

use crate::terrain::TerrainGpu;

fn array_space(array: usize) -> ColorSpace {
    if array == 0 {
        ColorSpace::Srgb
    } else {
        ColorSpace::Linear
    }
}

type LayerKey = [Option<TextureHandle>; 3];

const DEFAULTS: [[u8; 4]; 3] = [[255; 4], [128, 128, 255, 255], [255, 255, 255, 128]];

const RUNG_NAMES: [&str; 2] = ["_mid", "_far"];
const TILINGS: [&str; 2] = ["mid_tiling", "far_tiling"];
const SOFT_DEPTH: &str = "soft_depth";
const HEIGHT_DEVIATION: &str = "height_deviation";
const SITE_SHIFT: &str = "site_shift";
const UPRIGHT: &str = "upright";
const STRATA: [&str; 3] = ["strata", "strata_ledge", "strata_thickness"];

const BC7_DEFAULTS: [[u8; 16]; 3] = [
    [
        0x20, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x03, 0, 0, 0, 0, 0, 0, 0,
    ],
    [
        0x20, 0xc0, 0x1f, 0xf0, 0xf7, 0xff, 0xff, 0xff, 0xaf, 0xaa, 0xaa, 0xaa, 0, 0, 0, 0,
    ],
    [
        0x20, 0xff, 0xff, 0xff, 0xff, 0xff, 0x03, 0x02, 0x02, 0, 0, 0, 0, 0, 0, 0,
    ],
];

const ASTC_DEFAULTS: [[u8; 16]; 3] = [
    [
        0xfc, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff,
    ],
    [
        0xfc, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x80, 0x80, 0x80, 0x80, 0xff, 0xff, 0xff,
        0xff,
    ],
    [
        0xfc, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x80,
        0x80,
    ],
];

pub(crate) struct TerrainMaps {
    pub(crate) albedo: engine_render::Texture<TerrainGpu>,
    pub(crate) data: engine_render::Texture<TerrainGpu>,
    pub(crate) sampler: Sampler<TerrainGpu>,
    features: wgpu::Features,
    filled: bool,
    layers: Vec<LayerKey>,
    listed: Vec<LayerKey>,
    listing: Vec<LayerKey>,
    fine: Vec<LayerKey>,
    coarser: Vec<[LayerKey; 2]>,
    provided: ProvidedAssets,
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
    pub tilings: [f32; 2],
    pub soft_depth: f32,
    pub height_deviation: f32,
    pub site_shift: f32,
    pub upright: f32,
    pub strata: [f32; 3],
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
            sampler: cx.sampler(engine_core::Sampler::REPEAT),
            features: cx.features(),
            filled: false,
            layers: Vec::new(),
            listed: Vec::new(),
            listing: Vec::new(),
            fine: Vec::new(),
            coarser: Vec::new(),
            provided: ProvidedAssets::default(),
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

    fn rung_keys(&self, fine: LayerKey, assets: &AssetLibrary) -> [LayerKey; 2] {
        let Some(at) = self.fine.iter().position(|held| *held == fine) else {
            return [fine; 2];
        };
        let mut finer = fine;
        self.coarser[at].map(|named| {
            let held = named.map(|handle| handle.filter(|&handle| assets.texture(handle).is_some()));
            if held != [None; 3] {
                finer = held;
            }
            finer
        })
    }

    pub(crate) fn rungs(&self, maps: &SurfaceMaps, assets: &AssetLibrary) -> [Option<u32>; 2] {
        let fine = key(maps, assets);
        let [mid, far] = self.rung_keys(fine, assets);
        let held = |key: LayerKey| {
            self.layers
                .iter()
                .position(|layer| *layer == key)
                .map(|layer| layer as u32)
        };
        let mid = held(mid).or_else(|| held(fine));
        [mid, held(far).or(mid)]
    }

    pub(crate) fn means(&self, layer: u32) -> Means {
        self.means[layer as usize]
    }

    pub(crate) fn material(maps: &SurfaceMaps, assets: &AssetLibrary) -> Option<MapMaterial> {
        let material = assets.material(maps.material)?;
        let color = material.color();
        let surface = SurfaceValues::read(
            |name| material.surface_value(name),
            |name| material.value(name),
        );
        let packed = material.map(OCCLUSION_MAP).map(|map| map.texture)
            == material.map(METALLIC_ROUGHNESS_MAP).map(|map| map.texture);
        Some(MapMaterial {
            color: [color[0], color[1], color[2]],
            roughness: surface.roughness,
            normal_scale: material.normal_scale(),
            packed,
            occlusion_strength: surface.occlusion_strength,
            metalness: surface.metalness,
            tilings: TILINGS.map(|name| amount(material.value(name))),
            soft_depth: amount(material.value(SOFT_DEPTH)),
            height_deviation: amount(material.value(HEIGHT_DEVIATION)),
            site_shift: amount(material.value(SITE_SHIFT)),
            upright: engine_math::min(amount(material.value(UPRIGHT)), 1.0),
            strata: STRATA.map(|name| amount(material.value(name))),
        })
    }

    pub(crate) fn prepare(
        &mut self,
        cx: &mut Prepare<'_, TerrainGpu>,
        (tiles, assets, provided): (&TerrainTiles, &AssetLibrary, &ProvidedAssets),
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
        if self.listing != self.fine || *provided != self.provided {
            self.coarser = self
                .listing
                .iter()
                .map(|&fine| coarser(fine, provided))
                .collect();
            self.fine.clone_from(&self.listing);
            self.provided.clone_from(provided);
        }
        for at in 0..self.fine.len() {
            for key in self.rung_keys(self.fine[at], assets) {
                if !self.listing.contains(&key) {
                    self.listing.push(key);
                }
            }
        }
        if self.listing == self.listed {
            return;
        }
        self.listed.clone_from(&self.listing);
        let features = self.features;
        let shape = self
            .listing
            .iter()
            .find_map(|&key| shape_of(key, assets, features));
        let kept: Vec<LayerKey> = self
            .listing
            .iter()
            .copied()
            .filter(|&key| shape.is_some() && shape_of(key, assets, features) == shape)
            .collect();
        self.dropped = (self.listing.len() - kept.len()) as u32;
        self.means = kept.iter().map(|&key| means(key, assets)).collect();
        match shape.filter(|_| !kept.is_empty()) {
            Some(shape) => self.fill(cx, (&kept, assets), shape),
            None => self.defaults(cx),
        }
        self.materials = kept.len().max(1) as u32;
        self.size = shape.map_or((1, 1), |shape| shape.size);
        self.layers = kept;
    }

    fn fill(
        &self,
        cx: &mut Prepare<'_, TerrainGpu>,
        (kept, assets): (&[LayerKey], &AssetLibrary),
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
            let format = engine_render::texture_format(shape.formats[array], array_space(array)).0;
            let layers = count * if array == 0 { 1 } else { 2 };
            let size = wgpu::Extent3d {
                width: shape.size.0,
                height: shape.size.1,
                depth_or_array_layers: layers,
            };
            let (Allocation::Fresh | Allocation::Kept) =
                cx.allocate_as(texture, format, size, levels);
            write_array(cx, (texture, array), (kept, assets), shape);
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
            let format = engine_render::texture_format(TextureFormat::Rgba8, array_space(array)).0;
            let size = wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1 + array as u32,
            };
            let (Allocation::Fresh | Allocation::Kept) = cx.allocate_as(texture, format, size, 1);
            for (default, (held, layer)) in DEFAULTS.iter().zip([(0, 0), (1, 0), (1, 1)]) {
                if held == array {
                    cx.write_level(texture, (0, layer), (1, 1), TextureFormat::Rgba8, default);
                }
            }
        }
    }
}

fn write_array(
    cx: &mut Prepare<'_, TerrainGpu>,
    (texture, array): (engine_render::Texture<TerrainGpu>, usize),
    (kept, assets): (&[LayerKey], &AssetLibrary),
    shape: Shape,
) {
    let format = shape.formats[array];
    for (layer, &key) in kept.iter().enumerate() {
        let layer = layer as u32;
        let places = [(0, layer), (1, 2 * layer), (1, 2 * layer + 1)];
        for (default, (source, (held, at))) in textures(key, assets).iter().zip(places).enumerate()
        {
            if held != array {
                continue;
            }
            for level in 0..shape.levels {
                let bytes = match source {
                    Some(source) => source.levels()[level].to_vec(),
                    None => solid_level(format, shape.size, level as u32, default),
                };
                cx.write_level(texture, (level as u32, at), shape.size, format, &bytes);
            }
        }
    }
}

fn shape_of(key: LayerKey, assets: &AssetLibrary, features: wgpu::Features) -> Option<Shape> {
    let textures = textures(key, assets);
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
        let sampled =
            features.contains(engine_render::texture_format(format, array_space(array)).1);
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

fn solid_level(format: TextureFormat, size: (u32, u32), level: u32, default: usize) -> Vec<u8> {
    let (width, height) = ((size.0 >> level).max(1), (size.1 >> level).max(1));
    let block = match format {
        TextureFormat::Bc7 => BC7_DEFAULTS[default],
        TextureFormat::Astc4x4 => ASTC_DEFAULTS[default],
        TextureFormat::Rgba8 | TextureFormat::Etc2Rgba8 => {
            return DEFAULTS[default].repeat((width * height) as usize);
        }
    };
    block.repeat(format.level_bytes(width, height) / 16)
}

fn means(key: LayerKey, assets: &AssetLibrary) -> Means {
    let [base_color, normal, metallic_roughness] =
        textures(key, assets).map(|texture| texture.and_then(|texture| texture.summary()));
    let middle = f32::from(DEFAULTS[2][3]) / 255.0;
    Means {
        color: base_color.map_or([1.0; 3], |summary| summary.color),
        roughness: metallic_roughness.map_or(1.0, |summary| summary.mean[1]),
        occlusion: metallic_roughness.map_or(1.0, |summary| summary.mean[0]),
        slopes: normal.map_or(0.0, |summary| averaged_slopes(summary.mean)),
        height: metallic_roughness.map_or(middle, |summary| summary.mean[3]),
    }
}

fn amount(value: Option<f32>) -> f32 {
    engine_math::max(value.unwrap_or(0.0), 0.0)
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

fn coarser(fine: LayerKey, provided: &ProvidedAssets) -> [LayerKey; 2] {
    let names = fine.map(|handle| {
        let handle = handle?;
        provided
            .texture_names()
            .find(|name| provided.texture(name) == Some(handle))
    });
    RUNG_NAMES.map(|rung| names.map(|name| provided.texture(&format!("{}{rung}", name?))))
}

fn textures(key: LayerKey, assets: &AssetLibrary) -> [Option<&Arc<Texture>>; 3] {
    key.map(|handle| handle.and_then(|handle| assets.texture(handle)))
}
