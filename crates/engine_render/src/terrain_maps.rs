use engine_core::{Texture, TextureFormat, TextureHandle};
use engine_terrain::{RenderSurfaceMaps, RenderTerrains};

use crate::gpu_calls::Queue;
use crate::gpu_data::mip_levels;
use crate::materials::{gpu_format, write_level};
use crate::mipmaps::MipGenerator;

type LayerKey = [Option<TextureHandle>; 3];

const DEFAULTS: [[u8; 4]; 3] = [[255; 4], [128, 128, 255, 255], [255, 255, 255, 128]];

pub(crate) struct TerrainMaps {
    layers: Vec<LayerKey>,
    listed: Vec<LayerKey>,
    listing: Vec<LayerKey>,
    arrays: [wgpu::Texture; 2],
    pub(crate) views: [wgpu::TextureView; 2],
    pub(crate) materials: u32,
    pub(crate) size: (u32, u32),
    pub(crate) sampler: wgpu::Sampler,
    pub(crate) views_generation: u64,
    pub(crate) dropped: u32,
    bytes: u64,
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

#[derive(Clone, Copy, PartialEq, Eq)]
struct Shape {
    size: (u32, u32),
    levels: usize,
    formats: [TextureFormat; 2],
}

impl TerrainMaps {
    pub(crate) fn new(device: &wgpu::Device, queue: &crate::gpu_calls::Queue) -> Self {
        let arrays = defaults(device, queue);
        let views = [view(&arrays[0]), view(&arrays[1])];
        Self {
            layers: Vec::new(),
            listed: Vec::new(),
            listing: Vec::new(),
            arrays,
            views,
            materials: 1,
            size: (1, 1),
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("engine/terrain-material-sampler"),
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                anisotropy_clamp: crate::gpu_layouts::constants::MATERIAL_ANISOTROPY,
                ..Default::default()
            }),
            views_generation: 0,
            dropped: 0,
            bytes: 0,
            means: Vec::new(),
        }
    }

    pub(crate) fn layer(&self, maps: &RenderSurfaceMaps) -> Option<u32> {
        let key = key(maps);
        self.layers
            .iter()
            .position(|layer| *layer == key)
            .map(|layer| layer as u32)
    }

    pub(crate) fn means(&self, layer: u32) -> Means {
        self.means[layer as usize]
    }

    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.bytes
    }

    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &Queue,
        mips: &MipGenerator,
        terrains: &RenderTerrains,
    ) {
        let every = move || {
            terrains
                .terrains
                .iter()
                .flat_map(|terrain| terrain.maps.iter().flatten())
        };
        self.listing.clear();
        for maps in every() {
            let key = key(maps);
            if key != [None; 3] && !self.listing.contains(&key) {
                self.listing.push(key);
            }
        }
        if self.listing == self.listed {
            return;
        }
        let listed: Vec<(LayerKey, &RenderSurfaceMaps)> = self
            .listing
            .iter()
            .filter_map(|&wanted| Some((wanted, every().find(|maps| key(maps) == wanted)?)))
            .collect();
        self.listed.clone_from(&self.listing);
        let shape = listed.iter().find_map(|(_, maps)| shape_of(maps, device));
        let kept: Vec<&RenderSurfaceMaps> = listed
            .iter()
            .filter(|(_, maps)| shape.is_some() && shape_of(maps, device) == shape)
            .map(|(_, maps)| *maps)
            .collect();
        self.dropped = (listed.len() - kept.len()) as u32;
        self.layers = kept.iter().map(|maps| key(maps)).collect();
        self.means = kept.iter().map(|maps| means(maps)).collect();
        let (arrays, bytes) = match shape.filter(|_| !kept.is_empty()) {
            Some(shape) => (
                fill(device, queue, mips, &kept, shape),
                array_bytes(shape, kept.len() as u32),
            ),
            None => (defaults(device, queue), 0),
        };
        let count = kept.len().max(1) as u32;
        self.views = [view(&arrays[0]), view(&arrays[1])];
        self.arrays = arrays;
        self.materials = count;
        self.size = shape.map_or((1, 1), |shape| shape.size);
        self.bytes = bytes;
        self.views_generation += 1;
    }
}

fn shape_of(maps: &RenderSurfaceMaps, device: &wgpu::Device) -> Option<Shape> {
    let textures = textures(maps);
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
        let sampled = device.features().contains(gpu_format(format, array == 0).1);
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

fn fill(
    device: &wgpu::Device,
    queue: &Queue,
    mips: &MipGenerator,
    kept: &[&RenderSurfaceMaps],
    shape: Shape,
) -> [wgpu::Texture; 2] {
    let count = kept.len() as u32;
    let rendered = shape.levels == 1;
    let levels = if rendered {
        mip_levels(shape.size.0, shape.size.1)
    } else {
        shape.levels as u32
    };
    let arrays = [0, 1].map(|array| {
        let format = gpu_format(shape.formats[array], array == 0).0;
        let layers = count * if array == 0 { 1 } else { 2 };
        array_texture(device, shape.size, layers, levels, format, rendered)
    });
    for (layer, maps) in kept.iter().enumerate() {
        let layer = layer as u32;
        let places = [(0, layer), (1, 2 * layer), (1, 2 * layer + 1)];
        for ((texture, (array, at)), default) in textures(maps).iter().zip(places).zip(DEFAULTS) {
            let format = shape.formats[array];
            for level in 0..shape.levels {
                let bytes = match texture {
                    Some(texture) => texture.levels()[level].to_vec(),
                    None => solid_level(format, shape.size, level as u32, default),
                };
                let place = (level as u32, at);
                write_level(queue, &arrays[array], place, shape.size, format, &bytes);
            }
        }
    }
    if rendered && levels > 1 {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("engine/terrain-material-mips"),
        });
        for (array, layers) in arrays.iter().zip([count, 2 * count]) {
            for layer in 0..layers {
                mips.record_layer(device, &mut encoder, array, layer);
            }
        }
        queue.submit_now([encoder.finish()]);
    }
    arrays
}

fn solid_level(format: TextureFormat, size: (u32, u32), level: u32, color: [u8; 4]) -> Vec<u8> {
    let (width, height) = ((size.0 >> level).max(1), (size.1 >> level).max(1));
    match engine_textures::solid_block(format, color) {
        Some(block) => block.repeat(format.level_bytes(width, height) / 16),
        None => color.repeat((width * height) as usize),
    }
}

fn array_bytes(shape: Shape, materials: u32) -> u64 {
    let rendered = shape.levels == 1;
    let levels = if rendered {
        mip_levels(shape.size.0, shape.size.1)
    } else {
        shape.levels as u32
    };
    let layer = |format: TextureFormat| -> u64 {
        (0..levels)
            .map(|level| {
                let width = (shape.size.0 >> level).max(1);
                let height = (shape.size.1 >> level).max(1);
                format.level_bytes(width, height) as u64
            })
            .sum()
    };
    u64::from(materials) * (layer(shape.formats[0]) + 2 * layer(shape.formats[1]))
}

fn means(maps: &RenderSurfaceMaps) -> Means {
    let [base_color, normal, metallic_roughness] =
        textures(maps).map(|texture| texture.and_then(Texture::summary));
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
    let kept = engine_core::math::clamp(length, 1.0e-4, 1.0);
    2.0 * (1.0 - kept) / kept
}

fn key(maps: &RenderSurfaceMaps) -> LayerKey {
    textures_of(maps).map(|texture| texture.map(|texture| texture.handle))
}

fn textures_of(maps: &RenderSurfaceMaps) -> [Option<&engine_core::RenderTexture>; 3] {
    [
        maps.base_color.as_ref(),
        maps.normal.as_ref(),
        maps.metallic_roughness.as_ref(),
    ]
}

fn textures(maps: &RenderSurfaceMaps) -> [Option<&Texture>; 3] {
    textures_of(maps).map(|texture| texture.map(|texture| &*texture.texture))
}

fn defaults(device: &wgpu::Device, queue: &crate::gpu_calls::Queue) -> [wgpu::Texture; 2] {
    let arrays = [0, 1].map(|array| {
        let format = gpu_format(TextureFormat::Rgba8, array == 0).0;
        array_texture(device, (1, 1), 1 + array, 1, format, false)
    });
    for (default, (array, layer)) in DEFAULTS.iter().zip([(0, 0), (1, 0), (1, 1)]) {
        write_level(
            queue,
            &arrays[array],
            (0, layer),
            (1, 1),
            TextureFormat::Rgba8,
            default,
        );
    }
    arrays
}

fn array_texture(
    device: &wgpu::Device,
    size: (u32, u32),
    layers: u32,
    levels: u32,
    format: wgpu::TextureFormat,
    rendered: bool,
) -> wgpu::Texture {
    let mut usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
    if rendered {
        usage |= wgpu::TextureUsages::RENDER_ATTACHMENT;
    }
    crate::targets::texture_2d(
        device,
        "engine/terrain-material-maps",
        size,
        layers,
        levels,
        format,
        usage,
    )
}

fn view(array: &wgpu::Texture) -> wgpu::TextureView {
    array.create_view(&wgpu::TextureViewDescriptor {
        label: Some("engine/terrain-material-maps"),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    })
}
