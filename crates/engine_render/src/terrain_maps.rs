//! Terrain surface materials' maps (decision 054): every terrain's
//! materials with maps share three texture arrays, base color, normal, and
//! occlusion, roughness, metalness and height, one layer per material, all
//! the size of the first material's maps, their mip chains rendered as they
//! upload. The arrays rebuild only when the set of materials with maps
//! changes; materials whose maps differ in size or are block compressed
//! show their colors instead and count as dropped. Each layer's mean color,
//! roughness and occlusion stand in for its maps where they repeat too
//! small on screen to show.

use engine_core::{RenderSurfaceMaps, RenderTerrains, Texture, TextureFormat, TextureHandle};

use crate::gpu_data::{mip_chain_bytes, mip_levels};
use crate::mipmaps::MipGenerator;

/// The textures one layer holds: a material's base color, normal and
/// metallic-roughness maps.
type LayerKey = [Option<TextureHandle>; 3];

/// The arrays' formats: base color decodes sRGB, the others read linear.
const FORMATS: [wgpu::TextureFormat; 3] = [
    wgpu::TextureFormat::Rgba8UnormSrgb,
    wgpu::TextureFormat::Rgba8Unorm,
    wgpu::TextureFormat::Rgba8Unorm,
];

/// What a layer holds where its material lacks a map: white, a flat normal,
/// and full occlusion, roughness and metalness factors at middle height.
const DEFAULTS: [[u8; 4]; 3] = [[255; 4], [128, 128, 255, 255], [255, 255, 255, 128]];

pub(crate) struct TerrainMaps {
    // The layers' textures, and every material with maps the frames
    // listed, including those dropped, which rebuild nothing while they
    // stand.
    layers: Vec<LayerKey>,
    listed: Vec<LayerKey>,
    _arrays: [wgpu::Texture; 3],
    pub(crate) views: [wgpu::TextureView; 3],
    pub(crate) sampler: wgpu::Sampler,
    /// Advances whenever the arrays are replaced.
    pub(crate) generation: u64,
    /// Materials whose maps the arrays cannot hold.
    pub(crate) dropped: u32,
    bytes: u64,
    // Each layer's mean linear color, roughness and occlusion.
    means: Vec<Means>,
}

/// A layer's mean linear base color, roughness and occlusion texels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Means {
    pub color: [f32; 3],
    pub roughness: f32,
    pub occlusion: f32,
}

impl TerrainMaps {
    /// One layer of defaults, until materials with maps arrive.
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let (arrays, views) = arrays(device, (1, 1), 1, 1);
        for (array, texel) in arrays.iter().zip(DEFAULTS) {
            write_layer(queue, array, 0, (1, 1), &texel);
        }
        Self {
            layers: Vec::new(),
            listed: Vec::new(),
            _arrays: arrays,
            views,
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("engine/terrain-material-sampler"),
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                // Ground seen at grazing angles stays sharp along its
                // steep axis.
                anisotropy_clamp: 16,
                ..Default::default()
            }),
            generation: 0,
            dropped: 0,
            bytes: 0,
            means: Vec::new(),
        }
    }

    /// The layer holding `maps`' textures, if the arrays hold them.
    pub(crate) fn layer(&self, maps: &RenderSurfaceMaps) -> Option<u32> {
        let key = key(maps);
        self.layers
            .iter()
            .position(|layer| *layer == key)
            .map(|layer| layer as u32)
    }

    /// Layer `layer`'s means.
    pub(crate) fn means(&self, layer: u32) -> Means {
        self.means[layer as usize]
    }

    /// Bytes of the arrays and their mip chains.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.bytes
    }

    /// Rebuilds the arrays when the materials with maps `terrains` list
    /// change: each layer's maps upload, then their mip levels render
    /// in a submission of their own.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mips: &MipGenerator,
        terrains: &RenderTerrains,
    ) {
        let mut listed: Vec<(LayerKey, &RenderSurfaceMaps)> = Vec::new();
        for maps in terrains
            .terrains
            .iter()
            .flat_map(|terrain| terrain.maps.iter().flatten())
        {
            let key = key(maps);
            if key != [None; 3] && !listed.iter().any(|(listed, _)| *listed == key) {
                listed.push((key, maps));
            }
        }
        if listed.len() == self.listed.len()
            && listed
                .iter()
                .zip(&self.listed)
                .all(|((key, _), known)| key == known)
        {
            return;
        }
        self.listed = listed.iter().map(|(key, _)| *key).collect();
        // The first material's first map sets the size; materials whose
        // maps are compressed or of another size drop.
        let size = listed
            .iter()
            .flat_map(|(_, maps)| textures(maps).into_iter().flatten())
            .next()
            .map(|texture| (texture.width(), texture.height()));
        let kept: Vec<&RenderSurfaceMaps> = listed
            .iter()
            .filter(|(_, maps)| {
                textures(maps).into_iter().flatten().all(|texture| {
                    texture.format() == TextureFormat::Rgba8
                        && Some((texture.width(), texture.height())) == size
                })
            })
            .map(|(_, maps)| *maps)
            .collect();
        self.dropped = (listed.len() - kept.len()) as u32;
        self.layers = kept.iter().map(|maps| key(maps)).collect();
        self.means = kept.iter().map(|maps| means(maps)).collect();
        let size = size.unwrap_or((1, 1));
        let levels = if kept.is_empty() {
            1
        } else {
            mip_levels(size.0, size.1)
        };
        let count = kept.len().max(1) as u32;
        let (arrays, views) = arrays(device, size, count, levels);
        let texels = size.0 as usize * size.1 as usize;
        for (layer, maps) in kept.iter().enumerate() {
            for ((array, texture), texel) in arrays.iter().zip(textures(maps)).zip(DEFAULTS) {
                match texture {
                    Some(texture) => {
                        write_layer(queue, array, layer as u32, size, texture.pixels())
                    }
                    None => write_layer(queue, array, layer as u32, size, &texel.repeat(texels)),
                }
            }
        }
        if kept.is_empty() {
            for (array, texel) in arrays.iter().zip(DEFAULTS) {
                write_layer(queue, array, 0, (1, 1), &texel);
            }
        } else if levels > 1 {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("engine/terrain-material-mips"),
            });
            for array in &arrays {
                for layer in 0..count {
                    mips.record_layer(device, &mut encoder, array, layer);
                }
            }
            queue.submit([encoder.finish()]);
        }
        self.bytes = 3
            * u64::from(count)
            * (texels as u64 * 4
                + if levels > 1 {
                    mip_chain_bytes(size.0, size.1)
                } else {
                    0
                });
        self._arrays = arrays;
        self.views = views;
        self.generation += 1;
    }
}

/// The means of `maps`' base color, decoded from sRGB, and of its
/// metallic-roughness map's roughness and occlusion; 1 without them.
fn means(maps: &RenderSurfaceMaps) -> Means {
    let decode: Vec<f32> = (0..=255_u8)
        .map(|value| {
            let value = f32::from(value) / 255.0;
            if value <= 0.040_45 {
                value / 12.92
            } else {
                ((value + 0.055) / 1.055).powf(2.4)
            }
        })
        .collect();
    let [base_color, _, metallic_roughness] = textures(maps);
    let mut result = Means {
        color: [1.0; 3],
        roughness: 1.0,
        occlusion: 1.0,
    };
    if let Some(texture) = base_color {
        let pixels = texture.pixels().as_chunks::<4>().0;
        let mut sum = [0.0_f64; 3];
        for pixel in pixels {
            for (total, &value) in sum.iter_mut().zip(pixel) {
                *total += f64::from(decode[usize::from(value)]);
            }
        }
        result.color = sum.map(|total| (total / pixels.len().max(1) as f64) as f32);
    }
    if let Some(texture) = metallic_roughness {
        let pixels = texture.pixels().as_chunks::<4>().0;
        let mut sum = [0_u64; 2];
        for pixel in pixels {
            sum[0] += u64::from(pixel[0]);
            sum[1] += u64::from(pixel[1]);
        }
        let mean = |total: u64| total as f32 / (255.0 * pixels.len().max(1) as f32);
        result.occlusion = mean(sum[0]);
        result.roughness = mean(sum[1]);
    }
    result
}

/// The textures `maps` names, as its layer holds them.
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

/// The three arrays of `layers` layers of `size`, with `levels` mip levels,
/// and their views.
fn arrays(
    device: &wgpu::Device,
    size: (u32, u32),
    layers: u32,
    levels: u32,
) -> ([wgpu::Texture; 3], [wgpu::TextureView; 3]) {
    let arrays = FORMATS.map(|format| {
        crate::targets::texture_2d(
            device,
            "engine/terrain-material-maps",
            size,
            layers,
            levels,
            format,
            wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
        )
    });
    let views = arrays.each_ref().map(|array| {
        array.create_view(&wgpu::TextureViewDescriptor {
            label: Some("engine/terrain-material-maps"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        })
    });
    (arrays, views)
}

/// Writes `pixels`, RGBA8 rows of `size`, into the first level of `layer`.
fn write_layer(
    queue: &wgpu::Queue,
    array: &wgpu::Texture,
    layer: u32,
    size: (u32, u32),
    pixels: &[u8],
) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: array,
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: 0,
                y: 0,
                z: layer,
            },
            aspect: wgpu::TextureAspect::All,
        },
        pixels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(size.0 * 4),
            rows_per_image: Some(size.1),
        },
        wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
    );
}
