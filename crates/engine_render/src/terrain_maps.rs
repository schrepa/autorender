//! Terrain surface materials' maps: every terrain's materials with maps
//! share two texture arrays, base color, and data, which holds each
//! material's normal map then its occlusion, roughness, metalness and
//! height map, two layers a material. Every layer has the first material's
//! size, levels and format, base color's read as sRGB and data's linear:
//! cooked maps upload every level they carry, block-compressed where the
//! device samples their family, and RGBA8 maps holding their base level
//! alone have their levels rendered as they upload. The arrays rebuild
//! only when the materials with maps or their textures change; materials
//! whose maps differ in size, levels or format, or carry no summary, show
//! their colors instead and count as dropped. Each layer's summaries
//! ([`engine_core::Texture::summary`]) stand in for its maps where they
//! repeat too small on screen to show: the variance of its normal map's
//! slopes roughens it there, and its height map's mean and deviation share
//! the ground as height blending does on average. The runtime virtual texture
//! (`terrain_vt.rs`) keeps its sheets in a renderable array of their own,
//! three layers a sheet, as many as it asks for (up to the device's
//! layers), which grows by copying itself; it has sheets only while the
//! arrays hold materials.

use engine_core::{RenderSurfaceMaps, RenderTerrains, Texture, TextureFormat, TextureHandle};

use crate::gpu_calls::Queue;
use crate::gpu_data::mip_levels;
use crate::materials::{gpu_format, write_level};
use crate::mipmaps::MipGenerator;

/// The textures one material's layers hold, by handle: its base color,
/// normal and metallic-roughness maps.
type LayerKey = [Option<TextureHandle>; 3];

/// Texels along a side of the virtual texture's sheets.
pub(crate) const SHEET_TEXELS: u32 = 1024;
pub(crate) use crate::gpu_layouts::constants::VT_SHEET_LAYERS as SHEET_LAYERS;

/// What a layer holds where its material lacks a map: white, a flat
/// normal, and full occlusion, roughness and metalness factors at middle
/// height.
const DEFAULTS: [[u8; 4]; 3] = [[255; 4], [128, 128, 255, 255], [255, 255, 255, 128]];

pub(crate) struct TerrainMaps {
    // The layers' textures, and every material with maps the frames
    // listed, including those dropped, which rebuild nothing while they
    // stand; and those the frame lists, made in place of the last frame's.
    layers: Vec<LayerKey>,
    listed: Vec<LayerKey>,
    listing: Vec<LayerKey>,
    arrays: [wgpu::Texture; 2],
    sheet_array: wgpu::Texture,
    // What the bakes bind for the sheets they draw into.
    stand_in: wgpu::Texture,
    /// The views the terrain group binds: base color, data, and the
    /// virtual texture's sheets.
    pub(crate) views: [wgpu::TextureView; 3],
    /// The views its bakes bind: the materials' arrays, and a stand-in for
    /// the sheets they draw into.
    pub(crate) bake_views: [wgpu::TextureView; 3],
    /// Layers of materials, the virtual texture's sheets and the pages it
    /// asked them for, and the material arrays' size.
    pub(crate) materials: u32,
    pub(crate) sheets: u32,
    pages: u32,
    pub(crate) size: (u32, u32),
    pub(crate) sampler: wgpu::Sampler,
    /// Advances whenever the arrays' materials change, which empties the
    /// sheets.
    pub(crate) generation: u64,
    /// Advances whenever an array is replaced, also as the sheets grow.
    pub(crate) views_generation: u64,
    /// Materials whose maps the arrays cannot hold.
    pub(crate) dropped: u32,
    bytes: u64,
    // Each layer's means.
    means: Vec<Means>,
}

/// A layer's mean linear base color, roughness and occlusion texels, the
/// mean square of its normal map's slopes at scale 1, and its height map's
/// mean and deviation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Means {
    pub color: [f32; 3],
    pub roughness: f32,
    pub occlusion: f32,
    pub slopes: f32,
    pub height: f32,
    pub height_deviation: f32,
}

/// What every layer of the arrays shares: the size, levels and format of
/// the first material's maps, base color's and the data maps'.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Shape {
    size: (u32, u32),
    levels: usize,
    formats: [TextureFormat; 2],
}

impl TerrainMaps {
    /// One layer of defaults, until materials with maps arrive.
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let arrays = defaults(device, queue);
        let sheet_array = sheets(device, 0);
        let stand_in = sheets(device, 0);
        let views = [view(&arrays[0]), view(&arrays[1]), view(&sheet_array)];
        let bake_views = [view(&arrays[0]), view(&arrays[1]), view(&stand_in)];
        Self {
            layers: Vec::new(),
            listed: Vec::new(),
            listing: Vec::new(),
            arrays,
            sheet_array,
            stand_in,
            views,
            bake_views,
            materials: 1,
            sheets: 0,
            pages: 0,
            size: (1, 1),
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("engine/terrain-material-sampler"),
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                // Ground seen at grazing angles stays sharp along its
                // steep axis.
                anisotropy_clamp: crate::gpu_layouts::constants::MATERIAL_ANISOTROPY,
                ..Default::default()
            }),
            generation: 0,
            views_generation: 0,
            dropped: 0,
            bytes: 0,
            means: Vec::new(),
        }
    }

    /// Pages a sheet holds: its slots along a side squared, or none while
    /// the arrays hold no materials.
    pub(crate) fn per_sheet(&self) -> u32 {
        if self.layers.is_empty() {
            0
        } else {
            (SHEET_TEXELS / crate::terrain_vt::SLOT_TEXELS).pow(2)
        }
    }

    /// The most sheets the device's texture array layers leave room for.
    pub(crate) fn sheet_room(&self, device: &wgpu::Device) -> u32 {
        device.limits().max_texture_array_layers / SHEET_LAYERS
    }

    /// Grows the sheets to `sheets`, copying every layer the array holds;
    /// the pages baked into them stay.
    fn grow(&mut self, device: &wgpu::Device, queue: &Queue, sheets_now: u32) {
        let grown = sheets(device, sheets_now);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("engine/terrain-vt-grow"),
        });
        if self.sheets > 0 {
            encoder.copy_texture_to_texture(
                self.sheet_array.as_image_copy(),
                grown.as_image_copy(),
                wgpu::Extent3d {
                    width: SHEET_TEXELS,
                    height: SHEET_TEXELS,
                    depth_or_array_layers: self.sheets * SHEET_LAYERS,
                },
            );
        }
        queue.submit_now([encoder.finish()]);
        self.views[2] = view(&grown);
        self.sheet_array = grown;
        self.bytes += u64::from(sheets_now - self.sheets) * sheet_bytes();
        self.sheets = sheets_now;
        self.views_generation += 1;
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

    /// Bytes of the arrays with their levels, and of the sheets.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.bytes
    }

    /// The virtual texture's sheets, three layers a sheet, which its far
    /// pages upload into and are read back from.
    pub(crate) fn sheet_texture(&self) -> &wgpu::Texture {
        &self.sheet_array
    }

    /// The views a bake of the virtual texture's sheet `sheet` draws into:
    /// its base color, canopy and normal turn layers.
    pub(crate) fn sheet_views(&self, sheet: u32) -> [wgpu::TextureView; 3] {
        std::array::from_fn(|index| {
            self.sheet_array.create_view(&wgpu::TextureViewDescriptor {
                label: Some("engine/terrain-vt-sheet"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_mip_level: 0,
                mip_level_count: Some(1),
                base_array_layer: sheet * SHEET_LAYERS + index as u32,
                array_layer_count: Some(1),
                ..Default::default()
            })
        })
    }

    /// Rebuilds the arrays when the materials with maps `terrains` list
    /// change, or the virtual texture's `pages` do (0 without it), with the
    /// sheets holding them: each layer's levels upload, and maps holding
    /// their base level alone have their levels rendered in a submission of
    /// their own.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &Queue,
        mips: &MipGenerator,
        terrains: &RenderTerrains,
        pages: u32,
    ) {
        // The engine materials of every terrain's surface materials.
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
        // The same materials: the sheets grow to the pages asked for, and
        // drop only with them all.
        if self.listing == self.listed && (pages > 0 || self.sheets == 0) {
            let per_sheet = self.per_sheet();
            if pages > self.pages && per_sheet > 0 {
                let sheets = pages.div_ceil(per_sheet).min(self.sheet_room(device));
                if sheets > self.sheets {
                    self.grow(device, queue, sheets);
                }
            }
            self.pages = pages;
            return;
        }
        // Each material listed, with the maps it was first listed with.
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
        // Sheets drop beyond the device's layers.
        self.sheets = 0;
        let per_sheet = self.per_sheet();
        let sheets_now = if per_sheet == 0 {
            0
        } else {
            pages.div_ceil(per_sheet).min(self.sheet_room(device))
        };
        self.sheet_array = sheets(device, sheets_now);
        self.views = [view(&arrays[0]), view(&arrays[1]), view(&self.sheet_array)];
        self.bake_views = [view(&arrays[0]), view(&arrays[1]), view(&self.stand_in)];
        self.arrays = arrays;
        self.materials = count;
        self.sheets = sheets_now;
        self.pages = pages;
        self.size = shape.map_or((1, 1), |shape| shape.size);
        self.bytes = bytes + u64::from(sheets_now) * sheet_bytes();
        self.generation += 1;
        self.views_generation += 1;
    }
}

/// The shape `maps`' textures share, when the device samples them: one
/// size, one level count (a full chain, or RGBA8 base levels alone), and
/// one format for the data maps; none when any texture differs or lacks a
/// summary.
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
    // A material without data maps or without a base color takes the
    // other's format for its defaults.
    let fallback = formats.iter().flatten().next().copied()?;
    Some(Shape {
        size,
        levels,
        formats: formats.map(|format| format.unwrap_or(fallback)),
    })
}

/// The two arrays holding `kept`'s maps in `shape`, every level written,
/// and levels rendered where they hold their base alone.
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

/// Level `level` of a texture of `size` in `format`, every texel `color`.
fn solid_level(format: TextureFormat, size: (u32, u32), level: u32, color: [u8; 4]) -> Vec<u8> {
    let (width, height) = ((size.0 >> level).max(1), (size.1 >> level).max(1));
    match engine_textures::solid_block(format, color) {
        Some(block) => block.repeat(format.level_bytes(width, height) / 16),
        None => color.repeat((width * height) as usize),
    }
}

/// Bytes of the two arrays of `materials` in `shape`, every level.
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

/// Bytes of a sheet's layers.
fn sheet_bytes() -> u64 {
    u64::from(SHEET_LAYERS) * u64::from(SHEET_TEXELS) * u64::from(SHEET_TEXELS) * 4
}

/// The means of `maps`' base color, decoded from sRGB, and of its
/// metallic-roughness map's roughness and occlusion, 1 without them; of
/// the squares of its normal map's slopes, 0 without one; and the mean and
/// deviation of the height its metallic-roughness map's alpha holds, the
/// default layer's middle height without one: each map's summary.
fn means(maps: &RenderSurfaceMaps) -> Means {
    let [base_color, normal, metallic_roughness] =
        textures(maps).map(|texture| texture.and_then(Texture::summary));
    let middle = f32::from(DEFAULTS[2][3]) / 255.0;
    Means {
        color: base_color.map_or([1.0; 3], |summary| summary.color),
        roughness: metallic_roughness.map_or(1.0, |summary| summary.mean[1]),
        occlusion: metallic_roughness.map_or(1.0, |summary| summary.mean[0]),
        slopes: normal.map_or(0.0, |summary| summary.slopes),
        height: metallic_roughness.map_or(middle, |summary| summary.mean[3]),
        height_deviation: metallic_roughness.map_or(0.0, |summary| summary.deviation[3]),
    }
}

/// The textures `maps` names, as its layers hold them.
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

/// The arrays of one layer of defaults (two in the data array), 1 × 1.
fn defaults(device: &wgpu::Device, queue: &wgpu::Queue) -> [wgpu::Texture; 2] {
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

/// An array of `layers` of `size` with `levels` mip levels in `format`;
/// `rendered` ones render their levels.
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

/// The virtual texture's sheets: `count` of them, rendered into and
/// copied as they grow; without any, a texel of each layer, which the
/// terrain group binds while no sheet is drawn.
fn sheets(device: &wgpu::Device, count: u32) -> wgpu::Texture {
    let side = if count == 0 { 1 } else { SHEET_TEXELS };
    crate::targets::texture_2d(
        device,
        "engine/terrain-vt-sheets",
        (side, side),
        count.max(1) * SHEET_LAYERS,
        1,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
    )
}

/// A view of every layer of `array`.
fn view(array: &wgpu::Texture) -> wgpu::TextureView {
    array.create_view(&wgpu::TextureViewDescriptor {
        label: Some("engine/terrain-material-maps"),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    })
}
