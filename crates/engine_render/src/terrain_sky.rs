//! Terrain sky visibility: each terrain's sky map, cooked with its
//! lighting ([`engine_core::TerrainLighting`]): a grid over the whole
//! terrain holding at every texel the share of the sky its ground sees
//! about its normal, its large-scale variety (hollows and ridges) and the
//! turn of its normal toward the middle of the sky it sees. Terrain scales
//! its ambient light by the share, taking the hidden sky's share back as its
//! own light, and takes the sky's light around the turned normal.
//!
//! A terrain's two layers upload as its lighting arrives, and again when it
//! changes; until then its ground sees the whole sky.

use std::sync::Arc;

use engine_core::terrain_data::lighting::SKY_LAYERS;
use engine_core::{RenderTerrains, TerrainLighting, TerrainSettings};

/// A sky map texel: the share of the sky seen, hollows and ridges; then the
/// normal's turn.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

struct Maps {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    size: u32,
    terrains: usize,
}

pub(crate) struct SkyMaps {
    maps: Option<Maps>,
    // A one-texel map of open sky bound while no terrain has one.
    empty: wgpu::TextureView,
    // The lighting each terrain's layers hold, and the uploads each has
    // taken, which pages of the runtime virtual texture baked from it
    // follow.
    held: Vec<Option<Arc<TerrainLighting>>>,
    uploads: Vec<u64>,
    // Advances when the bound view is replaced.
    generation: u64,
    /// Terrains whose maps uploaded this frame.
    pub(crate) uploaded: u32,
}

impl SkyMaps {
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let empty = crate::targets::texture_2d(
            device,
            "engine/terrain-sky-empty",
            (1, 1),
            1,
            1,
            FORMAT,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        queue.write_texture(
            empty.as_image_copy(),
            &[255; 4],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        Self {
            maps: None,
            empty: array_view(&empty),
            held: Vec::new(),
            uploads: Vec::new(),
            generation: 0,
            uploaded: 0,
        }
    }

    /// Advances when [`Self::view`] is another view.
    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    /// The maps, two layers per terrain, or a texel of open sky.
    pub(crate) fn view(&self) -> &wgpu::TextureView {
        self.maps.as_ref().map_or(&self.empty, |maps| &maps.view)
    }

    /// The layer of terrain `index`'s share, hollows and ridges plus one
    /// once uploaded, which is the layer of its turn; 0 before.
    pub(crate) fn layer(&self, index: usize) -> u32 {
        match self.held.get(index) {
            Some(Some(_)) => (SKY_LAYERS * index) as u32 + 1,
            _ => 0,
        }
    }

    /// The uploads terrain `index`'s map took, which pages baked from it
    /// follow; `None` until its first; 0 without maps, which leave open
    /// sky everywhere.
    pub(crate) fn settled(&self, index: usize) -> Option<u64> {
        if self.maps.is_none() {
            return Some(0);
        }
        match self.held.get(index) {
            Some(Some(_)) => self.uploads.get(index).copied(),
            _ => None,
        }
    }

    /// Terrains whose maps are here.
    pub(crate) fn count(&self) -> u32 {
        self.held.iter().flatten().count() as u32
    }

    /// Bytes of the maps.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.maps.as_ref().map_or(0, |maps| {
            u64::from(maps.size) * u64::from(maps.size) * (SKY_LAYERS * maps.terrains) as u64 * 4
        })
    }

    /// Uploads the maps of every terrain whose lighting arrived or changed
    /// since, with terrain `settings` asking for sky visibility at their
    /// resolution; without it the maps are released.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        terrains: &RenderTerrains,
        settings: &TerrainSettings,
    ) {
        self.uploaded = 0;
        if !(settings.enabled && settings.sky_visibility) || terrains.terrains.is_empty() {
            if self.maps.take().is_some() {
                self.generation += 1;
            }
            self.held.clear();
            return;
        }
        let size = settings.sky_resolution;
        let count = terrains.terrains.len();
        if self
            .maps
            .as_ref()
            .is_none_or(|maps| maps.size != size || maps.terrains != count)
        {
            self.maps = Some(maps(device, size, count));
            self.held = vec![None; count];
            self.uploads.resize(count, 0);
            self.generation += 1;
        }
        let Some(maps) = &self.maps else {
            return;
        };
        for (index, terrain) in terrains.terrains.iter().enumerate() {
            let lighting = terrain
                .lighting
                .as_ref()
                .filter(|lighting| lighting.sky_size == size);
            let held = &mut self.held[index];
            match (lighting, &*held) {
                (Some(lighting), Some(current)) if Arc::ptr_eq(lighting, current) => {}
                (Some(lighting), _) => {
                    write_layers(queue, &maps.texture, index, size, &lighting.sky_layers());
                    *held = Some(lighting.clone());
                    self.uploads[index] += 1;
                    self.uploaded += 1;
                }
                (None, _) => *held = None,
            }
        }
    }
}

/// Writes terrain `index`'s layers of `texture`, `size` texels a side, from
/// `texels`.
fn write_layers(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    index: usize,
    size: u32,
    texels: &[u8],
) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: 0,
                y: 0,
                z: (SKY_LAYERS * index) as u32,
            },
            aspect: wgpu::TextureAspect::All,
        },
        texels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(size * 4),
            rows_per_image: Some(size),
        },
        wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: SKY_LAYERS as u32,
        },
    );
}

/// The maps of `terrains` terrains, `size` texels a side.
fn maps(device: &wgpu::Device, size: u32, terrains: usize) -> Maps {
    let texture = crate::targets::texture_2d(
        device,
        "engine/terrain-sky",
        (size, size),
        (SKY_LAYERS * terrains) as u32,
        1,
        FORMAT,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    );
    Maps {
        view: array_view(&texture),
        texture,
        size,
        terrains,
    }
}

fn array_view(texture: &wgpu::Texture) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("engine/terrain-sky"),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    })
}
