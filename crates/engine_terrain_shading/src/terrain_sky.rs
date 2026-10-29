use std::sync::Arc;

use engine_terrain::lighting::SKY_LAYERS;
use engine_terrain::lighting::TerrainLighting;
use engine_terrain::{RenderTerrains, TerrainSettings};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

struct Maps {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    size: u32,
    terrains: usize,
}

pub(crate) struct SkyMaps {
    maps: Option<Maps>,
    empty: wgpu::TextureView,
    held: Vec<Option<Arc<TerrainLighting>>>,
    generation: u64,
    pub(crate) uploaded: u32,
}

impl SkyMaps {
    pub(crate) fn new(device: &wgpu::Device, queue: &engine_render::gpu_calls::Queue) -> Self {
        let empty = engine_render::targets::texture_2d(
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
            generation: 0,
            uploaded: 0,
        }
    }

    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    pub(crate) fn view(&self) -> &wgpu::TextureView {
        self.maps.as_ref().map_or(&self.empty, |maps| &maps.view)
    }

    pub(crate) fn layer(&self, index: usize) -> u32 {
        match self.held.get(index) {
            Some(Some(_)) => (SKY_LAYERS * index) as u32 + 1,
            _ => 0,
        }
    }

    pub(crate) fn count(&self) -> u32 {
        self.held.iter().flatten().count() as u32
    }

    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.maps.as_ref().map_or(0, |maps| {
            u64::from(maps.size) * u64::from(maps.size) * (SKY_LAYERS * maps.terrains) as u64 * 4
        })
    }

    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &engine_render::gpu_calls::Queue,
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
        let count = terrains
            .terrains
            .len()
            .min(device.limits().max_texture_array_layers as usize / SKY_LAYERS);
        if self
            .maps
            .as_ref()
            .is_none_or(|maps| maps.size != size || maps.terrains != count)
        {
            self.maps = Some(maps(device, size, count));
            self.held = vec![None; count];
            self.generation += 1;
        }
        let Some(maps) = &self.maps else {
            return;
        };
        for (index, terrain) in terrains.terrains.iter().take(count).enumerate() {
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
                    self.uploaded += 1;
                }
                (None, _) => *held = None,
            }
        }
    }
}

fn write_layers(
    queue: &engine_render::gpu_calls::Queue,
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

fn maps(device: &wgpu::Device, size: u32, terrains: usize) -> Maps {
    let texture = engine_render::targets::texture_2d(
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
