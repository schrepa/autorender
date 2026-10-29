use std::sync::Arc;

use engine_core::{Texture, TextureFormat};
use engine_terrain::RenderTerrains;

#[derive(Clone, Copy, PartialEq, Eq)]
struct Shape {
    size: u32,
    levels: usize,
    format: TextureFormat,
}

impl Shape {
    fn of(texture: &Texture) -> Self {
        Self {
            size: texture.width(),
            levels: texture.levels().len(),
            format: texture.format(),
        }
    }
}

struct Maps {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    shape: Shape,
    terrains: usize,
}

pub(crate) struct MacroMaps {
    maps: Option<Maps>,
    empty: wgpu::TextureView,
    held: Vec<Option<Arc<Texture>>>,
    generation: u64,
}

impl MacroMaps {
    pub(crate) fn new(device: &wgpu::Device, queue: &engine_render::gpu_calls::Queue) -> Self {
        let empty = engine_render::targets::texture_2d(
            device,
            "engine/terrain-macro-empty",
            (1, 1),
            1,
            1,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        queue.write_texture(
            empty.as_image_copy(),
            &[128, 128, 128, 128],
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
            Some(Some(_)) => index as u32 + 1,
            _ => 0,
        }
    }

    pub(crate) fn count(&self) -> u32 {
        self.held.iter().flatten().count() as u32
    }

    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.maps.as_ref().map_or(0, |maps| {
            let shape = maps.shape;
            let layer: u64 = (0..shape.levels as u32)
                .map(|level| {
                    let side = (shape.size >> level).max(1);
                    shape.format.level_bytes(side, side) as u64
                })
                .sum();
            layer * maps.terrains as u64
        })
    }

    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &engine_render::gpu_calls::Queue,
        terrains: &RenderTerrains,
    ) {
        let first = terrains
            .terrains
            .iter()
            .find_map(|terrain| terrain.macro_map.as_ref());
        let Some(shape) = first.map(|map| Shape::of(&map.texture)) else {
            if self.maps.take().is_some() {
                self.generation += 1;
            }
            self.held.clear();
            return;
        };
        let count = terrains
            .terrains
            .len()
            .min(device.limits().max_texture_array_layers as usize);
        if self
            .maps
            .as_ref()
            .is_none_or(|maps| maps.shape != shape || maps.terrains != count)
        {
            self.maps = Some(maps(device, shape, count));
            self.held = vec![None; count];
            self.generation += 1;
        }
        let Some(maps) = &self.maps else {
            return;
        };
        for (index, terrain) in terrains.terrains.iter().take(count).enumerate() {
            let map = terrain
                .macro_map
                .as_ref()
                .map(|map| &map.texture)
                .filter(|texture| Shape::of(texture) == shape);
            let held = &mut self.held[index];
            match (map, &*held) {
                (Some(map), Some(current)) if Arc::ptr_eq(map, current) => {}
                (Some(map), _) => {
                    for (level, bytes) in map.levels().iter().enumerate() {
                        engine_render::materials::write_level(
                            queue,
                            &maps.texture,
                            (level as u32, index as u32),
                            (shape.size, shape.size),
                            shape.format,
                            bytes,
                        );
                    }
                    *held = Some(map.clone());
                }
                (None, Some(_)) => {
                    *held = None;
                }
                (None, None) => {}
            }
        }
    }
}

fn maps(device: &wgpu::Device, shape: Shape, terrains: usize) -> Maps {
    let texture = engine_render::targets::texture_2d(
        device,
        "engine/terrain-macro",
        (shape.size, shape.size),
        terrains as u32,
        shape.levels as u32,
        engine_render::materials::gpu_format(shape.format, false).0,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    );
    Maps {
        view: array_view(&texture),
        texture,
        shape,
        terrains,
    }
}

fn array_view(texture: &wgpu::Texture) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("engine/terrain-macro"),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    })
}
