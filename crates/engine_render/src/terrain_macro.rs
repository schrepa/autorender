//! Terrain's macro colors: each terrain's map of the tint its bakes give its
//! ground and how lush its plants grow there
//! ([`engine_core::RenderTerrain::macro_map`]), cooked with its mips in the
//! family of formats the device samples, a layer each of one array the
//! terrain group binds. Ground takes the tint over its materials' own
//! colors as the map holds it, the same however near or far it is seen
//! (`terrain_variety.wgsl`), the runtime virtual texture's pages carry it,
//! and ground cover's blades take it as their ground does. A terrain's
//! layer uploads as its map arrives and again when it changes; until then
//! its ground shows its materials' colors.
//! Terrains past the device's array layers, and maps of another size, level
//! count or format than the first's, show theirs too.

use std::sync::Arc;

use engine_core::{RenderTerrains, Texture, TextureFormat};

/// What every layer of the array shares: the first map's size, levels and
/// format.
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
    // A texel leaving every ground as it is, bound while no terrain has a
    // map.
    empty: wgpu::TextureView,
    // The map each terrain's layer holds, and the uploads each has taken,
    // which pages of the runtime virtual texture baked from it follow.
    held: Vec<Option<Arc<Texture>>>,
    uploads: Vec<u64>,
    // Advances when the bound view is replaced.
    generation: u64,
}

impl MacroMaps {
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let empty = crate::targets::texture_2d(
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
            uploads: Vec::new(),
            generation: 0,
        }
    }

    /// Advances when [`Self::view`] is another view.
    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    /// The maps, a layer per terrain, or a texel leaving the ground as it
    /// is.
    pub(crate) fn view(&self) -> &wgpu::TextureView {
        self.maps.as_ref().map_or(&self.empty, |maps| &maps.view)
    }

    /// Terrain `index`'s layer plus one once its map uploaded; 0 before.
    pub(crate) fn layer(&self, index: usize) -> u32 {
        match self.held.get(index) {
            Some(Some(_)) => index as u32 + 1,
            _ => 0,
        }
    }

    /// The uploads terrain `index`'s map took, which pages baked from it
    /// follow; 0 before its first.
    pub(crate) fn uploads(&self, index: usize) -> u64 {
        self.uploads.get(index).copied().unwrap_or(0)
    }

    /// Terrains whose maps are here.
    pub(crate) fn count(&self) -> u32 {
        self.held.iter().flatten().count() as u32
    }

    /// Bytes of the maps with their levels.
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

    /// Uploads the map of every terrain whose macro colors arrived or
    /// changed since; without any the maps are released.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        terrains: &RenderTerrains,
    ) {
        let first = terrains
            .terrains
            .iter()
            .find_map(|terrain| terrain.macro_map.as_ref());
        let Some(shape) = first.map(|map| Shape::of(&map.texture)) else {
            if self.maps.take().is_some() {
                self.generation += 1;
                // Pages baked from the maps follow them away.
                for uploads in &mut self.uploads {
                    *uploads += 1;
                }
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
            self.uploads.resize(count, 0);
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
                        crate::materials::write_level(
                            queue,
                            &maps.texture,
                            (level as u32, index as u32),
                            (shape.size, shape.size),
                            shape.format,
                            bytes,
                        );
                    }
                    *held = Some(map.clone());
                    self.uploads[index] += 1;
                }
                (None, Some(_)) => {
                    *held = None;
                    self.uploads[index] += 1;
                }
                (None, None) => {}
            }
        }
    }
}

/// The maps of `terrains` terrains in `shape`.
fn maps(device: &wgpu::Device, shape: Shape, terrains: usize) -> Maps {
    let texture = crate::targets::texture_2d(
        device,
        "engine/terrain-macro",
        (shape.size, shape.size),
        terrains as u32,
        shape.levels as u32,
        crate::materials::gpu_format(shape.format, false).0,
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
