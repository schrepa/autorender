use std::sync::Arc;

use engine_core::{AssetLibrary, Error, Texture, TextureFormat};
use engine_places::ground::TerrainTiles;
use engine_render::{Install, Lifetime, Prepare, wgpu};

use crate::state::Bakes;
use crate::terrain::TerrainGpu;

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

pub(crate) struct MacroMaps {
    pub(crate) texture: engine_render::Texture<TerrainGpu>,
    shape: Option<(Shape, usize)>,
    empty: bool,
    held: Vec<Option<Arc<Texture>>>,
}

impl MacroMaps {
    pub(crate) fn install(cx: &mut Install<'_, TerrainGpu>) -> Result<Self, Error> {
        Ok(Self {
            texture: cx.texture(
                "macro_colors",
                wgpu::TextureFormat::Rgba8Unorm,
                Lifetime::Kept,
            ),
            shape: None,
            empty: false,
            held: Vec::new(),
        })
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

    pub(crate) fn prepare(
        &mut self,
        cx: &mut Prepare<'_, TerrainGpu>,
        (tiles, bakes, assets, max_layers): (&TerrainTiles, &Bakes, &AssetLibrary, u32),
    ) {
        let map = |index: usize| {
            let terrain = tiles.terrains.get(index)?;
            assets.texture(bakes.macro_map(terrain.terrain)?)
        };
        let first = (0..tiles.terrains.len()).find_map(map);
        let Some(shape) = first.map(|texture| Shape::of(texture)) else {
            self.shape = None;
            self.held.clear();
            if !self.empty {
                self.empty = true;
                cx.allocate_as(
                    self.texture,
                    wgpu::TextureFormat::Rgba8Unorm,
                    wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    1,
                );
                cx.write_level(
                    self.texture,
                    (0, 0),
                    (1, 1),
                    TextureFormat::Rgba8,
                    &[128, 128, 128, 128],
                );
            }
            return;
        };
        let count = tiles.terrains.len().min(max_layers as usize);
        if self.shape != Some((shape, count)) {
            self.shape = Some((shape, count));
            self.empty = false;
            self.held = vec![None; count];
            cx.allocate_as(
                self.texture,
                engine_render::texture_format(shape.format, false).0,
                wgpu::Extent3d {
                    width: shape.size,
                    height: shape.size,
                    depth_or_array_layers: count as u32,
                },
                shape.levels as u32,
            );
        }
        for index in 0..count {
            let wanted = map(index).filter(|texture| Shape::of(texture) == shape);
            let held = &mut self.held[index];
            match (wanted, &*held) {
                (Some(map), Some(current)) if Arc::ptr_eq(map, current) => {}
                (Some(map), _) => {
                    for (level, bytes) in map.levels().iter().enumerate() {
                        cx.write_level(
                            self.texture,
                            (level as u32, index as u32),
                            (shape.size, shape.size),
                            shape.format,
                            bytes,
                        );
                    }
                    *held = Some(Arc::clone(map));
                }
                (None, Some(_)) => {
                    *held = None;
                }
                (None, None) => {}
            }
        }
    }
}
