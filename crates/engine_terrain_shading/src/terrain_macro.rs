use std::sync::Arc;

use engine_core::{AssetLibrary, ColorSpace, Error, Texture, TextureFormat};
use engine_open_world_kit::ground::TerrainTiles;
use engine_render::{Allocation, Install, Lifetime, Prepare, wgpu};

use crate::state::Bakes;
use crate::terrain::TerrainGpu;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct Shape {
    size: [u32; 2],
    levels: usize,
    format: TextureFormat,
}

impl Shape {
    pub(crate) fn of(texture: &Texture) -> Self {
        Self {
            size: [texture.width(), texture.height()],
            levels: texture.levels().len(),
            format: texture.format(),
        }
    }

    pub(crate) fn first(tiles: &TerrainTiles, bakes: &Bakes, assets: &AssetLibrary) -> Option<Self> {
        tiles
            .terrains
            .iter()
            .filter_map(|terrain| bakes.macro_map(terrain.terrain))
            .chain(bakes.lent().iter().copied().flatten())
            .find_map(|handle| assets.texture(handle))
            .map(|texture| Self::of(texture))
    }
}

pub(crate) struct MacroMaps {
    pub(crate) texture: engine_render::Texture<TerrainGpu>,
    shape: Option<(Shape, usize)>,
    empty: bool,
    lent: usize,
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
            lent: 0,
            held: Vec::new(),
        })
    }

    pub(crate) fn layer(&self, index: usize) -> u32 {
        let layer = self.lent + index;
        match self.held.get(layer) {
            Some(Some(_)) => layer as u32 + 1,
            _ => 0,
        }
    }

    pub(crate) fn count(&self) -> u32 {
        self.held.iter().skip(self.lent).flatten().count() as u32
    }

    pub(crate) fn prepare(
        &mut self,
        cx: &mut Prepare<'_, TerrainGpu>,
        (tiles, bakes, assets, max_layers): (&TerrainTiles, &Bakes, &AssetLibrary, u32),
    ) {
        let most = max_layers as usize;
        let lent = bakes.lent().len().min(most);
        let map = |index: usize| match index.checked_sub(lent) {
            Some(index) => {
                let terrain = tiles.terrains.get(index)?;
                assets.texture(bakes.macro_map(terrain.terrain)?)
            }
            None => assets.texture(bakes.lent()[index]?),
        };
        let Some(shape) = Shape::first(tiles, bakes, assets) else {
            self.shape = None;
            self.lent = 0;
            self.held.clear();
            let allocation = cx.allocate_as(
                self.texture,
                wgpu::TextureFormat::Rgba8Unorm,
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                1,
            );
            if allocation == Allocation::Fresh || !self.empty {
                self.empty = true;
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
        let count = (lent + tiles.terrains.len()).min(most);
        self.empty = false;
        let allocation = cx.allocate_as(
            self.texture,
            engine_render::texture_format(shape.format, ColorSpace::Linear).0,
            wgpu::Extent3d {
                width: shape.size[0],
                height: shape.size[1],
                depth_or_array_layers: count as u32,
            },
            shape.levels as u32,
        );
        if allocation == Allocation::Fresh
            || self.shape != Some((shape, count))
            || self.lent != lent
        {
            self.shape = Some((shape, count));
            self.lent = lent;
            self.held = vec![None; count];
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
                            (shape.size[0], shape.size[1]),
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
