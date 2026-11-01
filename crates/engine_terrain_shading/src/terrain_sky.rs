use std::sync::Arc;

use engine_core::Error;
use engine_places::ground::TerrainTiles;
use engine_render::{Install, Lifetime, Prepare, Texture, wgpu};

use crate::lighting::{SKY_LAYERS, TerrainLighting};
use crate::settings::TerrainShading;
use crate::state::Bakes;
use crate::terrain::TerrainGpu;

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

pub(crate) struct SkyMaps {
    pub(crate) texture: Texture<TerrainGpu>,
    shape: Option<(u32, usize)>,
    held: Vec<Option<Arc<TerrainLighting>>>,
    pub(crate) uploaded: u32,
}

impl SkyMaps {
    pub(crate) fn install(cx: &mut Install<'_, TerrainGpu>) -> Result<Self, Error> {
        Ok(Self {
            texture: cx.texture("sky", FORMAT, Lifetime::Kept),
            shape: None,
            held: Vec::new(),
            uploaded: 0,
        })
    }

    pub(crate) fn layer(&self, index: usize) -> u32 {
        if self.held.get(index).is_some_and(Option::is_some) {
            (SKY_LAYERS * index) as u32 + 1
        } else {
            0
        }
    }

    pub(crate) fn count(&self) -> u32 {
        self.held.iter().flatten().count() as u32
    }

    fn release(&mut self) {
        self.uploaded = 0;
        self.shape = None;
        self.held.clear();
    }

    pub(crate) fn prepare(
        &mut self,
        cx: &mut Prepare<'_, TerrainGpu>,
        (tiles, bakes, settings, layers): (&TerrainTiles, &Bakes, &TerrainShading, u32),
    ) {
        if !settings.sky_visibility || tiles.terrains.is_empty() {
            self.release();
            return;
        }
        self.uploaded = 0;
        let size = settings.sky_resolution;
        let count = tiles.terrains.len().min(layers as usize / SKY_LAYERS);
        if self.shape != Some((size, count)) {
            self.shape = Some((size, count));
            self.held = vec![None; count];
        }
        cx.allocate(
            self.texture,
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: (SKY_LAYERS * count).max(1) as u32,
            },
            1,
        );
        for (index, terrain) in tiles.terrains.iter().take(count).enumerate() {
            let lighting = bakes
                .of(terrain.terrain)
                .and_then(|held| held.lighting.made.as_ref())
                .filter(|lighting| lighting.sky_size == size);
            let held = &mut self.held[index];
            match (lighting, &*held) {
                (Some(lighting), Some(current)) if Arc::ptr_eq(lighting, current) => {}
                (Some(lighting), _) => {
                    cx.write_texture(
                        self.texture,
                        wgpu::Origin3d {
                            x: 0,
                            y: 0,
                            z: (SKY_LAYERS * index) as u32,
                        },
                        &lighting.sky_layers(),
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
                    *held = Some(Arc::clone(lighting));
                    self.uploaded += 1;
                }
                (None, _) => *held = None,
            }
        }
    }
}
