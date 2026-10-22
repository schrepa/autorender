//! Tiles' water on the GPU: each resident tile's water where it stands over
//! the ground, a patch at a time, in two texture arrays of blocks as many
//! samples wide as a patch's grid: their surfaces, and their words, their
//! packed flow, whitewater and kind, their state and their nearest obstacle
//! (`engine_core::terrain_data::TileWater::info`), and a fourth, 0. A tile's
//! blocks upload as it arrives and free as it leaves or changes, following
//! the tiles the atlas held anew; the camera's patches holding water lay
//! out each frame as instances water's pass draws on terrain's grid, with
//! its morph and projection, so water meets the ground it covers on the
//! same triangles (`crate::water`), written where they differ from the
//! last frame's.

#![expect(
    clippy::expect_used,
    reason = "the arrays are allocated before blocks upload"
)]

use super::*;
use crate::gpu_layouts::constants::{
    NO_WATER_BLOCK, NO_WATER_SURFACE, WATER_BLOCK as BLOCK, WATER_BLOCKS_SIDE as SIDE,
};
use crate::gpu_layouts::water::WaterPatchRecord;
use engine_core::terrain_data::{PATCHES_PER_TILE, Tile};

/// Blocks a layer holds, and its texels along a side.
const PER_LAYER: u32 = SIDE * SIDE;
const LAYER_TEXELS: u32 = BLOCK * SIDE;
/// The surface and info arrays' formats, and the bytes of each one's
/// texels: shaders unpack the info's words themselves.
const SURFACE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
const INFO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Uint;
const SURFACE_BYTES: u32 = 4;
const INFO_BYTES: u32 = 16;
/// Indices of the grid's triangles before its skirts': water hangs none.
pub(crate) const GRID_INDICES: u32 = (GRID - 1) * (GRID - 1) * 6;

/// The arrays and their layers.
struct Arrays {
    surface: wgpu::Texture,
    surface_view: wgpu::TextureView,
    info: wgpu::Texture,
    info_view: wgpu::TextureView,
    layers: u32,
}

/// A tile's blocks: the tile whose water they hold, each patch's block
/// (`NO_WATER_BLOCK` where its water stands nowhere over its ground) and
/// its water's highest surface.
struct TileBlocks {
    tile: Arc<Tile>,
    blocks: Box<[u32]>,
    tops: Box<[f32]>,
}

pub(crate) struct WaterBlocks {
    arrays: Option<Arrays>,
    /// Advances whenever the arrays are replaced.
    pub(crate) generation: u64,
    tiles: HashMap<(u64, TileKey), TileBlocks>,
    free: Vec<u32>,
    next: u32,
    /// Blocks uploaded this frame, and dropped for want of layers.
    pub(crate) uploads: u32,
    pub(crate) dropped: u32,
    // This frame's patches holding water and the last frame's, which the
    // instances hold unless they were made since (`fresh`); their
    // instances, and each terrain's uniform offset and range of them.
    records: Vec<WaterPatchRecord>,
    written: Vec<WaterPatchRecord>,
    fresh: bool,
    pub(crate) instances: wgpu::Buffer,
    pub(crate) terrains: Vec<(u32, Range<u32>)>,
}

impl WaterBlocks {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        Self {
            arrays: None,
            generation: 0,
            tiles: HashMap::default(),
            free: Vec::new(),
            next: 0,
            uploads: 0,
            dropped: 0,
            records: Vec::new(),
            written: Vec::new(),
            fresh: true,
            instances: instance_buffer(device, 64),
            terrains: Vec::new(),
        }
    }

    /// The surface and info arrays' views, once a block uploaded.
    pub(crate) fn views(&self) -> Option<(&wgpu::TextureView, &wgpu::TextureView)> {
        self.arrays
            .as_ref()
            .map(|arrays| (&arrays.surface_view, &arrays.info_view))
    }

    /// Blocks resident.
    pub(crate) fn resident(&self) -> u32 {
        self.next - self.free.len() as u32
    }

    /// Bytes of the arrays.
    pub(super) fn capacity_bytes(&self) -> u64 {
        self.arrays.as_ref().map_or(0, |arrays| {
            u64::from(arrays.layers)
                * u64::from(LAYER_TEXELS * LAYER_TEXELS)
                * u64::from(SURFACE_BYTES + INFO_BYTES)
        })
    }

    /// Uploads the blocks of tiles `held` (by terrain and key) that arrived
    /// or changed, and frees those of tiles gone, looking only at
    /// `changes`, the keys whose held tile changed since the last sync; a
    /// tile replaced under its key keeps the blocks of patches whose water
    /// stayed the same (its water made again where its rivers run
    /// otherwise). Layers grow up to `max_layers`.
    pub(super) fn sync(
        &mut self,
        device: &wgpu::Device,
        queue: &Queue,
        held: &HashMap<(u64, TileKey), Arc<Tile>>,
        changes: &[(u64, TileKey)],
        max_layers: u32,
    ) {
        self.uploads = 0;
        self.dropped = 0;
        let mut replaced = Vec::new();
        let mut arrived = Vec::new();
        for &slot in changes {
            let had = self.tiles.get(&slot).map(|blocks| Arc::clone(&blocks.tile));
            match (had, held.get(&slot)) {
                (Some(had), Some(tile)) if Arc::ptr_eq(&had, tile) => {}
                (Some(_), Some(tile)) => {
                    if let Some(blocks) = self.tiles.remove(&slot) {
                        replaced.push((slot, blocks, Arc::clone(tile)));
                    }
                }
                (Some(_), None) => {
                    if let Some(blocks) = self.tiles.remove(&slot) {
                        let used = blocks.blocks.iter().copied();
                        self.free
                            .extend(used.filter(|&block| block != NO_WATER_BLOCK));
                    }
                }
                (None, Some(tile)) => arrived.push((slot, Arc::clone(tile))),
                (None, None) => {}
            }
        }
        // Lower blocks first, so the arrays stay as small as they can.
        self.free.sort_unstable_by(|a, b| b.cmp(a));
        for (slot, before, tile) in replaced {
            let blocks = self.upload(device, queue, tile, Some(before), max_layers);
            self.tiles.insert(slot, blocks);
        }
        for (slot, tile) in arrived {
            let blocks = self.upload(device, queue, tile, None, max_layers);
            self.tiles.insert(slot, blocks);
        }
    }

    /// Whether it holds blocks for exactly the tiles `held` holds, each of
    /// that tile, which debug builds check.
    #[cfg(debug_assertions)]
    pub(super) fn follows(&self, held: &HashMap<(u64, TileKey), Arc<Tile>>) -> bool {
        self.tiles.len() == held.len()
            && self.tiles.iter().all(|(slot, blocks)| {
                held.get(slot)
                    .is_some_and(|tile| Arc::ptr_eq(tile, &blocks.tile))
            })
    }

    /// `tile`'s blocks, its patches holding water uploaded: in place of
    /// `before`'s, the tile it replaces under its key, whose blocks it keeps
    /// where its patches' water stayed the same and frees where they hold
    /// none.
    fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &Queue,
        tile: Arc<Tile>,
        before: Option<TileBlocks>,
        max_layers: u32,
    ) -> TileBlocks {
        let patches = PATCHES_PER_TILE as usize;
        let (mut blocks, mut tops, held) = match before {
            Some(before) => (before.blocks, before.tops, Some(before.tile)),
            None => (
                vec![NO_WATER_BLOCK; patches * patches].into_boxed_slice(),
                vec![f32::NEG_INFINITY; patches * patches].into_boxed_slice(),
                None,
            ),
        };
        let stride = LAYER_SIZE as usize;
        let side = BLOCK as usize;
        // A patch's samples in the tile's layout.
        let samples = |px: usize, pz: usize| {
            (0..side).flat_map(move |z| {
                (0..side).map(move |x| (pz * (side - 1) + z + 1) * stride + px * (side - 1) + x + 1)
            })
        };
        let mut surface_texels = vec![0.0_f32; side * side];
        let mut info_texels = vec![[0_u32; 4]; side * side];
        for pz in 0..patches {
            for px in 0..patches {
                let index = pz * patches + px;
                let water = tile
                    .water()
                    .filter(|water| water.patch_holds_water(px as u32, pz as u32));
                let Some(water) = water else {
                    if blocks[index] != NO_WATER_BLOCK {
                        self.free.push(blocks[index]);
                        blocks[index] = NO_WATER_BLOCK;
                        tops[index] = f32::NEG_INFINITY;
                    }
                    continue;
                };
                let kept = blocks[index] != NO_WATER_BLOCK
                    && held
                        .as_ref()
                        .and_then(|held| held.water())
                        .is_some_and(|before| {
                            samples(px, pz).all(|at| {
                                before.surfaces()[at].to_bits() == water.surfaces()[at].to_bits()
                                    && before.info()[at] == water.info()[at]
                            })
                        });
                if kept {
                    continue;
                }
                let block = match blocks[index] {
                    NO_WATER_BLOCK => match self.allocate(device, queue, max_layers) {
                        Some(block) => block,
                        None => {
                            self.dropped += 1;
                            continue;
                        }
                    },
                    block => block,
                };
                let mut top = f32::NEG_INFINITY;
                for (texel, at) in samples(px, pz).enumerate() {
                    let surface = water.surfaces()[at];
                    surface_texels[texel] = if surface.is_finite() {
                        top = engine_core::math::max(top, surface);
                        surface
                    } else {
                        NO_WATER_SURFACE
                    };
                    let [packed, state, obstacle] = water.info()[at];
                    info_texels[texel] = [packed, state, obstacle, 0];
                }
                self.write(queue, block, &surface_texels, &info_texels);
                blocks[index] = block;
                tops[index] = top;
                self.uploads += 1;
            }
        }
        TileBlocks { tile, blocks, tops }
    }

    /// A free block, the arrays grown to hold it; `None` beyond
    /// `max_layers`.
    fn allocate(&mut self, device: &wgpu::Device, queue: &Queue, max_layers: u32) -> Option<u32> {
        if let Some(block) = self.free.pop() {
            return Some(block);
        }
        let block = self.next;
        let needed = block / PER_LAYER + 1;
        if needed > max_layers {
            return None;
        }
        self.ensure_arrays(device, queue, needed, max_layers);
        self.next += 1;
        Some(block)
    }

    /// Grows the arrays to hold `needed` layers, doubling, keeping the
    /// blocks uploaded.
    fn ensure_arrays(
        &mut self,
        device: &wgpu::Device,
        queue: &Queue,
        needed: u32,
        max_layers: u32,
    ) {
        let current = self.arrays.as_ref().map_or(0, |arrays| arrays.layers);
        if needed <= current {
            return;
        }
        let layers = needed.next_power_of_two().min(max_layers.max(1));
        let array = |label: &'static str, format| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: LAYER_TEXELS,
                    height: LAYER_TEXELS,
                    depth_or_array_layers: layers,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let surface = array("engine/water-surfaces", SURFACE_FORMAT);
        let info = array("engine/water-info", INFO_FORMAT);
        if let Some(old) = &self.arrays {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("engine/water-grow"),
            });
            let extent = wgpu::Extent3d {
                width: LAYER_TEXELS,
                height: LAYER_TEXELS,
                depth_or_array_layers: old.layers,
            };
            encoder.copy_texture_to_texture(
                old.surface.as_image_copy(),
                surface.as_image_copy(),
                extent,
            );
            encoder.copy_texture_to_texture(old.info.as_image_copy(), info.as_image_copy(), extent);
            queue.submit_now([encoder.finish()]);
        }
        let view = |texture: &wgpu::Texture| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            })
        };
        self.arrays = Some(Arrays {
            surface_view: view(&surface),
            info_view: view(&info),
            surface,
            info,
            layers,
        });
        self.generation += 1;
    }

    /// Writes block `block`'s texels.
    fn write(&self, queue: &Queue, block: u32, surfaces: &[f32], info: &[[u32; 4]]) {
        let arrays = self.arrays.as_ref().expect("a block allocates its arrays");
        let index = block % PER_LAYER;
        let origin = wgpu::Origin3d {
            x: index % SIDE * BLOCK,
            y: index / SIDE * BLOCK,
            z: block / PER_LAYER,
        };
        let extent = wgpu::Extent3d {
            width: BLOCK,
            height: BLOCK,
            depth_or_array_layers: 1,
        };
        for (texture, bytes, texel) in [
            (&arrays.surface, bytemuck::cast_slice(surfaces), SURFACE_BYTES),
            (&arrays.info, bytemuck::cast_slice(info), INFO_BYTES),
        ] {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin,
                    aspect: wgpu::TextureAspect::All,
                },
                bytes,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(BLOCK * texel),
                    rows_per_image: Some(BLOCK),
                },
                extent,
            );
        }
    }

    /// Lays out this frame's patches of `terrains` holding water that the
    /// camera sees through `culling` (eye-relative) from `eye`, as terrain
    /// lays its own out in `parts`, and writes their instances where they
    /// differ from the last frame's. A tile's blocks, its parent's and what
    /// its patches' records share are looked up once for each run of them.
    pub(super) fn lay_out(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        terrains: &RenderTerrains,
        (culling, eye, fades): (Mat4, DVec3, bool),
        parts: &patches::Layout<'_>,
    ) {
        std::mem::swap(&mut self.records, &mut self.written);
        self.records.clear();
        self.terrains.clear();
        let planes = planes(culling, false);
        let patches = PATCHES_PER_TILE as usize;
        for (index, terrain) in terrains.terrains.iter().enumerate() {
            let anchors = patches::Anchors::of(terrain, eye);
            let relative_origin = (terrain.origin.as_dvec3() - eye).as_vec3();
            let start = self.records.len() as u32;
            // The tile of the patch before: its blocks, its parent's and
            // what its records share.
            let mut last: Option<TileKey> = None;
            let (mut blocks, mut parent, mut shared) = (None, None, None);
            for patch in &terrain.patches {
                if last != Some(patch.tile) {
                    last = Some(patch.tile);
                    blocks = self.tiles.get(&(terrain.id, patch.tile));
                    parent = (terrain.pyramid.parent(patch.tile))
                        .and_then(|parent| self.tiles.get(&(terrain.id, parent)));
                    shared = blocks
                        .and_then(|_| patches::TileRecord::of(parts, terrain, patch.tile, fades));
                }
                let Some(tile) = blocks else {
                    continue;
                };
                let at = usize::from(patch.z) * patches + usize::from(patch.x);
                let block = tile.blocks[at];
                if block == NO_WATER_BLOCK {
                    continue;
                }
                // Its box up to its water's highest surface.
                let top = Vec3::new(
                    patch.max.x,
                    engine_core::math::max(patch.max.y, tile.tops[at]),
                    patch.max.z,
                );
                if !visible(&planes, relative_origin + patch.min, relative_origin + top) {
                    continue;
                }
                let Some(record) = shared.map(|shared| shared.record(patch, anchors)) else {
                    continue;
                };
                // The parent tile's block over it, and where in that
                // block it starts.
                let (x, z) = (u32::from(patch.x), u32::from(patch.z));
                let over = parent.map_or(NO_WATER_BLOCK, |parent| {
                    let px = u32::from(patch.tile.x % 2) * 8 + x / 2;
                    let pz = u32::from(patch.tile.z % 2) * 8 + z / 2;
                    parent.blocks[(pz * PATCHES_PER_TILE + px) as usize]
                });
                let offset = ((x % 2) * 8) | (((z % 2) * 8) << 16);
                self.records.push(WaterPatchRecord {
                    first: record.first,
                    shape: record.shape,
                    atlas: record.atlas,
                    water: [block, over, offset, 0],
                    fade: record.fade,
                });
            }
            let range = start..self.records.len() as u32;
            if !range.is_empty() {
                self.terrains.push(((index as u64 * SLOT) as u32, range));
            }
        }
        if self.records.is_empty() {
            return;
        }
        let bytes = std::mem::size_of_val(self.records.as_slice()) as u64;
        if bytes > self.instances.size() {
            self.instances =
                instance_buffer(device, (self.records.len() as u64).next_power_of_two());
            self.fresh = true;
        }
        let run = if self.fresh {
            0..self.records.len()
        } else {
            changed_run(&self.written, &self.records)
        };
        self.fresh = false;
        if !run.is_empty() {
            let offset = (run.start * size_of::<WaterPatchRecord>()) as u64;
            let records = bytemuck::cast_slice(&self.records[run]);
            queue.write_buffer(&self.instances, offset, records);
        }
    }
}

/// `WaterPatchInput`'s instance locations.
pub(crate) fn water_patch_layout() -> wgpu::VertexBufferLayout<'static> {
    const ATTRIBUTES: [wgpu::VertexAttribute; 5] = crate::pipelines::vertex_attributes!(WaterPatchRecord {
        0 => first as Sint32x2,
        1 => shape as Float32x4,
        2 => atlas as Uint32x4,
        3 => water as Uint32x4,
        4 => fade as Float32,
    });
    wgpu::VertexBufferLayout {
        array_stride: size_of::<WaterPatchRecord>() as u64,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &ATTRIBUTES,
    }
}

fn instance_buffer(device: &wgpu::Device, records: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("engine/water-patches"),
        size: records * size_of::<WaterPatchRecord>() as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
