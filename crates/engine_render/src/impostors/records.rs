//! Impostors' records: a layer of an atlas, as hosts keep it and the cook
//! ships it. A record holds its header (its magic, then its format's tag,
//! frames a side, texels a frame, whether over the hemisphere and levels,
//! little-endian u32 each) and the four arrays' levels, first level first,
//! in its format's rows (of BC7 blocks or of RGBA8 texels). Loading one
//! writes it into a new layer; a bake stores its layer into the arrays,
//! encoded in BC7 on the GPU (`impostor_bc7.wgsl`) for BC7 atlases, and
//! reads it back as the record for hosts to keep.

use std::cell::RefCell;
use std::rc::Rc;

use super::*;
use crate::bindings::IMPOSTOR_BC7;
use crate::gpu_layouts::impostors::Bc7Level;
use crate::shaders::entry::impostor_bc7;

const RECORD_MAGIC: &[u8; 4] = b"EIMP";
const RECORD_HEADER: usize = 24;
/// Rows of copies between buffers and textures start this many bytes
/// apart.
const ROW: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

/// The identity of the record of an impostor held at `shape` as
/// `format`, of the mesh and the material whose contents are `content`.
pub(super) fn identity(shape: AtlasShape, format: AtlasFormat, content: (u128, u128)) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(48);
    for word in [
        format.tag(),
        shape.frames,
        shape.texels,
        u32::from(shape.hemisphere),
    ] {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes.extend_from_slice(&content.0.to_le_bytes());
    bytes.extend_from_slice(&content.1.to_le_bytes());
    bytes
}

/// The header of a record of a layer held at `shape` as `format`.
fn record_header(shape: AtlasShape, format: AtlasFormat) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(RECORD_HEADER);
    bytes.extend_from_slice(RECORD_MAGIC);
    for word in [
        format.tag(),
        shape.frames,
        shape.texels,
        u32::from(shape.hemisphere),
        shape.levels(),
    ] {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

/// Bytes of a record of a layer held at `shape` as `format`: its header
/// and four arrays' levels.
fn record_bytes(shape: AtlasShape, format: AtlasFormat) -> usize {
    RECORD_HEADER
        + 4 * (0..shape.levels())
            .map(|level| format.level_bytes(shape.size() >> level))
            .sum::<usize>()
}

/// Where a level's rows lie in a buffer: its offset, its rows (of blocks
/// in BC7), the bytes from one row's start to the next's, and those each
/// row holds.
#[derive(Clone, Copy)]
struct Region {
    offset: u64,
    rows: u32,
    stride: u32,
    row: u32,
}

/// A buffer holding every level of a layer's four textures, and each
/// one's region in it, the first texture's levels first.
struct Levels {
    buffer: wgpu::Buffer,
    regions: Vec<Region>,
}

/// The regions of the levels of four textures of `shape` in `format`,
/// each row padded to [`ROW`], and the bytes they take.
fn level_regions(shape: AtlasShape, format: AtlasFormat) -> (Vec<Region>, u64) {
    let mut regions = Vec::with_capacity(4 * shape.levels() as usize);
    let mut bytes = 0_u64;
    for _ in 0..4 {
        for level in 0..shape.levels() {
            let (row, rows) = format.rows(shape.size() >> level);
            let stride = row.next_multiple_of(ROW);
            regions.push(Region {
                offset: bytes,
                rows,
                stride,
                row,
            });
            bytes += u64::from(stride) * u64::from(rows);
        }
    }
    (regions, bytes)
}

/// Where a read back stands: recorded into a frame not yet submitted,
/// mapping, read, or failed.
enum Mapped {
    Recorded,
    Mapping,
    Read(Vec<u8>),
    Failed,
}

/// A record read back: its key, its header, the buffer mapped with its
/// regions, and where its map stands.
pub(super) struct Readback {
    record: RecordKey,
    header: Vec<u8>,
    levels: Levels,
    mapped: Rc<RefCell<Mapped>>,
}

impl ImpostorAtlas {
    /// Uploads the record `bytes` of `key`, drawing `source`, into a new
    /// layer with `gpu`'s queue; whether they were a record of the atlas's
    /// shape and format and found a layer within `budget`.
    pub(super) fn load(
        &mut self,
        gpu: &mut BakeGpu<'_>,
        key: ImpostorKey,
        source: &BakeSource<'_>,
        bytes: &[u8],
        budget: Option<u32>,
    ) -> bool {
        let header = record_header(self.shape, self.format);
        if bytes.len() != record_bytes(self.shape, self.format)
            || bytes.get(..RECORD_HEADER) != Some(header.as_slice())
        {
            return false;
        }
        let Some(layer) = self.layer(gpu.device, gpu.encoder, budget) else {
            return false;
        };
        let arrays = self.arrays.as_ref().expect("grown above");
        let mut at = RECORD_HEADER;
        for texture in &arrays.textures {
            for level in 0..self.shape.levels() {
                let size = self.shape.size() >> level;
                let (row, rows) = self.format.rows(size);
                let length = row as usize * rows as usize;
                gpu.queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture,
                        mip_level: level,
                        origin: wgpu::Origin3d {
                            x: 0,
                            y: 0,
                            z: layer,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    &bytes[at..at + length],
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(row),
                        rows_per_image: Some(rows),
                    },
                    wgpu::Extent3d {
                        width: size,
                        height: size,
                        depth_or_array_layers: 1,
                    },
                );
                at += length;
            }
        }
        self.loaded += bytes.len();
        self.resident(key, source, layer);
        true
    }

    /// Stores the layer the bake just drew into `layer` of the arrays in
    /// their format, and reads it back as the record hosts keep of the
    /// mesh and the material whose contents are `content`.
    pub(super) fn store(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pipelines: &mut ImpostorPipelines,
        layer: u32,
        content: (u128, u128),
    ) {
        let (shape, format) = (self.shape, self.format);
        let record = self.record_key(content);
        let baked = &self.targets.as_ref().expect("drawn above").layer;
        let arrays = self.arrays.as_ref().expect("grown above");
        let levels = match format {
            AtlasFormat::Bc7 => {
                let blocks = encode_bc7(device, encoder, pipelines, baked, shape);
                copy_blocks(encoder, &blocks, &arrays.textures, layer, shape.levels());
                read_buffer(device, encoder, &blocks)
            }
            AtlasFormat::Rgba8 => {
                copy_levels(encoder, baked, &arrays.textures, layer, shape);
                read_levels(device, encoder, baked, shape)
            }
        };
        self.readbacks.push(Readback {
            record,
            header: record_header(shape, format),
            levels,
            mapped: Rc::new(RefCell::new(Mapped::Recorded)),
        });
    }

    /// Records it made since the last call, for hosts to keep: the read
    /// backs mapped since.
    pub(crate) fn take_made(&mut self, out: &mut Vec<(RecordKey, Vec<u8>)>) {
        let mut kept = Vec::with_capacity(self.readbacks.len());
        for readback in std::mem::take(&mut self.readbacks) {
            let read = match &*readback.mapped.borrow() {
                Mapped::Read(bytes) => Some(Some(assemble(&readback, bytes))),
                Mapped::Failed => Some(None),
                Mapped::Recorded | Mapped::Mapping => None,
            };
            match read {
                Some(Some(bytes)) => out.push((readback.record, bytes)),
                Some(None) => {}
                None => kept.push(readback),
            }
        }
        self.readbacks = kept;
    }

    /// Maps the read backs the frame just submitted recorded, whose bytes
    /// come with a later frame; the next frame uploads within a new budget,
    /// the answers it held back first.
    pub(crate) fn after_submit(&mut self) {
        self.loaded = 0;
        if std::mem::take(&mut self.deferred) {
            self.generation += 1;
        }
        for readback in &self.readbacks {
            if !matches!(*readback.mapped.borrow(), Mapped::Recorded) {
                continue;
            }
            *readback.mapped.borrow_mut() = Mapped::Mapping;
            let buffer = readback.levels.buffer.clone();
            let mapped = Rc::clone(&readback.mapped);
            readback.levels.buffer.slice(..).map_async(
                wgpu::MapMode::Read,
                crate::platform::map_callback(move |result| {
                    let read = if result.is_ok() {
                        let bytes = buffer
                            .slice(..)
                            .get_mapped_range()
                            .ok()
                            .map(|view| view.to_vec());
                        buffer.unmap();
                        bytes
                    } else {
                        None
                    };
                    *mapped.borrow_mut() = read.map_or(Mapped::Failed, Mapped::Read);
                }),
            );
        }
    }
}

impl ImpostorPipelines {
    /// BC7's encoder: its group's layout and its pipeline.
    fn bc7(
        &mut self,
        device: &wgpu::Device,
    ) -> &(crate::pipelines::GroupLayout, wgpu::ComputePipeline) {
        self.bc7.get_or_insert_with(|| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("engine/impostor_bc7.wgsl"),
                source: wgpu::ShaderSource::Wgsl(crate::gpu_data::IMPOSTOR_BC7_SHADER.into()),
            });
            let layout = crate::pipelines::layout(device, &IMPOSTOR_BC7);
            let pipeline_layout = crate::pipelines::pipeline_layout(
                device,
                &crate::pipeline_layouts::IMPOSTOR_BC7_PIPELINES,
                &[&layout],
            );
            let pipeline = crate::pipelines::compute_entry(
                device,
                &pipeline_layout,
                &module,
                impostor_bc7::encode_bc7,
            );
            (layout, pipeline)
        })
    }
}

/// Every level of the bake layer `baked` (four textures at `shape`)
/// encoded in BC7 into a buffer.
fn encode_bc7(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    pipelines: &mut ImpostorPipelines,
    baked: &[wgpu::Texture; 4],
    shape: AtlasShape,
) -> Levels {
    let (regions, bytes) = level_regions(shape, AtlasFormat::Bc7);
    let buffer = crate::buffers::storage(
        device,
        "engine/impostor-bc7",
        bytes,
        wgpu::BufferUsages::COPY_SRC,
    );
    let (layout, pipeline) = pipelines.bc7(device);
    let levels = shape.levels() as usize;
    let mut groups = Vec::with_capacity(regions.len());
    for (index, region) in regions.iter().enumerate() {
        let level = (index % levels) as u32;
        // Texels as stored: sRGB levels read undecoded.
        let view = baked[index / levels].create_view(&wgpu::TextureViewDescriptor {
            label: Some("engine/impostor-bc7-source"),
            format: Some(wgpu::TextureFormat::Rgba8Unorm),
            dimension: Some(wgpu::TextureViewDimension::D2),
            base_mip_level: level,
            mip_level_count: Some(1),
            ..Default::default()
        });
        let uniform = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("engine/impostor-bc7-level"),
                contents: bytemuck::bytes_of(&Bc7Level {
                    blocks: [region.rows, region.rows],
                    first: (region.offset / 16) as u32,
                    row: region.stride / 16,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        );
        let group = crate::pipelines::group(
            device,
            "engine/impostor-bc7-group",
            &IMPOSTOR_BC7,
            layout,
            &[
                ("bc7_source", crate::pipelines::view(&view)),
                ("bc7_blocks", buffer.as_entire_binding()),
                ("bc7_level", uniform.as_entire_binding()),
            ],
        );
        groups.push((group, region.rows));
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("engine/impostor-bc7"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        for (group, blocks) in &groups {
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(blocks.div_ceil(8), blocks.div_ceil(8), 1);
        }
    }
    Levels { buffer, regions }
}

/// Copies BC7 `blocks` of a layer's `levels` into `layer` of `textures`.
fn copy_blocks(
    encoder: &mut wgpu::CommandEncoder,
    blocks: &Levels,
    textures: &[wgpu::Texture; 4],
    layer: u32,
    levels: u32,
) {
    for (index, region) in blocks.regions.iter().enumerate() {
        let size = region.rows * 4;
        encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: &blocks.buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: region.offset,
                    bytes_per_row: Some(region.stride),
                    rows_per_image: Some(region.rows),
                },
            },
            wgpu::TexelCopyTextureInfo {
                texture: &textures[index / levels as usize],
                mip_level: index as u32 % levels,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
        );
    }
}

/// Copies the levels of the bake layer `baked` (four textures at `shape`)
/// into `layer` of `textures`.
fn copy_levels(
    encoder: &mut wgpu::CommandEncoder,
    baked: &[wgpu::Texture; 4],
    textures: &[wgpu::Texture; 4],
    layer: u32,
    shape: AtlasShape,
) {
    for (from, to) in baked.iter().zip(textures) {
        for level in 0..shape.levels() {
            let size = shape.size() >> level;
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: from,
                    mip_level: level,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: to,
                    mip_level: level,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: layer,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: 1,
                },
            );
        }
    }
}

/// A buffer to map holding `blocks`' bytes, at their regions.
fn read_buffer(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    blocks: &Levels,
) -> Levels {
    let size = blocks.buffer.size();
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("engine/impostor-readback"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&blocks.buffer, 0, &buffer, 0, size);
    Levels {
        buffer,
        regions: blocks.regions.clone(),
    }
}

/// A buffer to map holding the levels of the bake layer `baked` (four
/// textures at `shape`) as RGBA8 texels.
fn read_levels(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    baked: &[wgpu::Texture; 4],
    shape: AtlasShape,
) -> Levels {
    let (regions, bytes) = level_regions(shape, AtlasFormat::Rgba8);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("engine/impostor-readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let levels = shape.levels();
    for (index, region) in regions.iter().enumerate() {
        let level = index as u32 % levels;
        let size = shape.size() >> level;
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &baked[index / levels as usize],
                mip_level: level,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: region.offset,
                    bytes_per_row: Some(region.stride),
                    rows_per_image: Some(region.rows),
                },
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
        );
    }
    Levels { buffer, regions }
}

/// A record's bytes from its read back's mapped `bytes`: its header, then
/// each region's rows less their padding.
fn assemble(readback: &Readback, bytes: &[u8]) -> Vec<u8> {
    let mut out = readback.header.clone();
    for region in &readback.levels.regions {
        for at in 0..region.rows {
            let start = (region.offset + u64::from(at) * u64::from(region.stride)) as usize;
            if let Some(row) = bytes.get(start..start + region.row as usize) {
                out.extend_from_slice(row);
            }
        }
    }
    out
}
