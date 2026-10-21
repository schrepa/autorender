//! Impostors' records: a layer of an atlas, as hosts keep it and the cook
//! ships it. A record holds its header (its magic, then its format's tag,
//! frames a side, texels a frame, whether over the hemisphere and levels,
//! little-endian u32 each) and the four arrays' levels, first level first,
//! in its format's rows (of BC7 blocks or of RGBA8 texels). Loading one
//! writes it into a new layer. A bake's layer is read back as RGBA8 texels
//! and made a record: BC7 atlases' encoded by the cook's own encoder
//! (`engine_textures::compress`), the one every texture record is made
//! with. Hosts keep it, and it loads as any record does.

use std::cell::RefCell;
use std::rc::Rc;

use super::*;

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

/// Where a level's RGBA8 rows lie in a buffer: its offset, its rows, the
/// bytes from one row's start to the next's, and those each row holds.
#[derive(Clone, Copy)]
struct Region {
    offset: u64,
    rows: u32,
    stride: u32,
    row: u32,
}

/// The regions of the RGBA8 levels of four textures of `shape`, each row
/// padded to [`ROW`], the first texture's levels first, and the bytes they
/// take.
fn level_regions(shape: AtlasShape) -> (Vec<Region>, u64) {
    let mut regions = Vec::with_capacity(4 * shape.levels() as usize);
    let mut bytes = 0_u64;
    for _ in 0..4 {
        for level in 0..shape.levels() {
            let (row, rows) = AtlasFormat::Rgba8.rows(shape.size() >> level);
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

/// A bake's layer read back: the key of the record it makes, the buffer
/// mapped with its levels' regions, and where its map stands.
pub(super) struct Readback {
    record: RecordKey,
    buffer: wgpu::Buffer,
    regions: Vec<Region>,
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

    /// Reads back the layer the bake just drew, at the atlas's shape with
    /// its levels, into a buffer: the record of the mesh and the material
    /// whose contents are `content` once it maps ([`Self::take_made`]).
    pub(super) fn read_back(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        content: (u128, u128),
    ) {
        let shape = self.shape;
        let record = self.record_key(content);
        let baked = &self.targets.as_ref().expect("drawn above").layer;
        let (regions, bytes) = level_regions(shape);
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
        self.readbacks.push(Readback {
            record,
            buffer,
            regions,
            mapped: Rc::new(RefCell::new(Mapped::Recorded)),
        });
    }

    /// Records it made since the last call, for hosts to keep: the read
    /// backs mapped since, each made a record (encoded in BC7 for BC7
    /// atlases), which also answers the supply waiting for it, so it loads
    /// into a layer as any record does.
    pub(crate) fn take_made(&mut self, out: &mut Vec<(RecordKey, Vec<u8>)>) {
        let mut kept = Vec::with_capacity(self.readbacks.len());
        for readback in std::mem::take(&mut self.readbacks) {
            let read = match &*readback.mapped.borrow() {
                Mapped::Read(bytes) => Some(Some(self.assemble(&readback.regions, bytes))),
                Mapped::Failed => Some(None),
                Mapped::Recorded | Mapped::Mapping => None,
            };
            match read {
                Some(Some(bytes)) => {
                    out.push((readback.record, bytes.clone()));
                    self.answer(readback.record, Some(bytes));
                }
                // A layer that never read back bakes again.
                Some(None) => self.answer(readback.record, None),
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
            let buffer = readback.buffer.clone();
            let mapped = Rc::clone(&readback.mapped);
            readback.buffer.slice(..).map_async(
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

    /// A record's bytes from a read back's mapped `bytes`, its levels at
    /// `regions`: its header, then each level's RGBA8 rows less their
    /// padding, or for BC7 atlases each level's blocks.
    fn assemble(&self, regions: &[Region], bytes: &[u8]) -> Vec<u8> {
        let mut out = record_header(self.shape, self.format);
        let levels = self.shape.levels() as usize;
        for (index, region) in regions.iter().enumerate() {
            let mut texels = Vec::with_capacity(region.row as usize * region.rows as usize);
            for at in 0..region.rows {
                let start = (region.offset + u64::from(at) * u64::from(region.stride)) as usize;
                if let Some(row) = bytes.get(start..start + region.row as usize) {
                    texels.extend_from_slice(row);
                }
            }
            match self.format {
                AtlasFormat::Rgba8 => out.extend_from_slice(&texels),
                AtlasFormat::Bc7 => {
                    let size = self.shape.size() >> (index % levels);
                    out.extend_from_slice(&engine_textures::compress(
                        size,
                        size,
                        &texels,
                        engine_textures::Family::Bc7,
                    ));
                }
            }
        }
        out
    }
}
