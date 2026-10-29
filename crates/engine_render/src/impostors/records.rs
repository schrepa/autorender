use std::cell::RefCell;
use std::rc::Rc;

use super::*;

const RECORD_MAGIC: &[u8; 4] = b"EIMP";
const RECORD_HEADER: usize = 24;
const ROW: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

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

fn record_bytes(shape: AtlasShape, format: AtlasFormat) -> usize {
    RECORD_HEADER
        + 4 * (0..shape.levels())
            .map(|level| format.level_bytes(shape.size() >> level))
            .sum::<usize>()
}

#[derive(Clone, Copy)]
struct Region {
    offset: u64,
    rows: u32,
    stride: u32,
    row: u32,
}

fn level_rows(shape: AtlasShape, level: u32) -> (u32, u32, u32) {
    let (row, rows) = AtlasFormat::Rgba8.rows(shape.size() >> level);
    (rows, row.next_multiple_of(ROW), row)
}

fn level_regions(shape: AtlasShape) -> (Vec<Region>, u64) {
    let mut regions = Vec::with_capacity(4 * shape.levels() as usize);
    let mut bytes = 0_u64;
    for _ in 0..4 {
        for level in 0..shape.levels() {
            let (rows, stride, row) = level_rows(shape, level);
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

pub(super) fn readback_bytes(shape: AtlasShape) -> u64 {
    4 * (0..shape.levels())
        .map(|level| {
            let (rows, stride, _) = level_rows(shape, level);
            u64::from(stride) * u64::from(rows)
        })
        .sum::<u64>()
}

enum Mapped {
    Recorded,
    Mapping,
    Read(Vec<u8>),
    Failed,
}

pub(super) struct Readback {
    record: RecordKey,
    buffer: wgpu::Buffer,
    regions: Vec<Region>,
    mapped: Rc<RefCell<Mapped>>,
}

impl ImpostorAtlas {
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
        for kind in 0..KINDS {
            for level in 0..self.shape.levels() {
                let size = self.shape.size() >> level;
                let (row, rows) = self.format.rows(size);
                let length = row as usize * rows as usize;
                gpu.queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &arrays.texture,
                        mip_level: level,
                        origin: wgpu::Origin3d {
                            x: 0,
                            y: 0,
                            z: kind * arrays.layers + layer,
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
        let buffer = crate::memory::buffer(device, &wgpu::BufferDescriptor {
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
                Some(None) => self.answer(readback.record, None),
                None => kept.push(readback),
            }
        }
        self.readbacks = kept;
    }

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
            match self.format.family() {
                None => out.extend_from_slice(&texels),
                Some(family) => {
                    let size = self.shape.size() >> (index % levels);
                    out.extend_from_slice(&engine_textures::compress(size, size, &texels, family));
                }
            }
        }
        out
    }
}
