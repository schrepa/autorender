use std::cell::RefCell;
use std::rc::Rc;

use engine_core::{impostor_assembly_input, impostor_level_rows};

use super::*;

const ROW: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

#[derive(Clone, Copy)]
struct Region {
    offset: u64,
    rows: u32,
    stride: u32,
}

fn level_regions(shape: AtlasShape) -> (Vec<Region>, u64) {
    let mut regions = Vec::with_capacity(4 * shape.levels() as usize);
    let mut bytes = 0_u64;
    for _ in 0..4 {
        for level in 0..shape.levels() {
            let (rows, stride, _) = impostor_level_rows(shape, level, ROW);
            regions.push(Region {
                offset: bytes,
                rows,
                stride,
            });
            bytes += u64::from(stride) * u64::from(rows);
        }
    }
    (regions, bytes)
}

enum Mapped {
    Recorded,
    Mapping,
    Read(Vec<u8>),
    Failed,
}

pub(super) struct Readback {
    identity: ImpostorIdentity,
    shape: AtlasShape,
    buffer: wgpu::Buffer,
    mapped: Rc<RefCell<Mapped>>,
}

impl Atlas {
    pub(super) fn load(&mut self, queue: &crate::gpu_calls::Queue, layer: u32, bytes: &[u8]) {
        let arrays = self.arrays.as_ref().expect("arrays_grown");
        let mut at = engine_core::IMPOSTOR_RECORD_HEADER;
        for kind in 0..KINDS {
            for level in 0..self.shape.levels() {
                let size = self.shape.size() >> level;
                let (row, rows) = self.format.rows(size);
                let length = row as usize * rows as usize;
                queue.write_texture(
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
    }

    pub(super) fn read_back(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        (identity, shape): (ImpostorIdentity, AtlasShape),
    ) {
        let baked = &self.targets.as_ref().expect("targets_drawn").layer;
        let (regions, bytes) = level_regions(shape);
        let buffer = crate::memory::buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("engine/impostor-readback"),
                size: bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        );
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
            identity,
            shape,
            buffer,
            mapped: Rc::new(RefCell::new(Mapped::Recorded)),
        });
    }

    pub(super) fn take_made(&mut self, out: &mut Vec<(ImpostorIdentity, Option<Vec<u8>>)>) {
        self.readbacks.retain(|readback| {
            let read = match &mut *readback.mapped.borrow_mut() {
                Mapped::Read(input) => Some(std::mem::take(input)),
                Mapped::Failed => None,
                Mapped::Recorded | Mapped::Mapping => return true,
            };
            out.push((readback.identity, read));
            false
        });
    }

    pub(super) fn after_submit(&mut self) {
        let format = self.format;
        for readback in &self.readbacks {
            if !matches!(*readback.mapped.borrow(), Mapped::Recorded) {
                continue;
            }
            *readback.mapped.borrow_mut() = Mapped::Mapping;
            let buffer = readback.buffer.clone();
            let mapped = Rc::clone(&readback.mapped);
            let shape = readback.shape;
            readback.buffer.slice(..).map_async(
                wgpu::MapMode::Read,
                crate::platform::map_callback(move |result| {
                    let read = if result.is_ok() {
                        let input = buffer
                            .slice(..)
                            .get_mapped_range()
                            .ok()
                            .map(|view| impostor_assembly_input(shape, format, ROW, &view));
                        buffer.unmap();
                        input
                    } else {
                        None
                    };
                    *mapped.borrow_mut() = read.map_or(Mapped::Failed, Mapped::Read);
                }),
            );
        }
    }
}
