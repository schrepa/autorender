use engine_core::{FrameId, impostor_assembly_input, impostor_level_rows};

use super::*;
use crate::readback::{Arrival, Keep, Reader, Readbacks};

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

pub(super) struct Pending {
    serial: u32,
    identity: ImpostorIdentity,
    shape: AtlasShape,
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
        recording: (&wgpu::Device, &mut wgpu::CommandEncoder),
        (readbacks, written, serial): (&mut Readbacks, FrameId, u32),
        (identity, shape): (ImpostorIdentity, AtlasShape),
    ) {
        let baked = &self.targets.as_ref().expect("targets_drawn").layer;
        let (regions, bytes) = level_regions(shape);
        let levels = shape.levels();
        let recorded = readbacks.record_with(
            recording,
            bytes,
            (Reader::Impostor(serial), written, None),
            Keep::Once,
            |encoder, buffer| {
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
                            buffer,
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
            },
        );
        if recorded {
            self.readbacks.push(Pending {
                serial,
                identity,
                shape,
            });
        } else {
            self.made.push((identity, None));
        }
    }

    pub(super) fn arrived(&mut self, serial: u32, arrival: Arrival<'_>) {
        let Some(at) = self
            .readbacks
            .iter()
            .position(|pending| pending.serial == serial)
        else {
            return;
        };
        let pending = self.readbacks.swap_remove(at);
        let input = match arrival {
            Arrival::Read(bytes) => Some(impostor_assembly_input(
                pending.shape,
                self.format,
                ROW,
                bytes,
            )),
            Arrival::Failed => None,
        };
        self.made.push((pending.identity, input));
    }

    pub(super) fn take_made(&mut self, out: &mut Vec<(ImpostorIdentity, Option<Vec<u8>>)>) {
        out.append(&mut self.made);
    }
}
