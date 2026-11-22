use std::io::Cursor;
use std::sync::{Arc, Mutex};

use engine::math;
use engine::prelude::*;
use engine_core::gpu_struct;
use zune_jpeg::JpegDecoder;

pub const MAX_MATERIALS: usize = 32;
pub const NO_TEX_LAYER: u32 = u32::MAX;

gpu_struct! {
    pub struct GpuMaterialParams {
        pub base_color_rough: [f32; 4],
        pub emissive_metal: [f32; 4],
        pub layer_flags: [u32; 4],
    }
}

pub struct PbrUploadQueue {
    pub albedo_dim: u32,
    pub detail_dim: u32,
    pub layer_count: u32,
    pub pending_layers: Vec<DecodedLayer>,
    pub dirty_materials: Option<Vec<GpuMaterialParams>>,
}

impl Default for PbrUploadQueue {
    fn default() -> Self {
        Self {
            albedo_dim: 2,
            detail_dim: 2,
            layer_count: 2,
            pending_layers: Vec::new(),
            dirty_materials: Some(vec![GpuMaterialParams::fallback(); MAX_MATERIALS]),
        }
    }
}

#[derive(Resource, Clone, Default)]
pub struct PbrUploadBridge(pub Arc<Mutex<PbrUploadQueue>>);

impl GpuMaterialParams {
    #[must_use]
    pub const fn fallback() -> Self {
        Self {
            base_color_rough: [0.24, 0.22, 0.20, 0.80],
            emissive_metal: [0.0, 0.0, 0.0, 0.0],
            layer_flags: [NO_TEX_LAYER, 0, 0, 0],
        }
    }
}

#[derive(Clone, Copy)]
struct LayerBlobs {
    bc_off: usize,
    bc_len: usize,
    nm_off: usize,
    nm_len: usize,
    mr_off: usize,
    mr_len: usize,
}

pub struct DecodedMip {
    pub dim: u32,
    pub rgba: Vec<u8>,
}

pub struct DecodedLayer {
    pub layer_idx: u32,
    pub albedo_mips: Vec<DecodedMip>,
    pub normal_mips: Vec<DecodedMip>,
    pub orm_mips: Vec<DecodedMip>,
}

#[derive(Resource)]
pub struct PbrTexturePack {
    raw_bytes: Option<Vec<u8>>,
    pub materials: Vec<GpuMaterialParams>,
    target_layers: Vec<u32>,
    layer_blobs: Vec<LayerBlobs>,
    pub albedo_dim: u32,
    pub detail_dim: u32,
    pub layer_count: u32,
    next_layer: usize,
    pub materials_dirty: bool,
    pub pack_loaded: bool,
    pub uploaded_layers: u32,
}

impl Default for PbrTexturePack {
    fn default() -> Self {
        Self {
            raw_bytes: None,
            materials: vec![GpuMaterialParams::fallback(); MAX_MATERIALS],
            target_layers: vec![NO_TEX_LAYER; MAX_MATERIALS],
            layer_blobs: Vec::new(),
            albedo_dim: 2,
            detail_dim: 2,
            layer_count: 2,
            next_layer: 0,
            materials_dirty: true,
            pack_loaded: false,
            uploaded_layers: 0,
        }
    }
}

fn read_u32_le(bytes: &[u8], off: usize) -> Option<u32> {
    let slice: [u8; 4] = bytes.get(off..off + 4)?.try_into().ok()?;
    Some(u32::from_le_bytes(slice))
}

fn read_f32_le(bytes: &[u8], off: usize) -> Option<f32> {
    Some(f32::from_bits(read_u32_le(bytes, off)?))
}

fn srgb_to_linear_u8(v: u8) -> f32 {
    let x = f32::from(v) * (1.0 / 255.0);
    if x <= 0.04045 {
        x * (1.0 / 12.92)
    } else {
        ((x + 0.055) * (1.0 / 1.055)).powf(2.4)
    }
}

fn linear_to_srgb_u8(x: f32) -> u8 {
    let c = math::clamp(x, 0.0, 1.0);
    let s = if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (math::clamp(s, 0.0, 1.0) * 255.0 + 0.5) as u8
}

fn decode_jpeg_rgba(slice: &[u8], expected_dim: u32) -> Option<Vec<u8>> {
    let mut decoder = JpegDecoder::new(Cursor::new(slice));
    let rgb = decoder.decode().ok()?;
    let info = decoder.info()?;
    let w = u32::from(info.width);
    let h = u32::from(info.height);
    if w != expected_dim || h != expected_dim {
        return None;
    }
    let px = (w as usize) * (h as usize);
    let mut rgba = vec![255_u8; px * 4];
    if rgb.len() == px * 3 {
        for i in 0..px {
            rgba[i * 4] = rgb[i * 3];
            rgba[i * 4 + 1] = rgb[i * 3 + 1];
            rgba[i * 4 + 2] = rgb[i * 3 + 2];
        }
    } else if rgb.len() == px {
        for i in 0..px {
            let v = rgb[i];
            rgba[i * 4] = v;
            rgba[i * 4 + 1] = v;
            rgba[i * 4 + 2] = v;
        }
    } else if rgb.len() == px * 4 {
        rgba.copy_from_slice(&rgb);
    } else {
        return None;
    }
    Some(rgba)
}

fn downsample_srgb(src: &[u8], src_dim: u32, lut: &[f32; 256]) -> Vec<u8> {
    let dst_dim = (src_dim / 2).max(1) as usize;
    let sw = src_dim as usize;
    let mut dst = vec![255_u8; dst_dim * dst_dim * 4];
    for y in 0..dst_dim {
        let sy = (y * 2).min(sw - 1);
        let sy1 = (sy + 1).min(sw - 1);
        for x in 0..dst_dim {
            let sx = (x * 2).min(sw - 1);
            let sx1 = (sx + 1).min(sw - 1);
            let i00 = (sy * sw + sx) * 4;
            let i10 = (sy * sw + sx1) * 4;
            let i01 = (sy1 * sw + sx) * 4;
            let i11 = (sy1 * sw + sx1) * 4;
            let o = (y * dst_dim + x) * 4;
            for c in 0..3 {
                let avg = 0.25
                    * (lut[usize::from(src[i00 + c])]
                        + lut[usize::from(src[i10 + c])]
                        + lut[usize::from(src[i01 + c])]
                        + lut[usize::from(src[i11 + c])]);
                dst[o + c] = linear_to_srgb_u8(avg);
            }
        }
    }
    dst
}

fn downsample_normal(src: &[u8], src_dim: u32) -> Vec<u8> {
    let dst_dim = (src_dim / 2).max(1) as usize;
    let sw = src_dim as usize;
    let mut dst = vec![255_u8; dst_dim * dst_dim * 4];
    for y in 0..dst_dim {
        let sy = (y * 2).min(sw - 1);
        let sy1 = (sy + 1).min(sw - 1);
        for x in 0..dst_dim {
            let sx = (x * 2).min(sw - 1);
            let sx1 = (sx + 1).min(sw - 1);
            let idxs = [
                (sy * sw + sx) * 4,
                (sy * sw + sx1) * 4,
                (sy1 * sw + sx) * 4,
                (sy1 * sw + sx1) * 4,
            ];
            let mut nx = 0.0_f32;
            let mut ny = 0.0_f32;
            let mut nz = 0.0_f32;
            for i in idxs {
                nx += f32::from(src[i]) * (2.0 / 255.0) - 1.0;
                ny += f32::from(src[i + 1]) * (2.0 / 255.0) - 1.0;
                nz += f32::from(src[i + 2]) * (2.0 / 255.0) - 1.0;
            }
            let l2 = nx * nx + ny * ny + nz * nz;
            let (ux, uy, uz) = if l2 > 1.0e-6 {
                let inv = 1.0 / l2.sqrt();
                (nx * inv, ny * inv, nz * inv)
            } else {
                (0.0, 0.0, 1.0)
            };
            let o = (y * dst_dim + x) * 4;
            dst[o] = (math::clamp(ux * 0.5 + 0.5, 0.0, 1.0) * 255.0 + 0.5) as u8;
            dst[o + 1] = (math::clamp(uy * 0.5 + 0.5, 0.0, 1.0) * 255.0 + 0.5) as u8;
            dst[o + 2] = (math::clamp(uz * 0.5 + 0.5, 0.0, 1.0) * 255.0 + 0.5) as u8;
        }
    }
    dst
}

fn downsample_linear(src: &[u8], src_dim: u32) -> Vec<u8> {
    let dst_dim = (src_dim / 2).max(1) as usize;
    let sw = src_dim as usize;
    let mut dst = vec![255_u8; dst_dim * dst_dim * 4];
    for y in 0..dst_dim {
        let sy = (y * 2).min(sw - 1);
        let sy1 = (sy + 1).min(sw - 1);
        for x in 0..dst_dim {
            let sx = (x * 2).min(sw - 1);
            let sx1 = (sx + 1).min(sw - 1);
            let i00 = (sy * sw + sx) * 4;
            let i10 = (sy * sw + sx1) * 4;
            let i01 = (sy1 * sw + sx) * 4;
            let i11 = (sy1 * sw + sx1) * 4;
            let o = (y * dst_dim + x) * 4;
            for c in 0..3 {
                let sum = u32::from(src[i00 + c])
                    + u32::from(src[i10 + c])
                    + u32::from(src[i01 + c])
                    + u32::from(src[i11 + c]);
                dst[o + c] = ((sum + 2) >> 2) as u8;
            }
        }
    }
    dst
}

fn build_mips(base: Vec<u8>, base_dim: u32, mode: u8) -> Vec<DecodedMip> {
    let levels = mip_count_for(base_dim) as usize;
    let mut out = Vec::with_capacity(levels);
    let mut lut = [0.0_f32; 256];
    if mode == 1 {
        for (i, slot) in lut.iter_mut().enumerate() {
            *slot = srgb_to_linear_u8(i as u8);
        }
    }
    out.push(DecodedMip {
        dim: base_dim,
        rgba: base,
    });
    let mut cur_dim = base_dim;
    while cur_dim > 1 {
        let next = {
            let prev = &out[out.len() - 1].rgba;
            match mode {
                1 => downsample_srgb(prev, cur_dim, &lut),
                2 => downsample_normal(prev, cur_dim),
                _ => downsample_linear(prev, cur_dim),
            }
        };
        cur_dim = (cur_dim / 2).max(1);
        out.push(DecodedMip {
            dim: cur_dim,
            rgba: next,
        });
    }
    out
}

#[must_use]
pub fn mip_count_for(dim: u32) -> u32 {
    if dim == 0 {
        1
    } else {
        32 - dim.leading_zeros()
    }
}

impl PbrTexturePack {
    pub fn load_bytes(&mut self, bytes: Vec<u8>) -> bool {
        if bytes.len() < 24 || &bytes[0..4] != b"SPBR" {
            return false;
        }
        let Some(version) = read_u32_le(&bytes, 4) else {
            return false;
        };
        let Some(mat_count) = read_u32_le(&bytes, 8) else {
            return false;
        };
        let Some(layer_count) = read_u32_le(&bytes, 12) else {
            return false;
        };
        let Some(albedo_dim) = read_u32_le(&bytes, 16) else {
            return false;
        };
        let Some(detail_dim) = read_u32_le(&bytes, 20) else {
            return false;
        };
        if version != 1 || mat_count == 0 || mat_count as usize > MAX_MATERIALS || layer_count == 0 {
            return false;
        }
        let mat_table_end = 24 + (mat_count as usize) * 48;
        let layer_table_end = mat_table_end + (layer_count as usize) * 24;
        if bytes.len() < layer_table_end {
            return false;
        }
        let mut mats = vec![GpuMaterialParams::fallback(); MAX_MATERIALS];
        let mut targets = vec![NO_TEX_LAYER; MAX_MATERIALS];
        for i in 0..mat_count as usize {
            let off = 24 + i * 48;
            let (Some(r), Some(g), Some(b), Some(rough)) = (
                read_f32_le(&bytes, off),
                read_f32_le(&bytes, off + 4),
                read_f32_le(&bytes, off + 8),
                read_f32_le(&bytes, off + 12),
            ) else {
                return false;
            };
            let (Some(er), Some(eg), Some(eb), Some(metal)) = (
                read_f32_le(&bytes, off + 16),
                read_f32_le(&bytes, off + 20),
                read_f32_le(&bytes, off + 24),
                read_f32_le(&bytes, off + 28),
            ) else {
                return false;
            };
            let (Some(layer), Some(flags)) =
                (read_u32_le(&bytes, off + 32), read_u32_le(&bytes, off + 36))
            else {
                return false;
            };
            targets[i] = layer;
            mats[i] = GpuMaterialParams {
                base_color_rough: [r, g, b, rough],
                emissive_metal: [er, eg, eb, metal],
                layer_flags: [NO_TEX_LAYER, flags, 0, 0],
            };
        }
        let mut blobs = Vec::with_capacity(layer_count as usize);
        for i in 0..layer_count as usize {
            let off = mat_table_end + i * 24;
            let (Some(bc_off), Some(bc_len), Some(nm_off), Some(nm_len), Some(mr_off), Some(mr_len)) = (
                read_u32_le(&bytes, off),
                read_u32_le(&bytes, off + 4),
                read_u32_le(&bytes, off + 8),
                read_u32_le(&bytes, off + 12),
                read_u32_le(&bytes, off + 16),
                read_u32_le(&bytes, off + 20),
            ) else {
                return false;
            };
            blobs.push(LayerBlobs {
                bc_off: bc_off as usize,
                bc_len: bc_len as usize,
                nm_off: nm_off as usize,
                nm_len: nm_len as usize,
                mr_off: mr_off as usize,
                mr_len: mr_len as usize,
            });
        }
        self.raw_bytes = Some(bytes);
        self.materials = mats;
        self.target_layers = targets;
        self.layer_blobs = blobs;
        self.albedo_dim = albedo_dim;
        self.detail_dim = detail_dim;
        self.layer_count = layer_count.max(2);
        self.next_layer = 0;
        self.uploaded_layers = 0;
        self.materials_dirty = true;
        self.pack_loaded = true;
        true
    }

    pub fn decode_next_layer(&mut self) -> Option<DecodedLayer> {
        let idx = self.next_layer;
        if idx >= self.layer_blobs.len() {
            return None;
        }
        let blob = self.layer_blobs[idx];
        let raw = self.raw_bytes.as_deref()?;
        let bc_slice = raw.get(blob.bc_off..blob.bc_off + blob.bc_len)?;
        let nm_slice = raw.get(blob.nm_off..blob.nm_off + blob.nm_len)?;
        let mr_slice = raw.get(blob.mr_off..blob.mr_off + blob.mr_len)?;
        let bc_rgba = decode_jpeg_rgba(bc_slice, self.albedo_dim)?;
        let nm_rgba = decode_jpeg_rgba(nm_slice, self.detail_dim)?;
        let mr_rgba = decode_jpeg_rgba(mr_slice, self.detail_dim)?;
        let albedo_mips = build_mips(bc_rgba, self.albedo_dim, 1);
        let normal_mips = build_mips(nm_rgba, self.detail_dim, 2);
        let orm_mips = build_mips(mr_rgba, self.detail_dim, 0);

        self.next_layer = idx + 1;
        self.uploaded_layers = self.uploaded_layers.saturating_add(1);
        let layer_u32 = idx as u32;
        for (mat, &target) in self.materials.iter_mut().zip(self.target_layers.iter()) {
            if target == layer_u32 {
                mat.layer_flags[0] = layer_u32;
                self.materials_dirty = true;
            }
        }
        if self.next_layer >= self.layer_blobs.len() {
            self.raw_bytes = None;
        }
        Some(DecodedLayer {
            layer_idx: layer_u32,
            albedo_mips,
            normal_mips,
            orm_mips,
        })
    }

    pub fn pump_uploads(&mut self, bridge: &PbrUploadBridge, max_layers: usize) {
        if !self.pack_loaded {
            return;
        }
        let mut decoded = Vec::new();
        for _ in 0..max_layers {
            if let Some(layer) = self.decode_next_layer() {
                decoded.push(layer);
            } else {
                break;
            }
        }
        if decoded.is_empty() && !self.materials_dirty {
            return;
        }
        if let Ok(mut queue) = bridge.0.lock() {
            queue.albedo_dim = self.albedo_dim;
            queue.detail_dim = self.detail_dim;
            queue.layer_count = self.layer_count;
            queue.pending_layers.extend(decoded);
            if self.materials_dirty {
                queue.dirty_materials = Some(self.materials.clone());
                self.materials_dirty = false;
            }
        }
    }
}
