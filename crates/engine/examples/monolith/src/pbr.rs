use std::io::Read as _;
use std::sync::{Arc, Mutex};

use engine::prelude::*;
use engine_core::gpu_struct;
use flate2::read::DeflateDecoder;

pub const MAX_MATERIALS: usize = 32;
pub const NO_TEX_LAYER: u32 = u32::MAX;
pub const SPB2_MAGIC: [u8; 4] = *b"SPB2";
pub const SPB2_VERSION: u32 = 2;

gpu_struct! {
    pub struct GpuMaterialParams {
        pub base_color_rough: [f32; 4],
        pub emissive_metal: [f32; 4],
        pub layer_flags: [u32; 4],
    }
}

impl GpuMaterialParams {
    #[must_use]
    pub const fn fallback() -> Self {
        Self {
            base_color_rough: [0.24, 0.22, 0.20, 0.80],
            emissive_metal: [0.0, 0.0, 0.0, 0.0],
            layer_flags: [NO_TEX_LAYER, 0, 3, 0],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PbrChannel {
    Albedo,
    Normal,
    Orm,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PbrTier {
    Mid,
    Hi,
}

#[derive(Clone, Copy, Debug)]
pub struct PbrHdRequest {
    pub layer_idx: u32,
    pub tier: PbrTier,
    pub byte_offset: u64,
    pub byte_len: u32,
}

pub struct PbrMipUpload {
    pub channel: PbrChannel,
    pub layer_idx: u32,
    pub mip_level: u32,
    pub dim: u32,
    pub bc_bytes: Vec<u8>,
}

pub struct PbrUploadQueue {
    pub albedo_dim: u32,
    pub normal_dim: u32,
    pub orm_dim: u32,
    pub layer_count: u32,
    pub pending_mips: Vec<PbrMipUpload>,
    pub dirty_materials: Option<Vec<GpuMaterialParams>>,
}

impl Default for PbrUploadQueue {
    fn default() -> Self {
        Self {
            albedo_dim: 4,
            normal_dim: 4,
            orm_dim: 4,
            layer_count: 2,
            pending_mips: Vec::new(),
            dirty_materials: Some(vec![GpuMaterialParams::fallback(); MAX_MATERIALS]),
        }
    }
}

#[derive(Resource, Clone, Default)]
pub struct PbrUploadBridge(pub Arc<Mutex<PbrUploadQueue>>);

#[derive(Clone, Copy)]
struct BaseLayerSpan {
    bc_off: usize,
    bc_len: usize,
    nm_off: usize,
    nm_len: usize,
    orm_off: usize,
    orm_len: usize,
}

#[derive(Clone, Copy)]
struct HdLayerSpan {
    mid_off: u64,
    mid_len: u32,
    hi_off: u64,
    hi_len: u32,
}

#[derive(Resource)]
pub struct PbrTexturePack {
    base_bytes: Option<Vec<u8>>,
    pub materials: Vec<GpuMaterialParams>,
    target_layers: Vec<u32>,
    base_spans: Vec<BaseLayerSpan>,
    hd_spans: Vec<HdLayerSpan>,
    pub albedo_dim: u32,
    pub normal_dim: u32,
    pub orm_dim: u32,
    pub base_dim: u32,
    pub layer_count: u32,
    next_base_layer: usize,
    hd_order: Vec<usize>,
    next_mid_req: usize,
    next_hi_req: usize,
    retry_hd: Vec<PbrHdRequest>,
    hd_failures: Vec<u8>,
    pending_uploads: Vec<PbrMipUpload>,
    pub materials_dirty: bool,
    pub pack_loaded: bool,
    pub base_done: bool,
    pub uploaded_base_layers: u32,
    pub uploaded_mid_layers: u32,
    pub uploaded_hi_layers: u32,
}

impl Default for PbrTexturePack {
    fn default() -> Self {
        Self {
            base_bytes: None,
            materials: vec![GpuMaterialParams::fallback(); MAX_MATERIALS],
            target_layers: vec![NO_TEX_LAYER; MAX_MATERIALS],
            base_spans: Vec::new(),
            hd_spans: Vec::new(),
            albedo_dim: 4,
            normal_dim: 4,
            orm_dim: 4,
            base_dim: 4,
            layer_count: 2,
            next_base_layer: 0,
            hd_order: Vec::new(),
            next_mid_req: 0,
            next_hi_req: 0,
            retry_hd: Vec::new(),
            hd_failures: Vec::new(),
            pending_uploads: Vec::new(),
            materials_dirty: true,
            pack_loaded: false,
            base_done: false,
            uploaded_base_layers: 0,
            uploaded_mid_layers: 0,
            uploaded_hi_layers: 0,
        }
    }
}

#[must_use]
pub fn mip_count_for(dim: u32) -> u32 {
    if dim == 0 {
        1
    } else {
        32 - dim.leading_zeros()
    }
}

#[must_use]
pub fn bc_level_bytes(dim: u32) -> usize {
    let blocks = (dim.max(1) as usize).div_ceil(4);
    blocks * blocks * 16
}

fn read_u32_le(bytes: &[u8], off: usize) -> Option<u32> {
    let slice: [u8; 4] = bytes.get(off..off + 4)?.try_into().ok()?;
    Some(u32::from_le_bytes(slice))
}

fn read_u64_le(bytes: &[u8], off: usize) -> Option<u64> {
    let slice: [u8; 8] = bytes.get(off..off + 8)?.try_into().ok()?;
    Some(u64::from_le_bytes(slice))
}

fn read_f32_le(bytes: &[u8], off: usize) -> Option<f32> {
    Some(f32::from_bits(read_u32_le(bytes, off)?))
}

fn inflate_raw(slice: &[u8], expected_len: usize) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(expected_len);
    let mut dec = DeflateDecoder::new(slice);
    dec.read_to_end(&mut out).ok()?;
    if out.len() != expected_len {
        return None;
    }
    Some(out)
}

fn tail_total_bytes(base_dim: u32) -> usize {
    let mut sum = 0_usize;
    let mut d = base_dim.max(1);
    loop {
        sum += bc_level_bytes(d);
        if d <= 1 {
            break;
        }
        d /= 2;
    }
    sum
}

fn split_tail_mips(
    raw: &[u8],
    channel: PbrChannel,
    layer_idx: u32,
    full_dim: u32,
    base_dim: u32,
    out: &mut Vec<PbrMipUpload>,
) -> bool {
    let start_mip = mip_count_for(full_dim).saturating_sub(mip_count_for(base_dim));
    let mut cursor = 0_usize;
    let mut d = base_dim.max(1);
    let mut mip = start_mip;
    loop {
        let len = bc_level_bytes(d);
        let Some(slice) = raw.get(cursor..cursor + len) else {
            return false;
        };
        out.push(PbrMipUpload {
            channel,
            layer_idx,
            mip_level: mip,
            dim: d,
            bc_bytes: slice.to_vec(),
        });
        cursor += len;
        if d <= 1 {
            break;
        }
        d /= 2;
        mip += 1;
    }
    cursor == raw.len()
}

fn parse_materials(
    bytes: &[u8],
    mat_count: usize,
) -> Option<(Vec<GpuMaterialParams>, Vec<u32>)> {
    let mut mats = vec![GpuMaterialParams::fallback(); MAX_MATERIALS];
    let mut targets = vec![NO_TEX_LAYER; MAX_MATERIALS];
    for i in 0..mat_count {
        let off = 32 + i * 48;
        let base_color_rough = [
            read_f32_le(bytes, off)?,
            read_f32_le(bytes, off + 4)?,
            read_f32_le(bytes, off + 8)?,
            read_f32_le(bytes, off + 12)?,
        ];
        let emissive_metal = [
            read_f32_le(bytes, off + 16)?,
            read_f32_le(bytes, off + 20)?,
            read_f32_le(bytes, off + 24)?,
            read_f32_le(bytes, off + 28)?,
        ];
        let layer = read_u32_le(bytes, off + 32)?;
        let flags = read_u32_le(bytes, off + 36)?;
        let min_mip = read_u32_le(bytes, off + 40)?;
        targets[i] = layer;
        mats[i] = GpuMaterialParams {
            base_color_rough,
            emissive_metal,
            layer_flags: [NO_TEX_LAYER, flags, min_mip, 0],
        };
    }
    Some((mats, targets))
}

fn build_hd_layer_order(
    layer_count: usize,
    materials: &[GpuMaterialParams],
    targets: &[u32],
) -> Vec<usize> {
    let mut is_curtain = vec![false; layer_count];
    let mut is_hero = vec![false; layer_count];
    for (mat, &target) in materials.iter().zip(targets.iter()) {
        let idx = target as usize;
        if idx < layer_count {
            let flags = mat.layer_flags[1];
            if (flags & 2) != 0 {
                is_curtain[idx] = true;
            }
            if (flags & 6) != 0 {
                is_hero[idx] = true;
            }
        }
    }
    let mut order = Vec::with_capacity(layer_count);
    for (idx, &curtain) in is_curtain.iter().enumerate() {
        if curtain {
            order.push(idx);
        }
    }
    for idx in 0..layer_count {
        if !is_curtain[idx] && (is_hero[idx] || idx >= 24) {
            order.push(idx);
        }
    }
    for idx in 0..layer_count {
        if !is_curtain[idx] && !is_hero[idx] && idx < 24 {
            order.push(idx);
        }
    }
    order
}

impl PbrTexturePack {
    pub fn load_base_bytes(&mut self, bytes: Vec<u8>) -> bool {
        if bytes.len() < 32 || bytes[0..4] != SPB2_MAGIC {
            return false;
        }
        let (
            Some(version),
            Some(mat_count),
            Some(layer_count),
            Some(albedo_dim),
            Some(normal_dim),
            Some(orm_dim),
            Some(base_dim),
        ) = (
            read_u32_le(&bytes, 4),
            read_u32_le(&bytes, 8),
            read_u32_le(&bytes, 12),
            read_u32_le(&bytes, 16),
            read_u32_le(&bytes, 20),
            read_u32_le(&bytes, 24),
            read_u32_le(&bytes, 28),
        )
        else {
            return false;
        };
        if version != SPB2_VERSION
            || mat_count == 0
            || mat_count as usize > MAX_MATERIALS
            || layer_count == 0
            || base_dim == 0
            || albedo_dim < base_dim
            || normal_dim < base_dim
            || orm_dim < base_dim
        {
            return false;
        }
        let mat_table_end = 32 + (mat_count as usize) * 48;
        let base_table_end = mat_table_end + (layer_count as usize) * 24;
        let hd_table_end = base_table_end + (layer_count as usize) * 32;
        if bytes.len() < hd_table_end {
            return false;
        }
        let Some((mats, targets)) = parse_materials(&bytes, mat_count as usize) else {
            return false;
        };
        let mut base_spans = Vec::with_capacity(layer_count as usize);
        for i in 0..layer_count as usize {
            let off = mat_table_end + i * 24;
            let (
                Some(bc_off),
                Some(bc_len),
                Some(nm_off),
                Some(nm_len),
                Some(orm_off),
                Some(orm_len),
            ) = (
                read_u32_le(&bytes, off),
                read_u32_le(&bytes, off + 4),
                read_u32_le(&bytes, off + 8),
                read_u32_le(&bytes, off + 12),
                read_u32_le(&bytes, off + 16),
                read_u32_le(&bytes, off + 20),
            )
            else {
                return false;
            };
            base_spans.push(BaseLayerSpan {
                bc_off: bc_off as usize,
                bc_len: bc_len as usize,
                nm_off: nm_off as usize,
                nm_len: nm_len as usize,
                orm_off: orm_off as usize,
                orm_len: orm_len as usize,
            });
        }
        let mut hd_spans = Vec::with_capacity(layer_count as usize);
        for i in 0..layer_count as usize {
            let off = base_table_end + i * 32;
            let (Some(mid_off), Some(mid_len), Some(hi_off), Some(hi_len)) = (
                read_u64_le(&bytes, off),
                read_u32_le(&bytes, off + 8),
                read_u64_le(&bytes, off + 16),
                read_u32_le(&bytes, off + 24),
            ) else {
                return false;
            };
            hd_spans.push(HdLayerSpan {
                mid_off,
                mid_len,
                hi_off,
                hi_len,
            });
        }
        let hd_order = build_hd_layer_order(layer_count as usize, &mats, &targets);
        self.base_bytes = Some(bytes);
        self.materials = mats;
        self.target_layers = targets;
        self.base_spans = base_spans;
        self.hd_spans = hd_spans;
        self.hd_order = hd_order;
        self.albedo_dim = albedo_dim;
        self.normal_dim = normal_dim;
        self.orm_dim = orm_dim;
        self.base_dim = base_dim;
        self.layer_count = layer_count.max(2);
        self.next_base_layer = 0;
        self.next_mid_req = 0;
        self.next_hi_req = 0;
        self.retry_hd.clear();
        self.hd_failures = vec![0; (layer_count as usize) * 2];
        self.pending_uploads.clear();
        self.uploaded_base_layers = 0;
        self.uploaded_mid_layers = 0;
        self.uploaded_hi_layers = 0;
        self.materials_dirty = true;
        self.pack_loaded = true;
        self.base_done = false;
        true
    }

    fn decode_next_base_layer(&mut self) -> bool {
        let idx = self.next_base_layer;
        if idx >= self.base_spans.len() {
            self.base_done = true;
            self.base_bytes = None;
            return false;
        }
        let span = self.base_spans[idx];
        let Some(raw) = self.base_bytes.as_deref() else {
            return false;
        };
        let expected = tail_total_bytes(self.base_dim);
        let (Some(bc_raw), Some(nm_raw), Some(orm_raw)) = (
            raw.get(span.bc_off..span.bc_off + span.bc_len)
                .and_then(|s| inflate_raw(s, expected)),
            raw.get(span.nm_off..span.nm_off + span.nm_len)
                .and_then(|s| inflate_raw(s, expected)),
            raw.get(span.orm_off..span.orm_off + span.orm_len)
                .and_then(|s| inflate_raw(s, expected)),
        ) else {
            self.next_base_layer = idx + 1;
            return true;
        };
        let layer_u32 = idx as u32;
        let ok = split_tail_mips(
            &bc_raw,
            PbrChannel::Albedo,
            layer_u32,
            self.albedo_dim,
            self.base_dim,
            &mut self.pending_uploads,
        ) && split_tail_mips(
            &nm_raw,
            PbrChannel::Normal,
            layer_u32,
            self.normal_dim,
            self.base_dim,
            &mut self.pending_uploads,
        ) && split_tail_mips(
            &orm_raw,
            PbrChannel::Orm,
            layer_u32,
            self.orm_dim,
            self.base_dim,
            &mut self.pending_uploads,
        );
        self.next_base_layer = idx + 1;
        if ok {
            self.uploaded_base_layers = self.uploaded_base_layers.saturating_add(1);
            let base_mip = mip_count_for(self.albedo_dim).saturating_sub(mip_count_for(self.base_dim));
            for (mat, &target) in self.materials.iter_mut().zip(self.target_layers.iter()) {
                if target == layer_u32 {
                    mat.layer_flags[0] = layer_u32;
                    mat.layer_flags[2] = mat.layer_flags[2].min(base_mip);
                    self.materials_dirty = true;
                }
            }
        }
        if self.next_base_layer >= self.base_spans.len() {
            self.base_done = true;
            self.base_bytes = None;
        }
        true
    }

    pub fn fail_hd_request(&mut self, req: PbrHdRequest) {
        let slot = (req.layer_idx as usize) * 2 + usize::from(req.tier == PbrTier::Hi);
        if let Some(count) = self.hd_failures.get_mut(slot) {
            *count = count.saturating_add(1);
            if *count <= 3 {
                self.retry_hd.push(req);
            }
        }
    }

    pub fn take_hd_requests(&mut self, max_count: usize) -> Vec<PbrHdRequest> {
        if !self.pack_loaded || !self.base_done || max_count == 0 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(max_count);
        let take_retry = max_count.min(self.retry_hd.len());
        out.extend(self.retry_hd.drain(..take_retry));
        while out.len() < max_count && self.next_mid_req < self.hd_order.len() {
            let idx = self.hd_order[self.next_mid_req];
            self.next_mid_req += 1;
            let span = self.hd_spans[idx];
            if span.mid_len > 0 {
                out.push(PbrHdRequest {
                    layer_idx: idx as u32,
                    tier: PbrTier::Mid,
                    byte_offset: span.mid_off,
                    byte_len: span.mid_len,
                });
            }
        }
        if self.next_mid_req >= self.hd_order.len() {
            while out.len() < max_count && self.next_hi_req < self.hd_order.len() {
                let idx = self.hd_order[self.next_hi_req];
                self.next_hi_req += 1;
                let span = self.hd_spans[idx];
                if span.hi_len > 0 {
                    out.push(PbrHdRequest {
                        layer_idx: idx as u32,
                        tier: PbrTier::Hi,
                        byte_offset: span.hi_off,
                        byte_len: span.hi_len,
                    });
                }
            }
        }
        out
    }

    pub fn deliver_hd_blob(&mut self, req: PbrHdRequest, compressed: &[u8]) -> bool {
        let d512 = self.albedo_dim / 2;
        let d256 = self.albedo_dim / 4;
        let s512 = bc_level_bytes(d512);
        let s256 = bc_level_bytes(d256);
        let s1024 = bc_level_bytes(self.albedo_dim);
        match req.tier {
            PbrTier::Mid => {
                let expected = (s512 + s256) * 3;
                let Some(raw) = inflate_raw(compressed, expected) else {
                    return false;
                };
                let mut cur = 0_usize;
                let specs = [
                    (PbrChannel::Albedo, 1_u32, d512, s512),
                    (PbrChannel::Albedo, 2_u32, d256, s256),
                    (PbrChannel::Normal, 1_u32, d512, s512),
                    (PbrChannel::Normal, 2_u32, d256, s256),
                    (PbrChannel::Orm, 0_u32, self.orm_dim, s512),
                    (PbrChannel::Orm, 1_u32, self.orm_dim / 2, s256),
                ];
                for (channel, mip_level, dim, len) in specs {
                    self.pending_uploads.push(PbrMipUpload {
                        channel,
                        layer_idx: req.layer_idx,
                        mip_level,
                        dim,
                        bc_bytes: raw[cur..cur + len].to_vec(),
                    });
                    cur += len;
                }
                self.uploaded_mid_layers = self.uploaded_mid_layers.saturating_add(1);
                for (mat, &target) in self.materials.iter_mut().zip(self.target_layers.iter()) {
                    if target == req.layer_idx {
                        mat.layer_flags[2] = mat.layer_flags[2].min(1);
                        self.materials_dirty = true;
                    }
                }
                true
            }
            PbrTier::Hi => {
                let expected = s1024 * 2;
                let Some(raw) = inflate_raw(compressed, expected) else {
                    return false;
                };
                self.pending_uploads.push(PbrMipUpload {
                    channel: PbrChannel::Albedo,
                    layer_idx: req.layer_idx,
                    mip_level: 0,
                    dim: self.albedo_dim,
                    bc_bytes: raw[..s1024].to_vec(),
                });
                self.pending_uploads.push(PbrMipUpload {
                    channel: PbrChannel::Normal,
                    layer_idx: req.layer_idx,
                    mip_level: 0,
                    dim: self.normal_dim,
                    bc_bytes: raw[s1024..s1024 * 2].to_vec(),
                });
                self.uploaded_hi_layers = self.uploaded_hi_layers.saturating_add(1);
                for (mat, &target) in self.materials.iter_mut().zip(self.target_layers.iter()) {
                    if target == req.layer_idx {
                        mat.layer_flags[2] = 0;
                        self.materials_dirty = true;
                    }
                }
                true
            }
        }
    }

    pub fn pump_uploads(&mut self, bridge: &PbrUploadBridge, max_base_layers: usize) {
        if !self.pack_loaded {
            return;
        }
        for _ in 0..max_base_layers {
            if !self.decode_next_base_layer() {
                break;
            }
        }
        if self.pending_uploads.is_empty() && !self.materials_dirty {
            return;
        }
        if let Ok(mut queue) = bridge.0.lock() {
            queue.albedo_dim = self.albedo_dim;
            queue.normal_dim = self.normal_dim;
            queue.orm_dim = self.orm_dim;
            queue.layer_count = self.layer_count;
            queue.pending_mips.append(&mut self.pending_uploads);
            if self.materials_dirty {
                queue.dirty_materials = Some(self.materials.clone());
                self.materials_dirty = false;
            }
        }
    }
}
