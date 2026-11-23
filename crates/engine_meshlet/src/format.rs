use std::mem::size_of;

use glam::{Vec2, Vec3};
use serde::Serialize;

pub const MAX_CLUSTER_VERTS: usize = 128;
pub const MAX_CLUSTER_TRIS: usize = 128;
pub const PAGE_SIZE_BYTES: usize = 16_384;
pub const PAGE_WORDS: usize = PAGE_SIZE_BYTES / 4;
pub const BVH_BRANCHING: usize = 4;
pub const CLUSTER_HEADER_WORDS: usize = 7;
pub const ROOT_PARENT_ERROR: f32 = 1.0e30;
pub const NO_BVH_CHILD: u32 = u32::MAX;
pub const MANIFEST_MAGIC: [u8; 4] = *b"MLM1";
pub const MANIFEST_VERSION: u32 = 5;
pub const FLAG_ROOT_CLUSTER: u32 = 1;
pub const FLAG_IN_ROOT_PAGE: u32 = 2;
pub const FLAG_PINNED_COARSE: u32 = 4;
pub const FLAG_HAS_VERTEX_COLOR: u32 = 8;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PageSpan {
    pub byte_offset: u64,
    pub byte_len: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ClusterHeader {
    pub bound_sphere: [f32; 4],
    pub lod_sphere: [f32; 4],
    pub parent_lod_sphere: [f32; 4],
    pub errors_and_cone: [u32; 4],
    pub page_and_fallback: [u32; 4],
}

const _: () = assert!(
    size_of::<ClusterHeader>() == 80,
    "ClusterHeader must be 80 bytes"
);

impl ClusterHeader {
    pub fn pack_meta(vert_count: usize, tri_count: usize, lod_level: u32, material_id: u8) -> u32 {
        ((vert_count as u32) & 0xFF)
            | (((tri_count as u32) & 0xFF) << 8)
            | ((lod_level & 0xFF) << 16)
            | (u32::from(material_id) << 24)
    }

    pub fn pack_offset_flags(word_offset: u32, flags: u32) -> u32 {
        (word_offset & 0xFFFF) | ((flags & 0xFFFF) << 16)
    }

    pub fn pack_fallback(start: u32, count: u32) -> u32 {
        (start & 0x00FF_FFFF) | ((count & 0xFF) << 24)
    }

    pub fn vert_count(&self) -> u32 {
        self.errors_and_cone[3] & 0xFF
    }

    pub fn tri_count(&self) -> u32 {
        (self.errors_and_cone[3] >> 8) & 0xFF
    }

    pub fn lod_level(&self) -> u32 {
        (self.errors_and_cone[3] >> 16) & 0xFF
    }

    pub fn material_id(&self) -> u8 {
        ((self.errors_and_cone[3] >> 24) & 0xFF) as u8
    }

    pub fn self_error(&self) -> f32 {
        f32::from_bits(self.errors_and_cone[0])
    }

    pub fn set_self_error(&mut self, error: f32) {
        self.errors_and_cone[0] = error.to_bits();
    }

    pub fn parent_error(&self) -> f32 {
        f32::from_bits(self.errors_and_cone[1])
    }

    pub fn set_parent_error(&mut self, error: f32) {
        self.errors_and_cone[1] = error.to_bits();
    }

    pub fn page_id(&self) -> u32 {
        self.page_and_fallback[0]
    }

    pub fn set_page_location(&mut self, page_id: u32, word_offset: u32) {
        self.page_and_fallback[0] = page_id;
        let flags = self.flags();
        self.page_and_fallback[1] = Self::pack_offset_flags(word_offset, flags);
    }

    pub fn page_word_offset(&self) -> u32 {
        self.page_and_fallback[1] & 0xFFFF
    }

    pub fn flags(&self) -> u32 {
        (self.page_and_fallback[1] >> 16) & 0xFFFF
    }

    pub fn set_flags(&mut self, flags: u32) {
        let offset = self.page_word_offset();
        self.page_and_fallback[1] = Self::pack_offset_flags(offset, flags);
    }

    pub fn fallback_start(&self) -> u32 {
        self.page_and_fallback[2] & 0x00FF_FFFF
    }

    pub fn fallback_count(&self) -> u32 {
        self.page_and_fallback[2] >> 24
    }

    pub fn set_fallback(&mut self, start: u32, count: u32) {
        self.page_and_fallback[2] = Self::pack_fallback(start, count);
    }

    pub fn group_id(&self) -> u32 {
        self.page_and_fallback[3]
    }

    pub fn set_group_id(&mut self, group_id: u32) {
        self.page_and_fallback[3] = group_id;
    }

    pub fn word_count(&self) -> u32 {
        let vc = self.vert_count();
        let tc = self.tri_count();
        let pos_words = (vc * 3).div_ceil(2);
        let tri_words = (tc * 3).div_ceil(4);
        let norm_words = (vc * 3).div_ceil(4);
        let uv_words = (vc * 3).div_ceil(4);
        let col_words = if self.has_vertex_color() {
            vc.div_ceil(2)
        } else {
            0
        };
        (CLUSTER_HEADER_WORDS as u32) + pos_words + tri_words + norm_words + uv_words + col_words
    }

    pub fn is_root(&self) -> bool {
        (self.flags() & FLAG_ROOT_CLUSTER) != 0
    }

    pub fn is_in_root_page(&self) -> bool {
        (self.flags() & FLAG_IN_ROOT_PAGE) != 0
    }

    pub fn is_pinned_coarse(&self) -> bool {
        (self.flags() & FLAG_PINNED_COARSE) != 0
    }

    pub fn has_vertex_color(&self) -> bool {
        (self.flags() & FLAG_HAS_VERTEX_COLOR) != 0
    }

    pub fn normal_cone(&self) -> ([i8; 3], i8) {
        let word = self.errors_and_cone[2];
        let ax = (word & 0xFF) as u8 as i8;
        let ay = ((word >> 8) & 0xFF) as u8 as i8;
        let az = ((word >> 16) & 0xFF) as u8 as i8;
        let cutoff = ((word >> 24) & 0xFF) as u8 as i8;
        ([ax, ay, az], cutoff)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BvhNode {
    pub sphere: [f32; 4],
    pub child_or_cluster: [u32; 4],
    pub max_parent_error: [f32; 4],
    pub meta: [u32; 4],
}

const _: () = assert!(size_of::<BvhNode>() == 64, "BvhNode must be 64 bytes");

impl BvhNode {
    pub fn child_count(&self) -> u32 {
        self.meta[0]
    }

    pub fn is_leaf(&self) -> bool {
        self.meta[1] != 0
    }

    pub fn leaf_cluster_start(&self) -> u32 {
        self.meta[2]
    }

    pub fn leaf_cluster_count(&self) -> u32 {
        self.meta[3]
    }
}

#[derive(Clone, Debug, Default)]
pub struct SourceMesh {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<Vec2>,
    pub colors: Vec<[f32; 3]>,
    pub ao: Vec<f32>,
    pub indices: Vec<u32>,
    pub material_ids: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClusterVertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
    pub color: [f32; 3],
    pub ao: f32,
}

impl Default for ClusterVertex {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            normal: Vec3::Y,
            uv: Vec2::ZERO,
            color: [1.0, 1.0, 1.0],
            ao: 1.0,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct DecodedCluster {
    pub origin: Vec3,
    pub step: f32,
    pub vertices: Vec<ClusterVertex>,
    pub triangles: Vec<[u8; 3]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MeshletManifest {
    pub source_tri_count: u32,
    pub lod_levels: u32,
    pub root_page_count: u32,
    pub page_count: u32,
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    pub clusters: Vec<ClusterHeader>,
    pub bvh: Vec<BvhNode>,
    pub page_spans: Vec<PageSpan>,
}

#[derive(Clone, Debug)]
pub struct BuiltMeshlet {
    pub manifest: MeshletManifest,
    pub page_words: Vec<u32>,
    pub compressed_pages: Vec<u8>,
    pub stats: BuildStats,
}

impl BuiltMeshlet {
    pub fn page(&self, page_id: u32) -> Option<&[u32]> {
        let start = (page_id as usize).checked_mul(PAGE_WORDS)?;
        let end = start.checked_add(PAGE_WORDS)?;
        self.page_words.get(start..end)
    }

    pub fn pages_to_bytes(&self) -> Vec<u8> {
        self.compressed_pages.clone()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct BuildStats {
    pub source_tri_count: u32,
    pub leaf_tri_count: u32,
    pub total_cluster_tris: u32,
    pub total_cluster_verts: u32,
    pub cluster_count: u32,
    pub leaf_cluster_count: u32,
    pub root_cluster_count: u32,
    pub lod_levels: u32,
    pub bvh_node_count: u32,
    pub page_count: u32,
    pub root_page_count: u32,
    pub geometry_bytes: u64,
    pub manifest_bytes: u64,
    pub page_blob_bytes: u64,
    pub bytes_per_cluster_tri: f64,
    pub bytes_per_source_tri: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct VerifyReport {
    pub cluster_count: u32,
    pub leaf_cluster_count: u32,
    pub root_cluster_count: u32,
    pub lod_levels: u32,
    pub bvh_node_count: u32,
    pub page_count: u32,
    pub root_page_count: u32,
    pub total_cluster_tris: u32,
    pub total_cluster_verts: u32,
    pub cullable_cone_clusters: u32,
    pub cut_checks_passed: u32,
    pub bytes_per_cluster_tri: f64,
    pub max_quant_error: f32,
}
