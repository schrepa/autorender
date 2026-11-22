use std::mem::size_of;

use glam::{Vec2, Vec3};
use serde::Serialize;

pub const MAX_CLUSTER_VERTS: usize = 64;
pub const MAX_CLUSTER_TRIS: usize = 64;
pub const PAGE_SIZE_BYTES: usize = 65_536;
pub const PAGE_WORDS: usize = PAGE_SIZE_BYTES / 4;
pub const BVH_BRANCHING: usize = 4;
pub const CLUSTER_VERTEX_WORDS: usize = 4;
pub const CLUSTER_TRI_WORDS: usize = 1;
pub const ROOT_PARENT_ERROR: f32 = 1.0e30;
pub const NO_FALLBACK_CLUSTER: u32 = u32::MAX;
pub const NO_BVH_CHILD: u32 = u32::MAX;
pub const MANIFEST_MAGIC: [u8; 4] = *b"MLM1";
pub const MANIFEST_VERSION: u32 = 1;
pub const FLAG_ROOT_CLUSTER: u32 = 1;
pub const FLAG_IN_ROOT_PAGE: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ClusterHeader {
    pub bound_sphere: [f32; 4],
    pub lod_sphere: [f32; 4],
    pub aabb_min: [f32; 4],
    pub aabb_extent: [f32; 4],
    pub parent_lod_sphere: [f32; 4],
    pub cone_and_meta: [u32; 4],
    pub streaming_meta: [u32; 4],
}

const _: () = assert!(
    size_of::<ClusterHeader>() == 112,
    "ClusterHeader must be 112 bytes"
);

impl ClusterHeader {
    pub fn pack_meta(vert_count: usize, tri_count: usize, lod_level: u32, material_id: u8) -> u32 {
        ((vert_count as u32) & 0xFF)
            | (((tri_count as u32) & 0xFF) << 8)
            | ((lod_level & 0xFF) << 16)
            | (u32::from(material_id) << 24)
    }

    pub fn vert_count(&self) -> u32 {
        self.cone_and_meta[1] & 0xFF
    }

    pub fn tri_count(&self) -> u32 {
        (self.cone_and_meta[1] >> 8) & 0xFF
    }

    pub fn lod_level(&self) -> u32 {
        (self.cone_and_meta[1] >> 16) & 0xFF
    }

    pub fn material_id(&self) -> u8 {
        ((self.cone_and_meta[1] >> 24) & 0xFF) as u8
    }

    pub fn page_id(&self) -> u32 {
        self.cone_and_meta[2]
    }

    pub fn page_word_offset(&self) -> u32 {
        self.cone_and_meta[3]
    }

    pub fn word_count(&self) -> u32 {
        self.vert_count() * (CLUSTER_VERTEX_WORDS as u32)
            + self.tri_count() * (CLUSTER_TRI_WORDS as u32)
    }

    pub fn self_error(&self) -> f32 {
        self.aabb_min[3]
    }

    pub fn parent_error(&self) -> f32 {
        self.aabb_extent[3]
    }

    pub fn group_id(&self) -> u32 {
        self.streaming_meta[0]
    }

    pub fn fallback_cluster(&self) -> u32 {
        self.streaming_meta[1]
    }

    pub fn flags(&self) -> u32 {
        self.streaming_meta[2]
    }

    pub fn is_root(&self) -> bool {
        (self.streaming_meta[2] & FLAG_ROOT_CLUSTER) != 0
    }

    pub fn is_in_root_page(&self) -> bool {
        (self.streaming_meta[2] & FLAG_IN_ROOT_PAGE) != 0
    }

    pub fn aabb_min_vec3(&self) -> Vec3 {
        Vec3::new(self.aabb_min[0], self.aabb_min[1], self.aabb_min[2])
    }

    pub fn aabb_extent_vec3(&self) -> Vec3 {
        Vec3::new(
            self.aabb_extent[0],
            self.aabb_extent[1],
            self.aabb_extent[2],
        )
    }

    pub fn normal_cone(&self) -> ([i8; 3], i8) {
        let word = self.cone_and_meta[0];
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
    pub ao: Vec<f32>,
    pub blend: Vec<f32>,
    pub indices: Vec<u32>,
    pub material_ids: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ClusterVertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
    pub ao: f32,
    pub blend: f32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct DecodedCluster {
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
}

#[derive(Clone, Debug)]
pub struct BuiltMeshlet {
    pub manifest: MeshletManifest,
    pub page_words: Vec<u32>,
    pub stats: BuildStats,
}

impl BuiltMeshlet {
    pub fn page(&self, page_id: u32) -> Option<&[u32]> {
        let start = (page_id as usize).checked_mul(PAGE_WORDS)?;
        let end = start.checked_add(PAGE_WORDS)?;
        self.page_words.get(start..end)
    }

    pub fn pages_to_bytes(&self) -> Vec<u8> {
        crate::encode::words_to_le_bytes(&self.page_words)
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
