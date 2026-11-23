#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod bounds;
mod bvh;
mod dag;
mod encode;
mod format;
mod group;
mod manifest;
mod partition;
mod quadric;
mod simplify;
mod verify;

pub use bounds::{
    aabb_of_points, bounding_sphere_points, compute_normal_cone, merge_spheres, morton_3d,
    pack_normal_cone,
};
pub use dag::build;
pub use encode::{
    compress_page_words, compute_base_quant_step, decode_cluster_words, decode_compressed_pages,
    decode_octahedral, decompress_page_words, dequantize_position, encode_cluster_words,
    encode_compressed_pages, encode_octahedral, page_bytes_as_words, words_to_le_bytes,
};
pub use format::{
    BVH_BRANCHING, BuildStats, BuiltMeshlet, BvhNode, CLUSTER_HEADER_WORDS, ClusterHeader,
    ClusterVertex, DecodedCluster, FLAG_HAS_VERTEX_COLOR, FLAG_IN_ROOT_PAGE, FLAG_PINNED_COARSE,
    FLAG_ROOT_CLUSTER, MANIFEST_MAGIC, MANIFEST_VERSION, MAX_CLUSTER_TRIS, MAX_CLUSTER_VERTS,
    MeshletManifest, NO_BVH_CHILD, PAGE_SIZE_BYTES, PAGE_WORDS, PageSpan, ROOT_PARENT_ERROR,
    SourceMesh, VerifyReport,
};
pub use verify::verify;
