//! `#[repr(C)]` mirrors of `vendor/meshoptimizer/demo/clusterlod.h` and the
//! meshoptimizer bounds API, behind safe wrappers: the only unsafe code of
//! the crate. Field order and types match the headers exactly; review them
//! on every meshoptimizer upgrade (`vendor/meshoptimizer/VENDORED.md`).

#![expect(unsafe_code, reason = "calls the vendored C++ meshoptimizer")]

use std::ffi::{c_int, c_uint, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct ClodConfig {
    pub max_vertices: usize,
    pub min_triangles: usize,
    pub max_triangles: usize,
    pub partition_spatial: bool,
    pub partition_sort: bool,
    pub partition_size: usize,
    pub cluster_spatial: bool,
    pub cluster_fill_weight: f32,
    pub cluster_split_factor: f32,
    pub simplify_ratio: f32,
    pub simplify_threshold: f32,
    pub simplify_error_merge_previous: f32,
    pub simplify_error_merge_additive: f32,
    pub simplify_error_factor_sloppy: f32,
    pub simplify_error_edge_limit: f32,
    pub simplify_error_clamped: bool,
    pub simplify_permissive: bool,
    pub simplify_fallback_permissive: bool,
    pub simplify_fallback_sloppy: bool,
    pub simplify_regularize: bool,
    pub simplify_preserve_folds: bool,
    pub simplify_dilate_borders: bool,
    pub optimize_bounds: bool,
    pub optimize_clusters: bool,
    pub optimize_clusters_level: c_int,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct ClodMesh {
    indices: *const c_uint,
    index_count: usize,
    vertex_count: usize,
    vertex_positions: *const f32,
    vertex_positions_stride: usize,
    vertex_attributes: *const f32,
    vertex_attributes_stride: usize,
    vertex_lock: *const u8,
    attribute_weights: *const f32,
    attribute_count: usize,
    attribute_protect_mask: c_uint,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct ClodBounds {
    pub center: [f32; 3],
    pub radius: f32,
    pub error: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct ClodCluster {
    refined: c_int,
    bounds: ClodBounds,
    indices: *const c_uint,
    index_count: usize,
    vertex_count: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ClodGroup {
    pub depth: c_int,
    pub simplified: ClodBounds,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ClodNode {
    pub bounds: ClodBounds,
    pub group: c_int,
    pub child_offset: c_uint,
    pub child_count: c_uint,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct MeshoptBounds {
    pub center: [f32; 3],
    pub radius: f32,
    pub cone_apex: [f32; 3],
    pub cone_axis: [f32; 3],
    pub cone_cutoff: f32,
    pub cone_axis_s8: [i8; 3],
    pub cone_cutoff_s8: i8,
}

type ClodOutput = extern "C" fn(
    context: *mut c_void,
    group: ClodGroup,
    clusters: *const ClodCluster,
    cluster_count: usize,
) -> c_int;

unsafe extern "C" {
    fn clodDefaultConfig(max_triangles: usize) -> ClodConfig;
    fn clodBuild(
        config: ClodConfig,
        mesh: ClodMesh,
        output_context: *mut c_void,
        output_callback: ClodOutput,
    );
    fn clodLocalIndices(
        vertices: *mut c_uint,
        triangles: *mut u8,
        indices: *const c_uint,
        index_count: usize,
    ) -> usize;
    fn clodBuildHierarchyBound(group_count: usize, node_width: usize, level_count: usize) -> usize;
    fn clodBuildHierarchy(
        nodes: *mut ClodNode,
        groups: *const ClodGroup,
        group_count: usize,
        node_width: usize,
        level_count: usize,
    ) -> usize;
    fn meshopt_computeClusterBounds(
        indices: *const c_uint,
        index_count: usize,
        vertex_positions: *const f32,
        vertex_count: usize,
        vertex_positions_stride: usize,
    ) -> MeshoptBounds;
    fn meshopt_simplify(
        destination: *mut c_uint,
        indices: *const c_uint,
        index_count: usize,
        vertex_positions: *const f32,
        vertex_count: usize,
        vertex_positions_stride: usize,
        target_index_count: usize,
        target_error: f32,
        options: c_uint,
        result_error: *mut f32,
    ) -> usize;
    fn meshopt_simplifyWithAttributes(
        destination: *mut c_uint,
        indices: *const c_uint,
        index_count: usize,
        vertex_positions: *const f32,
        vertex_count: usize,
        vertex_positions_stride: usize,
        vertex_attributes: *const f32,
        vertex_attributes_stride: usize,
        attribute_weights: *const f32,
        attribute_count: usize,
        vertex_lock: *const u8,
        target_index_count: usize,
        target_error: f32,
        options: c_uint,
        result_error: *mut f32,
    ) -> usize;
}

/// `clodDefaultConfig`: clusters of up to `max_triangles` (4 to 256).
pub(crate) fn default_config(max_triangles: usize) -> ClodConfig {
    assert!((4..=256).contains(&max_triangles));
    // SAFETY: a plain constructor over a checked argument.
    unsafe { clodDefaultConfig(max_triangles) }
}

/// One cluster `clodBuild` reports: the group that refined into it (or -1)
/// and its indices into the source vertices.
pub(crate) struct ClusterView<'a> {
    pub refined: c_int,
    pub indices: &'a [u32],
}

/// The mesh `clodBuild` reads: positions, per-vertex attributes of
/// `attribute_stride` floats (the first `weights.len()` weighted, those of
/// `protect_mask` kept at discontinuities) and triangle indices, all checked
/// by the caller (indices in range, whole triangles).
pub(crate) struct MeshInput<'a> {
    pub positions: &'a [[f32; 3]],
    pub attributes: &'a [f32],
    pub attribute_stride: usize,
    pub weights: &'a [f32],
    pub protect_mask: u32,
    pub indices: &'a [u32],
    /// `meshopt_SimplifyVertex_*` flags per vertex.
    pub locks: &'a [u8],
}

struct Context<'a> {
    output: &'a mut dyn FnMut(ClodGroup, &[ClusterView<'_>]) -> c_int,
    failed: bool,
}

#[expect(
    clippy::disallowed_methods,
    reason = "a tool's panic must not unwind into the C++ that called it"
)]
extern "C" fn trampoline(
    context: *mut c_void,
    group: ClodGroup,
    clusters: *const ClodCluster,
    cluster_count: usize,
) -> c_int {
    // SAFETY: `build` passes its `Context` to one `clodBuild` call, which
    // calls this on the same thread, and every cluster's indices stay valid
    // during the call.
    let (context, clusters) = unsafe {
        (
            &mut *context.cast::<Context<'_>>(),
            std::slice::from_raw_parts(clusters, cluster_count),
        )
    };
    let views: Vec<ClusterView<'_>> = clusters
        .iter()
        .map(|cluster| ClusterView {
            refined: cluster.refined,
            // SAFETY: as above.
            indices: unsafe { std::slice::from_raw_parts(cluster.indices, cluster.index_count) },
        })
        .collect();
    // Panics must not unwind into C++.
    match catch_unwind(AssertUnwindSafe(|| (context.output)(group, &views))) {
        Ok(value) => value,
        Err(_) => {
            context.failed = true;
            -1
        }
    }
}

/// `clodBuild`: calls `output` with each group and its clusters in order;
/// its result becomes `refined` of the clusters that group's simplification
/// makes. Returns false when `output` panicked.
pub(crate) fn build(
    config: ClodConfig,
    input: &MeshInput<'_>,
    output: &mut dyn FnMut(ClodGroup, &[ClusterView<'_>]) -> c_int,
) -> bool {
    assert_eq!(
        input.attributes.len(),
        input.positions.len() * input.attribute_stride
    );
    assert!(input.weights.len() <= input.attribute_stride);
    assert!(input.indices.len().is_multiple_of(3));
    assert!(
        input
            .indices
            .iter()
            .all(|&index| (index as usize) < input.positions.len())
    );
    assert_eq!(input.locks.len(), input.positions.len());
    let mesh = ClodMesh {
        indices: input.indices.as_ptr(),
        index_count: input.indices.len(),
        vertex_count: input.positions.len(),
        vertex_positions: input.positions.as_ptr().cast(),
        vertex_positions_stride: 12,
        vertex_attributes: input.attributes.as_ptr(),
        vertex_attributes_stride: input.attribute_stride * 4,
        vertex_lock: input.locks.as_ptr(),
        attribute_weights: input.weights.as_ptr(),
        attribute_count: input.weights.len(),
        attribute_protect_mask: input.protect_mask,
    };
    let mut context = Context {
        output,
        failed: false,
    };
    // SAFETY: every pointer in `mesh` borrows `input` for the call, the
    // indices were checked above, and `trampoline` only touches `context`.
    unsafe {
        clodBuild(config, mesh, (&raw mut context).cast(), trampoline);
    }
    !context.failed
}

/// `clodLocalIndices`: a cluster's vertices (indices into the source) and its
/// local triangle indices.
pub(crate) fn local_indices(indices: &[u32]) -> (Vec<u32>, Vec<u8>) {
    let mut vertices = vec![0; indices.len()];
    let mut triangles = vec![0; indices.len()];
    // SAFETY: both outputs hold `indices.len()` entries, the most unique
    // vertices there can be.
    let unique = unsafe {
        clodLocalIndices(
            vertices.as_mut_ptr(),
            triangles.as_mut_ptr(),
            indices.as_ptr(),
            indices.len(),
        )
    };
    vertices.truncate(unique);
    (vertices, triangles)
}

/// `meshopt_computeClusterBounds` of triangles `indices` over `positions`.
pub(crate) fn cluster_bounds(indices: &[u32], positions: &[[f32; 3]]) -> MeshoptBounds {
    assert!(
        indices
            .iter()
            .all(|&index| (index as usize) < positions.len())
    );
    // SAFETY: indices checked above address tightly packed positions.
    unsafe {
        meshopt_computeClusterBounds(
            indices.as_ptr(),
            indices.len(),
            positions.as_ptr().cast(),
            positions.len(),
            12,
        )
    }
}

/// `clodBuildHierarchy`: one tree per level over `groups`, `node_width`
/// children per node.
pub(crate) fn build_hierarchy(
    groups: &[ClodGroup],
    node_width: usize,
    levels: usize,
) -> Vec<ClodNode> {
    assert!(groups.iter().all(|group| (group.depth as usize) < levels));
    // SAFETY: plain arithmetic.
    let bound = unsafe { clodBuildHierarchyBound(groups.len(), node_width, levels) };
    let mut nodes = vec![ClodNode::default(); bound];
    // SAFETY: `nodes` holds the bound the library gives for these groups.
    let written = unsafe {
        clodBuildHierarchy(
            nodes.as_mut_ptr(),
            groups.as_ptr(),
            groups.len(),
            node_width,
            levels,
        )
    };
    nodes.truncate(written);
    nodes
}

/// What `meshopt_simplify` and `meshopt_simplifyWithAttributes` read: positions,
/// triangle indices, and optionally per-vertex attributes of
/// `attribute_stride` floats (the first `weights.len()` weighted) and
/// `meshopt_SimplifyVertex_*` flags per vertex.
pub struct SimplifyInput<'a> {
    pub positions: &'a [[f32; 3]],
    pub indices: &'a [u32],
    pub attributes: Option<(&'a [f32], usize, &'a [f32])>,
    pub locks: Option<&'a [u8]>,
}

/// meshoptimizer's own simplification of `input` toward `target_index_count`
/// indices within `target_error`, with `meshopt_Simplify*` `options`:
/// `meshopt_simplify` without attributes and flags, otherwise
/// `meshopt_simplifyWithAttributes`. Its indices and result error.
///
/// # Panics
///
/// Where `input` is not whole triangles of its vertices or its attributes
/// or flags do not cover every vertex, which the C++ would only assert.
pub fn simplify(
    input: &SimplifyInput<'_>,
    target_index_count: usize,
    target_error: f32,
    options: u32,
) -> (Vec<u32>, f32) {
    assert!(input.indices.len().is_multiple_of(3));
    assert!(
        input
            .indices
            .iter()
            .all(|&index| (index as usize) < input.positions.len())
    );
    if let Some((values, stride, weights)) = input.attributes {
        assert_eq!(values.len(), input.positions.len() * stride);
        assert!(weights.len() <= stride);
    }
    if let Some(locks) = input.locks {
        assert_eq!(locks.len(), input.positions.len());
    }
    let mut destination = vec![0u32; input.indices.len()];
    let mut error = 0.0f32;
    let written = if input.attributes.is_none() && input.locks.is_none() {
        // SAFETY: `destination` holds as many indices as the input, the most
        // simplification writes; the indices were checked above and address
        // tightly packed positions.
        unsafe {
            meshopt_simplify(
                destination.as_mut_ptr(),
                input.indices.as_ptr(),
                input.indices.len(),
                input.positions.as_ptr().cast(),
                input.positions.len(),
                12,
                target_index_count,
                target_error,
                options,
                &raw mut error,
            )
        }
    } else {
        let (values, stride, weights) = input.attributes.unwrap_or((&[], 0, &[]));
        // SAFETY: as above, and the attributes and flags, checked above,
        // cover every vertex (null pointers for none, as the library takes
        // them).
        unsafe {
            meshopt_simplifyWithAttributes(
                destination.as_mut_ptr(),
                input.indices.as_ptr(),
                input.indices.len(),
                input.positions.as_ptr().cast(),
                input.positions.len(),
                12,
                if stride == 0 {
                    std::ptr::null()
                } else {
                    values.as_ptr()
                },
                stride * 4,
                if weights.is_empty() {
                    std::ptr::null()
                } else {
                    weights.as_ptr()
                },
                weights.len(),
                input.locks.map_or(std::ptr::null(), <[u8]>::as_ptr),
                target_index_count,
                target_error,
                options,
                &raw mut error,
            )
        }
    };
    destination.truncate(written);
    (destination, error)
}
