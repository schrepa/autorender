//! `#[repr(C)]` mirrors of `vendor/meshoptimizer/demo/clusterlod.h` and the
//! meshoptimizer bounds API, behind safe wrappers: the only unsafe code of
//! the crate. Field order and types match the headers exactly; review them
//! on every meshoptimizer upgrade (`vendor/meshoptimizer/VENDORED.md`).

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

#[allow(unsafe_code)]
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
}

/// `clodDefaultConfig`: clusters of up to `max_triangles` (4 to 256).
pub(crate) fn default_config(max_triangles: usize) -> ClodConfig {
    assert!((4..=256).contains(&max_triangles));
    // SAFETY: a plain constructor over a checked argument.
    #[allow(unsafe_code)]
    unsafe {
        clodDefaultConfig(max_triangles)
    }
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

extern "C" fn trampoline(
    context: *mut c_void,
    group: ClodGroup,
    clusters: *const ClodCluster,
    cluster_count: usize,
) -> c_int {
    // SAFETY: `build` passes its `Context` to one `clodBuild` call, which
    // calls this on the same thread, and every cluster's indices stay valid
    // during the call.
    #[allow(unsafe_code)]
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
            #[allow(unsafe_code)]
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
    #[allow(unsafe_code)]
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
    #[allow(unsafe_code)]
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
    #[allow(unsafe_code)]
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
    #[allow(unsafe_code)]
    let bound = unsafe { clodBuildHierarchyBound(groups.len(), node_width, levels) };
    let mut nodes = vec![ClodNode::default(); bound];
    // SAFETY: `nodes` holds the bound the library gives for these groups.
    #[allow(unsafe_code)]
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

/// meshoptimizer's simplifier, which only the parity tests of the engine's
/// Rust port (`engine_assets::simplify`) call.
#[cfg(test)]
pub(crate) mod simplifier {
    use std::ffi::c_uint;

    #[allow(unsafe_code)]
    unsafe extern "C" {
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
        fn meshopt_simplifyScale(
            vertex_positions: *const f32,
            vertex_count: usize,
            vertex_positions_stride: usize,
        ) -> f32;
    }

    /// meshoptimizer's `meshopt_Simplify*` option bits.
    pub(crate) mod options {
        pub const LOCK_BORDER: u32 = 1 << 0;
        pub const ERROR_ABSOLUTE: u32 = 1 << 2;
        pub const PRUNE: u32 = 1 << 3;
        pub const REGULARIZE: u32 = 1 << 4;
        pub const PERMISSIVE: u32 = 1 << 5;
        pub const REGULARIZE_LIGHT: u32 = 1 << 6;
        pub const PRESERVE_FOLDS: u32 = 1 << 7;
        pub const ERROR_CLAMPED: u32 = 1 << 8;
        /// Every public option above; not the sparse one.
        pub const ALL: u32 = LOCK_BORDER
            | ERROR_ABSOLUTE
            | PRUNE
            | REGULARIZE
            | PERMISSIVE
            | REGULARIZE_LIGHT
            | PRESERVE_FOLDS
            | ERROR_CLAMPED;
    }

    /// What `meshopt_simplifyWithAttributes` reads: triangles over positions,
    /// `attribute_stride` attribute values per vertex (the first
    /// `weights.len()` weighted) and `meshopt_SimplifyVertex_*` flags per
    /// vertex (or none when empty).
    pub(crate) struct Input<'a> {
        pub indices: &'a [u32],
        pub positions: &'a [[f32; 3]],
        pub attributes: &'a [f32],
        pub attribute_stride: usize,
        pub weights: &'a [f32],
        pub locks: &'a [u8],
    }

    /// `meshopt_simplifyWithAttributes`: the simplified indices and the
    /// result error.
    pub(crate) fn simplify_with_attributes(
        input: &Input<'_>,
        target_index_count: usize,
        target_error: f32,
        options: u32,
    ) -> (Vec<u32>, f32) {
        let vertex_count = input.positions.len();
        // The library asserts these (or reads out of bounds without them).
        assert!(input.indices.len().is_multiple_of(3));
        assert!(
            input
                .indices
                .iter()
                .all(|&index| (index as usize) < vertex_count)
        );
        assert!(target_index_count <= input.indices.len());
        assert!(target_error >= 0.0);
        assert!(input.weights.len() <= input.attribute_stride.min(32));
        assert!(input.weights.iter().all(|&weight| weight >= 0.0));
        assert!(input.attribute_stride <= 64);
        assert_eq!(
            input.attributes.len(),
            vertex_count * input.attribute_stride
        );
        assert!(input.locks.is_empty() || input.locks.len() == vertex_count);
        assert_eq!(options & !options::ALL, 0);
        let attributes = !input.weights.is_empty();
        let mut destination = vec![0; input.indices.len()];
        let mut error = 0.0;
        // SAFETY: the destination holds every input index, the checks above
        // keep every read in bounds, and null pointers stand for absent
        // attributes and locks as the library documents.
        #[allow(unsafe_code)]
        let count = unsafe {
            meshopt_simplifyWithAttributes(
                destination.as_mut_ptr(),
                input.indices.as_ptr(),
                input.indices.len(),
                input.positions.as_ptr().cast(),
                vertex_count,
                12,
                if attributes {
                    input.attributes.as_ptr()
                } else {
                    std::ptr::null()
                },
                if attributes {
                    input.attribute_stride * 4
                } else {
                    0
                },
                if attributes {
                    input.weights.as_ptr()
                } else {
                    std::ptr::null()
                },
                input.weights.len(),
                if input.locks.is_empty() {
                    std::ptr::null()
                } else {
                    input.locks.as_ptr()
                },
                target_index_count,
                target_error,
                options,
                &raw mut error,
            )
        };
        destination.truncate(count);
        (destination, error)
    }

    /// `meshopt_simplifyScale`.
    pub(crate) fn scale(positions: &[[f32; 3]]) -> f32 {
        // SAFETY: the library reads `positions.len()` tightly packed
        // positions.
        #[allow(unsafe_code)]
        unsafe {
            meshopt_simplifyScale(positions.as_ptr().cast(), positions.len(), 12)
        }
    }
}
