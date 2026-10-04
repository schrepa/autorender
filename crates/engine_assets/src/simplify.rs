//! Mesh simplification for automatic levels of detail: a pure Rust port of
//! meshoptimizer 1.3's edge collapse simplifier (`meshopt_simplify`,
//! `meshopt_simplifyWithAttributes` and `meshopt_simplifyScale` in
//! `vendor/meshoptimizer/src/simplifier.cpp`), for browsers, which cannot
//! run the C++ library that native tools compile.
//!
//! [`simplify`] removes triangles by collapsing edges in passes, cheapest
//! first by quadric error, until the index count reaches a target or the
//! next collapse would exceed an error limit. It keeps the mesh's topology:
//! open borders and attribute seams (vertices sharing a position with
//! different attributes) collapse only along themselves, and vertices it
//! cannot classify stay. Weighted vertex attributes such as normals and
//! texture coordinates add to the error. The result indexes the input
//! vertices, which stay as they are.
//!
//! The port keeps meshoptimizer's structure, constants, tables and float
//! operation order. Rust never fuses multiply-adds, so it reproduces the C++
//! library's indices and error bit for bit wherever that is compiled without
//! floating-point contraction (`tools/vgbuild`'s parity tests compare them),
//! and gives the same bits on every target. Unlike the C++, it validates its
//! input instead of asserting it, and does not limit the attribute count.
//!
//! Based on Garland and Heckbert, "Surface simplification using quadric
//! error metrics" (1997), Hoppe, "New quadric metric for simplifying meshes
//! with appearance attributes" (1999), and the other works `simplifier.cpp`
//! cites.

use std::fmt;
use std::ops::{BitOr, BitOrAssign};

use engine_math::{max, min};

/// Options of [`simplify`]: meshoptimizer's `meshopt_Simplify*` flags. The
/// default (no options) is meshoptimizer's safe `0`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SimplifyOptions {
    /// Keep vertices on the topological border (edges without a paired
    /// triangle) in place, for simplifying part of a larger mesh
    /// (`meshopt_SimplifyLockBorder`).
    pub lock_border: bool,
    /// Treat the target error and the result error as distances in the
    /// positions' units instead of fractions of the mesh extent
    /// (`meshopt_SimplifyErrorAbsolute`; see [`simplify_scale`]).
    pub error_absolute: bool,
    /// Remove disconnected parts of the mesh as simplification proceeds,
    /// whatever their topology allows (`meshopt_SimplifyPrune`).
    pub prune: bool,
    /// Prefer regular triangle sizes and shapes, at some cost to geometric
    /// and attribute quality (`meshopt_SimplifyRegularize`).
    pub regularize: bool,
    /// As `regularize` at a smaller cost; takes precedence over it
    /// (`meshopt_SimplifyRegularizeLight`).
    pub regularize_light: bool,
    /// Allow collapses across attribute discontinuities, except at vertices
    /// flagged [`VertexFlags::PROTECT`] (`meshopt_SimplifyPermissive`).
    pub permissive: bool,
    /// Keep fold lines between opposite-facing triangles, at a small cost
    /// (`meshopt_SimplifyPreserveFolds`, experimental in meshoptimizer).
    pub preserve_folds: bool,
    /// Limit attribute error to the position error's scale where attributes
    /// vary strongly (`meshopt_SimplifyErrorClamped`, experimental in
    /// meshoptimizer).
    pub error_clamped: bool,
}

/// Per-vertex flags of [`SimplifyInput::vertex_flags`]: meshoptimizer's
/// `meshopt_SimplifyVertex_*` bits, combined with `|`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct VertexFlags(u8);

impl VertexFlags {
    /// No flags.
    pub const NONE: Self = Self(0);
    /// Never move this vertex; every vertex at its position stays
    /// (`meshopt_SimplifyVertex_Lock`).
    pub const LOCK: Self = Self(1);
    /// Keep the attribute discontinuity at this vertex's position under
    /// [`SimplifyOptions::permissive`] (`meshopt_SimplifyVertex_Protect`).
    pub const PROTECT: Self = Self(2);
    /// Make the vertex likelier to stay by weighting its position like its
    /// surface (`meshopt_SimplifyVertex_Priority`; read on the first vertex
    /// at a position).
    pub const PRIORITY: Self = Self(4);

    /// The flags of meshoptimizer's bits; unknown bits are kept and ignored.
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    /// meshoptimizer's bits.
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Whether every flag of `other` is set.
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}

impl BitOr for VertexFlags {
    type Output = Self;

    fn bitor(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl BitOrAssign for VertexFlags {
    fn bitor_assign(&mut self, other: Self) {
        self.0 |= other.0;
    }
}

/// Weighted per-vertex attributes that add to [`simplify`]'s error, such as
/// normals, texture coordinates or colors.
#[derive(Clone, Copy, Debug)]
pub struct VertexAttributes<'a> {
    /// `stride` values per vertex.
    pub values: &'a [f32],
    /// Values per vertex.
    pub stride: usize,
    /// The weights of the first `weights.len()` values of each vertex:
    /// finite and not negative, their priority relative to each other and to
    /// position error. Zero ignores a value.
    pub weights: &'a [f32],
}

/// The triangles [`simplify`] reduces.
#[derive(Clone, Copy, Debug)]
pub struct SimplifyInput<'a> {
    /// Triangle list indices into `positions`.
    pub indices: &'a [u32],
    /// Vertex positions. Vertices sharing a position exactly (seam wedges)
    /// move together.
    pub positions: &'a [[f32; 3]],
    /// Optional attributes of every vertex.
    pub attributes: Option<VertexAttributes<'a>>,
    /// Optional flags, one per vertex.
    pub vertex_flags: Option<&'a [VertexFlags]>,
}

impl<'a> SimplifyInput<'a> {
    /// Triangles without attributes or flags.
    pub fn new(indices: &'a [u32], positions: &'a [[f32; 3]]) -> Self {
        Self {
            indices,
            positions,
            attributes: None,
            vertex_flags: None,
        }
    }

    /// With `stride` attribute values per vertex, the first `weights.len()`
    /// of them weighted.
    pub fn with_attributes(mut self, values: &'a [f32], stride: usize, weights: &'a [f32]) -> Self {
        self.attributes = Some(VertexAttributes {
            values,
            stride,
            weights,
        });
        self
    }

    /// With one set of flags per vertex.
    pub fn with_vertex_flags(mut self, flags: &'a [VertexFlags]) -> Self {
        self.vertex_flags = Some(flags);
        self
    }
}

/// The result of [`simplify`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Simplified {
    /// Triangle list indices into the input vertices.
    pub indices: Vec<u32>,
    /// The largest error of the collapses and pruning it took, including
    /// attribute error: a fraction of the mesh extent, or a distance with
    /// [`SimplifyOptions::error_absolute`].
    pub error: f32,
}

/// Input [`simplify`] rejects instead of asserting as meshoptimizer does.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SimplifyError {
    /// The index count is not a multiple of 3.
    TriangleCount { indices: usize },
    /// An index names no vertex.
    IndexOutOfBounds {
        index: usize,
        value: u32,
        vertices: usize,
    },
    /// More vertices or indices than 32-bit indices address (`u32::MAX`
    /// marks "no vertex").
    TooLarge { vertices: usize, indices: usize },
    /// Attribute values are not `stride` per vertex, or there are more
    /// weights than values per vertex.
    AttributeLayout {
        values: usize,
        stride: usize,
        weights: usize,
        vertices: usize,
    },
    /// An attribute weight is negative or not finite.
    InvalidWeight { attribute: usize },
    /// Vertex flags are not one per vertex.
    VertexFlagCount { expected: usize, actual: usize },
    /// The target error is negative or NaN.
    InvalidTargetError,
}

impl fmt::Display for SimplifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TriangleCount { indices } => {
                write!(f, "simplify index count {indices} must be a multiple of 3")
            }
            Self::IndexOutOfBounds {
                index,
                value,
                vertices,
            } => write!(
                f,
                "simplify index {index} is {value}, outside {vertices} vertices"
            ),
            Self::TooLarge { vertices, indices } => write!(
                f,
                "simplify input of {vertices} vertices and {indices} indices exceeds 32-bit indices"
            ),
            Self::AttributeLayout {
                values,
                stride,
                weights,
                vertices,
            } => write!(
                f,
                "simplify attributes need {stride} values for each of {vertices} vertices and at most {stride} weights; received {values} values and {weights} weights"
            ),
            Self::InvalidWeight { attribute } => write!(
                f,
                "simplify attribute {attribute} weight must be finite and not negative"
            ),
            Self::VertexFlagCount { expected, actual } => write!(
                f,
                "simplify vertex flags need one value per vertex: {expected}; received {actual}"
            ),
            Self::InvalidTargetError => {
                write!(f, "simplify target error must not be negative or NaN")
            }
        }
    }
}

impl std::error::Error for SimplifyError {}

/// Reduces `input`'s triangles toward `target_index_count` indices while the
/// error stays at or below `target_error` (meshoptimizer's
/// `meshopt_simplifyWithAttributes`; without attributes and flags,
/// `meshopt_simplify`).
///
/// `target_error` is a fraction of the mesh extent (0.01 allows 1%
/// deformation) or, with [`SimplifyOptions::error_absolute`], a distance;
/// infinity sets no limit. Simplification stops short of the target where
/// topology or the error limit allows no further collapses, and a target at
/// or above the index count returns the indices unchanged. The same input
/// gives the same bits on every target.
///
/// # Errors
///
/// [`SimplifyError`] when indices are not whole triangles of existing
/// vertices, attributes or flags do not cover every vertex, a weight is
/// negative or not finite, or `target_error` is negative or NaN.
pub fn simplify(
    input: &SimplifyInput<'_>,
    target_index_count: usize,
    target_error: f32,
    options: SimplifyOptions,
) -> Result<Simplified, SimplifyError> {
    validate(input, target_error)?;
    Ok(simplify_edge(
        input,
        target_index_count,
        target_error,
        options,
    ))
}

/// The extent [`simplify`] measures relative errors against: the largest
/// side of the positions' bounding box (`meshopt_simplifyScale`). Multiply a
/// relative result error by it for a distance; divide a distance by it for a
/// relative target error.
pub fn simplify_scale(positions: &[[f32; 3]]) -> f32 {
    rescale_positions(None, positions)
}

/// Flags [`VertexFlags::PROTECT`] on every vertex whose values in `mask`
/// (bit `j` for value `j` of each vertex's `stride`) differ from those of
/// the first vertex at its position: the attribute seams that
/// [`SimplifyOptions::permissive`] then keeps. This is meshoptimizer's
/// `clusterlod` `attribute_protect_mask`; the core API has only the flag.
///
/// # Errors
///
/// [`SimplifyError::AttributeLayout`] when `values` are not `stride` per
/// position, [`SimplifyError::VertexFlagCount`] when `flags` are not one per
/// position.
pub fn protect_attribute_seams(
    positions: &[[f32; 3]],
    values: &[f32],
    stride: usize,
    mask: u32,
    flags: &mut [VertexFlags],
) -> Result<(), SimplifyError> {
    let vertices = positions.len();
    if vertices.checked_mul(stride) != Some(values.len()) {
        return Err(SimplifyError::AttributeLayout {
            values: values.len(),
            stride,
            weights: 0,
            vertices,
        });
    }
    if flags.len() != vertices {
        return Err(SimplifyError::VertexFlagCount {
            expected: vertices,
            actual: flags.len(),
        });
    }
    if vertices >= NONE as usize {
        return Err(SimplifyError::TooLarge {
            vertices,
            indices: 0,
        });
    }
    let (remap, _) = build_position_remap(positions, false);
    for (vertex, &first) in remap.iter().enumerate() {
        let first = first as usize;
        if first != vertex
            && (0..stride.min(32)).any(|value| {
                mask & (1 << value) != 0
                    && values[vertex * stride + value] != values[first * stride + value]
            })
        {
            flags[vertex] |= VertexFlags::PROTECT;
        }
    }
    Ok(())
}

fn validate(input: &SimplifyInput<'_>, target_error: f32) -> Result<(), SimplifyError> {
    let vertices = input.positions.len();
    let indices = input.indices.len();
    if !indices.is_multiple_of(3) {
        return Err(SimplifyError::TriangleCount { indices });
    }
    // Edge offsets count indices in 32 bits, and `NONE` is no vertex.
    if vertices >= NONE as usize || indices > NONE as usize {
        return Err(SimplifyError::TooLarge { vertices, indices });
    }
    if let Some(index) = input
        .indices
        .iter()
        .position(|&value| value as usize >= vertices)
    {
        return Err(SimplifyError::IndexOutOfBounds {
            index,
            value: input.indices[index],
            vertices,
        });
    }
    if let Some(attributes) = &input.attributes {
        if attributes.weights.len() > attributes.stride
            || vertices.checked_mul(attributes.stride) != Some(attributes.values.len())
        {
            return Err(SimplifyError::AttributeLayout {
                values: attributes.values.len(),
                stride: attributes.stride,
                weights: attributes.weights.len(),
                vertices,
            });
        }
        if let Some(attribute) = attributes
            .weights
            .iter()
            .position(|weight| !weight.is_finite() || *weight < 0.0)
        {
            return Err(SimplifyError::InvalidWeight { attribute });
        }
    }
    if let Some(flags) = input.vertex_flags
        && flags.len() != vertices
    {
        return Err(SimplifyError::VertexFlagCount {
            expected: vertices,
            actual: flags.len(),
        });
    }
    if target_error.is_nan() || target_error < 0.0 {
        return Err(SimplifyError::InvalidTargetError);
    }
    Ok(())
}

/// meshoptimizer's `~0u`: no vertex, in open-edge loops and hash buckets.
const NONE: u32 = u32::MAX;

/// Half-edges leaving each vertex (`EdgeAdjacency`): the edges of vertex
/// `v` are `data[offsets[v]..offsets[v + 1]]`.
struct EdgeAdjacency {
    offsets: Vec<u32>,
    data: Vec<Edge>,
}

/// A half-edge: the vertex it leads to and the third vertex of its triangle.
#[derive(Clone, Copy, Default)]
struct Edge {
    next: u32,
    prev: u32,
}

impl EdgeAdjacency {
    /// `prepareEdgeAdjacency`.
    fn new(index_count: usize, vertex_count: usize) -> Self {
        Self {
            offsets: vec![0; vertex_count + 1],
            data: vec![Edge::default(); index_count],
        }
    }

    /// `updateEdgeAdjacency`: the half-edges of `indices`, of the vertices
    /// `remap` maps them to if given.
    fn update(&mut self, indices: &[u32], remap: Option<&[u32]>) {
        match remap {
            Some(remap) => self.fill(indices, |vertex| remap[vertex as usize]),
            None => self.fill(indices, |vertex| vertex),
        }
    }

    fn fill(&mut self, indices: &[u32], map: impl Fn(u32) -> u32) {
        let Self { offsets, data } = self;
        // Counts, then starts, then ends of each vertex's edges one slot on.
        let counts = &mut offsets[1..];
        counts.fill(0);
        for &index in indices {
            counts[map(index) as usize] += 1;
        }
        let mut offset = 0;
        for entry in counts.iter_mut() {
            let count = *entry;
            *entry = offset;
            offset += count;
        }
        for triangle in indices.as_chunks::<3>().0 {
            let (a, b, c) = (map(triangle[0]), map(triangle[1]), map(triangle[2]));
            for (vertex, next, prev) in [(a, b, c), (b, c, a), (c, a, b)] {
                let slot = &mut counts[vertex as usize];
                data[*slot as usize] = Edge { next, prev };
                *slot += 1;
            }
        }
        offsets[0] = 0;
    }

    fn edges(&self, vertex: u32) -> &[Edge] {
        let vertex = vertex as usize;
        &self.data[self.offsets[vertex] as usize..self.offsets[vertex + 1] as usize]
    }
}

/// `hashBuckets2`: a power of two at least 1.25 times `count`.
fn hash_buckets2(count: usize) -> usize {
    let mut buckets = 1;
    while buckets < count + count / 4 {
        buckets *= 2;
    }
    buckets
}

/// `PositionHasher::hash`.
fn position_hash(position: &[f32; 3]) -> u32 {
    let [mut x, mut y, mut z] = position.map(f32::to_bits);
    // replace negative zero with zero
    x = if x == 0x8000_0000 { 0 } else { x };
    y = if y == 0x8000_0000 { 0 } else { y };
    z = if z == 0x8000_0000 { 0 } else { z };
    // scramble bits to make sure that integer coordinates have entropy in lower bits
    x ^= x >> 17;
    y ^= y >> 17;
    z ^= z >> 17;
    // Optimized Spatial Hashing for Collision Detection of Deformable Objects
    x.wrapping_mul(73_856_093) ^ y.wrapping_mul(19_349_663) ^ z.wrapping_mul(83_492_791)
}

/// `buildPositionRemap`: for each vertex the first vertex at its exact
/// position, and (if `wedges`) the next vertex at that position in a cycle.
fn build_position_remap(positions: &[[f32; 3]], wedges: bool) -> (Vec<u32>, Vec<u32>) {
    let vertex_count = positions.len();
    let table_size = hash_buckets2(vertex_count);
    let hashmod = table_size - 1;
    let mut table = vec![NONE; table_size];
    let mut remap = vec![0; vertex_count];

    // build forward remap: for each vertex, which other (canonical) vertex does it map to?
    // we use position equivalence for this, and remap vertices to other existing vertices
    for (index, position) in positions.iter().enumerate() {
        // hashLookup2 with quadratic probing; the table always has an empty bucket
        let mut bucket = position_hash(position) as usize & hashmod;
        let mut probe = 0;
        let entry = loop {
            let item = table[bucket];
            if item == NONE || positions[item as usize] == *position {
                break bucket;
            }
            bucket = (bucket + probe + 1) & hashmod;
            probe += 1;
        };
        if table[entry] == NONE {
            table[entry] = index as u32;
        }
        remap[index] = table[entry];
    }

    if !wedges {
        return (remap, Vec::new());
    }

    // build wedge table: for each vertex, which other vertex is the next wedge that also maps to the same vertex?
    // entries in table form a (cyclic) wedge loop per vertex; for manifold vertices, wedge[i] == remap[i] == i
    let mut wedge: Vec<u32> = (0..vertex_count as u32).collect();
    for i in 0..vertex_count {
        let r = remap[i] as usize;
        if r != i {
            wedge[i] = wedge[r];
            wedge[r] = i as u32;
        }
    }
    (remap, wedge)
}

/// `VertexKind`: how a vertex may move.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum Kind {
    /// Not on an attribute seam, not on any boundary.
    Manifold,
    /// Not on an attribute seam, has exactly two open edges.
    Border,
    /// On an attribute seam with exactly two attribute seam edges.
    Seam,
    /// None of the above; these vertices can move as long as all wedges move
    /// to the target vertex.
    Complex,
    /// Complex vertex on a geometric border; can only move to another fringe
    /// vertex.
    Fringe,
    /// None of the above; these vertices can't move.
    Locked,
}

const KIND_COUNT: usize = 6;

/// `kCanCollapse`: manifold vertices can collapse onto anything; border/seam
/// vertices onto border/seam respectively, or locked; complex vertices onto
/// complex/locked; fringe vertices only onto fringe vertices, while border,
/// manifold and complex vertices can collapse onto them. Collapsing kind A
/// into kind B preserves kind B in the target vertex.
const CAN_COLLAPSE: [[u8; KIND_COUNT]; KIND_COUNT] = [
    [1, 1, 1, 1, 1, 1],
    [0, 1, 0, 0, 1, 1],
    [0, 0, 1, 0, 0, 1],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0],
];

/// `kHasOpposite`: edges at manifold or seam vertices are guaranteed an
/// opposite edge (for seams in the position-only mesh). Many complex
/// collapses have one too, but keeping opposite edges separate improves
/// quality by considering both targets.
const HAS_OPPOSITE: [[u8; KIND_COUNT]; KIND_COUNT] = [
    [1, 1, 1, 1, 1, 1],
    [1, 0, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0],
];

fn can_collapse(from: Kind, to: Kind) -> u8 {
    CAN_COLLAPSE[from as usize][to as usize]
}

/// `hasEdge`.
fn has_edge(adjacency: &EdgeAdjacency, a: u32, b: u32) -> bool {
    adjacency.edges(a).iter().any(|edge| edge.next == b)
}

/// `hasEdge` over every wedge of `a`, comparing positions.
fn has_edge_wedged(
    adjacency: &EdgeAdjacency,
    a: u32,
    b: u32,
    remap: &[u32],
    wedge: &[u32],
) -> bool {
    let rb = remap[b as usize];
    let mut v = a;
    loop {
        if adjacency
            .edges(v)
            .iter()
            .any(|edge| remap[edge.next as usize] == rb)
        {
            return true;
        }
        v = wedge[v as usize];
        if v == a {
            return false;
        }
    }
}

/// Classification of every vertex, and the open edges (`loop_` leaving and
/// `loopback` entering each vertex) that border and seam collapses follow.
struct Topology {
    remap: Vec<u32>,
    wedge: Vec<u32>,
    kind: Vec<Kind>,
    loop_: Vec<u32>,
    loopback: Vec<u32>,
}

/// `classifyVertices`.
fn classify_vertices(
    adjacency: &EdgeAdjacency,
    remap: Vec<u32>,
    wedge: Vec<u32>,
    vertex_lock: Option<&[VertexFlags]>,
    options: SimplifyOptions,
) -> Topology {
    let vertex_count = remap.len();
    let mut result = vec![Kind::Manifold; vertex_count];

    // incoming & outgoing open edges: NONE if no open edges, i if there are more than 1
    // note that this is the same data as required in loop[] arrays; loop[] data is only used for border/seam by default
    // in permissive mode we also use it to guide complex-complex collapses, so we fill it for all vertices
    let mut openinc = vec![NONE; vertex_count]; // loopback
    let mut openout = vec![NONE; vertex_count]; // loop

    for vertex in 0..vertex_count as u32 {
        for edge in adjacency.edges(vertex) {
            let target = edge.next;

            if target == vertex {
                // degenerate triangles have two distinct edges instead of three, and the self edge
                // is bi-directional by definition; this can break border/seam classification by "closing"
                // the open edge from another triangle and falsely marking the vertex as manifold
                // instead we mark the vertex as having >1 open edges which turns it into locked/complex
                openinc[vertex as usize] = vertex;
                openout[vertex as usize] = vertex;
            } else if !has_edge(adjacency, target, vertex) {
                let t = target as usize;
                openinc[t] = if openinc[t] == NONE { vertex } else { target };
                let v = vertex as usize;
                openout[v] = if openout[v] == NONE { target } else { vertex };
            }
        }
    }

    for i in 0..vertex_count {
        let iu = i as u32;
        if remap[i] == iu {
            if wedge[i] == iu {
                // no attribute seam, need to check if it's manifold
                let openi = openinc[i];
                let openo = openout[i];

                // note: we classify any vertices with no open edges as manifold
                // this is technically incorrect - if 4 triangles share an edge, we'll classify vertices as manifold
                // it's unclear if this is a problem in practice
                result[i] = if openi == NONE && openo == NONE {
                    Kind::Manifold
                } else if openi != NONE
                    && openo != NONE
                    && remap[openi as usize] == remap[openo as usize]
                    && openi != iu
                {
                    // classify half-seams as seams (the branch below would mis-classify them as borders)
                    // half-seam is a single vertex that connects to both vertices of a potential seam
                    // treating these as seams allows collapsing the "full" seam vertex onto them
                    Kind::Seam
                } else if openi != iu && openo != iu {
                    Kind::Border
                } else {
                    Kind::Locked
                };
            } else if wedge[wedge[i] as usize] == iu {
                // attribute seam; need to distinguish between Seam and Locked
                let w = wedge[i];
                let (openiv, openov) = (openinc[i], openout[i]);
                let (openiw, openow) = (openinc[w as usize], openout[w as usize]);

                // seam should have one open half-edge for each vertex, and the edges need to "connect" - point to the same vertex post-remap
                result[i] = if openiv != NONE
                    && openiv != iu
                    && openov != NONE
                    && openov != iu
                    && openiw != NONE
                    && openiw != w
                    && openow != NONE
                    && openow != w
                    && remap[openiv as usize] == remap[openow as usize]
                    && remap[openov as usize] == remap[openiw as usize]
                    && remap[openiv as usize] != remap[openov as usize]
                {
                    Kind::Seam
                } else {
                    // mismatched seam endpoints or disconnected seam; we don't have classification available
                    Kind::Locked
                };
            } else {
                // more than one vertex maps to this one; we don't have classification available
                result[i] = Kind::Locked;
            }
        } else {
            result[i] = result[remap[i] as usize];
        }
    }

    if options.permissive {
        for i in 0..vertex_count {
            if result[i] != Kind::Seam && result[i] != Kind::Locked {
                continue;
            }
            if remap[i] != i as u32 {
                // only process primary vertices; wedges will be updated to match the primary vertex
                result[i] = result[remap[i] as usize];
                continue;
            }

            // vertex_lock may protect any wedge, not just the primary vertex, so we switch to complex only if no wedges are protected
            let mut protect = false;
            let mut v = i as u32;
            loop {
                protect |=
                    vertex_lock.is_some_and(|lock| lock[v as usize].contains(VertexFlags::PROTECT));
                v = wedge[v as usize];
                if v == i as u32 {
                    break;
                }
            }

            // check if any adjoining edge doesn't have an opposite edge (indicating vertex is on the border)
            let mut border = false;
            loop {
                for edge in adjacency.edges(v) {
                    border |= !has_edge_wedged(adjacency, edge.next, v, &remap, &wedge);
                }
                v = wedge[v as usize];
                if v == i as u32 {
                    break;
                }
            }

            // border vertices with attribute discontinuities can only collapse along the border, so they need a separate tagging
            if !protect {
                result[i] = if border { Kind::Fringe } else { Kind::Complex };
            }
        }
    }

    if let Some(lock) = vertex_lock {
        // vertex_lock may lock any wedge, not just the primary vertex, so we need to lock the primary vertex and relock any wedges
        for i in 0..vertex_count {
            if lock[i].contains(VertexFlags::LOCK) {
                result[remap[i] as usize] = Kind::Locked;
            }
        }
        for i in 0..vertex_count {
            if result[remap[i] as usize] == Kind::Locked {
                result[i] = Kind::Locked;
            }
        }
    }

    if options.lock_border {
        for kind in &mut result {
            if *kind == Kind::Border || *kind == Kind::Fringe {
                *kind = Kind::Locked;
            }
        }
    }

    Topology {
        remap,
        wedge,
        kind: result,
        loop_: openout,
        loopback: openinc,
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Vector3 {
    x: f32,
    y: f32,
    z: f32,
}

/// `a - b`, component by component.
fn sub(a: &Vector3, b: &Vector3) -> Vector3 {
    Vector3 {
        x: a.x - b.x,
        y: a.y - b.y,
        z: a.z - b.z,
    }
}

/// `a x b`, in meshoptimizer's operation order.
fn cross(a: &Vector3, b: &Vector3) -> Vector3 {
    Vector3 {
        x: a.y * b.z - a.z * b.y,
        y: a.z * b.x - a.x * b.z,
        z: a.x * b.y - a.y * b.x,
    }
}

fn dot(a: &Vector3, b: &Vector3) -> f32 {
    a.x * b.x + a.y * b.y + a.z * b.z
}

/// `rescalePositions`: fills `result` (if given) with the positions moved
/// to the origin and scaled to a unit extent, and returns the extent.
fn rescale_positions(result: Option<&mut [Vector3]>, positions: &[[f32; 3]]) -> f32 {
    let mut minv = [f32::MAX; 3];
    let mut maxv = [-f32::MAX; 3];

    for position in positions {
        for j in 0..3 {
            let vj = position[j];
            minv[j] = min(minv[j], vj);
            maxv[j] = max(maxv[j], vj);
        }
    }

    let mut extent = 0.0;
    extent = max(maxv[0] - minv[0], extent);
    extent = max(maxv[1] - minv[1], extent);
    extent = max(maxv[2] - minv[2], extent);

    if let Some(result) = result {
        let scale = if extent == 0.0 { 0.0 } else { 1.0 / extent };
        for (out, position) in result.iter_mut().zip(positions) {
            *out = Vector3 {
                x: (position[0] - minv[0]) * scale,
                y: (position[1] - minv[1]) * scale,
                z: (position[2] - minv[2]) * scale,
            };
        }
    }

    extent
}

/// `rescaleAttributes`: the weighted attributes `attribute_remap` selects,
/// `attribute_remap.len()` per vertex. The stride is not 0: validation
/// admits no more weights than values per vertex, and there are weights.
fn rescale_attributes(attributes: &VertexAttributes<'_>, attribute_remap: &[usize]) -> Vec<f32> {
    let mut result =
        Vec::with_capacity(attributes.values.len() / attributes.stride * attribute_remap.len());
    for vertex in attributes.values.chunks_exact(attributes.stride) {
        for &rk in attribute_remap {
            result.push(vertex[rk] * attributes.weights[rk]);
        }
    }
    result
}

/// `a00*x^2 + a11*y^2 + a22*z^2 + 2*a10*xy + 2*a20*xz + 2*a21*yz + 2*b0*x +
/// 2*b1*y + 2*b2*z + c`, with the accumulated weight `w`.
#[derive(Clone, Copy, Debug, Default)]
struct Quadric {
    a00: f32,
    a11: f32,
    a22: f32,
    a10: f32,
    a20: f32,
    a21: f32,
    b0: f32,
    b1: f32,
    b2: f32,
    c: f32,
    w: f32,
}

/// `gx*x + gy*y + gz*z + gw`.
#[derive(Clone, Copy, Debug, Default)]
struct QuadricGrad {
    gx: f32,
    gy: f32,
    gz: f32,
    gw: f32,
}

/// An edge collapse (`Collapse`): `bits` holds whether it collapses both
/// ways until ranking, then its error's bits (meshoptimizer's union).
#[derive(Clone, Copy, Debug, Default)]
struct Collapse {
    v0: u32,
    v1: u32,
    bits: u32,
}

impl Collapse {
    fn error(&self) -> f32 {
        f32::from_bits(self.bits)
    }
}

fn normalize(v: &mut Vector3) -> f32 {
    let length = (v.x * v.x + v.y * v.y + v.z * v.z).sqrt();
    if length > 0.0 {
        v.x /= length;
        v.y /= length;
        v.z /= length;
    }
    length
}

impl Quadric {
    fn add(&mut self, r: &Self) {
        self.a00 += r.a00;
        self.a11 += r.a11;
        self.a22 += r.a22;
        self.a10 += r.a10;
        self.a20 += r.a20;
        self.a21 += r.a21;
        self.b0 += r.b0;
        self.b1 += r.b1;
        self.b2 += r.b2;
        self.c += r.c;
        self.w += r.w;
    }

    /// `quadricEval`.
    fn eval(&self, v: &Vector3) -> f32 {
        let mut rx = (self.b0 + self.a10 * v.y) * 2.0;
        let mut ry = (self.b1 + self.a21 * v.z) * 2.0;
        let mut rz = (self.b2 + self.a20 * v.x) * 2.0;

        rx += self.a00 * v.x;
        ry += self.a11 * v.y;
        rz += self.a22 * v.z;

        self.c + rx * v.x + ry * v.y + rz * v.z
    }

    /// `quadricError`: the error at `v`, normalized by the weight.
    fn error(&self, v: &Vector3) -> f32 {
        let r = self.eval(v);
        let s = if self.w == 0.0 { 0.0 } else { 1.0 / self.w };
        r.abs() * s
    }

    /// `quadricFromPlane`.
    fn from_plane(a: f32, b: f32, c: f32, d: f32, w: f32) -> Self {
        let aw = a * w;
        let bw = b * w;
        let cw = c * w;
        let dw = d * w;
        Self {
            a00: a * aw,
            a11: b * bw,
            a22: c * cw,
            a10: a * bw,
            a20: a * cw,
            a21: b * cw,
            b0: a * dw,
            b1: b * dw,
            b2: c * dw,
            c: d * dw,
            w,
        }
    }

    /// `quadricFromPoint`.
    fn from_point(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self {
            a00: w,
            a11: w,
            a22: w,
            a10: 0.0,
            a20: 0.0,
            a21: 0.0,
            b0: -x * w,
            b1: -y * w,
            b2: -z * w,
            c: (x * x + y * y + z * z) * w,
            w,
        }
    }

    /// `quadricFromTriangle`.
    fn from_triangle(p0: &Vector3, p1: &Vector3, p2: &Vector3, weight: f32) -> Self {
        let p10 = sub(p1, p0);
        let p20 = sub(p2, p0);

        // normal = cross(p1 - p0, p2 - p0)
        let mut normal = cross(&p10, &p20);
        let area = normalize(&mut normal);

        let distance = dot(&normal, p0);

        // we use sqrtf(area) so that the error is scaled linearly; this tends to improve silhouettes
        Self::from_plane(
            normal.x,
            normal.y,
            normal.z,
            -distance,
            area.sqrt() * weight,
        )
    }

    /// `quadricFromTriangleEdge`.
    fn from_triangle_edge(p0: &Vector3, p1: &Vector3, p2: &Vector3, weight: f32) -> Self {
        let p10 = sub(p1, p0);

        // edge length; keep squared length around for projection correction
        let lengthsq = dot(&p10, &p10);
        let length = lengthsq.sqrt();

        // p20p = length of projection of p2-p0 onto p1-p0; note that p10 is unnormalized so we need to correct it later
        let p20 = sub(p2, p0);
        let p20p = dot(&p20, &p10);

        // perp = perpendicular vector from p2 to line segment p1-p0
        // note: since p10 is unnormalized we need to correct the projection; we scale p20 instead to take advantage of normalize below
        let mut perp = Vector3 {
            x: p20.x * lengthsq - p10.x * p20p,
            y: p20.y * lengthsq - p10.y * p20p,
            z: p20.z * lengthsq - p10.z * p20p,
        };
        normalize(&mut perp);

        let distance = dot(&perp, p0);

        // note: the weight is scaled linearly with edge length; this has to match the triangle weight
        Self::from_plane(perp.x, perp.y, perp.z, -distance, length * weight)
    }

    /// `quadricFromAttributes`: the attribute quadric of a triangle, with
    /// each attribute's gradient in `gradients`.
    fn from_attributes(
        gradients: &mut [QuadricGrad],
        p0: &Vector3,
        p1: &Vector3,
        p2: &Vector3,
        va0: &[f32],
        va1: &[f32],
        va2: &[f32],
    ) -> Self {
        // for each attribute we want to encode the following function into the quadric:
        // (eval(pos) - attr)^2
        // where eval(pos) interpolates attribute across the triangle like so:
        // eval(pos) = pos.x * gx + pos.y * gy + pos.z * gz + gw
        // where gx/gy/gz/gw are gradients
        let p10 = sub(p1, p0);
        let p20 = sub(p2, p0);

        // normal = cross(p1 - p0, p2 - p0)
        let normal = cross(&p10, &p20);
        let area = dot(&normal, &normal).sqrt() * 0.5;

        // quadric is weighted with the square of edge length (= area)
        // this equalizes the units with the positional error (which, after normalization, is a square of distance)
        // as a result, a change in weighted attribute of 1 along distance d is approximately equivalent to a change in position of d
        let w = area;

        // we compute gradients using barycentric coordinates; barycentric coordinates can be computed as follows:
        // v = (d11 * d20 - d01 * d21) / denom
        // w = (d00 * d21 - d01 * d20) / denom
        // u = 1 - v - w
        // here v0, v1 are triangle edge vectors, v2 is a vector from point to triangle corner, and dij = dot(vi, vj)
        // note: v2 and d20/d21 can not be evaluated here as v2 is effectively an unknown variable; we need these only as variables for derivation of gradients
        let v0 = &p10;
        let v1 = &p20;
        let d00 = dot(v0, v0);
        let d01 = dot(v0, v1);
        let d11 = dot(v1, v1);
        let denom = d00 * d11 - d01 * d01;
        let denomr = if denom == 0.0 { 0.0 } else { 1.0 / denom };

        // precompute gradient factors
        // these are derived by directly computing derivative of eval(pos) = a0 * u + a1 * v + a2 * w and factoring out expressions that are shared between attributes
        let gx1 = (d11 * v0.x - d01 * v1.x) * denomr;
        let gx2 = (d00 * v1.x - d01 * v0.x) * denomr;
        let gy1 = (d11 * v0.y - d01 * v1.y) * denomr;
        let gy2 = (d00 * v1.y - d01 * v0.y) * denomr;
        let gz1 = (d11 * v0.z - d01 * v1.z) * denomr;
        let gz2 = (d00 * v1.z - d01 * v0.z) * denomr;

        let mut q = Self {
            w,
            ..Self::default()
        };

        for (k, gradient) in gradients.iter_mut().enumerate() {
            let (a0, a1, a2) = (va0[k], va1[k], va2[k]);

            // compute gradient of eval(pos) for x/y/z/w
            // the formulas below are obtained by directly computing derivative of eval(pos) = a0 * u + a1 * v + a2 * w
            let gx = gx1 * (a1 - a0) + gx2 * (a2 - a0);
            let gy = gy1 * (a1 - a0) + gy2 * (a2 - a0);
            let gz = gz1 * (a1 - a0) + gz2 * (a2 - a0);
            let gw = a0 - p0.x * gx - p0.y * gy - p0.z * gz;

            // quadric encodes (eval(pos)-attr)^2; this means that the resulting expansion needs to compute, for example, pos.x * pos.y * K
            // since quadrics already encode factors for pos.x * pos.y, we can accumulate almost everything in basic quadric fields
            // note: for simplicity we scale all factors by weight here instead of outside the loop
            q.a00 += w * (gx * gx);
            q.a11 += w * (gy * gy);
            q.a22 += w * (gz * gz);

            q.a10 += w * (gy * gx);
            q.a20 += w * (gz * gx);
            q.a21 += w * (gz * gy);

            q.b0 += w * (gx * gw);
            q.b1 += w * (gy * gw);
            q.b2 += w * (gz * gw);

            q.c += w * (gw * gw);

            // the only remaining sum components are ones that depend on attr; these will be added during error evaluation, see quadricError
            *gradient = QuadricGrad {
                gx: w * gx,
                gy: w * gy,
                gz: w * gz,
                gw: w * gw,
            };
        }

        q
    }
}

impl QuadricGrad {
    fn add(&mut self, r: &Self) {
        self.gx += r.gx;
        self.gy += r.gy;
        self.gz += r.gz;
        self.gw += r.gw;
    }
}

/// `quadricError` with attributes: the error of attribute quadric `q` and
/// gradients `g` at position `v` with attribute values `va`.
fn quadric_error_attributes(
    q: &Quadric,
    g: &[QuadricGrad],
    v: &Vector3,
    va: &[f32],
    clamped: bool,
) -> f32 {
    let mut r = q.eval(v);

    // see quadricFromAttributes for general derivation; here we need to add the parts of (eval(pos) - attr)^2 that depend on attr
    for (gradient, &a) in g.iter().zip(va) {
        let g = v.x * gradient.gx + v.y * gradient.gy + v.z * gradient.gz + gradient.gw;
        r += a * (a * q.w - 2.0 * g);
    }

    // note: unlike position error, we do not normalize by Q.w to retain edge scaling as described in quadricFromAttributes
    // we do support clamping, with the error limited to the cumulative weight (= area), which makes perceptual error map to position error more closely
    let e = r.abs();
    if clamped { min(q.w, e) } else { e }
}

/// Weighted attributes with nonzero weight and their quadrics, per vertex
/// (not per position); empty when `count` is 0.
struct Attributes {
    count: usize,
    values: Vec<f32>,
    quadrics: Vec<Quadric>,
    gradients: Vec<QuadricGrad>,
}

impl Attributes {
    /// The attribute error of moving vertex `v` onto vertex `t`: `v`'s
    /// quadric at `t`'s position and attribute values.
    fn error(&self, v: u32, t: u32, vertex_positions: &[Vector3], clamped: bool) -> f32 {
        let count = self.count;
        let (v, t) = (v as usize, t as usize);
        quadric_error_attributes(
            &self.quadrics[v],
            &self.gradients[v * count..v * count + count],
            &vertex_positions[t],
            &self.values[t * count..t * count + count],
            clamped,
        )
    }
}

/// `fillFaceQuadrics`.
fn fill_face_quadrics(
    vertex_quadrics: &mut [Quadric],
    indices: &[u32],
    vertex_positions: &[Vector3],
    remap: &[u32],
) {
    for triangle in indices.as_chunks::<3>().0 {
        let [i0, i1, i2] = [triangle[0], triangle[1], triangle[2]].map(|index| index as usize);

        let q = Quadric::from_triangle(
            &vertex_positions[i0],
            &vertex_positions[i1],
            &vertex_positions[i2],
            1.0,
        );

        vertex_quadrics[remap[i0] as usize].add(&q);
        vertex_quadrics[remap[i1] as usize].add(&q);
        vertex_quadrics[remap[i2] as usize].add(&q);
    }
}

/// `fillVertexQuadrics`: a point quadric at each position for regularization.
fn fill_vertex_quadrics(
    vertex_quadrics: &mut [Quadric],
    vertex_positions: &[Vector3],
    remap: &[u32],
    vertex_lock: Option<&[VertexFlags]>,
    options: SimplifyOptions,
) {
    // by default, we use a very small weight to improve triangulation and numerical stability without affecting the shape or error
    let factor = if options.regularize_light {
        1e-2
    } else if options.regularize {
        1e-1
    } else {
        1e-7
    };

    for (i, quadric) in vertex_quadrics.iter_mut().enumerate() {
        if remap[i] != i as u32 {
            continue;
        }

        // increase regularization weight for vertices marked as priority; for now we only examine the primary vertex
        let priority = vertex_lock.is_some_and(|lock| lock[i].contains(VertexFlags::PRIORITY));

        let p = &vertex_positions[i];
        let w = quadric.w * if priority { 1.0 } else { factor };

        quadric.add(&Quadric::from_point(p.x, p.y, p.z, w));
    }
}

/// `fillEdgeQuadrics`: edge quadrics along borders and seams.
fn fill_edge_quadrics(
    vertex_quadrics: &mut [Quadric],
    indices: &[u32],
    vertex_positions: &[Vector3],
    topology: &Topology,
) {
    const NEXT: [usize; 4] = [1, 2, 0, 1];
    let Topology {
        remap,
        kind: vertex_kind,
        loop_,
        loopback,
        ..
    } = topology;

    for triangle in indices.as_chunks::<3>().0 {
        for e in 0..3 {
            let i0 = triangle[e];
            let i1 = triangle[NEXT[e]];

            let k0 = vertex_kind[i0 as usize];
            let k1 = vertex_kind[i1 as usize];

            // early out: manifold vertices never lie on a border
            if k0 == Kind::Manifold || k1 == Kind::Manifold {
                continue;
            }

            // one of the vertices must be border/seam/fringe; additional checks help validate the border
            if k0 != Kind::Border
                && k0 != Kind::Seam
                && k0 != Kind::Fringe
                && k1 != Kind::Border
                && k1 != Kind::Seam
                && k1 != Kind::Fringe
            {
                continue;
            }

            // if i0/i1 are border/seam we check if they are on the same edge loop
            // note that we need to add the error even for edges that connect e.g. border & locked
            // if we don't do that, the adjacent border->border edge won't have correct errors for corners
            if (k0 == Kind::Border || k0 == Kind::Seam) && loop_[i0 as usize] != i1 {
                continue;
            }

            if (k1 == Kind::Border || k1 == Kind::Seam) && loopback[i1 as usize] != i0 {
                continue;
            }

            // fringe vertices don't have proper edge loops, so we need to add edge quadrics to fringe-fringe or fringe-locked edges
            // note that border-fringe edges are still processed, guarded by the loop check above
            if (k0 == Kind::Fringe || k1 == Kind::Fringe)
                && (k0 == Kind::Complex || k1 == Kind::Complex)
            {
                continue;
            }

            let i2 = triangle[NEXT[e + 1]];

            // we try hard to maintain border edge geometry; seam edges can move more freely
            // due to topological restrictions on collapses, seam quadrics slightly improve collapse structure but aren't critical
            const EDGE_WEIGHT_SEAM: f32 = 0.5; // applied twice due to opposite edges
            const EDGE_WEIGHT_BORDER: f32 = 10.0;

            let edge_weight = if k0 == Kind::Seam || k1 == Kind::Seam {
                EDGE_WEIGHT_SEAM
            } else {
                EDGE_WEIGHT_BORDER
            };

            let (p0, p1, p2) = (
                &vertex_positions[i0 as usize],
                &vertex_positions[i1 as usize],
                &vertex_positions[i2 as usize],
            );
            let mut q = Quadric::from_triangle_edge(p0, p1, p2, edge_weight);
            let mut qt = Quadric::from_triangle(p0, p1, p2, edge_weight);

            // mix edge quadric with triangle quadric to stabilize collapses in both directions; both quadrics inherit edge weight so that their error is added
            qt.w = 0.0;
            q.add(&qt);

            vertex_quadrics[remap[i0 as usize] as usize].add(&q);
            vertex_quadrics[remap[i1 as usize] as usize].add(&q);
        }
    }
}

/// `oppositeVertex`: the third vertex of the unique triangle on edge `a`
/// to `b` (by position), or `NONE`.
fn opposite_vertex(adjacency: &EdgeAdjacency, a: u32, b: u32, remap: &[u32], wedge: &[u32]) -> u32 {
    let rb = remap[b as usize];
    let mut v = a;
    let mut r = NONE;

    loop {
        for edge in adjacency.edges(v) {
            if remap[edge.next as usize] == rb {
                r = if r == NONE { edge.prev } else { a };
            }
        }
        v = wedge[v as usize];
        if v == a {
            break;
        }
    }

    // we only return the opposite vertex if it's unique
    if r == a { NONE } else { r }
}

/// `fillFoldQuadrics`: edge quadrics on sharp folds.
fn fill_fold_quadrics(
    vertex_quadrics: &mut [Quadric],
    adjacency: &EdgeAdjacency,
    vertex_positions: &[Vector3],
    remap: &[u32],
    wedge: &[u32],
) {
    for i0 in 0..vertex_positions.len() as u32 {
        for edge in adjacency.edges(i0) {
            let i1 = edge.next;
            let i2 = edge.prev;

            // since we only process paired edges each edge should occur twice; skip redundant edges
            if remap[i1 as usize] > remap[i0 as usize] {
                continue;
            }

            let i3 = opposite_vertex(adjacency, i1, i0, remap, wedge);

            if i3 == NONE {
                continue;
            }

            let (p0, p1, p2, p3) = (
                &vertex_positions[i0 as usize],
                &vertex_positions[i1 as usize],
                &vertex_positions[i2 as usize],
                &vertex_positions[i3 as usize],
            );
            let p10 = sub(p1, p0);
            let p20 = sub(p2, p0);
            let p30 = sub(p3, p0);

            let normal = cross(&p10, &p20);
            let opposite = cross(&p30, &p10);
            let nl = dot(&normal, &normal).sqrt();
            let ol = dot(&opposite, &opposite).sqrt();

            // identify folds by checking dihedral angle against a somewhat arbitrary ~30 degree cutoff
            if dot(&normal, &opposite) >= -0.85 * nl * ol {
                continue;
            }

            // use both triangles for symmetry since we won't visit this edge again
            let mut q = Quadric::from_triangle_edge(p0, p1, p2, 1.0);
            let qo = Quadric::from_triangle_edge(p0, p1, p3, 1.0);
            q.add(&qo);

            vertex_quadrics[remap[i0 as usize] as usize].add(&q);
            vertex_quadrics[remap[i1 as usize] as usize].add(&q);
        }
    }
}

/// `fillAttributeQuadrics`.
fn fill_attribute_quadrics(
    attributes: &mut Attributes,
    indices: &[u32],
    vertex_positions: &[Vector3],
) {
    let count = attributes.count;
    let mut g = vec![QuadricGrad::default(); count];

    for triangle in indices.as_chunks::<3>().0 {
        let [i0, i1, i2] = [triangle[0], triangle[1], triangle[2]].map(|index| index as usize);

        let qa = Quadric::from_attributes(
            &mut g,
            &vertex_positions[i0],
            &vertex_positions[i1],
            &vertex_positions[i2],
            &attributes.values[i0 * count..i0 * count + count],
            &attributes.values[i1 * count..i1 * count + count],
            &attributes.values[i2 * count..i2 * count + count],
        );

        for vertex in [i0, i1, i2] {
            attributes.quadrics[vertex].add(&qa);
            for (gradient, add) in attributes.gradients[vertex * count..vertex * count + count]
                .iter_mut()
                .zip(&g)
            {
                gradient.add(add);
            }
        }
    }
}

/// `hasTriangleFlip`: does triangle ABC flip when C is replaced with D?
fn has_triangle_flip(a: &Vector3, b: &Vector3, c: &Vector3, d: &Vector3) -> bool {
    let eb = sub(b, a);
    let ec = sub(c, a);
    let ed = sub(d, a);

    let nbc = cross(&eb, &ec);
    let nbd = cross(&eb, &ed);

    let ndp = dot(&nbc, &nbd);
    let abc = dot(&nbc, &nbc);
    let abd = dot(&nbd, &nbd);

    // scale is cos(angle); somewhat arbitrarily set to ~75 degrees
    // note that the "pure" check is ndp <= 0 (90 degree cutoff) but that allows flipping through a series of close-to-90 collapses
    ndp <= 0.25 * (abc * abd).sqrt()
}

/// `hasTriangleFlips`: does collapsing `i0` onto `i1` flip a triangle?
fn has_triangle_flips(
    adjacency: &EdgeAdjacency,
    vertex_positions: &[Vector3],
    collapse_remap: &[u32],
    i0: u32,
    i1: u32,
) -> bool {
    let v0 = &vertex_positions[i0 as usize];
    let v1 = &vertex_positions[i1 as usize];

    for edge in adjacency.edges(i0) {
        let a = collapse_remap[edge.next as usize];
        let b = collapse_remap[edge.prev as usize];

        // skip triangles that will get collapsed by i0->i1 collapse or already got collapsed previously
        if a == i1 || b == i1 || a == b {
            continue;
        }

        // early-out when at least one triangle flips due to a collapse
        if has_triangle_flip(
            &vertex_positions[a as usize],
            &vertex_positions[b as usize],
            v0,
            v1,
        ) {
            return true;
        }
    }

    false
}

/// `getComplexTarget`: the wedge of `target`'s position that wedge `v`
/// moves to.
fn get_complex_target(v: u32, target: u32, remap: &[u32], loop_: &[u32], loopback: &[u32]) -> u32 {
    let r = remap[target as usize];
    let (forward, back) = (loop_[v as usize], loopback[v as usize]);

    // use loop metadata to guide complex collapses towards the correct wedge
    // this works for edges on attribute discontinuities because loop/loopback track the single half-edge without a pair, similar to seams
    if forward != NONE && remap[forward as usize] == r {
        forward
    } else if back != NONE && remap[back as usize] == r {
        back
    } else {
        target
    }
}

/// `boundEdgeCollapses`.
fn bound_edge_collapses(
    adjacency: &EdgeAdjacency,
    index_count: usize,
    vertex_kind: &[Kind],
) -> usize {
    let mut dual_count = 0;

    for (i, &k) in vertex_kind.iter().enumerate() {
        let e = (adjacency.offsets[i + 1] - adjacency.offsets[i]) as usize;
        dual_count += if k == Kind::Manifold || k == Kind::Seam {
            e
        } else {
            0
        };
    }

    // pad capacity by 3 so that we can check for overflow once per triangle instead of once per edge
    (index_count - dual_count / 2) + 3
}

/// `pickEdgeCollapses`: the collapsible edges of `indices` into `collapses`,
/// returning their count.
fn pick_edge_collapses(collapses: &mut [Collapse], indices: &[u32], topology: &Topology) -> usize {
    const NEXT: [usize; 3] = [1, 2, 0];
    let Topology {
        remap,
        kind: vertex_kind,
        loop_,
        loopback,
        ..
    } = topology;
    let collapse_capacity = collapses.len();
    let mut collapse_count = 0;

    for triangle in indices.as_chunks::<3>().0 {
        // this should never happen as boundEdgeCollapses should give an upper bound for the collapse count, but in an unlikely event it does we can just drop extra collapses
        if collapse_count + 3 > collapse_capacity {
            break;
        }

        for e in 0..3 {
            let i0 = triangle[e];
            let i1 = triangle[NEXT[e]];

            // this can happen either when input has a zero-length edge, or when we perform collapses for complex
            // topology w/seams and collapse a manifold vertex that connects to both wedges onto one of them
            // we leave edges like this alone since they may be important for preserving mesh integrity
            if remap[i0 as usize] == remap[i1 as usize] {
                continue;
            }

            let k0 = vertex_kind[i0 as usize];
            let k1 = vertex_kind[i1 as usize];

            // the edge has to be collapsible in at least one direction
            if (can_collapse(k0, k1) | can_collapse(k1, k0)) == 0 {
                continue;
            }

            // manifold and seam edges should occur twice (i0->i1 and i1->i0) - skip redundant edges
            if HAS_OPPOSITE[k0 as usize][k1 as usize] != 0
                && remap[i1 as usize] > remap[i0 as usize]
            {
                continue;
            }

            // two vertices are on a border or a seam, but there's no direct edge between them
            // this indicates that they belong to two different edge loops and we should not collapse this edge
            // loop[] and loopback[] track half edges so we only need to check one of them
            if (k0 == Kind::Border || k0 == Kind::Seam)
                && k1 != Kind::Manifold
                && loop_[i0 as usize] != i1
            {
                continue;
            }
            if (k1 == Kind::Border || k1 == Kind::Seam)
                && k0 != Kind::Manifold
                && loopback[i1 as usize] != i0
            {
                continue;
            }

            // edge can be collapsed in either direction - we will pick the one with minimum error
            // note: we evaluate error later during collapse ranking, here we just tag the edge as bidirectional
            collapses[collapse_count] = if (can_collapse(k0, k1) & can_collapse(k1, k0)) != 0 {
                Collapse {
                    v0: i0,
                    v1: i1,
                    bits: 1,
                }
            } else {
                // edge can only be collapsed in one direction
                let forward = can_collapse(k0, k1) != 0;
                Collapse {
                    v0: if forward { i0 } else { i1 },
                    v1: if forward { i1 } else { i0 },
                    bits: 0,
                }
            };
            collapse_count += 1;
        }
    }

    collapse_count
}

/// `rankEdgeCollapses`: each collapse's error, in the cheaper direction of
/// bidirectional edges.
fn rank_edge_collapses(
    collapses: &mut [Collapse],
    vertex_positions: &[Vector3],
    vertex_quadrics: &[Quadric],
    attributes: &Attributes,
    topology: &Topology,
    clamped: bool,
) {
    let Topology {
        remap,
        wedge,
        kind: vertex_kind,
        loop_,
        loopback,
    } = topology;

    for c in collapses {
        let i0 = c.v0;
        let i1 = c.v1;
        let bidi = c.bits != 0;

        let mut ei =
            vertex_quadrics[remap[i0 as usize] as usize].error(&vertex_positions[i1 as usize]);
        let mut ej = if bidi {
            vertex_quadrics[remap[i1 as usize] as usize].error(&vertex_positions[i0 as usize])
        } else {
            f32::MAX
        };

        if attributes.count > 0 {
            ei += attributes.error(i0, i1, vertex_positions, clamped);
            ej += if bidi {
                attributes.error(i1, i0, vertex_positions, clamped)
            } else {
                0.0
            };

            // seam edges need to aggregate attribute errors between primary and secondary edges, as attribute quadrics are separate
            if vertex_kind[i0 as usize] == Kind::Seam {
                // for seam collapses we need to find the seam pair; this is a bit tricky since we need to rely on edge loops as target vertex may be locked (and thus have more than two wedges)
                let s0 = wedge[i0 as usize];
                let s1 = if loop_[i0 as usize] == i1 {
                    loopback[s0 as usize]
                } else {
                    loop_[s0 as usize]
                };

                // note: this should never happen, but play it safe as meshoptimizer does
                let s1 = if s1 != NONE { s1 } else { wedge[i1 as usize] };

                ei += attributes.error(s0, s1, vertex_positions, clamped);
                ej += if bidi {
                    attributes.error(s1, s0, vertex_positions, clamped)
                } else {
                    0.0
                };
            } else {
                // complex edges can have multiple wedges, so we need to aggregate errors for all wedges based on the selected target
                if matches!(vertex_kind[i0 as usize], Kind::Complex | Kind::Fringe) {
                    let mut v = wedge[i0 as usize];
                    while v != i0 {
                        let t = get_complex_target(v, i1, remap, loop_, loopback);
                        ei += attributes.error(v, t, vertex_positions, clamped);
                        v = wedge[v as usize];
                    }
                }

                if matches!(vertex_kind[i1 as usize], Kind::Complex | Kind::Fringe) && bidi {
                    let mut v = wedge[i1 as usize];
                    while v != i1 {
                        let t = get_complex_target(v, i0, remap, loop_, loopback);
                        ej += attributes.error(v, t, vertex_positions, clamped);
                        v = wedge[v as usize];
                    }
                }
            }
        }

        // pick edge direction with minimal error (branchless: the comparison
        // is a coin flip, and compilers turn selects of it into branches)
        let rev = (u32::from(bidi) & u32::from(ej < ei)).wrapping_neg();
        let swap = (i0 ^ i1) & rev;

        c.v0 = i0 ^ swap;
        c.v1 = i1 ^ swap;
        c.bits = min(ei, ej).to_bits();
    }
}

/// `sortEdgeCollapses`: `sort_order` lists the collapses by error.
fn sort_edge_collapses(sort_order: &mut [u32], collapses: &[Collapse]) {
    // we use counting sort to order collapses by error; since the exact sort order is not as critical,
    // only top 12 bits of exponent+mantissa (8 bits of exponent and 4 bits of mantissa) are used.
    // to avoid excessive stack usage, we clamp the exponent range as collapses with errors much higher than 1 are not useful.
    const SORT_BITS: u32 = 12;
    const SORT_BINS: usize = 2048 + 512; // exponent range [-127, 32)

    // skip sign bit since error is non-negative
    let key = |collapse: &Collapse| {
        (((collapse.bits << 1) >> (32 - SORT_BITS)) as usize).min(SORT_BINS - 1)
    };

    // fill histogram for counting sort
    let mut histogram = [0u32; SORT_BINS];
    for collapse in collapses {
        histogram[key(collapse)] += 1;
    }

    // compute offsets based on histogram data
    let mut histogram_sum = 0;
    for entry in &mut histogram {
        let count = *entry;
        *entry = histogram_sum;
        histogram_sum += count;
    }

    // compute sort order based on offsets
    for (i, collapse) in collapses.iter().enumerate() {
        let entry = &mut histogram[key(collapse)];
        sort_order[*entry as usize] = i as u32;
        *entry += 1;
    }
}

/// `performEdgeCollapses`: collapses in error order until a limit, never
/// moving a vertex twice or onto a moved vertex in one pass. Returns the
/// number of collapses and raises `result_error` to their largest error.
#[allow(clippy::too_many_arguments)]
fn perform_edge_collapses(
    collapse_remap: &mut [u32],
    collapse_locked: &mut [bool],
    collapses: &[Collapse],
    collapse_order: &[u32],
    topology: &Topology,
    vertex_positions: &[Vector3],
    adjacency: &EdgeAdjacency,
    triangle_collapse_goal: usize,
    error_limit: f32,
    result_error: &mut f32,
) -> usize {
    let Topology {
        remap,
        wedge,
        kind: vertex_kind,
        loop_,
        loopback,
    } = topology;
    let collapse_count = collapses.len();
    let mut edge_collapses = 0;
    let mut triangle_collapses = 0;

    // most collapses remove 2 triangles; use this to establish a bound on the pass in terms of error limit
    // note that edge_collapse_goal is an estimate; triangle_collapse_goal will be used to actually limit collapses
    let mut edge_collapse_goal = triangle_collapse_goal / 2;

    for &order in collapse_order {
        let c = collapses[order as usize];
        let error = c.error();

        if error > error_limit {
            break;
        }

        if triangle_collapses >= triangle_collapse_goal {
            break;
        }

        // we limit the error in each pass based on the error of optimal last collapse; since many collapses will be locked
        // as they will share vertices with other successful collapses, we need to increase the acceptable error by some factor
        let error_goal = if edge_collapse_goal < collapse_count {
            1.5 * collapses[collapse_order[edge_collapse_goal] as usize].error()
        } else {
            f32::MAX
        };

        // on average, each collapse is expected to lock 6 other collapses; to avoid degenerate passes on meshes with odd
        // topology, we only abort if we got over 1/6 collapses accordingly.
        if error > error_goal
            && error > *result_error
            && triangle_collapses > triangle_collapse_goal / 6
        {
            break;
        }

        let i0 = c.v0;
        let i1 = c.v1;

        let r0 = remap[i0 as usize];
        let r1 = remap[i1 as usize];

        let kind = vertex_kind[i0 as usize];

        // we don't collapse vertices that had source or target vertex involved in a collapse
        // it's important to not move the vertices twice since it complicates the tracking/remapping logic
        // it's important to not move other vertices towards a moved vertex to preserve error since we don't re-rank collapses mid-pass
        if collapse_locked[r0 as usize] | collapse_locked[r1 as usize] {
            continue;
        }

        if has_triangle_flips(adjacency, vertex_positions, collapse_remap, r0, r1) {
            // adjust collapse goal since this collapse is invalid and shouldn't factor into error goal
            edge_collapse_goal += 1;
            continue;
        }

        if kind == Kind::Complex || kind == Kind::Fringe {
            collapse_remap[i0 as usize] = i1;

            // remap all vertices in the complex to the target vertex, using the same ranking that we used to evaluate the collapses
            let mut v = wedge[i0 as usize];
            while v != i0 {
                collapse_remap[v as usize] = get_complex_target(v, i1, remap, loop_, loopback);
                v = wedge[v as usize];
            }
        } else if kind == Kind::Seam {
            // for seam collapses we need to move the seam pair together; this is a bit tricky since we need to rely on edge loops as target vertex may be locked (and thus have more than two wedges)
            let s0 = wedge[i0 as usize];
            let s1 = if loop_[i0 as usize] == i1 {
                loopback[s0 as usize]
            } else {
                loop_[s0 as usize]
            };

            // note: this should never happen, but play it safe as meshoptimizer does
            let s1 = if s1 != NONE { s1 } else { wedge[i1 as usize] };

            collapse_remap[i0 as usize] = i1;
            collapse_remap[s0 as usize] = s1;
        } else {
            collapse_remap[i0 as usize] = i1;
        }

        // note: we technically don't need to lock r1 if it's a locked vertex, as it can't move and its quadric won't be used
        // however, this results in slightly worse error on some meshes because the locked collapses get an unfair advantage wrt scheduling
        collapse_locked[r0 as usize] = true;
        collapse_locked[r1 as usize] = true;

        // border edges collapse 1 triangle, other edges collapse 2 or more
        // fringe edges often collapse 1 but may also collapse 2 because they don't respect loops, so we conservatively assume 2
        triangle_collapses += if kind == Kind::Border { 1 } else { 2 };
        edge_collapses += 1;

        *result_error = max(*result_error, error);
    }

    edge_collapses
}

/// `updateQuadrics`: moves the quadrics of collapsed vertices onto their
/// targets, and with attributes raises `vertex_error` to the position error.
fn update_quadrics(
    collapse_remap: &[u32],
    vertex_quadrics: &mut [Quadric],
    attributes: &mut Attributes,
    vertex_positions: &[Vector3],
    remap: &[u32],
    vertex_error: &mut f32,
) {
    let count = attributes.count;

    for (i, &target) in collapse_remap.iter().enumerate() {
        if target == i as u32 {
            continue;
        }

        let i0 = i;
        let i1 = target as usize;

        let r0 = remap[i0] as usize;
        let r1 = remap[i1] as usize;

        // ensure we only update vertex_quadrics once: primary vertex must be moved if any wedge is moved
        if i0 == r0 {
            let q = vertex_quadrics[r0];
            vertex_quadrics[r1].add(&q);
        }

        if count > 0 {
            let q = attributes.quadrics[i0];
            attributes.quadrics[i1].add(&q);
            for k in 0..count {
                let g = attributes.gradients[i0 * count + k];
                attributes.gradients[i1 * count + k].add(&g);
            }

            if i0 == r0 {
                // when attributes are used, distance error needs to be recomputed as collapses don't track it; it is safe to do this after the quadric adjustment
                let derr = vertex_quadrics[r0].error(&vertex_positions[r1]);
                *vertex_error = max(*vertex_error, derr);
            }
        }
    }
}

/// `remapIndexBuffer`: applies the collapses and drops triangles they
/// degenerate, returning the new index count.
fn remap_index_buffer(indices: &mut [u32], collapse_remap: &[u32], remap: &[u32]) -> usize {
    let mut write = 0;

    for i in (0..indices.len()).step_by(3) {
        let v0 = collapse_remap[indices[i] as usize];
        let v1 = collapse_remap[indices[i + 1] as usize];
        let v2 = collapse_remap[indices[i + 2] as usize];

        // collapse zero area triangles even if they are not topologically degenerate
        // this is required to cleanup manifold->seam collapses when a vertex is collapsed onto a seam pair
        // as well as complex collapses and some other cases where cross wedge collapses are performed
        let r0 = remap[v0 as usize];
        let r1 = remap[v1 as usize];
        let r2 = remap[v2 as usize];

        if r0 != r1 && r0 != r2 && r1 != r2 {
            indices[write] = v0;
            indices[write + 1] = v1;
            indices[write + 2] = v2;
            write += 3;
        }
    }

    write
}

/// `remapEdgeLoops`.
fn remap_edge_loops(loop_: &mut [u32], collapse_remap: &[u32]) {
    for i in 0..loop_.len() {
        // note: this is a no-op for vertices that were remapped
        // ideally we would clear the loop entries for those for consistency, even though they aren't going to be used
        // however, the remapping process needs loop information for remapped vertices, so this would require a separate pass
        let l = loop_[i];
        if l != NONE {
            let r = collapse_remap[l as usize];

            // i == r is a special case when the seam edge is collapsed in a direction opposite to where loop goes
            loop_[i] = if i as u32 == r {
                let next = loop_[l as usize];
                if next != NONE {
                    collapse_remap[next as usize]
                } else {
                    NONE
                }
            } else {
                r
            };
        }
    }
}

/// `follow`: the root of `index` with path halving.
fn follow(parents: &mut [u32], mut index: u32) -> u32 {
    while index != parents[index as usize] {
        let parent = parents[index as usize];
        parents[index as usize] = parents[parent as usize];
        index = parent;
    }
    index
}

/// `buildComponents`: sequential connected component indices of every
/// vertex; returns their count.
fn build_components(components: &mut [u32], indices: &[u32], remap: &[u32]) -> usize {
    const NEXT: [usize; 4] = [1, 2, 0, 1];
    let vertex_count = components.len();
    for (i, component) in components.iter_mut().enumerate() {
        *component = i as u32;
    }

    // compute a unique (but not sequential!) index for each component via union-find
    for triangle in indices.as_chunks::<3>().0 {
        for e in 0..3 {
            let i0 = triangle[e];
            let i1 = triangle[NEXT[e]];

            let r0 = follow(components, remap[i0 as usize]);
            let r1 = follow(components, remap[i1 as usize]);

            // merge components with larger indices into components with smaller indices
            // this guarantees that the root of the component is always the one with the smallest index
            if r0 != r1 {
                components[r0.max(r1) as usize] = r0.min(r1);
            }
        }
    }

    // make sure each element points to the component root *before* we renumber the components
    for i in 0..vertex_count {
        if remap[i] == i as u32 {
            components[i] = follow(components, i as u32);
        }
    }

    let mut next_component = 0;

    // renumber components using sequential indices
    // a sequential pass is sufficient because component root always has the smallest index
    // note: it is unsafe to use follow() in this pass because we're replacing component links with sequential indices inplace
    for i in 0..vertex_count {
        if remap[i] == i as u32 {
            let root = components[i];
            components[i] = if root == i as u32 {
                next_component += 1;
                next_component - 1
            } else {
                components[root as usize]
            };
        } else {
            components[i] = components[remap[i] as usize];
        }
    }

    next_component as usize
}

/// `measureComponents`: each component's squared radius around the average
/// of its vertices.
fn measure_components(
    component_count: usize,
    components: &[u32],
    vertex_positions: &[Vector3],
) -> Vec<f32> {
    let mut component_errors = vec![0.0f32; component_count * 4];

    // compute approximate sphere center for each component as an average
    for (&c, v) in components.iter().zip(vertex_positions) {
        let c = c as usize * 4;
        component_errors[c] += v.x;
        component_errors[c + 1] += v.y;
        component_errors[c + 2] += v.z;
        component_errors[c + 3] += 1.0; // weight
    }

    // complete the center computation, and reinitialize [3] as a radius
    for sphere in component_errors.as_chunks_mut::<4>().0 {
        let w = sphere[3];
        let iw = if w == 0.0 { 0.0 } else { 1.0 / w };

        sphere[0] *= iw;
        sphere[1] *= iw;
        sphere[2] *= iw;
        sphere[3] = 0.0; // radius
    }

    // compute squared radius for each component
    for (&c, v) in components.iter().zip(vertex_positions) {
        let c = c as usize * 4;
        let dx = v.x - component_errors[c];
        let dy = v.y - component_errors[c + 1];
        let dz = v.z - component_errors[c + 2];
        let r = dx * dx + dy * dy + dz * dz;

        component_errors[c + 3] = max(component_errors[c + 3], r);
    }

    // note: we keep the squared error to make it match quadric error metric
    component_errors
        .as_chunks::<4>()
        .0
        .iter()
        .map(|sphere| sphere[3])
        .collect()
}

/// `pruneComponents`: drops the triangles of components with at most
/// `error_cutoff` error, and sets `nexterror` to the smallest remaining
/// error. Returns the new index count.
fn prune_components(
    indices: &mut [u32],
    components: &[u32],
    component_errors: &[f32],
    error_cutoff: f32,
    nexterror: &mut f32,
) -> usize {
    let mut write = 0;
    let mut min_error = f32::MAX;

    for i in (0..indices.len()).step_by(3) {
        let (v0, v1, v2) = (indices[i], indices[i + 1], indices[i + 2]);
        let c = components[v0 as usize] as usize;

        if component_errors[c] > error_cutoff {
            min_error = min(min_error, component_errors[c]);

            indices[write] = v0;
            indices[write + 1] = v1;
            indices[write + 2] = v2;
            write += 3;
        }
    }

    // update next error with the smallest error of the remaining components
    *nexterror = min_error;
    write
}

/// `meshopt_simplifyEdge` without the sparse, update and debug paths.
fn simplify_edge(
    input: &SimplifyInput<'_>,
    target_index_count: usize,
    target_error: f32,
    options: SimplifyOptions,
) -> Simplified {
    let index_count = input.indices.len();
    let vertex_count = input.positions.len();
    let vertex_lock = input.vertex_flags;
    let mut result = input.indices.to_vec();

    // build adjacency information
    let mut adjacency = EdgeAdjacency::new(index_count, vertex_count);
    adjacency.update(&result, None);

    // build position remap that maps each vertex to the one with identical position
    // wedge table stores next vertex with identical position for each vertex
    let (remap, wedge) = build_position_remap(input.positions, true);

    // classify vertices; vertex kind determines collapse rules, see kCanCollapse
    let mut topology = classify_vertices(&adjacency, remap, wedge, vertex_lock, options);

    let mut vertex_positions = vec![Vector3::default(); vertex_count];
    let vertex_scale = rescale_positions(Some(&mut vertex_positions), input.positions);

    let mut attributes = Attributes {
        count: 0,
        values: Vec::new(),
        quadrics: Vec::new(),
        gradients: Vec::new(),
    };

    if let Some(source) = input
        .attributes
        .as_ref()
        .filter(|source| !source.weights.is_empty())
    {
        // remap attributes to only include ones with weight > 0 to minimize memory/compute overhead for quadrics
        let attribute_remap: Vec<usize> = (0..source.weights.len())
            .filter(|&attribute| source.weights[attribute] > 0.0)
            .collect();

        attributes.count = attribute_remap.len();
        attributes.values = rescale_attributes(source, &attribute_remap);
    }

    let mut vertex_quadrics = vec![Quadric::default(); vertex_count];

    if attributes.count > 0 {
        attributes.quadrics = vec![Quadric::default(); vertex_count];
        attributes.gradients = vec![QuadricGrad::default(); vertex_count * attributes.count];
    }

    fill_face_quadrics(
        &mut vertex_quadrics,
        &result,
        &vertex_positions,
        &topology.remap,
    );
    fill_vertex_quadrics(
        &mut vertex_quadrics,
        &vertex_positions,
        &topology.remap,
        vertex_lock,
        options,
    );
    fill_edge_quadrics(&mut vertex_quadrics, &result, &vertex_positions, &topology);

    if options.preserve_folds {
        fill_fold_quadrics(
            &mut vertex_quadrics,
            &adjacency,
            &vertex_positions,
            &topology.remap,
            &topology.wedge,
        );
    }

    if attributes.count > 0 {
        fill_attribute_quadrics(&mut attributes, &result, &vertex_positions);
    }

    let mut components = Vec::new();
    let mut component_errors = Vec::new();
    let mut component_nexterror = 0.0;

    if options.prune {
        components = vec![0; vertex_count];
        let component_count = build_components(&mut components, &result, &topology.remap);

        component_errors = measure_components(component_count, &components, &vertex_positions);

        component_nexterror = f32::MAX;
        for &error in &component_errors {
            component_nexterror = min(component_nexterror, error);
        }
    }

    let collapse_capacity = bound_edge_collapses(&adjacency, index_count, &topology.kind);

    let mut edge_collapses = vec![Collapse::default(); collapse_capacity];
    let mut collapse_order = vec![0u32; collapse_capacity];
    let mut collapse_remap = vec![0u32; vertex_count];
    let mut collapse_locked = vec![false; vertex_count];

    let mut result_count = index_count;
    let mut result_error = 0.0f32;
    let mut vertex_error = 0.0f32;

    // target_error input is linear; we need to adjust it to match quadricError units
    let error_scale = if options.error_absolute {
        vertex_scale
    } else {
        1.0
    };
    let error_limit = (target_error * target_error) / (error_scale * error_scale);

    while result_count > target_index_count {
        // note: throughout the simplification process adjacency structure reflects welded topology for result-in-progress
        adjacency.update(&result[..result_count], Some(&topology.remap));

        let edge_collapse_count =
            pick_edge_collapses(&mut edge_collapses, &result[..result_count], &topology);

        // no edges can be collapsed any more due to topology restrictions
        if edge_collapse_count == 0 {
            break;
        }

        let collapses = &mut edge_collapses[..edge_collapse_count];
        rank_edge_collapses(
            collapses,
            &vertex_positions,
            &vertex_quadrics,
            &attributes,
            &topology,
            options.error_clamped,
        );

        sort_edge_collapses(&mut collapse_order, collapses);

        let triangle_collapse_goal = (result_count - target_index_count) / 3;

        for (i, target) in collapse_remap.iter_mut().enumerate() {
            *target = i as u32;
        }

        collapse_locked.fill(false);

        let collapses = perform_edge_collapses(
            &mut collapse_remap,
            &mut collapse_locked,
            collapses,
            &collapse_order[..edge_collapse_count],
            &topology,
            &vertex_positions,
            &adjacency,
            triangle_collapse_goal,
            error_limit,
            &mut result_error,
        );

        // no edges can be collapsed any more due to hitting the error limit or triangle collapse limit
        if collapses == 0 {
            break;
        }

        update_quadrics(
            &collapse_remap,
            &mut vertex_quadrics,
            &mut attributes,
            &vertex_positions,
            &topology.remap,
            &mut vertex_error,
        );

        // updateQuadrics will update vertex error if we use attributes, but if we don't then result_error and vertex_error are equivalent
        if attributes.count == 0 {
            vertex_error = result_error;
        }

        // note: we update loops following edge collapses, but after this we might still have stale loop data
        // this can happen when a triangle with a loop edge gets collapsed along a non-loop edge
        // that works since a loop that points to a vertex that is no longer connected is not affecting collapse logic
        remap_edge_loops(&mut topology.loop_, &collapse_remap);
        remap_edge_loops(&mut topology.loopback, &collapse_remap);

        result_count = remap_index_buffer(
            &mut result[..result_count],
            &collapse_remap,
            &topology.remap,
        );

        if options.prune && result_count > target_index_count && component_nexterror <= vertex_error
        {
            result_count = prune_components(
                &mut result[..result_count],
                &components,
                &component_errors,
                vertex_error,
                &mut component_nexterror,
            );
        }
    }

    // at this point, component_nexterror might be stale: component it references may have been removed through a series of edge collapses
    let mut component_nextstale = true;

    // we're done with the regular simplification but we're still short of the target; try pruning more aggressively towards error_limit
    while options.prune && result_count > target_index_count && component_nexterror <= error_limit {
        let component_cutoff = min(error_limit, component_nexterror * 1.5);

        // track maximum error in eligible components as we are increasing resulting error
        let mut component_maxerror = 0.0f32;
        for &error in &component_errors {
            if error > component_maxerror && error <= component_cutoff {
                component_maxerror = error;
            }
        }

        let new_count = prune_components(
            &mut result[..result_count],
            &components,
            &component_errors,
            component_cutoff,
            &mut component_nexterror,
        );
        if new_count == result_count && !component_nextstale {
            break;
        }

        component_nextstale = false; // pruneComponents guarantees next error is up to date
        result_count = new_count;
        result_error = max(result_error, component_maxerror);
    }

    result.truncate(result_count);

    // result_error is quadratic; we need to remap it back to linear
    Simplified {
        indices: result,
        error: result_error.sqrt() * error_scale,
    }
}
