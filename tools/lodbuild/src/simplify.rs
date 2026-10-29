use std::ops::{BitOr, BitOrAssign};

use engine_math::{max, min};
use engine_settings::{Error, ErrorCode};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SimplifyOptions {
    pub lock_border: bool,
    pub error_absolute: bool,
    pub prune: bool,
    pub regularize: bool,
    pub regularize_light: bool,
    pub permissive: bool,
    pub preserve_folds: bool,
    pub error_clamped: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct VertexFlags(u8);

impl VertexFlags {
    pub const NONE: Self = Self(0);
    pub const LOCK: Self = Self(1);
    pub const PROTECT: Self = Self(2);
    pub const PRIORITY: Self = Self(4);

    pub const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    pub const fn bits(self) -> u8 {
        self.0
    }

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

#[derive(Clone, Copy, Debug)]
pub struct VertexAttributes<'a> {
    pub values: &'a [f32],
    pub stride: usize,
    pub weights: &'a [f32],
}

#[derive(Clone, Copy, Debug)]
pub struct SimplifyInput<'a> {
    pub indices: &'a [u32],
    pub positions: &'a [[f32; 3]],
    pub attributes: Option<VertexAttributes<'a>>,
    pub vertex_flags: Option<&'a [VertexFlags]>,
}

impl<'a> SimplifyInput<'a> {
    pub fn new(indices: &'a [u32], positions: &'a [[f32; 3]]) -> Self {
        Self {
            indices,
            positions,
            attributes: None,
            vertex_flags: None,
        }
    }

    #[must_use]
    pub fn with_attributes(mut self, values: &'a [f32], stride: usize, weights: &'a [f32]) -> Self {
        self.attributes = Some(VertexAttributes {
            values,
            stride,
            weights,
        });
        self
    }

    #[must_use]
    pub fn with_vertex_flags(mut self, flags: &'a [VertexFlags]) -> Self {
        self.vertex_flags = Some(flags);
        self
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Simplified {
    pub indices: Vec<u32>,
    pub error: f32,
}

fn mesh_error(path: impl Into<String>) -> Error {
    Error::new(ErrorCode::InvalidMesh, path)
}

fn exactly(path: &str, wanted: usize, got: usize) -> Error {
    mesh_error(path)
        .range(wanted as f64, wanted as f64)
        .got(got)
}

fn too_large(vertices: usize, indices: usize) -> Error {
    if vertices >= NONE as usize {
        mesh_error("$.positions")
            .at_most(u64::from(NONE - 1))
            .got(vertices)
    } else {
        mesh_error("$.indices")
            .at_most(u64::from(NONE))
            .got(indices)
    }
}

pub fn simplify(
    input: &SimplifyInput<'_>,
    target_index_count: usize,
    target_error: f32,
    options: SimplifyOptions,
) -> Result<Simplified, Error> {
    validate(input, target_error)?;
    Ok(simplify_edge(
        input,
        target_index_count,
        target_error,
        options,
    ))
}

pub fn simplify_scale(positions: &[[f32; 3]]) -> f32 {
    rescale_positions(None, positions)
}

pub fn protect_attribute_seams(
    positions: &[[f32; 3]],
    values: &[f32],
    stride: usize,
    mask: u32,
    flags: &mut [VertexFlags],
) -> Result<(), Error> {
    let vertices = positions.len();
    if vertices.checked_mul(stride) != Some(values.len()) {
        return Err(exactly(
            "$.values",
            vertices.saturating_mul(stride),
            values.len(),
        ));
    }
    if flags.len() != vertices {
        return Err(exactly("$.flags", vertices, flags.len()));
    }
    if vertices >= NONE as usize {
        return Err(too_large(vertices, 0));
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

fn validate(input: &SimplifyInput<'_>, target_error: f32) -> Result<(), Error> {
    let vertices = input.positions.len();
    let indices = input.indices.len();
    if !indices.is_multiple_of(3) {
        return Err(mesh_error("$.indices").got(indices));
    }
    if vertices >= NONE as usize || indices > NONE as usize {
        return Err(too_large(vertices, indices));
    }
    if let Some(index) = input
        .indices
        .iter()
        .position(|&value| value as usize >= vertices)
    {
        return Err(mesh_error(format!("$.indices[{index}]"))
            .at_most(vertices.saturating_sub(1) as u64)
            .got(input.indices[index]));
    }
    if let Some(attributes) = &input.attributes {
        if attributes.weights.len() > attributes.stride {
            return Err(mesh_error("$.attributes.weights")
                .at_most(attributes.stride as u64)
                .got(attributes.weights.len()));
        }
        if vertices.checked_mul(attributes.stride) != Some(attributes.values.len()) {
            return Err(exactly(
                "$.attributes.values",
                vertices.saturating_mul(attributes.stride),
                attributes.values.len(),
            ));
        }
        if let Some(attribute) = attributes
            .weights
            .iter()
            .position(|weight| !weight.is_finite() || *weight < 0.0)
        {
            return Err(mesh_error(format!("$.attributes.weights[{attribute}]"))
                .at_least(0)
                .got(attributes.weights[attribute]));
        }
    }
    if let Some(flags) = input.vertex_flags
        && flags.len() != vertices
    {
        return Err(exactly("$.vertex_flags", vertices, flags.len()));
    }
    if target_error.is_nan() || target_error < 0.0 {
        return Err(mesh_error("$.target_error").at_least(0).got(target_error));
    }
    Ok(())
}

const NONE: u32 = u32::MAX;

struct EdgeAdjacency {
    offsets: Vec<u32>,
    data: Vec<Edge>,
}

#[derive(Clone, Copy, Default)]
struct Edge {
    next: u32,
    prev: u32,
}

impl EdgeAdjacency {
    fn new(index_count: usize, vertex_count: usize) -> Self {
        Self {
            offsets: vec![0; vertex_count + 1],
            data: vec![Edge::default(); index_count],
        }
    }

    fn update(&mut self, indices: &[u32], remap: Option<&[u32]>) {
        match remap {
            Some(remap) => self.fill(indices, |vertex| remap[vertex as usize]),
            None => self.fill(indices, |vertex| vertex),
        }
    }

    fn fill(&mut self, indices: &[u32], map: impl Fn(u32) -> u32) {
        let Self { offsets, data } = self;
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

fn hash_buckets2(count: usize) -> usize {
    let mut buckets = 1;
    while buckets < count + count / 4 {
        buckets *= 2;
    }
    buckets
}

fn position_hash(position: &[f32; 3]) -> u32 {
    let [mut x, mut y, mut z] = position.map(f32::to_bits);
    x = if x == 0x8000_0000 { 0 } else { x };
    y = if y == 0x8000_0000 { 0 } else { y };
    z = if z == 0x8000_0000 { 0 } else { z };
    x ^= x >> 17;
    y ^= y >> 17;
    z ^= z >> 17;
    x.wrapping_mul(73_856_093) ^ y.wrapping_mul(19_349_663) ^ z.wrapping_mul(83_492_791)
}

fn build_position_remap(positions: &[[f32; 3]], wedges: bool) -> (Vec<u32>, Vec<u32>) {
    let vertex_count = positions.len();
    let table_size = hash_buckets2(vertex_count);
    let hashmod = table_size - 1;
    let mut table = vec![NONE; table_size];
    let mut remap = vec![0; vertex_count];

    for (index, position) in positions.iter().enumerate() {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum Kind {
    Manifold,
    Border,
    Seam,
    Complex,
    Fringe,
    Locked,
}

const KIND_COUNT: usize = 6;

const CAN_COLLAPSE: [[u8; KIND_COUNT]; KIND_COUNT] = [
    [1, 1, 1, 1, 1, 1],
    [0, 1, 0, 0, 1, 1],
    [0, 0, 1, 0, 0, 1],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0],
];

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

fn has_edge(adjacency: &EdgeAdjacency, a: u32, b: u32) -> bool {
    adjacency.edges(a).iter().any(|edge| edge.next == b)
}

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

struct Topology {
    remap: Vec<u32>,
    wedge: Vec<u32>,
    kind: Vec<Kind>,
    loop_: Vec<u32>,
    loopback: Vec<u32>,
}

#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
fn classify_vertices(
    adjacency: &EdgeAdjacency,
    remap: Vec<u32>,
    wedge: Vec<u32>,
    vertex_lock: Option<&[VertexFlags]>,
    options: SimplifyOptions,
) -> Topology {
    let vertex_count = remap.len();
    let mut result = vec![Kind::Manifold; vertex_count];

    let mut openinc = vec![NONE; vertex_count];
    let mut openout = vec![NONE; vertex_count];

    for vertex in 0..vertex_count as u32 {
        for edge in adjacency.edges(vertex) {
            let target = edge.next;

            if target == vertex {
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
                let openi = openinc[i];
                let openo = openout[i];

                result[i] = if openi == NONE && openo == NONE {
                    Kind::Manifold
                } else if openi != NONE
                    && openo != NONE
                    && remap[openi as usize] == remap[openo as usize]
                    && openi != iu
                {
                    Kind::Seam
                } else if openi != iu && openo != iu {
                    Kind::Border
                } else {
                    Kind::Locked
                };
            } else if wedge[wedge[i] as usize] == iu {
                let w = wedge[i];
                let (openiv, openov) = (openinc[i], openout[i]);
                let (openiw, openow) = (openinc[w as usize], openout[w as usize]);

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
                    Kind::Locked
                };
            } else {
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
                result[i] = result[remap[i] as usize];
                continue;
            }

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

            if !protect {
                result[i] = if border { Kind::Fringe } else { Kind::Complex };
            }
        }
    }

    if let Some(lock) = vertex_lock {
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

fn sub(a: &Vector3, b: &Vector3) -> Vector3 {
    Vector3 {
        x: a.x - b.x,
        y: a.y - b.y,
        z: a.z - b.z,
    }
}

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

#[derive(Clone, Copy, Debug, Default)]
struct QuadricGrad {
    gx: f32,
    gy: f32,
    gz: f32,
    gw: f32,
}

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

    fn eval(&self, v: &Vector3) -> f32 {
        let mut rx = (self.b0 + self.a10 * v.y) * 2.0;
        let mut ry = (self.b1 + self.a21 * v.z) * 2.0;
        let mut rz = (self.b2 + self.a20 * v.x) * 2.0;

        rx += self.a00 * v.x;
        ry += self.a11 * v.y;
        rz += self.a22 * v.z;

        self.c + rx * v.x + ry * v.y + rz * v.z
    }

    fn error(&self, v: &Vector3) -> f32 {
        let r = self.eval(v);
        let s = if self.w == 0.0 { 0.0 } else { 1.0 / self.w };
        r.abs() * s
    }

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

    fn from_triangle(p0: &Vector3, p1: &Vector3, p2: &Vector3, weight: f32) -> Self {
        let p10 = sub(p1, p0);
        let p20 = sub(p2, p0);

        let mut normal = cross(&p10, &p20);
        let area = normalize(&mut normal);

        let distance = dot(&normal, p0);

        Self::from_plane(
            normal.x,
            normal.y,
            normal.z,
            -distance,
            area.sqrt() * weight,
        )
    }

    fn from_triangle_edge(p0: &Vector3, p1: &Vector3, p2: &Vector3, weight: f32) -> Self {
        let p10 = sub(p1, p0);

        let lengthsq = dot(&p10, &p10);
        let length = lengthsq.sqrt();

        let p20 = sub(p2, p0);
        let p20p = dot(&p20, &p10);

        let mut perp = Vector3 {
            x: p20.x * lengthsq - p10.x * p20p,
            y: p20.y * lengthsq - p10.y * p20p,
            z: p20.z * lengthsq - p10.z * p20p,
        };
        normalize(&mut perp);

        let distance = dot(&perp, p0);

        Self::from_plane(perp.x, perp.y, perp.z, -distance, length * weight)
    }

    fn from_attributes(
        gradients: &mut [QuadricGrad],
        p0: &Vector3,
        p1: &Vector3,
        p2: &Vector3,
        va0: &[f32],
        va1: &[f32],
        va2: &[f32],
    ) -> Self {
        let p10 = sub(p1, p0);
        let p20 = sub(p2, p0);

        let normal = cross(&p10, &p20);
        let area = dot(&normal, &normal).sqrt() * 0.5;

        let w = area;

        let v0 = &p10;
        let v1 = &p20;
        let d00 = dot(v0, v0);
        let d01 = dot(v0, v1);
        let d11 = dot(v1, v1);
        let denom = d00 * d11 - d01 * d01;
        let denomr = if denom == 0.0 { 0.0 } else { 1.0 / denom };

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

            let gx = gx1 * (a1 - a0) + gx2 * (a2 - a0);
            let gy = gy1 * (a1 - a0) + gy2 * (a2 - a0);
            let gz = gz1 * (a1 - a0) + gz2 * (a2 - a0);
            let gw = a0 - p0.x * gx - p0.y * gy - p0.z * gz;

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

fn quadric_error_attributes(
    q: &Quadric,
    g: &[QuadricGrad],
    v: &Vector3,
    va: &[f32],
    clamped: bool,
) -> f32 {
    let mut r = q.eval(v);

    for (gradient, &a) in g.iter().zip(va) {
        let g = v.x * gradient.gx + v.y * gradient.gy + v.z * gradient.gz + gradient.gw;
        r += a * (a * q.w - 2.0 * g);
    }

    let e = r.abs();
    if clamped { min(q.w, e) } else { e }
}

struct Attributes {
    count: usize,
    values: Vec<f32>,
    quadrics: Vec<Quadric>,
    gradients: Vec<QuadricGrad>,
}

impl Attributes {
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

fn fill_vertex_quadrics(
    vertex_quadrics: &mut [Quadric],
    vertex_positions: &[Vector3],
    remap: &[u32],
    vertex_lock: Option<&[VertexFlags]>,
    options: SimplifyOptions,
) {
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

        let priority = vertex_lock.is_some_and(|lock| lock[i].contains(VertexFlags::PRIORITY));

        let p = &vertex_positions[i];
        let w = quadric.w * if priority { 1.0 } else { factor };

        quadric.add(&Quadric::from_point(p.x, p.y, p.z, w));
    }
}

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

            if k0 == Kind::Manifold || k1 == Kind::Manifold {
                continue;
            }

            if k0 != Kind::Border
                && k0 != Kind::Seam
                && k0 != Kind::Fringe
                && k1 != Kind::Border
                && k1 != Kind::Seam
                && k1 != Kind::Fringe
            {
                continue;
            }

            if (k0 == Kind::Border || k0 == Kind::Seam) && loop_[i0 as usize] != i1 {
                continue;
            }

            if (k1 == Kind::Border || k1 == Kind::Seam) && loopback[i1 as usize] != i0 {
                continue;
            }

            if (k0 == Kind::Fringe || k1 == Kind::Fringe)
                && (k0 == Kind::Complex || k1 == Kind::Complex)
            {
                continue;
            }

            let i2 = triangle[NEXT[e + 1]];

            const EDGE_WEIGHT_SEAM: f32 = 0.5;
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

            qt.w = 0.0;
            q.add(&qt);

            vertex_quadrics[remap[i0 as usize] as usize].add(&q);
            vertex_quadrics[remap[i1 as usize] as usize].add(&q);
        }
    }
}

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

    if r == a { NONE } else { r }
}

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

            if dot(&normal, &opposite) >= -0.85 * nl * ol {
                continue;
            }

            let mut q = Quadric::from_triangle_edge(p0, p1, p2, 1.0);
            let qo = Quadric::from_triangle_edge(p0, p1, p3, 1.0);
            q.add(&qo);

            vertex_quadrics[remap[i0 as usize] as usize].add(&q);
            vertex_quadrics[remap[i1 as usize] as usize].add(&q);
        }
    }
}

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

fn has_triangle_flip(a: &Vector3, b: &Vector3, c: &Vector3, d: &Vector3) -> bool {
    let eb = sub(b, a);
    let ec = sub(c, a);
    let ed = sub(d, a);

    let nbc = cross(&eb, &ec);
    let nbd = cross(&eb, &ed);

    let ndp = dot(&nbc, &nbd);
    let abc = dot(&nbc, &nbc);
    let abd = dot(&nbd, &nbd);

    ndp <= 0.25 * (abc * abd).sqrt()
}

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

        if a == i1 || b == i1 || a == b {
            continue;
        }

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

fn get_complex_target(v: u32, target: u32, remap: &[u32], loop_: &[u32], loopback: &[u32]) -> u32 {
    let r = remap[target as usize];
    let (forward, back) = (loop_[v as usize], loopback[v as usize]);

    if forward != NONE && remap[forward as usize] == r {
        forward
    } else if back != NONE && remap[back as usize] == r {
        back
    } else {
        target
    }
}

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

    (index_count - dual_count / 2) + 3
}

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
        if collapse_count + 3 > collapse_capacity {
            break;
        }

        for e in 0..3 {
            let i0 = triangle[e];
            let i1 = triangle[NEXT[e]];

            if remap[i0 as usize] == remap[i1 as usize] {
                continue;
            }

            let k0 = vertex_kind[i0 as usize];
            let k1 = vertex_kind[i1 as usize];

            if (can_collapse(k0, k1) | can_collapse(k1, k0)) == 0 {
                continue;
            }

            if HAS_OPPOSITE[k0 as usize][k1 as usize] != 0
                && remap[i1 as usize] > remap[i0 as usize]
            {
                continue;
            }

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

            collapses[collapse_count] = if (can_collapse(k0, k1) & can_collapse(k1, k0)) != 0 {
                Collapse {
                    v0: i0,
                    v1: i1,
                    bits: 1,
                }
            } else {
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

            if vertex_kind[i0 as usize] == Kind::Seam {
                let s0 = wedge[i0 as usize];
                let s1 = if loop_[i0 as usize] == i1 {
                    loopback[s0 as usize]
                } else {
                    loop_[s0 as usize]
                };

                let s1 = if s1 != NONE { s1 } else { wedge[i1 as usize] };

                ei += attributes.error(s0, s1, vertex_positions, clamped);
                ej += if bidi {
                    attributes.error(s1, s0, vertex_positions, clamped)
                } else {
                    0.0
                };
            } else {
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

        let rev = (u32::from(bidi) & u32::from(ej < ei)).wrapping_neg();
        let swap = (i0 ^ i1) & rev;

        c.v0 = i0 ^ swap;
        c.v1 = i1 ^ swap;
        c.bits = min(ei, ej).to_bits();
    }
}

fn sort_edge_collapses(sort_order: &mut [u32], collapses: &[Collapse]) {
    const SORT_BITS: u32 = 12;
    const SORT_BINS: usize = 2048 + 512;

    let key = |collapse: &Collapse| {
        (((collapse.bits << 1) >> (32 - SORT_BITS)) as usize).min(SORT_BINS - 1)
    };

    let mut histogram = [0u32; SORT_BINS];
    for collapse in collapses {
        histogram[key(collapse)] += 1;
    }

    let mut histogram_sum = 0;
    for entry in &mut histogram {
        let count = *entry;
        *entry = histogram_sum;
        histogram_sum += count;
    }

    for (i, collapse) in collapses.iter().enumerate() {
        let entry = &mut histogram[key(collapse)];
        sort_order[*entry as usize] = i as u32;
        *entry += 1;
    }
}

#[expect(clippy::too_many_arguments, reason = "meshoptimizer's own signature")]
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

        let error_goal = if edge_collapse_goal < collapse_count {
            1.5 * collapses[collapse_order[edge_collapse_goal] as usize].error()
        } else {
            f32::MAX
        };

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

        if collapse_locked[r0 as usize] | collapse_locked[r1 as usize] {
            continue;
        }

        if has_triangle_flips(adjacency, vertex_positions, collapse_remap, r0, r1) {
            edge_collapse_goal += 1;
            continue;
        }

        if kind == Kind::Complex || kind == Kind::Fringe {
            collapse_remap[i0 as usize] = i1;

            let mut v = wedge[i0 as usize];
            while v != i0 {
                collapse_remap[v as usize] = get_complex_target(v, i1, remap, loop_, loopback);
                v = wedge[v as usize];
            }
        } else if kind == Kind::Seam {
            let s0 = wedge[i0 as usize];
            let s1 = if loop_[i0 as usize] == i1 {
                loopback[s0 as usize]
            } else {
                loop_[s0 as usize]
            };

            let s1 = if s1 != NONE { s1 } else { wedge[i1 as usize] };

            collapse_remap[i0 as usize] = i1;
            collapse_remap[s0 as usize] = s1;
        } else {
            collapse_remap[i0 as usize] = i1;
        }

        collapse_locked[r0 as usize] = true;
        collapse_locked[r1 as usize] = true;

        triangle_collapses += if kind == Kind::Border { 1 } else { 2 };
        edge_collapses += 1;

        *result_error = max(*result_error, error);
    }

    edge_collapses
}

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
                let derr = vertex_quadrics[r0].error(&vertex_positions[r1]);
                *vertex_error = max(*vertex_error, derr);
            }
        }
    }
}

fn remap_index_buffer(indices: &mut [u32], collapse_remap: &[u32], remap: &[u32]) -> usize {
    let mut write = 0;

    for i in (0..indices.len()).step_by(3) {
        let v0 = collapse_remap[indices[i] as usize];
        let v1 = collapse_remap[indices[i + 1] as usize];
        let v2 = collapse_remap[indices[i + 2] as usize];

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

fn remap_edge_loops(loop_: &mut [u32], collapse_remap: &[u32]) {
    for i in 0..loop_.len() {
        let l = loop_[i];
        if l != NONE {
            let r = collapse_remap[l as usize];

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

fn follow(parents: &mut [u32], mut index: u32) -> u32 {
    while index != parents[index as usize] {
        let parent = parents[index as usize];
        parents[index as usize] = parents[parent as usize];
        index = parent;
    }
    index
}

fn build_components(components: &mut [u32], indices: &[u32], remap: &[u32]) -> usize {
    const NEXT: [usize; 4] = [1, 2, 0, 1];
    let vertex_count = components.len();
    for (i, component) in components.iter_mut().enumerate() {
        *component = i as u32;
    }

    for triangle in indices.as_chunks::<3>().0 {
        for e in 0..3 {
            let i0 = triangle[e];
            let i1 = triangle[NEXT[e]];

            let r0 = follow(components, remap[i0 as usize]);
            let r1 = follow(components, remap[i1 as usize]);

            if r0 != r1 {
                components[r0.max(r1) as usize] = r0.min(r1);
            }
        }
    }

    for i in 0..vertex_count {
        if remap[i] == i as u32 {
            components[i] = follow(components, i as u32);
        }
    }

    let mut next_component = 0;

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

fn measure_components(
    component_count: usize,
    components: &[u32],
    vertex_positions: &[Vector3],
) -> Vec<f32> {
    let mut component_errors = vec![0.0f32; component_count * 4];

    for (&c, v) in components.iter().zip(vertex_positions) {
        let c = c as usize * 4;
        component_errors[c] += v.x;
        component_errors[c + 1] += v.y;
        component_errors[c + 2] += v.z;
        component_errors[c + 3] += 1.0;
    }

    for sphere in component_errors.as_chunks_mut::<4>().0 {
        let w = sphere[3];
        let iw = if w == 0.0 { 0.0 } else { 1.0 / w };

        sphere[0] *= iw;
        sphere[1] *= iw;
        sphere[2] *= iw;
        sphere[3] = 0.0;
    }

    for (&c, v) in components.iter().zip(vertex_positions) {
        let c = c as usize * 4;
        let dx = v.x - component_errors[c];
        let dy = v.y - component_errors[c + 1];
        let dz = v.z - component_errors[c + 2];
        let r = dx * dx + dy * dy + dz * dz;

        component_errors[c + 3] = max(component_errors[c + 3], r);
    }

    component_errors
        .as_chunks::<4>()
        .0
        .iter()
        .map(|sphere| sphere[3])
        .collect()
}

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

    *nexterror = min_error;
    write
}

#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
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

    let mut adjacency = EdgeAdjacency::new(index_count, vertex_count);
    adjacency.update(&result, None);

    let (remap, wedge) = build_position_remap(input.positions, true);

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

    let error_scale = if options.error_absolute {
        vertex_scale
    } else {
        1.0
    };
    let error_limit = (target_error * target_error) / (error_scale * error_scale);

    while result_count > target_index_count {
        adjacency.update(&result[..result_count], Some(&topology.remap));

        let edge_collapse_count =
            pick_edge_collapses(&mut edge_collapses, &result[..result_count], &topology);

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

        if attributes.count == 0 {
            vertex_error = result_error;
        }

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

    let mut component_nextstale = true;

    while options.prune && result_count > target_index_count && component_nexterror <= error_limit {
        let component_cutoff = min(error_limit, component_nexterror * 1.5);

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

        component_nextstale = false;
        result_count = new_count;
        result_error = max(result_error, component_maxerror);
    }

    result.truncate(result_count);

    Simplified {
        indices: result,
        error: result_error.sqrt() * error_scale,
    }
}
