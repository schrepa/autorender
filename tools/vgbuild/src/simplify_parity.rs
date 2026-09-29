//! Parity of `engine_assets::simplify`, the engine's Rust port of
//! meshoptimizer's edge collapse simplifier, with the vendored C++ library:
//! both run on generated meshes over option sets, attributes and targets,
//! and must return the same indices and the same error bits. The C++ is
//! built for x86-64 without fused multiply-add (MSVC `/fp:precise`, no
//! `/arch:AVX2`), and Rust never fuses, so every operation rounds alike.
//! Run `cargo test -p engine_vgbuild --release simplify -- --nocapture` for
//! the table and the timings.

use std::time::{Duration, Instant};

use engine_assets::simplify::{self as port, SimplifyInput, SimplifyOptions, VertexFlags};

use crate::ffi::simplifier::{self as meshopt, options as bits};
use crate::procedural;

/// Attribute values per vertex: a normal and texture coordinates.
const STRIDE: usize = 5;

/// A test mesh with a normal and texture coordinates per vertex.
struct TestMesh {
    name: &'static str,
    positions: Vec<[f32; 3]>,
    attributes: Vec<f32>,
    indices: Vec<u32>,
}

impl TestMesh {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            positions: Vec::new(),
            attributes: Vec::new(),
            indices: Vec::new(),
        }
    }

    fn vertex(&mut self, position: [f32; 3], normal: [f32; 3], uv: [f32; 2]) -> u32 {
        self.positions.push(position);
        self.attributes.extend(normal);
        self.attributes.extend(uv);
        (self.positions.len() - 1) as u32
    }

    fn from_geometry(name: &'static str, geometry: &procedural::Geometry) -> Self {
        let mut mesh = Self::new(name);
        for (vertex, (&position, &normal)) in
            geometry.positions.iter().zip(&geometry.normals).enumerate()
        {
            let uv = geometry.uvs.as_ref().map_or([0.0; 2], |uvs| uvs[vertex]);
            mesh.vertex(position, normal, uv);
        }
        mesh.indices.clone_from(&geometry.indices);
        mesh
    }

    /// Appends `other` scaled by `scale` and moved by `offset`.
    fn append(&mut self, other: &Self, scale: f32, offset: [f32; 3]) {
        let base = self.positions.len() as u32;
        for (position, attributes) in other
            .positions
            .iter()
            .zip(other.attributes.as_chunks::<STRIDE>().0)
        {
            let moved = [0, 1, 2].map(|axis| position[axis] * scale + offset[axis]);
            self.positions.push(moved);
            self.attributes.extend(attributes);
        }
        self.indices
            .extend(other.indices.iter().map(|index| base + index));
    }

    fn triangles(&self) -> usize {
        self.indices.len() / 3
    }
}

/// A sphere of `rings` x `segments` quads, its radius rippled by `bump`,
/// with a texture seam and poles whose vertices share positions exactly.
fn uv_sphere(name: &'static str, rings: u32, segments: u32, bump: f32) -> TestMesh {
    let mut mesh = TestMesh::new(name);
    for ring in 0..=rings {
        let v = ring as f32 / rings as f32;
        let theta = v * std::f32::consts::PI;
        let (sin_theta, cos_theta) = engine_math::sin_cos(theta);
        for segment in 0..=segments {
            let u = segment as f32 / segments as f32;
            let phi = (segment % segments) as f32 / segments as f32 * std::f32::consts::TAU;
            let (sin_phi, cos_phi) = engine_math::sin_cos(phi);
            let (direction, radius) = match ring {
                0 => ([0.0, 1.0, 0.0], 1.0),
                _ if ring == rings => ([0.0, -1.0, 0.0], 1.0),
                _ => (
                    [sin_theta * cos_phi, cos_theta, sin_theta * sin_phi],
                    1.0 + bump * engine_math::sin(theta * 5.0) * engine_math::cos(phi * 3.0),
                ),
            };
            mesh.vertex(direction.map(|axis| axis * radius), direction, [u, v]);
        }
    }
    let row = segments + 1;
    for ring in 0..rings {
        for segment in 0..segments {
            let a = ring * row + segment;
            let b = a + row;
            // The poles' quads are single triangles.
            if ring != 0 {
                mesh.indices.extend([a, a + 1, b]);
            }
            if ring != rings - 1 {
                mesh.indices.extend([a + 1, b + 1, b]);
            }
        }
    }
    mesh
}

/// A unit box whose faces are `cells` x `cells` grids with their own
/// normals: every edge is a seam and every corner has three wedges. One
/// cell per face is the 24-vertex faceted cube.
fn faceted_box(name: &'static str, cells: u32) -> TestMesh {
    // Normal, then two tangents whose cross product is the normal.
    let faces: [[[f32; 3]; 3]; 6] = [
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    ];
    let mut mesh = TestMesh::new(name);
    let row = cells + 1;
    for [normal, tangent, bitangent] in faces {
        let base = mesh.positions.len() as u32;
        for y in 0..row {
            for x in 0..row {
                // Axis-aligned tangents keep shared edges bit-identical.
                let (s, t) = (x as f32 / cells as f32 - 0.5, y as f32 / cells as f32 - 0.5);
                let position = [0, 1, 2]
                    .map(|axis| normal[axis] * 0.5 + tangent[axis] * s + bitangent[axis] * t);
                mesh.vertex(
                    position,
                    normal,
                    [x as f32 / cells as f32, y as f32 / cells as f32],
                );
            }
        }
        for y in 0..cells {
            for x in 0..cells {
                let a = base + y * row + x;
                let (b, c) = (a + 1, a + row);
                mesh.indices.extend([a, b, c + 1, a, c + 1, c]);
            }
        }
    }
    mesh
}

/// A flat grid of `cells` x `cells` quads facing up, with open borders.
fn grid(name: &'static str, cells: u32) -> TestMesh {
    let mut mesh = TestMesh::new(name);
    let row = cells + 1;
    for z in 0..row {
        for x in 0..row {
            let (u, v) = (x as f32 / cells as f32, z as f32 / cells as f32);
            mesh.vertex([u, 0.0, v], [0.0, 1.0, 0.0], [u, v]);
        }
    }
    for z in 0..cells {
        for x in 0..cells {
            let a = z * row + x;
            let b = a + row;
            mesh.indices.extend([a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    mesh
}

/// A capped cylinder of `segments` x `rings` side quads with a texture seam;
/// the caps have rims of their own for split normals.
fn cylinder(name: &'static str, segments: u32, rings: u32) -> TestMesh {
    let mut mesh = TestMesh::new(name);
    let around = |segment: u32| {
        let phi = (segment % segments) as f32 / segments as f32 * std::f32::consts::TAU;
        engine_math::sin_cos(phi)
    };
    let row = segments + 1;
    for ring in 0..=rings {
        let v = ring as f32 / rings as f32;
        let y = v * 2.0 - 1.0;
        for segment in 0..=segments {
            let (sin, cos) = around(segment);
            mesh.vertex(
                [cos, y, sin],
                [cos, 0.0, sin],
                [segment as f32 / segments as f32, v],
            );
        }
    }
    for ring in 0..rings {
        for segment in 0..segments {
            let a = ring * row + segment;
            let (b, c) = (a + 1, a + row);
            mesh.indices.extend([a, c, b, b, c, c + 1]);
        }
    }
    for (y, top) in [(-1.0, false), (1.0, true)] {
        let normal = [0.0, y, 0.0];
        let center = mesh.vertex([0.0, y, 0.0], normal, [0.5, 0.5]);
        for segment in 0..segments {
            let (sin, cos) = around(segment);
            mesh.vertex([cos, y, sin], normal, [0.5 + cos * 0.5, 0.5 + sin * 0.5]);
        }
        for segment in 0..segments {
            let (a, b) = (center + 1 + segment, center + 1 + (segment + 1) % segments);
            if top {
                mesh.indices.extend([center, b, a]);
            } else {
                mesh.indices.extend([center, a, b]);
            }
        }
    }
    mesh
}

/// Disconnected parts of very different sizes: spheres, cubes, a patch.
fn components() -> TestMesh {
    let mut mesh = TestMesh::new("components");
    mesh.append(&uv_sphere("", 16, 32, 0.0), 1.0, [0.0; 3]);
    mesh.append(&uv_sphere("", 10, 20, 0.02), 0.3, [2.0, 0.0, 0.0]);
    mesh.append(&uv_sphere("", 6, 12, 0.0), 0.08, [0.0, 2.0, 0.0]);
    for cube in 0..6 {
        let offset = [-2.0 + cube as f32 * 0.25, -1.5, 0.5];
        mesh.append(&faceted_box("", 2), 0.04 + cube as f32 * 0.01, offset);
    }
    mesh.append(&grid("", 6), 0.3, [1.5, 1.5, 1.5]);
    mesh
}

/// A grid with degenerate triangles: repeated indices, and triangles of
/// three distinct vertices where two share a position.
fn degenerate() -> TestMesh {
    let mut mesh = grid("degenerate", 12);
    let triangles = mesh.triangles() as u32;
    for triangle in (0..triangles).step_by(7) {
        let [a, b, _] = [0, 1, 2].map(|corner| mesh.indices[(triangle * 3 + corner) as usize]);
        mesh.indices.extend([a, a, b]);
    }
    let copy = mesh.vertex([0.5, 0.0, 0.5], [0.0, 1.0, 0.0], [0.5, 0.5]);
    mesh.indices.extend([6 * 13 + 6, 6 * 13 + 7, copy]);
    mesh
}

/// The meshes every option set runs on.
fn meshes() -> Vec<TestMesh> {
    vec![
        uv_sphere("sphere", 24, 48, 0.0),
        faceted_box("cube24", 1),
        faceted_box("box16", 16),
        grid("grid", 32),
        cylinder("cylinder", 48, 12),
        TestMesh::from_geometry("terrain", &procedural::terrain(48, 16.0, 5)),
        TestMesh::from_geometry("column", &procedural::column(48, 24)),
        TestMesh::from_geometry("boulder", &procedural::boulder(5, 7)),
        components(),
        degenerate(),
    ]
}

/// Which vertex flags a case sets.
#[derive(Clone, Copy, PartialEq)]
enum Flags {
    None,
    /// `PROTECT` where texture coordinates differ at a position.
    Protect,
    /// `LOCK` on every 11th vertex, `PRIORITY` on every 7th.
    LockPriority,
}

/// Option sets: a name, meshoptimizer's bits and the vertex flags.
const OPTION_SETS: [(&str, u32, Flags); 10] = [
    ("none", 0, Flags::None),
    ("lock_border", bits::LOCK_BORDER, Flags::None),
    ("error_absolute", bits::ERROR_ABSOLUTE, Flags::None),
    ("permissive", bits::PERMISSIVE, Flags::None),
    ("prune", bits::PRUNE, Flags::None),
    ("regularize", bits::REGULARIZE, Flags::None),
    ("permissive+protect", bits::PERMISSIVE, Flags::Protect),
    ("lock+priority", 0, Flags::LockPriority),
    (
        "folds+clamped+light",
        bits::PRESERVE_FOLDS | bits::ERROR_CLAMPED | bits::REGULARIZE_LIGHT,
        Flags::None,
    ),
    (
        "prune+border+permissive",
        bits::PRUNE | bits::LOCK_BORDER | bits::PERMISSIVE,
        Flags::None,
    ),
];

/// Attribute weights: positions only, normals and texture coordinates, and
/// weights with zeros (which the simplifiers drop).
const WEIGHTS: [(&str, &[f32]); 3] = [
    ("pos", &[]),
    ("nrm+uv", &[0.5, 0.5, 0.5, 1.0, 1.0]),
    ("sparse", &[1.0, 0.0, 1.0, 0.0, 0.25]),
];

fn port_options(options: u32) -> SimplifyOptions {
    SimplifyOptions {
        lock_border: options & bits::LOCK_BORDER != 0,
        error_absolute: options & bits::ERROR_ABSOLUTE != 0,
        prune: options & bits::PRUNE != 0,
        regularize: options & bits::REGULARIZE != 0,
        regularize_light: options & bits::REGULARIZE_LIGHT != 0,
        permissive: options & bits::PERMISSIVE != 0,
        preserve_folds: options & bits::PRESERVE_FOLDS != 0,
        error_clamped: options & bits::ERROR_CLAMPED != 0,
    }
}

fn vertex_flags(mesh: &TestMesh, flags: Flags) -> Vec<VertexFlags> {
    let mut result = vec![VertexFlags::NONE; mesh.positions.len()];
    match flags {
        Flags::None => {}
        Flags::Protect => {
            port::protect_attribute_seams(
                &mesh.positions,
                &mesh.attributes,
                STRIDE,
                0b11000,
                &mut result,
            )
            .unwrap();
        }
        Flags::LockPriority => {
            for (vertex, flag) in result.iter_mut().enumerate() {
                if vertex % 11 == 0 {
                    *flag |= VertexFlags::LOCK;
                }
                if vertex % 7 == 0 {
                    *flag |= VertexFlags::PRIORITY;
                }
            }
        }
    }
    result
}

/// One simplification by both libraries.
struct Case<'a> {
    mesh: &'a TestMesh,
    weights: &'a [f32],
    flags: &'a [VertexFlags],
    options: u32,
    target_index_count: usize,
    target_error: f32,
}

/// Indices, error and time of a run.
struct Run {
    indices: Vec<u32>,
    error: f32,
    time: Duration,
}

impl Case<'_> {
    fn cpp(&self) -> Run {
        let locks: Vec<u8> = if self.flags.iter().all(|&flag| flag == VertexFlags::NONE) {
            Vec::new()
        } else {
            self.flags.iter().map(|flag| flag.bits()).collect()
        };
        let input = meshopt::Input {
            indices: &self.mesh.indices,
            positions: &self.mesh.positions,
            attributes: &self.mesh.attributes,
            attribute_stride: STRIDE,
            weights: self.weights,
            locks: &locks,
        };
        let start = Instant::now();
        let (indices, error) = meshopt::simplify_with_attributes(
            &input,
            self.target_index_count,
            self.target_error,
            self.options,
        );
        Run {
            indices,
            error,
            time: start.elapsed(),
        }
    }

    fn rust(&self) -> Run {
        let mut input = SimplifyInput::new(&self.mesh.indices, &self.mesh.positions);
        if !self.weights.is_empty() {
            input = input.with_attributes(&self.mesh.attributes, STRIDE, self.weights);
        }
        if self.flags.iter().any(|&flag| flag != VertexFlags::NONE) {
            input = input.with_vertex_flags(self.flags);
        }
        let start = Instant::now();
        let result = port::simplify(
            &input,
            self.target_index_count,
            self.target_error,
            port_options(self.options),
        )
        .unwrap();
        Run {
            indices: result.indices,
            error: result.error,
            time: start.elapsed(),
        }
    }
}

/// Runs `case` in both libraries, prints a table row and returns whether
/// they agree bit for bit.
fn compare(case: &Case<'_>, label: &str) -> bool {
    let (cpp, rust) = (case.cpp(), case.rust());
    let identical = cpp.indices == rust.indices && cpp.error.to_bits() == rust.error.to_bits();
    println!(
        "{:<10} {label:<46} | C++ {:>7} {:<12.6e} | Rust {:>7} {:<12.6e} | {}",
        case.mesh.name,
        cpp.indices.len(),
        cpp.error,
        rust.indices.len(),
        rust.error,
        if identical { "identical" } else { "DIFFERENT" }
    );
    if !identical {
        let first = cpp
            .indices
            .iter()
            .zip(&rust.indices)
            .position(|(a, b)| a != b);
        println!(
            "           first differing index {first:?}; errors {:#010x} {:#010x}",
            cpp.error.to_bits(),
            rust.error.to_bits()
        );
    }
    identical
}

/// Runs every option set, attribute mode, ratio and error limit on `mesh`;
/// returns (cases, identical cases).
fn compare_mesh(
    mesh: &TestMesh,
    option_sets: &[(&str, u32, Flags)],
    errors: &[f32],
) -> (usize, usize) {
    let scale = port::simplify_scale(&mesh.positions);
    assert_eq!(
        scale.to_bits(),
        meshopt::scale(&mesh.positions).to_bits(),
        "{} scale",
        mesh.name
    );
    let (mut cases, mut identical) = (0, 0);
    for &(options_name, options, flags) in option_sets {
        let flags = vertex_flags(mesh, flags);
        for (weights_name, weights) in WEIGHTS {
            // Zero weights only need the plainest options.
            if weights_name == "sparse" && options != 0 {
                continue;
            }
            for ratio in [0.5, 0.25, 0.1] {
                for &error in errors {
                    let target_error = if options & bits::ERROR_ABSOLUTE != 0 && error != f32::MAX {
                        error * scale
                    } else {
                        error
                    };
                    let case = Case {
                        mesh,
                        weights,
                        flags: &flags,
                        options,
                        target_index_count: (mesh.indices.len() as f32 * ratio) as usize / 3 * 3,
                        target_error,
                    };
                    let error_name = if error == f32::MAX {
                        "max".to_owned()
                    } else {
                        format!("{error}")
                    };
                    let label =
                        format!("{options_name} {weights_name} ratio {ratio} error {error_name}");
                    cases += 1;
                    identical += usize::from(compare(&case, &label));
                }
            }
        }
    }
    (cases, identical)
}

#[test]
fn simplify_port_matches_meshoptimizer() {
    let (mut cases, mut identical) = (0, 0);
    for mesh in meshes() {
        println!(
            "{}: {} triangles, {} vertices",
            mesh.name,
            mesh.triangles(),
            mesh.positions.len()
        );
        let (mesh_cases, mesh_identical) = compare_mesh(&mesh, &OPTION_SETS, &[1.0e-2, f32::MAX]);
        cases += mesh_cases;
        identical += mesh_identical;
    }
    println!("{identical} of {cases} cases identical");
    assert_eq!(identical, cases);
}

/// A small deterministic generator (SplitMix64).
struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound.max(1) as u64) as usize
    }

    fn chance(&mut self, percent: usize) -> bool {
        self.below(100) < percent
    }
}

/// A mesh of arbitrary triangles over coincident lattice positions:
/// non-manifold edges, degenerate triangles, wedges without seams.
fn arbitrary_mesh(random: &mut Random) -> TestMesh {
    let mut mesh = TestMesh::new("arbitrary");
    let vertices = 1 + random.below(48);
    let lattice = 1 + random.below(4);
    for _ in 0..vertices {
        let position = [0; 3].map(|_| random.below(lattice) as f32 * 0.5);
        let normal = [0; 3].map(|_| random.below(3) as f32 - 1.0);
        let uv = [0; 2].map(|_| random.below(3) as f32 * 0.5);
        mesh.vertex(position, normal, uv);
    }
    mesh.indices = (0..random.below(96) * 3)
        .map(|_| random.below(vertices) as u32)
        .collect();
    mesh
}

/// A generated mesh with holes, split vertices (new seams with their own
/// attributes), flipped and extra triangles.
fn damaged_mesh(random: &mut Random) -> TestMesh {
    let mut mesh = match random.below(4) {
        0 => uv_sphere(
            "damaged",
            6 + random.below(6) as u32,
            8 + random.below(8) as u32,
            0.1,
        ),
        1 => faceted_box("damaged", 1 + random.below(4) as u32),
        2 => grid("damaged", 2 + random.below(8) as u32),
        _ => cylinder(
            "damaged",
            6 + random.below(10) as u32,
            1 + random.below(4) as u32,
        ),
    };
    let triangles = mesh.triangles();
    let mut indices = Vec::with_capacity(mesh.indices.len());
    for triangle in 0..triangles {
        let mut corners = [0, 1, 2].map(|corner| mesh.indices[triangle * 3 + corner]);
        if random.chance(8) {
            continue;
        }
        if random.chance(4) {
            corners.swap(1, 2);
        }
        if random.chance(6) {
            // A new wedge for this corner only.
            let corner = random.below(3);
            let vertex = corners[corner] as usize;
            let position = mesh.positions[vertex];
            let mut attributes = [0.0; STRIDE];
            attributes.copy_from_slice(&mesh.attributes[vertex * STRIDE..][..STRIDE]);
            attributes[random.below(STRIDE)] += 0.25;
            corners[corner] = mesh.vertex(
                position,
                [attributes[0], attributes[1], attributes[2]],
                [attributes[3], attributes[4]],
            );
        }
        indices.extend(corners);
    }
    for _ in 0..random.below(4) {
        indices.extend([0; 3].map(|_| random.below(mesh.positions.len()) as u32));
    }
    mesh.indices = indices;
    mesh
}

#[test]
fn simplify_port_matches_meshoptimizer_on_random_meshes() {
    let mut random = Random(0x5eed);
    let (mut cases, mut identical) = (0, 0);
    for iteration in 0..4000 {
        let mesh = if iteration % 2 == 0 {
            arbitrary_mesh(&mut random)
        } else {
            damaged_mesh(&mut random)
        };
        let options = (random.next() as u32) & bits::ALL;
        let weights: &[f32] = [
            &[][..],
            &[0.5, 0.5, 0.5, 1.0, 1.0][..],
            &[1.0, 0.0, 2.0][..],
        ][random.below(3)];
        let mut flags = vec![VertexFlags::NONE; mesh.positions.len()];
        match random.below(3) {
            0 => {}
            1 => flags.iter_mut().for_each(|flag| {
                *flag = VertexFlags::from_bits(random.below(8) as u8);
            }),
            _ => port::protect_attribute_seams(
                &mesh.positions,
                &mesh.attributes,
                STRIDE,
                0b11111,
                &mut flags,
            )
            .unwrap(),
        }
        let case = Case {
            mesh: &mesh,
            weights,
            flags: &flags,
            options,
            target_index_count: random.below(mesh.indices.len() + 1) / 3 * 3,
            target_error: [0.0, 1.0e-3, 1.0e-2, 0.1, 1.0, f32::MAX][random.below(6)],
        };
        let (cpp, rust) = (case.cpp(), case.rust());
        cases += 1;
        if cpp.indices == rust.indices && cpp.error.to_bits() == rust.error.to_bits() {
            identical += 1;
        } else if cases - identical <= 5 {
            println!(
                "iteration {iteration}: {} triangles, options {options:#x}, target {} error {}: C++ {} {:e}, Rust {} {:e}",
                mesh.triangles(),
                case.target_index_count,
                case.target_error,
                cpp.indices.len(),
                cpp.error,
                rust.indices.len(),
                rust.error
            );
        }
    }
    println!("{identical} of {cases} random cases identical");
    assert_eq!(identical, cases);
}

/// The ~100k-triangle sphere of the timing test.
fn large_sphere() -> TestMesh {
    uv_sphere("large", 160, 320, 0.05)
}

#[test]
fn simplify_port_matches_meshoptimizer_on_a_large_mesh() {
    let mesh = large_sphere();
    println!(
        "{}: {} triangles, {} vertices",
        mesh.name,
        mesh.triangles(),
        mesh.positions.len()
    );
    let option_sets: Vec<_> = OPTION_SETS
        .iter()
        .copied()
        .filter(|(name, ..)| {
            [
                "none",
                "lock_border",
                "permissive",
                "prune",
                "permissive+protect",
            ]
            .contains(name)
        })
        .collect();
    let (cases, identical) = compare_mesh(&mesh, &option_sets, &[f32::MAX]);
    println!("{identical} of {cases} cases identical");
    assert_eq!(identical, cases);
}

#[test]
fn simplify_port_timing_against_meshoptimizer() {
    let mesh = large_sphere();
    let repeats = if cfg!(debug_assertions) { 1 } else { 7 };
    for (weights_name, weights) in &WEIGHTS[..2] {
        for ratio in [0.5, 0.1] {
            let case = Case {
                mesh: &mesh,
                weights,
                flags: &[],
                options: 0,
                target_index_count: (mesh.indices.len() as f32 * ratio) as usize / 3 * 3,
                target_error: f32::MAX,
            };
            let (mut cpp, mut rust) = (Vec::new(), Vec::new());
            for _ in 0..repeats {
                let (a, b) = (case.cpp(), case.rust());
                assert_eq!(a.indices, b.indices);
                cpp.push(a.time);
                rust.push(b.time);
            }
            cpp.sort();
            rust.sort();
            println!(
                "{} triangles, {weights_name}, ratio {ratio}: C++ min {:.1} ms median {:.1} ms, Rust min {:.1} ms median {:.1} ms ({} runs{})",
                mesh.triangles(),
                cpp[0].as_secs_f64() * 1e3,
                cpp[repeats / 2].as_secs_f64() * 1e3,
                rust[0].as_secs_f64() * 1e3,
                rust[repeats / 2].as_secs_f64() * 1e3,
                repeats,
                if cfg!(debug_assertions) {
                    ", debug build: not representative"
                } else {
                    ""
                }
            );
        }
    }
}
