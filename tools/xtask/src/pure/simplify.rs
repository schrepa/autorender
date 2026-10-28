use std::path::Path;

use engine_core::math::{cos, sin};
use engine_core::{MeshLods, MeshVertex};
use engine_lodbuild::simplify::{SimplifyInput, SimplifyOptions, VertexFlags, simplify};
use serde_json::json;
use sha2::{Digest, Sha256};

use super::Observations;

pub const PROP: &str = "tools/xtask/pure/prop.glb";
const TARGETS: [f32; 3] = [0.5, 0.25, 0.05];
const LIMITS: [f32; 2] = [0.01, 1.0];

struct Mesh {
    name: String,
    positions: Vec<[f32; 3]>,
    indices: Vec<u32>,
    attributes: Option<(Vec<f32>, usize, Vec<f32>)>,
    flags: Option<Vec<u8>>,
}

fn options(mask: u32) -> SimplifyOptions {
    let set = |bit: u32| mask & (1 << bit) != 0;
    SimplifyOptions {
        lock_border: set(0),
        error_absolute: set(1),
        prune: set(2),
        regularize: set(3),
        regularize_light: set(4),
        permissive: set(5),
        preserve_folds: set(6),
        error_clamped: set(7),
    }
}

pub fn run(root: &Path, observed: &mut Observations) -> Result<(), String> {
    let mut meshes = vec![
        wavy_grid(24, 24),
        seamed_grid(16),
        sphere(20, 12),
        torus(28, 12),
    ];
    meshes.extend(prop(root)?);
    for mesh in &meshes {
        let mut hash = Sha256::new();
        let (mut runs, mut kept) = (0usize, 0usize);
        let flags: Option<Vec<VertexFlags>> = mesh.flags.as_ref().map(|flags| {
            flags
                .iter()
                .map(|&bits| VertexFlags::from_bits(bits))
                .collect()
        });
        let mut input = SimplifyInput::new(&mesh.indices, &mesh.positions);
        if let Some((values, stride, weights)) = &mesh.attributes {
            input = input.with_attributes(values, *stride, weights);
        }
        if let Some(flags) = &flags {
            input = input.with_vertex_flags(flags);
        }
        for mask in 0..256 {
            let options = options(mask);
            for fraction in TARGETS {
                let target = ((mesh.indices.len() as f32 * fraction) as usize / 3) * 3;
                for limit in LIMITS {
                    let ported = simplify(&input, target, limit, options)
                        .map_err(|error| format!("{}: {error}", mesh.name))?;
                    runs += 1;
                    kept += ported.indices.len();
                    for index in &ported.indices {
                        hash.update(index.to_le_bytes());
                    }
                    hash.update(ported.error.to_bits().to_le_bytes());
                }
            }
        }
        let digest: String = hash
            .finalize()
            .iter()
            .take(8)
            .map(|byte| format!("{byte:02x}"))
            .collect();
        observed.record(
            format!("simplify {}", mesh.name),
            json!({
                "triangles": mesh.indices.len() / 3,
                "runs": runs,
                "kept indices": kept,
                "hash": digest,
            }),
        );
        levels(mesh, observed)?;
    }
    Ok(())
}

fn levels(mesh: &Mesh, observed: &mut Observations) -> Result<(), String> {
    let vertices: Vec<MeshVertex> = mesh
        .positions
        .iter()
        .enumerate()
        .map(|(vertex, &position)| {
            let values = mesh
                .attributes
                .as_ref()
                .filter(|(_, stride, _)| *stride == 5)
                .map(|(values, _, _)| &values[vertex * 5..vertex * 5 + 5]);
            MeshVertex {
                position,
                normal: values.map_or([0.0, 1.0, 0.0], |values| [values[0], values[1], values[2]]),
                uv: values.map_or([0.0; 2], |values| [values[3], values[4]]),
            }
        })
        .collect();
    let lods = engine_lodbuild::levels(&vertices, &mesh.indices)
        .map_err(|error| format!("levels of {}: {error}", mesh.name))?;
    let bytes = lods.to_bytes();
    observed.check(
        MeshLods::from_bytes(&bytes).is_ok_and(|read| read == lods),
        || {
            format!(
                "{}: its levels do not read back from their record",
                mesh.name
            )
        },
    );
    observed.check(
        lods.check(mesh.indices.len(), vertices.len()).is_ok(),
        || format!("{}: its levels do not suit it", mesh.name),
    );
    observed.record(
        format!("levels {}", mesh.name),
        json!({
            "triangles": lods.levels.iter().map(|level| level.index_count / 3).collect::<Vec<_>>(),
            "errors": lods.levels.iter().map(|level| level.error).collect::<Vec<_>>(),
            "casters": lods.casters.len(),
            "hash": super::hash(&bytes),
        }),
    );
    Ok(())
}

fn quads(columns: u32, rows: u32, at: impl Fn(u32, u32) -> u32) -> Vec<u32> {
    let mut indices = Vec::new();
    for row in 0..rows - 1 {
        for column in 0..columns - 1 {
            let (a, b) = (at(column, row), at(column + 1, row));
            let (c, d) = (at(column + 1, row + 1), at(column, row + 1));
            indices.extend([a, d, b, b, d, c]);
        }
    }
    indices
}

fn wave(x: f32, z: f32) -> f32 {
    sin(x * 0.35) * cos(z * 0.25) * 0.8
}

fn wavy_grid(columns: u32, rows: u32) -> Mesh {
    let mut positions = Vec::new();
    for row in 0..rows {
        for column in 0..columns {
            let (x, z) = (column as f32, row as f32);
            positions.push([x, wave(x, z), z]);
        }
    }
    Mesh {
        name: format!("grid {columns}x{rows}"),
        indices: quads(columns, rows, |column, row| row * columns + column),
        positions,
        attributes: None,
        flags: None,
    }
}

fn seamed_grid(side: u32) -> Mesh {
    let seam = side / 2;
    let stride = side + 1;
    let (mut positions, mut values, mut flags) = (Vec::new(), Vec::new(), Vec::new());
    for row in 0..side {
        for slot in 0..stride {
            let column = if slot <= seam { slot } else { slot - 1 };
            let (x, z) = (column as f32, row as f32);
            let y = wave(x, z);
            let slope = [
                wave(x + 0.5, z) - wave(x - 0.5, z),
                wave(x, z + 0.5) - wave(x, z - 0.5),
            ];
            let length = (slope[0] * slope[0] + 1.0 + slope[1] * slope[1]).sqrt();
            let u = if slot > seam { 1.0 } else { 0.0 } + column as f32 / side as f32;
            positions.push([x, y, z]);
            values.extend([-slope[0] / length, 1.0 / length, -slope[1] / length]);
            values.extend([u, row as f32 / side as f32]);
            let corner = (column == 0 || column == side - 1) && (row == 0 || row == side - 1);
            let mut bits = 0u8;
            if corner {
                bits |= VertexFlags::LOCK.bits();
            }
            if column == seam {
                bits |= VertexFlags::PROTECT.bits();
            }
            if column == row {
                bits |= VertexFlags::PRIORITY.bits();
            }
            flags.push(bits);
        }
    }
    let at = |column: u32, row: u32, right: bool| {
        row * stride
            + if column > seam || (column == seam && right) {
                column + 1
            } else {
                column
            }
    };
    let mut indices = Vec::new();
    for row in 0..side - 1 {
        for column in 0..side - 1 {
            let right = column >= seam;
            let (a, b) = (at(column, row, right), at(column + 1, row, right));
            let (c, d) = (at(column + 1, row + 1, right), at(column, row + 1, right));
            indices.extend([a, d, b, b, d, c]);
        }
    }
    Mesh {
        name: format!("seamed grid {side}x{side}"),
        positions,
        indices,
        attributes: Some((values, 5, vec![1.0, 1.0, 1.0, 0.5, 0.5])),
        flags: Some(flags),
    }
}

fn sphere(segments: u32, rings: u32) -> Mesh {
    let (mut positions, mut values) = (Vec::new(), Vec::new());
    for ring in 0..=rings {
        let polar = std::f32::consts::PI * ring as f32 / rings as f32;
        for segment in 0..=segments {
            let around = std::f32::consts::TAU * segment as f32 / segments as f32;
            let point = [
                sin(polar) * cos(around),
                cos(polar),
                sin(polar) * sin(around),
            ];
            positions.push(point);
            values.extend(point);
            values.extend([segment as f32 / segments as f32, ring as f32 / rings as f32]);
        }
    }
    Mesh {
        name: format!("sphere {segments}x{rings}"),
        indices: quads(segments + 1, rings + 1, |segment, ring| {
            ring * (segments + 1) + segment
        }),
        positions,
        attributes: Some((values, 5, vec![1.0, 1.0, 1.0, 0.25, 0.25])),
        flags: None,
    }
}

fn torus(around: u32, across: u32) -> Mesh {
    let mut positions = Vec::new();
    for i in 0..around {
        let a = std::f32::consts::TAU * i as f32 / around as f32;
        for j in 0..across {
            let b = std::f32::consts::TAU * j as f32 / across as f32;
            let radius = 2.0 + 0.6 * cos(b);
            positions.push([radius * cos(a), 0.6 * sin(b), radius * sin(a)]);
        }
    }
    Mesh {
        name: format!("torus {around}x{across}"),
        indices: quads(around + 1, across + 1, |i, j| {
            (i % around) * across + (j % across)
        }),
        positions,
        attributes: None,
        flags: None,
    }
}

fn prop(root: &Path) -> Result<Vec<Mesh>, String> {
    let bytes =
        std::fs::read(root.join(PROP)).map_err(|error| format!("cannot read {PROP}: {error}"))?;
    let glb = gltf::Gltf::from_slice(&bytes).map_err(|error| format!("{PROP}: {error}"))?;
    let binary = glb
        .blob
        .as_deref()
        .ok_or_else(|| format!("{PROP} has no binary chunk"))?;
    let mut meshes = Vec::new();
    for mesh in glb.document.meshes() {
        for (index, primitive) in mesh.primitives().enumerate() {
            let reader = primitive.reader(|buffer| (buffer.index() == 0).then_some(binary));
            let positions: Vec<[f32; 3]> = reader
                .read_positions()
                .ok_or("a prop primitive has no positions")?
                .collect();
            let normals = reader
                .read_normals()
                .ok_or("a prop primitive has no normals")?;
            let uvs = reader
                .read_tex_coords(0)
                .ok_or("a prop primitive has no texture coordinates")?
                .into_f32();
            let values = normals
                .zip(uvs)
                .flat_map(|(normal, uv)| [normal[0], normal[1], normal[2], uv[0], uv[1]])
                .collect();
            let indices = reader
                .read_indices()
                .ok_or("a prop primitive has no indices")?
                .into_u32()
                .collect();
            meshes.push(Mesh {
                name: format!("prop {index}"),
                positions,
                indices,
                attributes: Some((values, 5, vec![1.0, 1.0, 1.0, 0.5, 0.5])),
                flags: None,
            });
        }
    }
    Ok(meshes)
}
