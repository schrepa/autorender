//! Automatic levels of detail (decision 047): coarser index lists of a mesh
//! over its own vertices, each simplified from the last by [`simplify`] with
//! meshoptimizer's recommended chain settings, then an empty level, and the
//! geometric error each may show in mesh units. Renderers draw the coarsest
//! level whose error projects within a pixel budget: past the empty level's,
//! nothing.

use crate::Mesh;
use crate::simplify::{
    SimplifyInput, SimplifyOptions, VertexFlags, protect_attribute_seams, simplify,
};

/// The empty level's error over the error at which pruning removes the
/// mesh's last part (that part's radius about its center): the mesh vanishes
/// only where its largest part's diameter projects within the pixel budget.
const EMPTY_ERROR_SCALE: f32 = 2.0;
/// Each level targets this share of the last one's triangles.
const LEVEL_RATIO: f32 = 0.5;
/// A level keeping more than this share of the last one's triangles stops
/// the chain: topology or the error allows no worthwhile simplification.
const LEVEL_KEEP_LIMIT: f32 = 0.85;
/// Each level's error is at least the last one's times this, so errors grow
/// strictly and a level never looks better than a finer one.
const ERROR_GROWTH: f32 = 1.5;
/// Attribute weights against positions (normalized to the mesh's extent):
/// normals, then texture coordinates per unit of their own range.
const NORMAL_WEIGHT: f32 = 1.0;
const UV_WEIGHT: f32 = 1.0;

/// A level of detail: where its indices start in [`MeshLods::indices`], how
/// many there are (none for the empty level), and the largest geometric
/// deviation from the mesh in mesh units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LodLevel {
    pub first_index: u32,
    pub index_count: u32,
    pub error: f32,
}

/// A mesh's indices followed by its levels of detail, finest first.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MeshLods {
    pub indices: Vec<u32>,
    pub levels: Vec<LodLevel>,
}

/// `mesh`'s levels of detail, at most `max_levels` of them, the last empty:
/// none for meshes with joints (which move their vertices) or virtual
/// geometry (which has its own). The indices always start with the mesh's
/// own.
pub fn mesh_lods(mesh: &Mesh, max_levels: usize) -> MeshLods {
    let mut lods = MeshLods {
        indices: mesh.indices().to_vec(),
        levels: Vec::new(),
    };
    if mesh.joint_count() != 0 || mesh.virtual_geometry().is_some() || max_levels == 0 {
        return lods;
    }
    let vertices = mesh.vertices();
    let positions: Vec<[f32; 3]> = vertices.iter().map(|vertex| vertex.position).collect();
    // Normals and texture coordinates, the latter per unit of their range so
    // tiled and atlas coordinates weigh alike.
    let (mut low, mut high) = ([f32::INFINITY; 2], [f32::NEG_INFINITY; 2]);
    for vertex in vertices {
        for axis in 0..2 {
            low[axis] = engine_math::min(low[axis], vertex.uv[axis]);
            high[axis] = engine_math::max(high[axis], vertex.uv[axis]);
        }
    }
    let uv_range = engine_math::max(high[0] - low[0], high[1] - low[1]);
    let uv_weight = if uv_range > 1.0e-6 {
        UV_WEIGHT / uv_range
    } else {
        0.0
    };
    let attributes: Vec<f32> = vertices
        .iter()
        .flat_map(|vertex| {
            let [nx, ny, nz] = vertex.normal;
            let [u, v] = vertex.uv;
            [nx, ny, nz, u, v]
        })
        .collect();
    let weights = [
        NORMAL_WEIGHT,
        NORMAL_WEIGHT,
        NORMAL_WEIGHT,
        uv_weight,
        uv_weight,
    ];
    // Faceted meshes split every edge by its normals: collapses may cross
    // those seams, but not texture seams, which would stretch the texture.
    let mut flags = vec![VertexFlags::NONE; vertices.len()];
    if protect_attribute_seams(&positions, &attributes, 5, 0b11000, &mut flags).is_err() {
        return lods;
    }
    let options = SimplifyOptions {
        error_absolute: true,
        prune: true,
        permissive: true,
        error_clamped: true,
        ..SimplifyOptions::default()
    };
    let mut current = mesh.indices().to_vec();
    let mut error = 0.0_f32;
    // Simplified levels, leaving room for the empty one.
    while lods.levels.len() + 1 < max_levels {
        let target = ((current.len() / 3) as f32 * LEVEL_RATIO) as usize * 3;
        let input = SimplifyInput::new(&current, &positions)
            .with_attributes(&attributes, 5, &weights)
            .with_vertex_flags(&flags);
        // No error limit: the renderer only draws a level where its error
        // projects within its pixel budget. Low-poly props built of closed
        // parts have no edge to collapse, so pruning whole parts is how
        // their levels shrink.
        let Ok(simplified) = simplify(&input, target, f32::INFINITY, options) else {
            break;
        };
        let triangles = simplified.indices.len() / 3;
        if triangles == 0 || triangles as f32 > (current.len() / 3) as f32 * LEVEL_KEEP_LIMIT {
            break;
        }
        error = engine_math::max(error * ERROR_GROWTH, simplified.error);
        lods.levels.push(LodLevel {
            first_index: lods.indices.len() as u32,
            index_count: simplified.indices.len() as u32,
            error,
        });
        lods.indices.extend_from_slice(&simplified.indices);
        current = simplified.indices;
    }
    // The empty level, past the error at which pruning removes every part.
    let input = SimplifyInput::new(&current, &positions)
        .with_attributes(&attributes, 5, &weights)
        .with_vertex_flags(&flags);
    if let Ok(gone) = simplify(&input, 0, f32::INFINITY, options)
        && gone.indices.is_empty()
    {
        lods.levels.push(LodLevel {
            first_index: lods.indices.len() as u32,
            index_count: 0,
            error: engine_math::max(error * ERROR_GROWTH, gone.error * EMPTY_ERROR_SCALE),
        });
    }
    lods
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MeshVertex;

    // A UV sphere of `rings` by `segments` quads, with a texture seam.
    fn sphere(rings: u32, segments: u32) -> Mesh {
        let mut vertices = Vec::new();
        for ring in 0..=rings {
            let v = ring as f32 / rings as f32;
            let (sin_theta, cos_theta) = engine_math::sin_cos(v * std::f32::consts::PI);
            for segment in 0..=segments {
                let u = segment as f32 / segments as f32;
                let (sin_phi, cos_phi) = engine_math::sin_cos(u * std::f32::consts::TAU);
                let normal = [sin_theta * cos_phi, cos_theta, sin_theta * sin_phi];
                vertices.push(MeshVertex {
                    position: normal,
                    normal,
                    uv: [u, v],
                });
            }
        }
        let mut indices = Vec::new();
        for ring in 0..rings {
            for segment in 0..segments {
                let a = ring * (segments + 1) + segment;
                let b = a + segments + 1;
                indices.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
            }
        }
        Mesh::new(vertices, indices).unwrap()
    }

    #[test]
    fn levels_shrink_and_grow_their_errors() {
        let mesh = sphere(24, 48);
        let lods = mesh_lods(&mesh, 3);
        assert_eq!(lods.levels.len(), 3);
        assert_eq!(&lods.indices[..mesh.indices().len()], mesh.indices());
        // The last level is empty, past twice the radius.
        let (empty, simplified) = lods.levels.split_last().unwrap();
        assert_eq!(empty.index_count, 0);
        assert_eq!(empty.first_index as usize, lods.indices.len());
        assert!(
            empty.error > simplified[1].error && empty.error >= 1.9,
            "{empty:?}"
        );
        let mut previous = (mesh.indices().len() as u32, 0.0);
        for level in simplified {
            assert!(level.index_count < previous.0 * 3 / 4, "{level:?}");
            assert!(level.error > previous.1, "{level:?}");
            // A unit sphere's simplification stays within its radius.
            assert!(level.error < 0.5, "{level:?}");
            let range =
                level.first_index as usize..(level.first_index + level.index_count) as usize;
            assert!(
                lods.indices[range]
                    .iter()
                    .all(|&index| (index as usize) < mesh.vertices().len())
            );
            previous = (level.index_count, level.error);
        }
    }

    #[test]
    fn small_meshes_simplify_and_vanish() {
        let mesh = sphere(4, 6);
        let lods = mesh_lods(&mesh, 2);
        assert_eq!(lods.levels.len(), 2);
        assert!(lods.levels[0].index_count > 0);
        assert!((lods.levels[0].index_count as usize) < mesh.indices().len());
        assert_eq!(lods.levels[1].index_count, 0);
        assert_eq!(&lods.indices[..mesh.indices().len()], mesh.indices());
        assert!(mesh_lods(&sphere(24, 48), 0).levels.is_empty());
        // One level allowed: the empty one.
        let one = mesh_lods(&sphere(24, 48), 1);
        assert_eq!(one.levels.len(), 1);
        assert_eq!(one.levels[0].index_count, 0);
        assert_eq!(one.indices, sphere(24, 48).indices());
    }
}
