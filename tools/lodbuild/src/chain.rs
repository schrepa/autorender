//! A mesh's levels of detail: coarser index lists over its own vertices,
//! each simplified from the mesh itself (never from the level before, so
//! errors do not pile up) toward a halving share of its triangles with
//! meshoptimizer's chain settings, then an empty level. Each level's error
//! is measured, not estimated: how far its surface strays from the mesh's
//! both ways ([`crate::distance`]), so renderers drawing the coarsest level
//! whose error projects within a pixel budget show every level only where
//! its deviation truly fits it, its silhouette included.

use engine_assets::{LodInput, LodLevel, MeshLods};
use engine_math::max;
use glam::Vec3;

use crate::casters::caster_ranges;
use crate::distance::{mean_edge, surface_distance};
use crate::simplify::{
    SimplifyError, SimplifyInput, SimplifyOptions, VertexFlags, protect_attribute_seams, simplify,
};

/// The share of the mesh's triangles each simplified level targets, finest
/// first: renderers list a placed object's mesh and four coarser levels,
/// the last of them empty.
const LEVEL_SHARES: [f32; 3] = [0.5, 0.25, 0.125];
/// A level keeping more than this share of the last one's triangles stops
/// the chain: topology or the error allows no worthwhile simplification.
const LEVEL_KEEP_LIMIT: f32 = 0.85;
/// Each level's error is at least the last one's times this, so errors grow
/// strictly and a level never shows where a finer one would not.
const ERROR_GROWTH: f32 = 1.05;
/// The empty level's error over the error at which pruning removes the
/// mesh's last part (that part's radius about its center): the mesh vanishes
/// only where its largest part's diameter projects within the pixel budget.
const EMPTY_ERROR_SCALE: f32 = 2.0;
/// Attribute weights against positions (normalized to the mesh's extent):
/// normals, then texture coordinates per unit of their own range.
const NORMAL_WEIGHT: f32 = 1.0;
const UV_WEIGHT: f32 = 1.0;
/// Points sampled along a triangle's edges, measuring errors, at most
/// this many times as far apart as the mesh's mean edge is long.
const SAMPLE_STEP: f32 = 0.5;

/// `input`'s levels of detail, the last empty where pruning removes every
/// part, and the triangles shadow maps draw of each: none where
/// simplification refuses the mesh (beyond what 32-bit indices address),
/// or where no level is worth drawing.
pub(crate) fn levels(input: &LodInput) -> Result<MeshLods, SimplifyError> {
    let positions: Vec<[f32; 3]> = input.vertices.iter().map(|vertex| vertex.position).collect();
    let points: Vec<Vec3> = positions.iter().map(|&point| Vec3::from(point)).collect();
    // Normals and texture coordinates, the latter per unit of their range so
    // tiled and atlas coordinates weigh alike.
    let (mut low, mut high) = ([f32::INFINITY; 2], [f32::NEG_INFINITY; 2]);
    for vertex in &input.vertices {
        for axis in 0..2 {
            low[axis] = engine_math::min(low[axis], vertex.uv[axis]);
            high[axis] = engine_math::max(high[axis], vertex.uv[axis]);
        }
    }
    let uv_range = max(high[0] - low[0], high[1] - low[1]);
    let uv_weight = if uv_range > 1.0e-6 {
        UV_WEIGHT / uv_range
    } else {
        0.0
    };
    let attributes: Vec<f32> = input
        .vertices
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
    let mut flags = vec![VertexFlags::NONE; positions.len()];
    protect_attribute_seams(&positions, &attributes, 5, 0b11000, &mut flags)?;
    let options = SimplifyOptions {
        error_absolute: true,
        prune: true,
        permissive: true,
        error_clamped: true,
        ..SimplifyOptions::default()
    };
    let own = &input.indices;
    let simplified = |target: usize| {
        let request = SimplifyInput::new(own, &positions)
            .with_attributes(&attributes, 5, &weights)
            .with_vertex_flags(&flags);
        // No error limit: renderers draw a level only where its error
        // projects within their pixel budget. Low-poly props built of
        // closed parts have no edge to collapse, so pruning whole parts is
        // how their levels shrink.
        simplify(&request, target, f32::INFINITY, options)
    };
    let step = mean_edge(&points, own) * SAMPLE_STEP;
    let triangles = own.len() / 3;
    let mut lods = MeshLods::default();
    let mut indices = own.clone();
    let mut kept = triangles;
    let mut error = 0.0_f32;
    for share in LEVEL_SHARES {
        let target = ((triangles as f32 * share) as usize) * 3;
        let level = simplified(target)?.indices;
        let count = level.len() / 3;
        if count == 0 || count as f32 > kept as f32 * LEVEL_KEEP_LIMIT {
            break;
        }
        error = max(
            surface_distance(&points, own, &level, step),
            error * ERROR_GROWTH,
        );
        lods.levels.push(LodLevel {
            first_index: indices.len() as u32,
            index_count: level.len() as u32,
            error,
        });
        indices.extend_from_slice(&level);
        kept = count;
    }
    // The empty level, past the error at which pruning removes every part;
    // none where pruning keeps a part.
    let gone = simplified(0)?;
    if gone.indices.is_empty() {
        lods.levels.push(LodLevel {
            first_index: indices.len() as u32,
            index_count: 0,
            error: max(error * ERROR_GROWTH, gone.error * EMPTY_ERROR_SCALE),
        });
    }
    let ranges: Vec<(u32, u32, f32)> = lods
        .levels
        .iter()
        .map(|level| (level.first_index, level.index_count, level.error))
        .collect();
    lods.casters = caster_ranges(&mut indices, own.len() as u32, &ranges);
    lods.indices = indices.split_off(own.len());
    Ok(lods)
}
