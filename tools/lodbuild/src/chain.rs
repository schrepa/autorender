use engine_codec::Error;
use engine_math::max;
use glam::Vec3;

use crate::casters::caster_ranges;
use crate::distance::{mean_edge, surface_distance};
use crate::{LodInput, LodLevel, MeshLods};

use crate::simplify::{
    SimplifyInput, SimplifyOptions, VertexFlags, protect_attribute_seams, simplify,
};

const LEVEL_SHARES: [f32; 3] = [0.5, 0.25, 0.125];
const LEVEL_KEEP_LIMIT: f32 = 0.85;
const ERROR_GROWTH: f32 = 1.05;
const EMPTY_ERROR_SCALE: f32 = 2.0;
const NORMAL_WEIGHT: f32 = 1.0;
const UV_WEIGHT: f32 = 1.0;
const SAMPLE_STEP: f32 = 0.5;

pub(crate) fn levels(input: &LodInput) -> Result<MeshLods, Error> {
    let positions: Vec<[f32; 3]> = input
        .vertices
        .iter()
        .map(|vertex| vertex.position)
        .collect();
    let points: Vec<Vec3> = positions.iter().map(|&point| Vec3::from(point)).collect();
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
