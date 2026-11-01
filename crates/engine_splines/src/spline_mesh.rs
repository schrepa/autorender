use bevy_ecs::prelude::*;
use engine_codec::Expected;
use engine_core::{
    AssetLibrary, Error, ErrorCode, Mesh, Mesh3d, MeshHandle, MeshVertex, Parent, Spawner,
    Transform,
};
use engine_math::collections::HashMap;
use engine_math::{clamp, max};
use engine_places::splines::SplineMesh;
use glam::Vec3;

use crate::SplinesBrick;

pub const CHUNK_LENGTH: f32 = 256.0;
pub const MAX_SPLINE_POINTS: usize = 4096;
const STEP: f32 = 0.5;
const MAX_COORDINATE: f32 = 1.0e7;

#[derive(Resource, Default)]
pub(crate) struct SplineBuilds {
    built: HashMap<Entity, SplineChunks>,
}

impl SplineBuilds {
    pub(crate) fn of(&self, entity: Entity) -> Option<&SplineChunks> {
        self.built.get(&entity)
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct SplineChunks {
    pub(crate) entities: Vec<Entity>,
    pub(crate) meshes: Vec<MeshHandle>,
    pub(crate) length: f32,
    pub(crate) copies: u32,
    pub(crate) vertices: u32,
    pub(crate) error: Option<Error>,
    built: Option<SplineMesh>,
}

fn discard(
    entity: Entity,
    chunks: &SplineChunks,
    drawn: &mut Query<&mut Mesh3d>,
    assets: &mut AssetLibrary,
    spawner: &mut Spawner<SplinesBrick>,
) {
    for &child in &chunks.entities {
        spawner.despawn(child);
    }
    if let (Some(&first), Ok(mut shown)) = (chunks.meshes.first(), drawn.get_mut(entity))
        && shown.mesh == first
    {
        *shown = Mesh3d::default();
    }
    for &mesh in &chunks.meshes {
        assets.remove_mesh(mesh);
    }
}

pub(crate) fn build(
    splines: Query<(Entity, &SplineMesh, Has<Transform>)>,
    mut drawn: Query<&mut Mesh3d>,
    mut assets: ResMut<AssetLibrary>,
    mut builds: ResMut<SplineBuilds>,
    mut spawner: Spawner<SplinesBrick>,
) {
    let built = &mut builds.built;
    let mut gone: Vec<Entity> = built
        .keys()
        .copied()
        .filter(|&entity| !splines.contains(entity))
        .collect();
    gone.sort_unstable();
    for entity in gone {
        if let Some(old) = built.remove(&entity) {
            discard(entity, &old, &mut drawn, &mut assets, &mut spawner);
        }
    }
    let stale: Vec<(Entity, SplineMesh, bool)> = splines
        .iter()
        .filter(|(entity, spline, _)| {
            built
                .get(entity)
                .is_none_or(|chunks| chunks.built.as_ref() != Some(*spline))
        })
        .map(|(entity, spline, parented)| (entity, spline.clone(), parented))
        .collect();
    for (entity, spline, parented) in stale {
        if let Some(old) = built.remove(&entity) {
            discard(entity, &old, &mut drawn, &mut assets, &mut spawner);
        }
        let bent = match bend(&assets, &spline) {
            Ok(bent) => bent,
            Err(error) => {
                built.insert(
                    entity,
                    SplineChunks {
                        error: Some(error),
                        built: Some(spline),
                        ..SplineChunks::default()
                    },
                );
                continue;
            }
        };
        let mut chunks = SplineChunks {
            length: bent.length,
            copies: bent.copies,
            vertices: bent
                .meshes
                .iter()
                .map(|mesh| mesh.vertices().len() as u32)
                .sum(),
            built: Some(spline.clone()),
            ..SplineChunks::default()
        };
        for mesh in bent.meshes {
            let handle = match assets.insert_mesh(mesh) {
                Ok(handle) => handle,
                Err(error) => {
                    chunks.error = Some(error);
                    break;
                }
            };
            let first = chunks.meshes.is_empty();
            chunks.meshes.push(handle);
            let shown = Mesh3d {
                mesh: handle,
                material: spline.material,
            };
            if first && parented && let Ok(mut own) = drawn.get_mut(entity) {
                *own = shown;
                continue;
            }
            let child = if parented {
                spawner.spawn((Transform::default(), shown, Parent::of(entity)))
            } else {
                spawner.spawn((Transform::default(), shown))
            };
            chunks.entities.push(child);
        }
        built.insert(entity, chunks);
    }
}

struct Bent {
    meshes: Vec<Mesh>,
    length: f32,
    copies: u32,
}

fn curve(points: &[Vec3]) -> (Vec<Vec3>, Vec<f32>) {
    let count = points.len();
    let at = |index: isize| -> Vec3 {
        if index < 0 {
            2.0 * points[0] - points[1]
        } else if index as usize >= count {
            2.0 * points[count - 1] - points[count - 2]
        } else {
            points[index as usize]
        }
    };
    let mut samples = vec![points[0]];
    for segment in 0..count - 1 {
        let index = segment as isize;
        let [p0, p1, p2, p3] = [at(index - 1), at(index), at(index + 1), at(index + 2)];
        let knot = |a: Vec3, b: Vec3| max(a.distance(b).sqrt(), 1.0e-4);
        let t1 = knot(p0, p1);
        let t2 = t1 + knot(p1, p2);
        let t3 = t2 + knot(p2, p3);
        let steps = max((p1.distance(p2) / STEP).ceil(), 1.0) as usize;
        for step in 1..=steps {
            let t = t1 + (t2 - t1) * step as f32 / steps as f32;
            let a1 = p0 * ((t1 - t) / t1) + p1 * (t / t1);
            let a2 = p1 * ((t2 - t) / (t2 - t1)) + p2 * ((t - t1) / (t2 - t1));
            let a3 = p2 * ((t3 - t) / (t3 - t2)) + p3 * ((t - t2) / (t3 - t2));
            let b1 = a1 * ((t2 - t) / t2) + a2 * (t / t2);
            let b2 = a2 * ((t3 - t) / (t3 - t1)) + a3 * ((t - t1) / (t3 - t1));
            samples.push(b1 * ((t2 - t) / (t2 - t1)) + b2 * ((t - t1) / (t2 - t1)));
        }
    }
    let mut along = Vec::with_capacity(samples.len());
    let mut length = 0.0;
    for (index, &sample) in samples.iter().enumerate() {
        if index > 0 {
            length += samples[index - 1].distance(sample);
        }
        along.push(length);
    }
    (samples, along)
}

fn point_at(samples: &[Vec3], along: &[f32], distance: f32) -> (Vec3, Vec3) {
    let last = samples.len() - 1;
    let index = along
        .partition_point(|&value| value <= distance)
        .saturating_sub(1)
        .min(last - 1);
    let (a, b) = (samples[index], samples[index + 1]);
    let run = along[index + 1] - along[index];
    let t = if run > 0.0 {
        clamp((distance - along[index]) / run, 0.0, 1.0)
    } else {
        0.0
    };
    let back = samples[index.saturating_sub(1)];
    let ahead = samples[(index + 2).min(last)];
    let direction = (b - a) * 2.0 + (ahead - back);
    (a + (b - a) * t, direction.normalize_or(b - a))
}

fn bend(assets: &AssetLibrary, spline: &SplineMesh) -> Result<Bent, Error> {
    let invalid = |path: &str| Error::new(ErrorCode::InvalidValue, path);
    if !(2..=MAX_SPLINE_POINTS).contains(&spline.points.len()) {
        return Err(invalid("$.points")
            .expected(Expected::Array)
            .range(2.0, MAX_SPLINE_POINTS as f64)
            .got(spline.points.len()));
    }
    if let Some(index) = spline.points.iter().position(|point| {
        !point
            .to_array()
            .iter()
            .all(|value| value.is_finite() && value.abs() <= MAX_COORDINATE)
    }) {
        return Err(invalid(&format!("$.points[{index}]"))
            .expected(Expected::Finite)
            .range(-f64::from(MAX_COORDINATE), f64::from(MAX_COORDINATE))
            .got(spline.points[index].to_array()));
    }
    let mesh = assets
        .mesh(spline.mesh)
        .ok_or_else(|| invalid("$.mesh").expected(Expected::Live))?
        .clone();
    let bounds = mesh.bounds();
    let (start, span) = (bounds.min().x, bounds.max().x - bounds.min().x);
    if span <= 1.0e-3 {
        return Err(invalid("$.mesh")
            .expected(Expected::Above)
            .range(1.0e-3, f64::MAX)
            .got(span));
    }
    let (samples, along) = curve(&spline.points);
    let length = *along.last().expect("curve_has_samples");
    if length <= 1.0e-3 {
        return Err(invalid("$.points")
            .expected(Expected::Above)
            .range(1.0e-3, f64::MAX)
            .got(length));
    }
    let copies = max((length / span).round(), 1.0) as u32;
    let copy = length / copies as f32;
    let stretch = copy / span;
    let per_chunk = max((CHUNK_LENGTH / copy).floor(), 1.0) as u32;
    let mut meshes = Vec::new();
    let mut first = 0;
    while first < copies {
        let end = (first + per_chunk).min(copies);
        let mut vertices = Vec::with_capacity(mesh.vertices().len() * (end - first) as usize);
        let mut indices = Vec::with_capacity(mesh.indices().len() * (end - first) as usize);
        for index in first..end {
            let base = vertices.len() as u32;
            for vertex in mesh.vertices() {
                let [x, y, z] = vertex.position;
                let distance = index as f32 * copy + (x - start) * stretch;
                let (point, tangent) = point_at(&samples, &along, distance);
                let up = (Vec3::Y - tangent * Vec3::Y.dot(tangent)).normalize_or(Vec3::Z);
                let across = tangent.cross(up);
                let [nx, ny, nz] = vertex.normal;
                let normal = (tangent * (nx / stretch) + up * ny + across * nz).normalize_or(up);
                vertices.push(MeshVertex {
                    position: (point + up * y + across * z).to_array(),
                    normal: normal.to_array(),
                    uv: vertex.uv,
                });
            }
            indices.extend(mesh.indices().iter().map(|&vertex| base + vertex));
        }
        meshes.push(Mesh::new(vertices, indices)?);
        first = end;
    }
    Ok(Bent {
        meshes,
        length,
        copies,
    })
}
