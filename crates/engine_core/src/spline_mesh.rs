//! Spline meshes (decision 071): one mesh repeated end to end and bent
//! along a curve, for bridges' decks, walls, fences, rails and pipes. The
//! curve is a centripetal Catmull-Rom spline through the component's
//! points; copies of the mesh lie along it, stretched so a whole number of
//! them spans it, and build into meshes of at most [`CHUNK_LENGTH`] meters
//! of curve each, drawn by child entities, so distant chunks cull. Chunks
//! build on the extraction path before transforms refresh, so paused
//! frames show them, and rebuild when the component changes.

use bevy_ecs::{lifecycle::HookContext, prelude::*, world::DeferredWorld};
use engine_assets::{AssetLibrary, MaterialHandle, Mesh, MeshHandle, MeshVertex};
use glam::Vec3;

use crate::hierarchy::set_parents;
use crate::math::max;
use crate::{Mesh3d, Transform};

/// Meters of curve a chunk spans at most: long enough that a bridge or
/// wall is a draw or two (each chunk is one), short enough that long curves
/// still cull in pieces.
pub const CHUNK_LENGTH: f32 = 256.0;
/// Most points a curve names.
pub const MAX_SPLINE_POINTS: usize = 4096;
/// Meters between the points the curve is evaluated at.
const STEP: f32 = 0.5;

/// A mesh bent along a curve through `points` (decision 071), in the
/// entity's space: 2 to 4096 of them, joined by a centripetal Catmull-Rom
/// spline. Copies of `mesh` lie end to end along it, the mesh's x axis
/// along the curve, its y up and its z across, each stretched along x so a
/// whole number of them spans the curve. Its frames keep y up: they turn
/// about the vertical as the curve does and tilt only where it climbs.
/// Child entities draw it with `material`, a chunk of curve each.
#[derive(Component, Clone, Debug, PartialEq)]
#[component(on_remove = release)]
pub struct SplineMesh {
    pub points: Vec<Vec3>,
    pub mesh: MeshHandle,
    pub material: MaterialHandle,
}

/// What a spline mesh built (decision 071): its chunks' entities and meshes,
/// the curve's length, the copies along it and their vertices, or why it
/// built nothing.
#[derive(Component, Clone, Debug, Default)]
pub struct SplineChunks {
    pub entities: Vec<Entity>,
    pub meshes: Vec<MeshHandle>,
    pub length: f32,
    pub copies: u32,
    pub vertices: u32,
    pub error: Option<String>,
    built: Option<SplineMesh>,
}

/// Despawns a spline mesh's chunks and frees their meshes when it goes.
fn release(mut world: DeferredWorld, context: HookContext) {
    let Some(chunks) = world.get::<SplineChunks>(context.entity) else {
        return;
    };
    let (entities, meshes) = (chunks.entities.clone(), chunks.meshes.clone());
    world.commands().queue(move |world: &mut World| {
        discard(world, &entities, &meshes);
    });
}

fn discard(world: &mut World, entities: &[Entity], meshes: &[MeshHandle]) {
    for &entity in entities {
        if let Ok(entity) = world.get_entity_mut(entity) {
            entity.despawn();
        }
    }
    let mut assets = world.resource_mut::<AssetLibrary>();
    for &mesh in meshes {
        assets.remove_mesh(mesh);
    }
}

/// Builds the chunks of every spline mesh that changed since it last
/// built.
pub(crate) fn build(world: &mut World) {
    let mut query = world.query::<(Entity, &SplineMesh, Option<&SplineChunks>)>();
    let stale: Vec<(Entity, SplineMesh)> = query
        .iter(world)
        .filter(|(_, spline, chunks)| {
            chunks.is_none_or(|chunks| chunks.built.as_ref() != Some(*spline))
        })
        .map(|(entity, spline, _)| (entity, spline.clone()))
        .collect();
    for (entity, spline) in stale {
        if let Some(old) = world.get::<SplineChunks>(entity) {
            let (entities, meshes) = (old.entities.clone(), old.meshes.clone());
            discard(world, &entities, &meshes);
        }
        let built = match bend(world.resource::<AssetLibrary>(), &spline) {
            Ok(built) => built,
            Err(error) => {
                world.entity_mut(entity).insert(SplineChunks {
                    error: Some(error),
                    built: Some(spline),
                    ..SplineChunks::default()
                });
                continue;
            }
        };
        let mut chunks = SplineChunks {
            length: built.length,
            copies: built.copies,
            vertices: built
                .meshes
                .iter()
                .map(|mesh| mesh.vertices().len() as u32)
                .sum(),
            built: Some(spline.clone()),
            ..SplineChunks::default()
        };
        let parented = world.get::<Transform>(entity).is_some();
        for mesh in built.meshes {
            let Ok(handle) = world.resource_mut::<AssetLibrary>().insert_mesh(mesh) else {
                chunks.error = Some("the asset library refused a chunk's mesh".into());
                break;
            };
            chunks.meshes.push(handle);
            let child = world
                .spawn((
                    Transform::default(),
                    Mesh3d {
                        mesh: handle,
                        material: spline.material,
                    },
                ))
                .id();
            chunks.entities.push(child);
        }
        if parented {
            let changes: Vec<_> = chunks
                .entities
                .iter()
                .map(|&child| (child, Some(entity)))
                .collect();
            if let Err(error) = set_parents(world, &changes) {
                chunks.error = Some(error.to_string());
            }
        }
        world.entity_mut(entity).insert(chunks);
    }
}

/// A curve's chunk meshes, its length and the copies along it.
struct Bent {
    meshes: Vec<Mesh>,
    length: f32,
    copies: u32,
}

/// The curve through `points` every [`STEP`] meters or so (a centripetal
/// Catmull-Rom spline, its ends continued straight), and the meters along
/// it at each.
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
        // Knots spaced by the square roots of the distances (centripetal).
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

/// The curve's point and direction `distance` meters along it.
fn point_at(samples: &[Vec3], along: &[f32], distance: f32) -> (Vec3, Vec3) {
    let last = samples.len() - 1;
    let index = along
        .partition_point(|&value| value <= distance)
        .saturating_sub(1)
        .min(last - 1);
    let (a, b) = (samples[index], samples[index + 1]);
    let run = along[index + 1] - along[index];
    let t = if run > 0.0 {
        crate::math::clamp((distance - along[index]) / run, 0.0, 1.0)
    } else {
        0.0
    };
    // The direction across a few samples, smooth through the curve's turns.
    let back = samples[index.saturating_sub(1)];
    let ahead = samples[(index + 2).min(last)];
    let direction = (b - a) * 2.0 + (ahead - back);
    (a + (b - a) * t, direction.normalize_or(b - a))
}

/// Bends copies of `spline`'s mesh along its curve into chunk meshes.
fn bend(assets: &AssetLibrary, spline: &SplineMesh) -> Result<Bent, String> {
    if !(2..=MAX_SPLINE_POINTS).contains(&spline.points.len()) {
        return Err(format!("a curve names 2 to {MAX_SPLINE_POINTS} points"));
    }
    if spline.points.iter().any(|point| !point.is_finite()) {
        return Err("curve points must be finite".into());
    }
    let mesh = assets
        .mesh(spline.mesh)
        .ok_or("its mesh is not resident")?
        .clone();
    let bounds = mesh.bounds();
    let (start, span) = (bounds.min().x, bounds.max().x - bounds.min().x);
    if span <= 1.0e-3 {
        return Err("its mesh must span more than a millimeter along x".into());
    }
    let (samples, along) = curve(&spline.points);
    let length = *along.last().expect("curves hold samples");
    if length <= 1.0e-3 {
        return Err("its curve must be longer than a millimeter".into());
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
                // Frames keep up as up: turning about the vertical with the
                // curve, tilting only where it climbs.
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
        meshes.push(Mesh::new(vertices, indices).map_err(|error| error.to_string())?);
        first = end;
    }
    Ok(Bent {
        meshes,
        length,
        copies,
    })
}
