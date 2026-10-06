//! Waterfalls (decision 075): where terrains' hydrology bakes find rivers
//! falling, the engine spawns what draws them and despawns it as terrains'
//! bakes finish and terrains go. White water pours down each fall's course
//! as a translucent ribbon the river's width (a spline mesh, decision 071)
//! just over the ground, spray runs down each step of it, bowed a little by
//! gravity, and mist rises from its foot (particles, decision 026): the
//! river carries no water surface over the drop, which a grid of its
//! samples could not follow. Water that does not draw spawns none.

use std::collections::HashMap;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_assets::{
    AssetLibrary, Material, MaterialHandle, Mesh, MeshHandle, MeshVertex, Surface, Texture,
    TextureRef,
};
use glam::Vec3;

use crate::math::clamp;
use crate::terrain::Terrains;
use crate::terrain_data::Fall;
use crate::{ParticleBlend, ParticleEmitter, SplineMesh, Transform, Water};

/// The gravity spray falls under.
const GRAVITY: f32 = 9.8;

/// The entities spawned for each terrain's falls, and the ribbons (by
/// width) and white water's material they share.
#[derive(Resource, Default)]
pub(crate) struct TerrainFalls {
    spawned: HashMap<Entity, (Arc<[Fall]>, Vec<Entity>)>,
    ribbons: Vec<(f32, MeshHandle)>,
    white: Option<MaterialHandle>,
}

/// Meters a ribbon lies over the ground, the meters of it a section spans
/// along its course (its texture's length), and the rows of vertices across
/// it that bend a section with the course.
const RIBBON_RISE: f32 = 0.6;
const RIBBON_SECTION: f32 = 16.0;
const RIBBON_ROWS: u32 = 8;

/// White water's texture: streaks running along the flow (u) that vary
/// quickly across it (v), foam broken by gaps the water and rock show
/// through, fading out toward its edges.
fn foam_texture() -> Texture {
    const ALONG: u32 = 128;
    const ACROSS: u32 = 64;
    let hash = |x: u32, y: u32| {
        let mut h = x.wrapping_mul(0x9e37_79b9) ^ y.wrapping_mul(0x85eb_ca6b);
        h ^= h >> 15;
        h = h.wrapping_mul(0x2c1b_3c6d);
        h ^= h >> 12;
        (h & 0xffff) as f32 / 65_535.0
    };
    // Value noise on a lattice `cells` across each way, wrapping.
    let noise = |u: f32, v: f32, cells: [u32; 2]| {
        let (x, y) = (u * cells[0] as f32, v * cells[1] as f32);
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let (sx, sy) = (fx * fx * (3.0 - 2.0 * fx), fy * fy * (3.0 - 2.0 * fy));
        let at = |dx: u32, dy: u32| hash((x0 as u32 + dx) % cells[0], (y0 as u32 + dy) % cells[1]);
        let low = at(0, 0) + (at(1, 0) - at(0, 0)) * sx;
        let high = at(0, 1) + (at(1, 1) - at(0, 1)) * sx;
        low + (high - low) * sy
    };
    let channel = |value: f32| (clamp(value, 0.0, 1.0) * 255.0 + 0.5) as u8;
    let mut rgba = Vec::with_capacity((ALONG * ACROSS * 4) as usize);
    for row in 0..ACROSS {
        let v = (row as f32 + 0.5) / ACROSS as f32;
        for column in 0..ALONG {
            let u = (column as f32 + 0.5) / ALONG as f32;
            let streaks = 0.6 * noise(u, v, [3, 20]) + 0.4 * noise(u, v, [8, 44]);
            let edge = clamp((1.0 - (2.0 * v - 1.0).abs()) / 0.35, 0.0, 1.0);
            let gaps = clamp((streaks - 0.3) / 0.4, 0.0, 1.0);
            let foam = gaps * gaps * (3.0 - 2.0 * gaps);
            let shade = 0.8 + 0.2 * streaks;
            let alpha = edge * (0.15 + 0.85 * foam);
            rgba.extend_from_slice(&[
                channel(shade * 0.96),
                channel(shade * 0.98),
                channel(shade),
                channel(alpha),
            ]);
        }
    }
    Texture::with_alpha(ALONG, ACROSS, rgba).expect("the foam texture is valid")
}

/// A ribbon section `width` meters across: a flat strip, x along its
/// course, facing up, its texture once along it.
fn ribbon(width: f32) -> Mesh {
    let half = 0.5 * width;
    let mut vertices = Vec::with_capacity(2 * RIBBON_ROWS as usize + 2);
    let mut indices = Vec::with_capacity(6 * RIBBON_ROWS as usize);
    for row in 0..=RIBBON_ROWS {
        let along = row as f32 / RIBBON_ROWS as f32;
        for side in [-1.0, 1.0] {
            vertices.push(MeshVertex {
                position: [along * RIBBON_SECTION, 0.0, side * half],
                normal: [0.0, 1.0, 0.0],
                uv: [along, 0.5 + 0.5 * side],
            });
        }
        if row > 0 {
            let base = 2 * row - 2;
            indices.extend_from_slice(&[base, base + 1, base + 3, base, base + 3, base + 2]);
        }
    }
    Mesh::new(vertices, indices).expect("ribbon sections are valid meshes")
}

/// Meters above a fall's ground its spray runs, and the seconds a spray
/// particle takes down a step of its course.
const SPRAY_RISE: f32 = 1.2;
const STEP_SECONDS: f32 = 1.1;

/// A fall's emitters at `origin`, seeded by `seed`: spray down each step
/// of its course, hugging the ground as the river falls over it, and mist
/// rising from its foot.
fn emitters(fall: &Fall, origin: Vec3, seed: u32) -> Vec<(Transform, ParticleEmitter)> {
    let width = fall.width;
    let at = |point: Vec3| Transform {
        translation: origin + point,
        ..Transform::default()
    };
    let mut emitters: Vec<_> = fall
        .path
        .windows(2)
        .enumerate()
        .map(|(step, pair)| {
            let (from, to) = (Vec3::from_array(pair[0]), Vec3::from_array(pair[1]));
            let line = to - from;
            let spray = ParticleEmitter {
                rate: 6.0 + 3.0 * width,
                lifetime: STEP_SECONDS,
                speed: line.length() / STEP_SECONDS,
                direction: line,
                spread: 0.08,
                // A little gravity bows the spray over the rocks.
                acceleration: Vec3::new(0.0, -0.5 * GRAVITY, 0.0),
                size: [1.6, 4.0],
                color: [[0.88, 0.92, 0.96, 0.7], [0.92, 0.95, 1.0, 0.1]],
                emission: 1.0,
                blend: ParticleBlend::Alpha,
                stretch: 0.15,
                offset: Vec3::ZERO,
                radius: 0.4 * width,
                seed: seed.wrapping_mul(97).wrapping_add(step as u32),
            };
            (
                at(from + Vec3::Y * (SPRAY_RISE + 0.25 * GRAVITY * STEP_SECONDS * STEP_SECONDS)),
                spray,
            )
        })
        .collect();
    let bottom = Vec3::from_array(fall.bottom);
    emitters.push((
        at(bottom + Vec3::Y),
        ParticleEmitter {
            rate: 5.0 + 0.8 * width,
            lifetime: 3.5,
            speed: 1.2,
            direction: Vec3::Y,
            spread: 1.2,
            acceleration: Vec3::new(0.0, 0.5, 0.0),
            size: [3.0, 9.0],
            color: [[0.94, 0.96, 1.0, 0.26], [1.0, 1.0, 1.0, 0.0]],
            emission: 1.0,
            blend: ParticleBlend::Alpha,
            stretch: 0.0,
            offset: Vec3::ZERO,
            radius: 0.5 * width,
            seed: seed ^ 0x9e37_79b9,
        },
    ));
    emitters
}

/// Spawns and despawns waterfall emitters to match terrains' falls.
pub(crate) fn sync(world: &mut World) {
    let drawn = world
        .get_resource::<Water>()
        .is_some_and(|water| water.enabled);
    let wanted: Vec<(Entity, Arc<[Fall]>, Vec3)> = match world.get_resource::<Terrains>() {
        Some(terrains) if drawn => terrains.fall_sets().collect(),
        _ => Vec::new(),
    };
    let mut falls = world.remove_resource::<TerrainFalls>().unwrap_or_default();
    let mut stale = Vec::new();
    falls.spawned.retain(|terrain, (set, entities)| {
        let keep = wanted
            .iter()
            .any(|(entity, wanted, _)| entity == terrain && Arc::ptr_eq(wanted, set));
        if !keep {
            stale.append(entities);
        }
        keep
    });
    for entity in stale {
        if let Ok(entity) = world.get_entity_mut(entity) {
            entity.despawn();
        }
    }
    for (terrain, set, origin) in wanted {
        if set.is_empty() || falls.spawned.contains_key(&terrain) {
            continue;
        }
        let mut entities = Vec::with_capacity(3 * set.len());
        // White water: translucent, streaked and rippling, its edges fading.
        let mut assets = world.resource_mut::<AssetLibrary>();
        let white = match falls.white {
            Some(white) => Some(white),
            None => assets
                .insert_texture(foam_texture())
                .ok()
                .and_then(|foam| {
                    Material::new([0.95, 0.97, 1.0, 1.0])
                        .and_then(|material| material.with_opacity(0.9))
                        .and_then(|material| {
                            material.with_surface(Surface {
                                roughness: 0.45,
                                ripples: 1.0,
                                ..Surface::MATTE
                            })
                        })
                        .and_then(|material| {
                            material.with_base_color_texture(TextureRef::new(foam))
                        })
                        .ok()
                })
                .and_then(|material| assets.insert_material(material).ok()),
        };
        falls.white = white;
        let mut ribbons = Vec::with_capacity(set.len());
        for fall in set.iter() {
            let mesh = match falls.ribbons.iter().find(|(width, _)| *width == fall.width) {
                Some(&(_, mesh)) => Some(mesh),
                None => assets
                    .insert_mesh(ribbon(fall.width))
                    .ok()
                    .inspect(|&mesh| {
                        falls.ribbons.push((fall.width, mesh));
                    }),
            };
            ribbons.push(mesh);
        }
        for (index, (fall, mesh)) in set.iter().zip(ribbons).enumerate() {
            if let (Some(mesh), Some(material)) = (mesh, white) {
                let points = fall
                    .path
                    .iter()
                    .map(|&point| Vec3::from_array(point) + Vec3::Y * RIBBON_RISE)
                    .collect();
                let transform = Transform {
                    translation: origin,
                    ..Transform::default()
                };
                let spline = SplineMesh {
                    points,
                    mesh,
                    material,
                };
                entities.push(world.spawn((transform, spline)).id());
            }
            for (transform, emitter) in emitters(fall, origin, index as u32) {
                entities.push(world.spawn((transform, emitter)).id());
            }
        }
        falls.spawned.insert(terrain, (set, entities));
    }
    world.insert_resource(falls);
}
