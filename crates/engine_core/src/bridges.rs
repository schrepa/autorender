//! Bridges (decision 072) and tunnels (decision 073): what terrains' roads
//! span instead of filling and bore instead of cutting, drawn by entities
//! the engine spawns and despawns as terrains' bakes finish and terrains
//! go. A bridge's deck is a [`SplineMesh`] of a deck section (a slab a
//! road's width and a half meter more on each side, with parapets) along
//! its centerline, and its piers are boxes standing on the ground or water
//! bed under it; a tunnel's tube is a spline mesh of an arched section (a
//! floor slab, walls and a vault) along its road; all are in their
//! terrain's bridge material or a weathered stone.

use std::collections::HashMap;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_assets::{
    AssetLibrary, Material, MaterialHandle, Mesh, MeshHandle, MeshVertex, Surface,
};
use glam::Vec3;

use crate::terrain::Terrains;
use crate::terrain_data::Bridge;
use crate::{Mesh3d, SplineMesh, Transform};

/// Meters a deck's road surface stands above its road's profile, and its
/// slab reaches below it.
pub const DECK_RISE: f32 = 0.2;
pub const DECK_DEPTH: f32 = 1.1;
/// Meters of slab past each side of the road, and its parapets' height and
/// thickness.
pub const DECK_MARGIN: f32 = 0.5;
const PARAPET_HEIGHT: f32 = 0.9;
const PARAPET_THICKNESS: f32 = 0.4;
/// Meters of deck section a copy spans along its curve.
const SECTION_LENGTH: f32 = 4.0;
/// A pier's side and how far it reaches into the ground.
const PIER_SIDE: f32 = 1.8;
const PIER_FOOTING: f32 = 1.0;
/// Meters of clear space past each side of a tunnel's road, its walls'
/// height before the vault springs, its shell's thickness and the vault's
/// segments.
const TUBE_MARGIN: f32 = 1.0;
const TUBE_WALL: f32 = 4.0;
const TUBE_SHELL: f32 = 0.7;
const VAULT_SEGMENTS: usize = 10;

/// The bridge entities spawned for each terrain, the bridges they draw, and
/// the meshes and material they share.
#[derive(Resource, Default)]
pub(crate) struct TerrainBridges {
    spawned: HashMap<Entity, (Arc<[Bridge]>, Vec<Entity>)>,
    decks: Vec<(f32, MeshHandle)>,
    tubes: Vec<(f32, MeshHandle)>,
    pier: Option<MeshHandle>,
    stone: Option<MaterialHandle>,
}

/// A box from `min` to `max` into `vertices` and `indices`, its faces
/// outward.
fn cuboid(min: Vec3, max: Vec3, vertices: &mut Vec<MeshVertex>, indices: &mut Vec<u32>) {
    let corner = |x: bool, y: bool, z: bool| {
        Vec3::new(
            if x { max.x } else { min.x },
            if y { max.y } else { min.y },
            if z { max.z } else { min.z },
        )
    };
    // Each face: its normal and its corners counterclockwise seen from
    // outside.
    let faces = [
        (
            Vec3::X,
            [
                corner(true, false, false),
                corner(true, true, false),
                corner(true, true, true),
                corner(true, false, true),
            ],
        ),
        (
            Vec3::NEG_X,
            [
                corner(false, false, true),
                corner(false, true, true),
                corner(false, true, false),
                corner(false, false, false),
            ],
        ),
        (
            Vec3::Y,
            [
                corner(false, true, false),
                corner(false, true, true),
                corner(true, true, true),
                corner(true, true, false),
            ],
        ),
        (
            Vec3::NEG_Y,
            [
                corner(false, false, true),
                corner(false, false, false),
                corner(true, false, false),
                corner(true, false, true),
            ],
        ),
        (
            Vec3::Z,
            [
                corner(true, false, true),
                corner(true, true, true),
                corner(false, true, true),
                corner(false, false, true),
            ],
        ),
        (
            Vec3::NEG_Z,
            [
                corner(false, false, false),
                corner(false, true, false),
                corner(true, true, false),
                corner(true, false, false),
            ],
        ),
    ];
    for (normal, corners) in faces {
        let base = vertices.len() as u32;
        for position in corners {
            // A repeat of texture every two meters across the face.
            let uv = if normal.x != 0.0 {
                [position.z, position.y]
            } else if normal.y != 0.0 {
                [position.x, position.z]
            } else {
                [position.x, position.y]
            };
            vertices.push(MeshVertex {
                position: position.to_array(),
                normal: normal.to_array(),
                uv: [0.5 * uv[0], 0.5 * uv[1]],
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

/// A deck section for a road `width` meters wide: x along it, y up from the
/// road's profile, z across.
fn deck_section(width: f32) -> Mesh {
    let (mut vertices, mut indices) = (Vec::new(), Vec::new());
    let half = 0.5 * width;
    let outer = half + DECK_MARGIN;
    cuboid(
        Vec3::new(0.0, -DECK_DEPTH, -outer),
        Vec3::new(SECTION_LENGTH, DECK_RISE, outer),
        &mut vertices,
        &mut indices,
    );
    for side in [-1.0_f32, 1.0] {
        let (near, far) = (side * (outer - PARAPET_THICKNESS), side * outer);
        cuboid(
            Vec3::new(0.0, DECK_RISE, if side < 0.0 { far } else { near }),
            Vec3::new(
                SECTION_LENGTH,
                DECK_RISE + PARAPET_HEIGHT,
                if side < 0.0 { near } else { far },
            ),
            &mut vertices,
            &mut indices,
        );
    }
    Mesh::new(vertices, indices).expect("deck sections are valid meshes")
}

/// A terrain, its bridges, its origin and its bridge material.
pub(crate) type BridgeSet = (Entity, Arc<[Bridge]>, Vec3, Option<MaterialHandle>);

/// A tunnel's section for a road `width` meters wide: x along it, y up
/// from the road's profile, z across. Its shell's inner face looks in and
/// its outer face out, its floor is a slab under the road, and rings close
/// each end.
fn tube_section(width: f32) -> Mesh {
    let (mut vertices, mut indices) = (Vec::new(), Vec::new());
    let inner = 0.5 * width + TUBE_MARGIN;
    let outer = inner + TUBE_SHELL;
    cuboid(
        Vec3::new(0.0, -DECK_DEPTH, -outer),
        Vec3::new(SECTION_LENGTH, DECK_RISE, outer),
        &mut vertices,
        &mut indices,
    );
    // The shell's profile across, from the right wall's foot over the vault
    // to the left's: each point at `radius` from the axis it bends about and
    // its outward direction.
    let profile = |radius: f32| -> Vec<(f32, f32, f32, f32)> {
        let mut points = vec![(radius, DECK_RISE, 1.0, 0.0)];
        for segment in 0..=VAULT_SEGMENTS {
            let angle = std::f32::consts::PI * segment as f32 / VAULT_SEGMENTS as f32;
            let (sin, cos) = crate::math::sin_cos(angle);
            points.push((radius * cos, TUBE_WALL + radius * sin, cos, sin));
        }
        points.push((-radius, DECK_RISE, -1.0, 0.0));
        points
    };
    let (inside, outside) = (profile(inner), profile(outer));
    // A quad from `a` to `b` along x, facing `normal`, wound to face it.
    let mut quad = |corners: [Vec3; 4], normal: Vec3, vertices: &mut Vec<MeshVertex>| {
        let base = vertices.len() as u32;
        for corner in &corners {
            // A repeat of texture every two meters, along and around.
            vertices.push(MeshVertex {
                position: corner.to_array(),
                normal: normal.to_array(),
                uv: [0.5 * corner.x, 0.5 * (corner.y + corner.z)],
            });
        }
        let facing = (corners[1] - corners[0])
            .cross(corners[2] - corners[0])
            .dot(normal)
            >= 0.0;
        if facing {
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        } else {
            indices.extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);
        }
    };
    for pair in 0..inside.len() - 1 {
        for (points, sign) in [(&inside, -1.0_f32), (&outside, 1.0)] {
            let (z0, y0, nz0, ny0) = points[pair];
            let (z1, y1, nz1, ny1) = points[pair + 1];
            let normal =
                Vec3::new(0.0, 0.5 * (ny0 + ny1), 0.5 * (nz0 + nz1)).normalize_or(Vec3::Y) * sign;
            quad(
                [
                    Vec3::new(0.0, y0, z0),
                    Vec3::new(SECTION_LENGTH, y0, z0),
                    Vec3::new(SECTION_LENGTH, y1, z1),
                    Vec3::new(0.0, y1, z1),
                ],
                normal,
                &mut vertices,
            );
        }
        // The rings closing each end, between the faces.
        for (x, normal) in [(0.0, Vec3::NEG_X), (SECTION_LENGTH, Vec3::X)] {
            let (zi0, yi0, ..) = inside[pair];
            let (zi1, yi1, ..) = inside[pair + 1];
            let (zo0, yo0, ..) = outside[pair];
            let (zo1, yo1, ..) = outside[pair + 1];
            quad(
                [
                    Vec3::new(x, yi0, zi0),
                    Vec3::new(x, yo0, zo0),
                    Vec3::new(x, yo1, zo1),
                    Vec3::new(x, yi1, zi1),
                ],
                normal,
                &mut vertices,
            );
        }
    }
    Mesh::new(vertices, indices).expect("tunnel sections are valid meshes")
}

/// Spawns and despawns bridge entities to match terrains' bridges.
pub(crate) fn sync(world: &mut World) {
    let wanted: Vec<BridgeSet> = match world.get_resource::<Terrains>() {
        Some(terrains) => terrains.bridge_sets().collect(),
        None => Vec::new(),
    };
    let mut bridges = world
        .remove_resource::<TerrainBridges>()
        .unwrap_or_default();
    // Terrains gone, or whose bridges changed.
    let mut stale = Vec::new();
    bridges.spawned.retain(|terrain, (set, entities)| {
        let keep = wanted
            .iter()
            .any(|(entity, wanted, ..)| entity == terrain && Arc::ptr_eq(wanted, set));
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
    for (terrain, set, origin, material) in wanted {
        if set.is_empty() || bridges.spawned.contains_key(&terrain) {
            continue;
        }
        let mut entities = Vec::new();
        let mut assets = world.resource_mut::<AssetLibrary>();
        let material = match material.or(bridges.stone) {
            Some(material) => material,
            None => {
                let stone = Material::new([0.56, 0.53, 0.49, 1.0])
                    .and_then(|material| {
                        material.with_surface(Surface {
                            roughness: 0.85,
                            ..Surface::MATTE
                        })
                    })
                    .expect("the stone material is valid");
                let Ok(stone) = assets.insert_material(stone) else {
                    continue;
                };
                bridges.stone = Some(stone);
                stone
            }
        };
        let pier = match bridges.pier {
            Some(pier) => pier,
            None => {
                let Ok(pier) = assets.insert_mesh(Mesh::cube()) else {
                    continue;
                };
                bridges.pier = Some(pier);
                pier
            }
        };
        let mut spawns = Vec::new();
        for bridge in set.iter() {
            // A deck for bridges, a tube for tunnels, a mesh a road width.
            let (sections, build): (_, fn(f32) -> Mesh) = if bridge.tunnel {
                (&mut bridges.tubes, tube_section)
            } else {
                (&mut bridges.decks, deck_section)
            };
            let deck = match sections.iter().find(|(width, _)| *width == bridge.width) {
                Some(&(_, deck)) => deck,
                None => {
                    let Ok(deck) = assets.insert_mesh(build(bridge.width)) else {
                        continue;
                    };
                    sections.push((bridge.width, deck));
                    deck
                }
            };
            spawns.push((
                Transform {
                    translation: origin,
                    ..Transform::default()
                },
                Some(SplineMesh {
                    points: bridge.deck.clone(),
                    mesh: deck,
                    material,
                }),
                None,
            ));
            for &(top, ground) in &bridge.piers {
                let (high, low) = (top.y - DECK_DEPTH, ground - PIER_FOOTING);
                if high <= low {
                    continue;
                }
                spawns.push((
                    Transform {
                        translation: origin + Vec3::new(top.x, 0.5 * (high + low), top.z),
                        scale: Vec3::new(PIER_SIDE, high - low, PIER_SIDE),
                        ..Transform::default()
                    },
                    None,
                    Some(Mesh3d {
                        mesh: pier,
                        material,
                    }),
                ));
            }
        }
        for (transform, spline, mesh) in spawns {
            let mut entity = world.spawn(transform);
            if let Some(spline) = spline {
                entity.insert(spline);
            }
            if let Some(mesh) = mesh {
                entity.insert(mesh);
            }
            entities.push(entity.id());
        }
        bridges.spawned.insert(terrain, (set, entities));
    }
    world.insert_resource(bridges);
}
