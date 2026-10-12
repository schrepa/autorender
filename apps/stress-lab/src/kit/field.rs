//! The tower field: a grid of towers bobbing and turning about their
//! anchors, which the field stands on open ground and the highlands on
//! their plateau; the dense world's moving objects move the same way.

use engine::prelude::*;

use super::random::hash;
use super::{StressConfig, Workload};
use crate::{FrameMetrics, metrics};

const GROUP_SIZE: u32 = 32;
pub(crate) const SPACING: f32 = 2.4;

#[derive(Component)]
pub(crate) struct Motion {
    pub anchor: Vec3,
    pub rotation: Quat,
    pub phase: f32,
    pub speed: f32,
}

/// The tower field: a grid of towers `lift` above the ground, optionally
/// grouped under parents of 32, with the first `colliders` carrying
/// colliders. Returns the moving node count.
pub(crate) fn field(world: &mut World, config: &StressConfig, lift: f32) -> Result<u32, String> {
    let meshes = create_assets(&mut world.resource_mut::<AssetLibrary>(), config)?;
    let mut parent_changes = Vec::with_capacity(if config.hierarchy {
        config.renderables as usize
    } else {
        0
    });
    let mut probes = Vec::with_capacity(config.colliders as usize);
    let mut parent = None;
    let mut parent_inverse = Mat4::IDENTITY;
    let mut parent_count = 0;
    let mut moving_nodes = 0;
    let mut sensor_nodes = 0;
    for index in 0..config.renderables {
        let position = position_for_index(index, config) + Vec3::Y * lift;
        if config.hierarchy && index % GROUP_SIZE == 0 {
            let end = (index + GROUP_SIZE - 1).min(config.renderables - 1);
            let anchor = (position + position_for_index(end, config) + Vec3::Y * lift) * 0.5;
            let transform = Transform {
                translation: anchor,
                rotation: Quat::from_rotation_y(0.25 + (index % 5) as f32 * 0.1),
                scale: Vec3::new(1.2, 0.9, 0.8),
            };
            parent_inverse = transform.matrix().inverse();
            parent = Some(world.spawn(transform).id());
            parent_count += 1;
        }
        let random = hash(index ^ config.seed);
        let tall = if config.dense_colliders && index < config.colliders {
            1.0
        } else {
            1.0 + ((random >> 8) % 70) as f32 * 0.11
        };
        let transform = Transform {
            translation: if parent.is_some() {
                parent_inverse.transform_point3(position)
            } else {
                position
            },
            rotation: Quat::from_rotation_y((random % 8) as f32 * 0.16),
            scale: Vec3::new(
                0.7 + (random % 5) as f32 * 0.08,
                tall,
                0.7 + ((random >> 4) % 5) as f32 * 0.08,
            ),
        };
        let mut entity = world.spawn((transform, meshes[index as usize % meshes.len()]));
        if index % 100 < config.moving_percent {
            entity.insert(Motion {
                anchor: transform.translation,
                rotation: transform.rotation,
                phase: (random % 10_000) as f32 * 0.001,
                speed: 0.3 + ((random >> 12) % 15) as f32 * 0.05,
            });
            if config.interpolation {
                entity.insert(InterpolateTransform);
            }
            moving_nodes += 1;
        }
        if index < config.colliders {
            let sensor = index % 100 < config.sensor_percent;
            // Alternate primitives exercise affine boxes and stretched spheres.
            let collider = if index % 4 == 0 {
                Collider::sphere(0.5)
            } else {
                Collider::cuboid(Vec3::splat(0.5))
            }
            .map_err(|error| error.to_string())?
            .with_sensor(sensor);
            entity.insert(collider);
            probes.push(entity.id());
            sensor_nodes += u32::from(sensor);
        }
        if let Some(parent) = parent {
            parent_changes.push((entity.id(), Some(parent)));
        }
    }
    set_parents(world, &parent_changes).map_err(|error| error.to_string())?;
    world.insert_resource(Workload {
        transform_nodes: config.renderables + parent_count,
        moving_nodes,
        sensor_nodes,
        probes,
        overlap_hits: Vec::with_capacity(if config.queries_per_tick > 0 { 128 } else { 0 }),
    });
    Ok(moving_nodes)
}

pub(crate) fn position_for_index(index: u32, config: &StressConfig) -> Vec3 {
    if config.dense_colliders && index < config.colliders {
        let side = math::max((config.colliders as f32).cbrt().ceil(), 1.0) as u32;
        let center = (side - 1) as f32 * 0.125;
        return Vec3::new(
            (index % side) as f32 * 0.25 - center,
            ((index / side) % side) as f32 * 0.25 + 3.0,
            (index / (side * side)) as f32 * 0.25 - center,
        );
    }
    let side = (config.renderables as f32).sqrt().ceil() as u32;
    let center = (side - 1) as f32 * SPACING * 0.5;
    let random = hash(index ^ config.seed);
    let height = 1.0 + ((random >> 8) % 70) as f32 * 0.11;
    Vec3::new(
        (index % side) as f32 * SPACING - center,
        height * 0.5,
        (index / side) as f32 * SPACING - center,
    )
}

fn create_assets(assets: &mut AssetLibrary, config: &StressConfig) -> Result<Vec<Mesh3d>, String> {
    let colors = [
        [0.09, 0.88, 0.78, 1.0],
        [0.10, 0.49, 1.0, 1.0],
        [0.55, 0.20, 1.0, 1.0],
        [1.0, 0.22, 0.52, 1.0],
        [1.0, 0.61, 0.13, 1.0],
        [0.25, 0.93, 0.54, 1.0],
        [0.45, 0.72, 1.0, 1.0],
        [0.92, 0.37, 0.97, 1.0],
    ];
    (0..config.mesh_variety)
        .map(|index| {
            let source = if index % 3 == 2 {
                Mesh::octahedron()
            } else {
                Mesh::cube()
            };
            // Distinct immutable meshes make the batch/upload workload explicit.
            let mut vertices = source.vertices().to_vec();
            for vertex in &mut vertices {
                let scale_x = 0.85 + (index % 4) as f32 * 0.05;
                let scale_z = 0.85 + (index % 7) as f32 * 0.025;
                vertex.position[0] *= scale_x;
                vertex.position[2] *= scale_z;
                vertex.normal[0] /= scale_x;
                vertex.normal[2] /= scale_z;
            }
            let mesh = assets
                .insert_mesh(
                    Mesh::new(vertices, source.indices().to_vec())
                        .map_err(|error| error.to_string())?,
                )
                .map_err(|error| error.to_string())?;
            let color = colors[index as usize % colors.len()];
            // Surfaces from glossy metal to matte; every
            // fourth tower's bright stripes light up.
            let lit = config.textured && index % 4 == 1;
            let surface = Surface {
                roughness: [0.25, 0.5, 0.8, 0.35][index as usize % 4],
                metalness: if index % 3 == 0 { 0.7 } else { 0.0 },
                emission: if lit { 1.2 } else { 0.0 },
                ripples: 0.0,
            };
            let material = if config.textured {
                let mut pixels = Vec::with_capacity(64 * 64 * 4);
                for y in 0..64 {
                    for x in 0..64 {
                        let stripe = (x / 4 + y / 4 + index) % 2 == 0;
                        let value = if x % 8 == 0 || y % 8 == 0 {
                            90
                        } else if stripe {
                            255
                        } else {
                            195
                        };
                        let emits = lit && value == 255;
                        pixels.extend_from_slice(&[
                            value,
                            value,
                            value,
                            if emits { 255 } else { 0 },
                        ]);
                    }
                }
                let texture = assets
                    .insert_texture(
                        Texture::from_rgba(64, 64, pixels).map_err(|error| error.to_string())?,
                    )
                    .map_err(|error| error.to_string())?;
                Material::textured(color, texture)
            } else {
                Material::new(color)
            }
            .and_then(|material| material.with_surface(surface))
            .map_err(|error| error.to_string())?;
            let material = assets
                .insert_material(material)
                .map_err(|error| error.to_string())?;
            Ok(Mesh3d { mesh, material })
        })
        .collect()
}

pub(crate) fn animate(
    time: Res<FixedTime>,
    mut objects: Query<(&Motion, &mut Transform)>,
    mut metrics: ResMut<FrameMetrics>,
) {
    let start = metrics::now_ms();
    let t = (time.elapsed_seconds % 10_000.0) as f32;
    move_lanes(t, objects.iter_mut());
    metrics.animation_ms += metrics::now_ms() - start;
}

/// Each object bobs and turns about its anchor, four objects at a time with
/// the engine's SIMD helpers: the operations of `move_each` per lane, so
/// the same bits (glam's rotation constructors take `sin_cos` of half the
/// angle).
fn move_lanes<'a, T: std::ops::DerefMut<Target = Transform>>(
    t: f32,
    objects: impl Iterator<Item = (&'a Motion, T)>,
) {
    math::for_each4(objects, |lanes| {
        let angle =
            lanes.load(|(motion, _)| motion.speed) * t + lanes.load(|(motion, _)| motion.phase);
        let (wave, _) = math::sin_cos4(angle);
        let (yaw_sin, yaw_cos) = math::sin_cos4(angle * 0.7 * 0.5);
        let (roll_sin, roll_cos) = math::sin_cos4(wave * 0.12 * 0.5);
        lanes.for_each(|lane, (motion, transform)| {
            transform.translation = motion.anchor + Vec3::Y * (wave[lane] * 0.6);
            transform.rotation = motion.rotation
                * Quat::from_xyzw(0.0, yaw_sin[lane], 0.0, yaw_cos[lane])
                * Quat::from_xyzw(0.0, 0.0, roll_sin[lane], roll_cos[lane]);
        });
    });
}
