use engine::prelude::*;

use std::collections::HashMap;

use crate::{FrameMetrics, StressConfig, StressScene, city, dense, highlands, metrics, metro};

const GROUP_SIZE: u32 = 32;
pub(crate) const SPACING: f32 = 2.4;

#[derive(Resource)]
pub(crate) struct Workload {
    pub transform_nodes: u32,
    pub moving_nodes: u32,
    pub sensor_nodes: u32,
    pub probes: Vec<Entity>,
    pub overlap_hits: Vec<Entity>,
}

#[derive(Component)]
pub(crate) struct Motion {
    pub anchor: Vec3,
    pub rotation: Quat,
    pub phase: f32,
    pub speed: f32,
}

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Animation;

/// Creates an independently owned world. Failed configuration leaves any active
/// BrowserApp untouched. Distinct AssetLibrary identities prevent GPU aliasing.
pub fn create_workload(config: StressConfig) -> Result<Engine, String> {
    create_workload_with(config, &HashMap::new())
}

/// A file a workload reads: its bytes (a virtual mesh's may be the prefix)
/// and the URL the rest streams from, or an image's RGBA8 pixels and their
/// width and height.
#[derive(Clone, Debug, Default)]
pub struct StressFile {
    pub bytes: Vec<u8>,
    pub url: Option<String>,
    pub size: Option<[u32; 2]>,
}

/// [`create_workload`] reading the files a scene needs from `files` (the
/// dense scene's virtual meshes, decision 044, and the highlands' terrain
/// materials, decision 054).
pub fn create_workload_with(
    config: StressConfig,
    files: &HashMap<String, StressFile>,
) -> Result<Engine, String> {
    config.validate()?;
    let mut engine = Engine::new(EngineConfig {
        frustum_culling: config.culling,
        ..EngineConfig::default()
    });
    let collision_enabled = config.colliders > 0 || config.queries_per_tick > 0;
    if collision_enabled {
        engine.app.add_plugins(CollisionPlugin);
    }
    engine.app.insert_resource(config.clone());
    let (atmosphere, lamps) = look(config.scene, "scene")?;
    engine.app.insert_resource(atmosphere);
    engine.app.init_resource::<FrameMetrics>();
    let world = engine.app.world_mut();
    let (moving_nodes, bounds) = match config.scene {
        StressScene::Field => {
            // The grid, the height objects bob to, and a margin.
            let half = (config.renderables as f32).sqrt().ceil() * SPACING * 0.5 + 4.0;
            let bounds = (Vec3::new(-half, -1.0, -half), Vec3::new(half, 12.0, half));
            (field(world, &config, 0.0)?, bounds)
        }
        StressScene::Highlands => {
            let (bounds, plateau) =
                highlands::populate(world, &config, highlands::hamlet_radius(&config), files)?;
            (field(world, &config, plateau)?, bounds)
        }
        StressScene::City => {
            let city = city::populate(world, &config)?;
            world.insert_resource(Workload {
                transform_nodes: city.transform_nodes,
                moving_nodes: city.moving_nodes,
                sensor_nodes: city.sensor_nodes,
                probes: city.probes,
                overlap_hits: Vec::with_capacity(if config.queries_per_tick > 0 { 128 } else { 0 }),
            });
            (city.moving_nodes, city.bounds)
        }
        StressScene::Metro => {
            let city = metro::populate(world, &config)?;
            world.insert_resource(Workload {
                transform_nodes: city.transform_nodes,
                moving_nodes: city.moving_nodes,
                sensor_nodes: city.sensor_nodes,
                probes: city.probes,
                overlap_hits: Vec::with_capacity(if config.queries_per_tick > 0 { 128 } else { 0 }),
            });
            // Glass refracts what is behind it (decision 038).
            world.insert_resource(Refraction::default());
            (city.moving_nodes, city.bounds)
        }
        StressScene::Dense => dense::populate(world, &config, files)?,
    };
    // A brisk wind sways the highlands' grass and spruces (decision 058),
    // and drives clouds (decision 084) over them twice as fast aloft.
    if config.scene == StressScene::Highlands {
        world.insert_resource(Wind {
            velocity: [5.5, 2.0],
            gusts: 0.55,
        });
        world.insert_resource(Clouds {
            coverage: 0.38,
            base: 1500.0,
            thickness: 1500.0,
            wind: [11.0, 4.0],
            shadows: 0.6,
            ..Clouds::default()
        });
    }
    // Shadows (decision 020) keep the whole scene in one map, seen from
    // every view; the highlands' reach across the valleys, and two kept
    // levels (decision 065) keep still casters sharper within a kilometer
    // of the camera: a quarter meter a texel within 256 meters, one beyond.
    world.insert_resource(if config.scene == StressScene::Highlands {
        Shadows {
            bounds: Some(bounds),
            distance: 6000.0,
            near_distance: 160.0,
            resolution: 4096,
            levels: 2,
            level_extent: 512.0,
            ..Shadows::default()
        }
    } else {
        Shadows {
            bounds: Some(bounds),
            distance: 1600.0,
            near_distance: 90.0,
            resolution: 4096,
            ..Shadows::default()
        }
    });
    // Sun shafts (decision 024) and screen-space reflections (decision 025)
    // are opt-in engine effects; street lamps near the camera light the
    // city (decision 023), brighter as the look darkens.
    world.insert_resource(stress_shafts());
    // The highlands' lakes (decision 067) mirror shores kilometers away.
    world.insert_resource(if config.scene == StressScene::Highlands {
        Reflections {
            distance: 4000.0,
            ..Reflections::default()
        }
    } else {
        Reflections::default()
    });
    world.insert_resource(LocalLights {
        distance: 220.0,
        intensity: lamps,
        ..LocalLights::default()
    });
    if collision_enabled {
        refresh_collision_world(world).map_err(|error| error.to_string())?;
    }
    set_view(&mut engine, "overview")?;
    engine.app.add_plugins(FlyCameraPlugin);
    // The highlands' cave opens once its terrain's bakes are done.
    if config.scene == StressScene::Highlands {
        highlands::add_systems(&mut engine.app);
    }
    // The metro's billboards change pictures whether or not anything moves.
    if config.scene == StressScene::Metro {
        metro::add_systems(&mut engine.app);
    } else if moving_nodes > 0 {
        match config.scene {
            StressScene::Field => {
                engine
                    .app
                    .add_systems(FixedSimulation, animate.in_set(Animation));
            }
            _ => city::add_systems(&mut engine.app),
        }
    }
    if config.queries_per_tick > 0 {
        engine
            .app
            .add_systems(FixedSimulation, query_workload.after(Animation));
    }
    if collision_enabled {
        metrics::add_timers(&mut engine.app);
    }
    Ok(engine)
}

/// The tower field: a grid of towers `lift` above the ground, optionally
/// grouped under parents of 32, with the first `colliders` carrying
/// colliders. Returns the moving node count.
fn field(world: &mut World, config: &StressConfig, lift: f32) -> Result<u32, String> {
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

pub(crate) fn hash(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 16)
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
            // Surfaces (decision 019) from glossy metal to matte; every
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
                        Texture::with_emission_mask(64, 64, pixels)
                            .map_err(|error| error.to_string())?,
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

fn animate(
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

fn query_workload(
    config: Res<StressConfig>,
    time: Res<FixedTime>,
    globals: Res<GlobalTransforms>,
    mut collision: ResMut<CollisionWorld>,
    mut workload: ResMut<Workload>,
    mut metrics: ResMut<FrameMetrics>,
) {
    let start = metrics::now_ms();
    for index in 0..config.queries_per_tick {
        let point = if workload.probes.is_empty() {
            Vec3::ZERO
        } else {
            let entity =
                workload.probes[(time.tick as usize + index as usize * 37) % workload.probes.len()];
            globals
                .get(entity)
                .and_then(|global| global.matrix())
                .map_or(Vec3::ZERO, |matrix| matrix.w_axis.truncate())
        };
        let result = match index % 3 {
            0 => collision
                .raycast(
                    CollisionRay {
                        origin: point + Vec3::Y * 12.0,
                        direction: Vec3::NEG_Y,
                        max_distance: 24.0,
                    },
                    QueryFilter::default(),
                )
                .map(|hit| usize::from(hit.is_some())),
            1 => collision
                .overlap(
                    ColliderShape::Cuboid {
                        half_extents: Vec3::splat(0.8),
                    },
                    Mat4::from_translation(point),
                    QueryFilter::default(),
                    &mut workload.overlap_hits,
                    128,
                )
                .map(|()| workload.overlap_hits.len()),
            _ => collision
                .cast_shape(
                    ColliderShape::Sphere { radius: 0.3 },
                    Mat4::from_translation(point + Vec3::X * 3.0),
                    Vec3::NEG_X * 6.0,
                    QueryFilter::default(),
                    ShapeCastOptions::default(),
                )
                .map(|hit| usize::from(hit.is_some())),
        };
        metrics.query_result(result);
    }
    metrics.queries_ms += metrics::now_ms() - start;
}

/// Sun shafts strong enough to read over the scenes' light fog.
pub fn stress_shafts() -> SunShafts {
    SunShafts {
        intensity: 1.5,
        ..SunShafts::default()
    }
}

/// A lighting preset: its name, atmosphere and street lamp brightness.
type Look = (&'static str, fn() -> Atmosphere, f32);

/// The lighting presets; lamps are off by day.
const LOOKS: [Look; 4] = [
    ("golden", Atmosphere::golden_hour, 0.35),
    ("dusk", Atmosphere::dusk, 0.8),
    ("night", Atmosphere::night, 1.0),
    ("day", Atmosphere::day, 0.0),
];

/// The lighting preset `name` and its street lamp brightness: `golden`,
/// `dusk`, `night` or `day`, or `scene` for the scene's own (night over the
/// tower field, golden hour over the city, day over dense geometry).
pub fn look(scene: StressScene, name: &str) -> Result<(Atmosphere, f32), String> {
    let name = match (name, scene) {
        ("scene", StressScene::Field) => "night",
        ("scene", StressScene::City | StressScene::Metro) => "golden",
        ("scene", StressScene::Dense) => "day",
        ("scene", StressScene::Highlands) => "golden",
        _ => name,
    };
    LOOKS
        .iter()
        .find(|(look, ..)| *look == name)
        .map(|&(_, atmosphere, lamps)| (atmosphere(), lamps))
        .ok_or_else(|| "look must be scene, golden, dusk, night or day".into())
}

pub fn set_view(engine: &mut Engine, view: &str) -> Result<(), String> {
    let config = engine.app.world().resource::<StressConfig>();
    let span = (config.renderables as f32).sqrt().ceil() * SPACING;
    let camera = match view {
        "overview" if config.scene == StressScene::Highlands => {
            let (position, target) = highlands::overview(engine.app.world(), config);
            Camera {
                position,
                target,
                near: 0.5,
                far: 20_000.0,
                vertical_fov_radians: 55.0_f32.to_radians(),
            }
        }
        "close" if config.scene == StressScene::Highlands => {
            let (position, target) = highlands::close_view(engine.app.world(), config);
            Camera {
                position,
                target,
                near: 0.2,
                far: 20_000.0,
                ..Camera::default()
            }
        }
        "overview" if config.scene == StressScene::Metro => {
            let (position, target) = metro::overview(config);
            Camera {
                position,
                target,
                far: 5000.0,
                vertical_fov_radians: 55.0_f32.to_radians(),
                ..Camera::default()
            }
        }
        "close" if config.scene == StressScene::Metro => {
            let (position, target) = metro::close_view(config);
            Camera {
                position,
                target,
                far: 5000.0,
                ..Camera::default()
            }
        }
        "overview" => Camera {
            position: Vec3::new(span * 0.46, span * 0.62, span * 0.8),
            target: Vec3::ZERO,
            far: 5000.0,
            vertical_fov_radians: 55.0_f32.to_radians(),
            ..Camera::default()
        },
        // Across the river toward the city's core.
        "close" if config.scene == StressScene::City => {
            let (position, target) = city::close_view(config);
            Camera {
                position,
                target,
                far: 5000.0,
                ..Camera::default()
            }
        }
        "close" => Camera {
            position: Vec3::new(15.0, 15.0, 23.0),
            target: Vec3::new(0.0, 3.0, 0.0),
            far: 5000.0,
            ..Camera::default()
        },
        "away" if config.scene == StressScene::Highlands => {
            let (position, target) = highlands::away_view(engine.app.world());
            Camera {
                position,
                target,
                near: 0.5,
                far: 20_000.0,
                ..Camera::default()
            }
        }
        "away" => {
            let position = Vec3::new(0.0, span, span * 1.4);
            Camera {
                position,
                target: position + Vec3::Z,
                far: 5000.0,
                ..Camera::default()
            }
        }
        _ => return Err("view must be overview, close or away".into()),
    };
    // The highlands' camera crosses kilometers.
    let speed = if config.scene == StressScene::Highlands {
        160.0
    } else {
        math::clamp(span * 0.3, 22.0, 220.0)
    };
    engine.app.insert_resource(camera);
    engine
        .app
        .insert_resource(FlyCameraController::from_camera(&camera).with_speed(speed));
    Ok(())
}
