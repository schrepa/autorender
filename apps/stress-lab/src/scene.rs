use engine::prelude::*;

use engine::collections::HashMap;

use crate::kit::field::SPACING;
use crate::kit::motion::Animation;
use crate::{FrameMetrics, StressConfig, StressScene, metrics};

#[derive(Resource)]
pub(crate) struct Workload {
    pub transform_nodes: u32,
    pub moving_nodes: u32,
    pub sensor_nodes: u32,
    pub probes: Vec<Entity>,
    pub overlap_hits: Vec<Entity>,
}

/// Creates an independently owned world. Failed configuration leaves the
/// running engine untouched. Distinct AssetLibrary identities prevent GPU aliasing.
pub fn create_workload(config: StressConfig) -> Result<Engine, String> {
    create_workload_with(config, &HashMap::default())
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
/// dense scene's virtual meshes and the highlands' terrain materials).
pub fn create_workload_with(
    config: StressConfig,
    files: &HashMap<String, StressFile>,
) -> Result<Engine, String> {
    config.validate().map_err(|error| error.to_string())?;
    let mut engine = Engine::new(EngineConfig {
        frustum_culling: config.culling,
        ..EngineConfig::default()
    });
    // A workload with neither colliders nor queries leaves collision out.
    let collision_enabled = config.colliders > 0 || config.queries_per_tick > 0;
    if collision_enabled {
        engine
            .app
            .add_plugins(EnginePlugins)
            .insert_resource(stress_collision_budgets());
    } else {
        engine
            .app
            .add_plugins(EnginePlugins::without([Feature::Collision]));
    }
    engine.app.insert_resource(config.clone());
    let (atmosphere, lamps) = look(config.scene, "scene")?;
    engine.app.insert_resource(atmosphere);
    apply_sky(engine.app.world_mut(), config.scene, "scene");
    engine.app.init_resource::<FrameMetrics>();
    let content = config.scene.world();
    let world = engine.app.world_mut();
    let (moving_nodes, bounds) = content.populate(world, &config, files)?;
    content.environment(world);
    world.insert_resource(content.shadows(bounds));
    // Sun shafts and screen-space reflections
    // are opt-in engine effects; street lamps near the camera light the
    // city, brighter as the look darkens.
    world.insert_resource(stress_shafts());
    world.insert_resource(content.reflections());
    world.insert_resource(LocalLightSettings {
        distance: 220.0,
        intensity: lamps,
        ..LocalLightSettings::default()
    });
    if collision_enabled {
        refresh_collision_world(world).map_err(|error| error.to_string())?;
    }
    set_view(&mut engine, "overview")?;
    engine.app.add_plugins(FlyCameraPlugin);
    content.add_systems(&mut engine.app, moving_nodes);
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

fn query_workload(
    config: Res<StressConfig>,
    time: Res<FixedTime>,
    globals: Res<GlobalTransforms>,
    collision: Res<CollisionWorld>,
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
/// The collision budgets workloads measure under: the dense sensor
/// workload exceeds the pair budgets on purpose.
pub(crate) fn stress_collision_budgets() -> engine::collision::CollisionConfig {
    engine::collision::CollisionConfig {
        max_colliders: Some(10_000),
        max_candidate_pairs: Some(100_000),
        max_sensor_pairs: Some(16_384),
        max_query_candidates: Some(10_000),
        max_query_hits: Some(1024),
    }
}

pub fn stress_shafts() -> SunShafts {
    SunShafts {
        intensity: 1.5,
        ..SunShafts::default()
    }
}

/// Screen-space reflections as `scene` has them.
pub fn stress_reflections(scene: StressScene) -> Reflections {
    scene.world().reflections()
}

/// A lighting preset: its name, its name in menus, its atmosphere and its
/// street lamp brightness.
pub(crate) type Look = (&'static str, &'static str, fn() -> Atmosphere, f32);

/// The look `scene`: each scene's own (see [`look`]).
pub(crate) const SCENE_LOOK: (&str, &str) = ("scene", "Scene default");

/// The lighting presets; lamps are off by day. `sky` is the physically
/// based sky at a time of day, under a neutral grade.
pub(crate) const LOOKS: [Look; 5] = [
    ("golden", "Golden hour", Atmosphere::golden_hour, 0.35),
    ("dusk", "Dusk", Atmosphere::dusk, 0.8),
    ("night", "Night", Atmosphere::night, 1.0),
    ("day", "Day", Atmosphere::day, 0.0),
    ("sky", "Physical sky", physical_look, 0.5),
];

/// The rendering effects the page can switch off (`no-NAME`): each one's
/// name and what it is in menus.
pub(crate) const EFFECTS: [(&str, &str); 9] = [
    ("shadows", "shadows"),
    ("occlusion", "ambient occlusion"),
    ("bloom", "bloom"),
    ("lights", "local lights"),
    ("shafts", "sun shafts"),
    ("reflections", "reflections"),
    ("particles", "particles"),
    ("refraction", "refraction"),
    ("probes", "reflection probes"),
];

/// The camera views: each one's name and its name in menus.
pub(crate) const VIEWS: [(&str, &str); 3] = [
    ("overview", "Overview"),
    ("close", "Inside the field"),
    ("away", "Look away"),
];

/// The grade under the physically based sky: its light at
/// full strength, close to authored colors, a little contrast; the sky
/// gives the sun's and the sky's colors and the haze.
fn physical_look() -> Atmosphere {
    Atmosphere {
        ambient: 1.0,
        rim: 0.2,
        emission: 0.5,
        exposure: 1.0,
        saturation: 1.05,
        contrast: 1.1,
        vibrance: 0.15,
        shadow_tint: Vec3::new(0.95, 0.98, 1.06),
        highlight_tint: Vec3::new(1.04, 1.0, 0.96),
        ..Atmosphere::day()
    }
}

/// Puts the physically based sky and the time of day into
/// `world` for `scene`'s look `name`, or takes them out for the others.
pub fn apply_sky(world: &mut World, scene: StressScene, name: &str) {
    if resolve(scene, name) == "sky" {
        world.insert_resource(Sky::default());
        world.insert_resource(TimeOfDay {
            hours: 16.5,
            latitude: 42.0,
            north: 30.0,
            ..TimeOfDay::default()
        });
    } else {
        world.remove_resource::<Sky>();
        world.remove_resource::<TimeOfDay>();
    }
}

/// The lighting preset `name` and its street lamp brightness: `golden`,
/// `dusk`, `night` or `day`, or `scene` for the scene's own (night over the
/// tower field, golden hour over the city, day over dense geometry).
pub fn look(scene: StressScene, name: &str) -> Result<(Atmosphere, f32), String> {
    let name = resolve(scene, name);
    LOOKS
        .iter()
        .find(|(look, ..)| *look == name)
        .map(|&(_, _, atmosphere, lamps)| (atmosphere(), lamps))
        .ok_or_else(|| {
            let names: Vec<&str> = std::iter::once(SCENE_LOOK.0)
                .chain(LOOKS.iter().map(|(name, ..)| *name))
                .collect();
            format!("look must be one of {}", names.join(", "))
        })
}

/// Look `name` of `scene`, `scene` resolved to the scene's own.
fn resolve(scene: StressScene, name: &str) -> &str {
    if name != SCENE_LOOK.0 {
        return name;
    }
    scene.world().look()
}

pub fn set_view(engine: &mut Engine, view: &str) -> Result<(), String> {
    let config = engine.app.world().resource::<StressConfig>();
    let span = (config.renderables as f32).sqrt().ceil() * SPACING;
    let content = config.scene.world();
    let camera = match content.camera(engine.app.world(), config, view) {
        Some(camera) => camera,
        None => match view {
            "overview" => Camera {
                position: Vec3::new(span * 0.46, span * 0.62, span * 0.8),
                target: Vec3::ZERO,
                far: 5000.0,
                vertical_fov_radians: 55.0_f32.to_radians(),
                ..Camera::default()
            },
            "close" => Camera {
                position: Vec3::new(15.0, 15.0, 23.0),
                target: Vec3::new(0.0, 3.0, 0.0),
                far: 5000.0,
                ..Camera::default()
            },
            "away" => {
                let position = Vec3::new(0.0, span, span * 1.4);
                Camera {
                    position,
                    target: position + Vec3::Z,
                    far: 5000.0,
                    ..Camera::default()
                }
            }
            _ => {
                let names: Vec<&str> = VIEWS.iter().map(|(name, _)| *name).collect();
                return Err(format!("view must be one of {}", names.join(", ")));
            }
        },
    };
    let speed = content.camera_speed(span);
    engine.app.insert_resource(camera);
    engine
        .app
        .insert_resource(FlyCameraController::from_camera(&camera).with_speed(speed));
    Ok(())
}
