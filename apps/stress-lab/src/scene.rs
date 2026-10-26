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
pub fn create_workload(config: StressConfig) -> engine::Result<Engine> {
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
) -> engine::Result<Engine> {
    config.validate()?;
    let mut engine = Engine::new(EngineConfig {
        frustum_culling: config.culling,
        ..EngineConfig::default()
    });
    // A workload with neither colliders nor queries leaves collision out,
    // and physics, whose bodies meet collision's solids.
    let collision_enabled =
        config.colliders > 0 || config.queries_per_tick > 0 || config.bodies > 0;
    if collision_enabled {
        engine
            .app
            .add_plugins(EnginePlugins)
            .insert_resource(stress_collision_budgets());
    } else {
        engine.app.add_plugins(EnginePlugins::without([
            Feature::Collision,
            Feature::Physics,
        ]));
    }
    engine.app.insert_resource(config.clone());
    let (atmosphere, lamps) = look(config.scene, "scene")?;
    engine.app.insert_resource(atmosphere);
    apply_sky(engine.app.world_mut(), config.scene, "scene");
    engine.app.init_resource::<FrameMetrics>();
    let content = config.scene.world();
    let world = engine.app.world_mut();
    let moving_nodes = content.populate(world, &config, files)?;
    content.environment(world);
    // Screen-space reflections are an opt-in engine effect; street lamps
    // near the camera light the city, brighter as the look darkens.
    world.insert_resource(content.reflections());
    world.insert_resource(LocalLightSettings {
        distance: 220.0,
        intensity: lamps,
        ..LocalLightSettings::default()
    });
    if collision_enabled {
        refresh_collision_world(world)?;
    }
    set_view(&mut engine, "overview")?;
    engine.app.add_plugins(FlyCameraPlugin);
    engine.app.add_systems(Presentation, aim_when_bodies_come);
    engine
        .app
        .add_systems(Presentation, hold_standing_eye.after(CameraSystems));
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
pub(crate) const EFFECTS: [(&str, &str); 8] = [
    ("shadows", "shadows"),
    ("occlusion", "ambient occlusion"),
    ("bloom", "bloom"),
    ("lights", "local lights"),
    ("reflections", "reflections"),
    ("particles", "particles"),
    ("refraction", "refraction"),
    ("probes", "reflection probes"),
];

/// The camera views: each one's name and its name in menus.
pub(crate) const VIEWS: [(&str, &str); 4] = [
    ("overview", "Overview"),
    ("close", "Inside the field"),
    ("away", "Look away"),
    ("bodies", "The bodies"),
];

/// The standing eye view's name before its numbers: `stand:x,z,yaw,pitch`
/// stands the eye at world `x` and `z` (meters), [`EYE_HEIGHT`] above the
/// ground under it, looking along `yaw` (degrees from -z toward +x) and
/// `pitch` (degrees from level, negative down), as the highlands' water
/// views give them.
const STAND: &str = "stand:";

/// A standing adult's eye height above the ground, in meters: the standing
/// eye view's eye stands this high over the ground under it.
const EYE_HEIGHT: f32 = 1.7;

/// The standing eye view, held over the ground each frame as the ground
/// under it streams in and settles: the eye's world x and z, the unit
/// direction it looks along, and where the hold last put it. It lets go
/// once the fly camera's input moves or turns the camera, or the camera
/// stands anywhere else.
#[derive(Resource, Clone, Copy)]
struct StandingEye {
    x: f32,
    z: f32,
    direction: Vec3,
    eye: Vec3,
}

impl StandingEye {
    /// The standing eye view `place` names (`x,z,yaw,pitch`, after
    /// [`STAND`]) in `world`: its eye over the ground under it where that
    /// has loaded, else over the world's level, y = 0.
    fn new(world: &World, place: &str) -> engine::Result<Self> {
        let malformed = || {
            engine::Error::game_rule(format!(
                "view {STAND}x,z,yaw,pitch takes x and z in world meters and yaw and pitch in degrees, pitch less than 90 from level, not {STAND}{place}"
            ))
        };
        let numbers: Vec<f32> = place
            .split(',')
            .map(|number| number.trim().parse::<f32>())
            .collect::<Result<_, _>>()
            .map_err(|_| malformed())?;
        let [x, z, yaw, pitch] = numbers[..] else {
            return Err(malformed());
        };
        if numbers.iter().any(|number| !number.is_finite()) || pitch.abs() >= 90.0 {
            return Err(malformed());
        }
        let (yaw_sin, yaw_cos) = math::sin_cos(yaw.to_radians());
        let (pitch_sin, pitch_cos) = math::sin_cos(pitch.to_radians());
        let ground = Ground::sample_in(world, x, z).map_or(0.0, |ground| ground.height);
        Ok(Self {
            x,
            z,
            direction: Vec3::new(yaw_sin * pitch_cos, pitch_sin, -yaw_cos * pitch_cos),
            eye: Vec3::new(x, ground + EYE_HEIGHT, z),
        })
    }

    /// Its camera: at the eye, its field of view 60 degrees high, its near
    /// plane a fifth of a meter out and its far plane 20 km out, as the
    /// water views' are.
    fn camera(&self) -> Camera {
        Camera {
            position: self.eye,
            target: self.eye + self.direction,
            near: 0.2,
            far: 20_000.0,
            vertical_fov_radians: 60.0_f32.to_radians(),
        }
    }
}

/// Holds the standing eye over the ground under it as the ground streams
/// in and settles, its target moving with it, until the fly camera's input
/// moves or turns the camera or the camera stands anywhere else.
fn hold_standing_eye(world: &mut World) {
    let Some(standing) = world.get_resource::<StandingEye>().copied() else {
        return;
    };
    let input = world
        .get_resource::<InputState>()
        .copied()
        .unwrap_or_default();
    let steered = [
        input.move_forward,
        input.move_right,
        input.move_up,
        input.look_yaw,
        input.look_pitch,
    ]
    .into_iter()
    .any(|amount| amount != 0.0);
    if steered || world.resource::<Camera>().position != standing.eye {
        world.remove_resource::<StandingEye>();
        return;
    }
    let Some(ground) = Ground::sample_in(world, standing.x, standing.z) else {
        return;
    };
    let held = StandingEye {
        eye: Vec3::new(standing.x, ground.height + EYE_HEIGHT, standing.z),
        ..standing
    };
    world.insert_resource(held.camera());
    world.insert_resource(held);
}

/// The bodies' view was asked for before there were any: the camera turns
/// to them when they come.
#[derive(Resource)]
struct AwaitingBodies;

/// Turns the camera to the bodies once they come, their view asked for
/// before there were any.
fn aim_when_bodies_come(world: &mut World) {
    if world.contains_resource::<AwaitingBodies>() && bodies_view(world).is_some() {
        aim(world, "bodies").expect("known view");
    }
}

/// Looking down at the world's rigid bodies from beside them, where most
/// stand: `None` without any.
fn bodies_view(world: &mut World) -> Option<Camera> {
    let mut bodies = world.query_filtered::<&Transform, With<RigidBody>>();
    let at: Vec<Vec3> = bodies
        .iter(world)
        .map(|transform| transform.translation)
        .collect();
    if at.is_empty() {
        return None;
    }
    let middle = at.iter().copied().sum::<Vec3>() / at.len() as f32;
    let reach = at
        .iter()
        .map(|point| point.distance(middle))
        .fold(4.0, engine::math::max);
    Some(Camera {
        position: middle + Vec3::new(1.0, 0.8, 1.2) * (reach * 1.6 + 6.0),
        target: middle,
        far: 20_000.0,
        ..Camera::default()
    })
}

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
pub fn look(scene: StressScene, name: &str) -> engine::Result<(Atmosphere, f32)> {
    let name = resolve(scene, name);
    LOOKS
        .iter()
        .find(|(look, ..)| *look == name)
        .map(|&(_, _, atmosphere, lamps)| (atmosphere(), lamps))
        .ok_or_else(|| {
            let names: Vec<&str> = std::iter::once(SCENE_LOOK.0)
                .chain(LOOKS.iter().map(|(name, ..)| *name))
                .collect();
            engine::Error::game_rule(format!("look must be one of {}", names.join(", ")))
        })
}

/// Look `name` of `scene`, `scene` resolved to the scene's own.
fn resolve(scene: StressScene, name: &str) -> &str {
    if name != SCENE_LOOK.0 {
        return name;
    }
    scene.world().look()
}

pub fn set_view(engine: &mut Engine, view: &str) -> engine::Result<()> {
    aim(engine.app.world_mut(), view)
}

/// Puts `world`'s camera at view `view`. The bodies' view, asked for before
/// there are any, looks over the world until they come; the standing eye
/// view ([`STAND`]) holds its eye over the ground as it loads
/// ([`hold_standing_eye`]).
fn aim(world: &mut World, view: &str) -> engine::Result<()> {
    let bodies = (view == "bodies").then(|| bodies_view(world)).flatten();
    let awaiting = view == "bodies" && bodies.is_none();
    if awaiting {
        world.insert_resource(AwaitingBodies);
    } else {
        world.remove_resource::<AwaitingBodies>();
    }
    let view = if awaiting { "overview" } else { view };
    let standing = view
        .strip_prefix(STAND)
        .map(|place| StandingEye::new(world, place))
        .transpose()?;
    let config = world.resource::<StressConfig>();
    let span = (config.renderables as f32).sqrt().ceil() * SPACING;
    let content = config.scene.world();
    let camera = match standing
        .as_ref()
        .map(StandingEye::camera)
        .or_else(|| content.camera(world, config, view))
        .or(bodies)
    {
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
                return Err(engine::Error::game_rule(format!(
                    "view must be one of {}, or {STAND}x,z,yaw,pitch",
                    names.join(", ")
                )));
            }
        },
    };
    let speed = content.camera_speed(span);
    if let Some(standing) = standing {
        world.insert_resource(standing);
    } else {
        world.remove_resource::<StandingEye>();
    }
    world.insert_resource(camera);
    world.insert_resource(FlyCameraController::from_camera(&camera).with_speed(speed));
    Ok(())
}
