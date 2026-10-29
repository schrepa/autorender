use engine::prelude::*;
use engine::{Error, ErrorCode, Expected};

const EYE_HEIGHT: f32 = 1.7;

const SPEED: f32 = 160.0;

#[derive(Resource, Clone, Copy)]
struct StartView(Camera);

pub(crate) fn add_systems(app: &mut App) {
    app.add_systems(Presentation, aim_when_bodies_come);
    app.add_systems(Presentation, hold_standing_eye.after(CameraSystems));
}

pub(crate) fn start(world: &mut World) {
    let camera = *world.resource::<Camera>();
    world.insert_resource(StartView(camera));
    look_from(world, camera);
}

pub(crate) fn reset(world: &mut World) {
    world.remove_resource::<StandingEye>();
    world.remove_resource::<AwaitingBodies>();
    if let Some(StartView(camera)) = world.get_resource::<StartView>().copied() {
        look_from(world, camera);
    }
}

fn look_from(world: &mut World, camera: Camera) {
    world.insert_resource(camera);
    world.insert_resource(FlyCameraController::from_camera(&camera).with_speed(SPEED));
}

pub(crate) fn stand(world: &mut World, place: &str) -> engine::Result<()> {
    let standing = StandingEye::new(world, place)?;
    world.insert_resource(standing);
    look_from(world, standing.camera());
    Ok(())
}

pub(crate) fn watch_bodies(world: &mut World, on: bool) {
    if !on {
        world.remove_resource::<AwaitingBodies>();
        return;
    }
    match bodies_view(world) {
        Some(camera) => {
            world.remove_resource::<AwaitingBodies>();
            world.remove_resource::<StandingEye>();
            look_from(world, camera);
        }
        None => world.insert_resource(AwaitingBodies),
    }
}

#[derive(Resource, Clone, Copy)]
struct StandingEye {
    x: f32,
    z: f32,
    direction: Vec3,
    eye: Vec3,
}

impl StandingEye {
    fn new(world: &World, place: &str) -> engine::Result<Self> {
        let malformed = |expected| {
            Error::new(ErrorCode::InvalidValue, "$.stand")
                .expected(expected)
                .got(place)
        };
        let numbers: Vec<f32> = place
            .split(',')
            .map(|number| number.trim().parse::<f32>())
            .collect::<Result<_, _>>()
            .map_err(|_| malformed(Expected::Number))?;
        let [x, z, yaw, pitch] = numbers[..] else {
            return Err(malformed(Expected::Array));
        };
        if numbers.iter().any(|number| !number.is_finite()) {
            return Err(malformed(Expected::Finite));
        }
        if pitch.abs() >= 90.0 {
            return Err(Error::new(ErrorCode::OutOfRange, "$.stand")
                .range(-90.0, 90.0)
                .got(pitch));
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

#[derive(Resource)]
struct AwaitingBodies;

fn aim_when_bodies_come(world: &mut World) {
    if world.contains_resource::<AwaitingBodies>() && bodies_view(world).is_some() {
        watch_bodies(world, true);
    }
}

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
