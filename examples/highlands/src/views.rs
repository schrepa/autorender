use engine::prelude::*;
use engine::{Error, ErrorCode, Expected};

use crate::field::hamlet_radius;
use crate::land::Plateau;

const VIEWS: [&str; 4] = ["overview", "close", "away", "bodies"];

const STAND: &str = "stand:";

const EYE_HEIGHT: f32 = 1.7;

const SPEED: f32 = 160.0;

fn view_names() -> impl Iterator<Item = &'static str> {
    VIEWS
        .into_iter()
        .chain(WATER_VIEWS.iter().map(|(name, ..)| *name))
}

pub(crate) fn names() -> String {
    let names: Vec<&str> = view_names().collect();
    format!("{}|{STAND}x,z,yaw,pitch", names.join("|"))
}

pub(crate) fn add_systems(app: &mut App) {
    app.add_systems(Presentation, aim_when_bodies_come);
    app.add_systems(Presentation, hold_standing_eye.after(CameraSystems));
}

pub(crate) fn set_view(world: &mut World, view: &str) -> engine::Result<()> {
    aim(world, view)
}

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
    let camera = standing
        .as_ref()
        .map(StandingEye::camera)
        .or_else(|| camera(world, view))
        .or(bodies)
        .ok_or_else(|| {
            Error::new(ErrorCode::InvalidValue, "$.view")
                .got(view)
                .variants(view_names())
        })?;
    if let Some(standing) = standing {
        world.insert_resource(standing);
    } else {
        world.remove_resource::<StandingEye>();
    }
    world.insert_resource(camera);
    world.insert_resource(FlyCameraController::from_camera(&camera).with_speed(SPEED));
    Ok(())
}

fn camera(world: &World, view: &str) -> Option<Camera> {
    match view {
        "overview" => {
            let (position, target) = overview(world);
            Some(Camera {
                position,
                target,
                near: 0.5,
                far: 20_000.0,
                vertical_fov_radians: 55.0_f32.to_radians(),
            })
        }
        "close" => {
            let (position, target) = close_view(world);
            Some(Camera {
                position,
                target,
                near: 0.2,
                far: 20_000.0,
                ..Camera::default()
            })
        }
        "away" => {
            let (position, target) = away_view(world);
            Some(Camera {
                position,
                target,
                near: 0.5,
                far: 20_000.0,
                ..Camera::default()
            })
        }
        _ => WATER_VIEWS
            .iter()
            .find(|(name, ..)| *name == view)
            .map(|&(_, position, yaw, pitch)| water_view(position, yaw, pitch)),
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
            Error::new(ErrorCode::InvalidValue, "$.view")
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
            return Err(Error::new(ErrorCode::OutOfRange, "$.view")
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
        aim(world, "bodies").expect("view_known");
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

fn ground(world: &World, x: f32, z: f32) -> f32 {
    let mut terrains = world.try_query::<&Terrain>().expect("terrain_registered");
    terrains
        .iter(world)
        .next()
        .map(|terrain| {
            terrain.origin.y
                + terrain.layers.height_at(
                    &terrain.pyramid,
                    Vec2::new(x - terrain.origin.x, z - terrain.origin.z),
                )
        })
        .unwrap_or_else(|| world.resource::<Plateau>().0)
}

fn directions() -> impl Iterator<Item = Vec2> {
    (0..8).map(|step| {
        let (sin, cos) = math::sin_cos(step as f32 * std::f32::consts::FRAC_PI_4);
        Vec2::new(cos, sin)
    })
}

fn lowest(world: &World, reach: f32) -> Vec2 {
    directions()
        .map(|direction| {
            let point = direction * reach;
            (ground(world, point.x, point.y), direction)
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map_or(Vec2::Y, |(_, direction)| direction)
}

fn clear_view(world: &World, target: Vec3, direction: Vec2, distance: f32, rise: f32) -> Vec3 {
    let mut height = distance * rise;
    for step in 1..=24 {
        let along = step as f32 / 24.0;
        let point = Vec2::new(target.x, target.z) + direction * distance * along;
        let needed = (ground(world, point.x, point.y) + 30.0 - target.y) / along;
        height = math::max(height, needed);
    }
    let point = Vec2::new(target.x, target.z) + direction * distance;
    Vec3::new(point.x, target.y + height, point.y)
}

fn overview(world: &World) -> (Vec3, Vec3) {
    let target = Vec3::new(0.0, world.resource::<Plateau>().0, 0.0);
    let direction = lowest(world, 1200.0);
    let distance = hamlet_radius() + 700.0;
    (clear_view(world, target, direction, distance, 0.45), target)
}

fn close_view(world: &World) -> (Vec3, Vec3) {
    let direction = lowest(world, 1500.0);
    let edge = direction * hamlet_radius() * 0.85;
    let far = direction * 1500.0;
    (
        Vec3::new(edge.x, ground(world, edge.x, edge.y) + 10.0, edge.y),
        Vec3::new(far.x, ground(world, far.x, far.y) + 60.0, far.y),
    )
}

const WATER_VIEWS: [(&str, [f32; 3], f32, f32); 5] = [
    ("torrent", [-1584.0, 675.0, -2973.0], 0.0, -34.0),
    ("rapids", [-3008.0, 228.0, -168.0], 0.0, -30.0),
    ("meadow_river", [-266.0, 671.0, -2986.0], 0.0, -32.0),
    ("band", [-300.0, 730.0, -3180.0], -10.0, -38.0),
    ("close_stream", [1474.5, 605.9, 743.3], 112.0, -18.0),
];

fn water_view(position: [f32; 3], yaw: f32, pitch: f32) -> Camera {
    let (yaw_sin, yaw_cos) = math::sin_cos(yaw.to_radians());
    let (pitch_sin, pitch_cos) = math::sin_cos(pitch.to_radians());
    let position = Vec3::from(position);
    Camera {
        position,
        target: position + Vec3::new(yaw_sin * pitch_cos, pitch_sin, -yaw_cos * pitch_cos),
        near: 0.2,
        far: 20_000.0,
        vertical_fov_radians: 60.0_f32.to_radians(),
    }
}

fn away_view(world: &World) -> (Vec3, Vec3) {
    let height = ground(world, 0.0, 0.0) + 700.0;
    (
        Vec3::new(0.0, height, 600.0),
        Vec3::new(0.0, height - 500.0, -3000.0),
    )
}
