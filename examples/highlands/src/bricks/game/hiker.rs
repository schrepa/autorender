use bevy_ecs::prelude::*;
use engine_core::{
    AssetLibrary, Error, FixedTime, InterpolateTransform, Material, Mesh, Mesh3d, Spawner,
    Transform,
};
use engine_places::ground::Grounded;
use engine_places::navigation::{NavAgent, NavStatus, NavWalk};
use glam::{Vec2, Vec3};

use crate::bricks::game::HighlandsBrick;

#[derive(Component)]
pub(crate) struct Hiker {
    ends: [[f32; 2]; 2],
    toward: usize,
    rest: f32,
}

const HIKER_REST: f32 = 10.0;

pub(crate) fn hiker(
    (assets, spawner): (&mut AssetLibrary, &mut Spawner<HighlandsBrick>),
    road: &[[f32; 2]],
    origin: Vec2,
) -> Result<(), Error> {
    let world_point = |[x, z]: [f32; 2]| [x + origin.x, z + origin.y];
    let start = world_point(road[0]);
    let far = road
        .iter()
        .map(|&point| world_point(point))
        .find(|point| Vec2::from(*point).distance(Vec2::from(start)) > 350.0)
        .unwrap_or(world_point(road[road.len() - 1]));
    let mesh = assets.insert_mesh(Mesh::cube())?;
    let material = assets.insert_material(Material::new([0.62, 0.1, 0.07, 1.0])?)?;
    let offset = 0.875;
    spawner.spawn((
        Transform {
            translation: Vec3::new(start[0], 0.0, start[1]),
            scale: Vec3::new(0.45, 1.75, 0.3),
            ..Transform::default()
        },
        Mesh3d { mesh, material },
        Grounded { offset },
        NavAgent {
            destination: Some(far),
            speed: 1.6,
            offset,
            ..NavAgent::default()
        },
        InterpolateTransform::default(),
        Hiker {
            ends: [start, far],
            toward: 1,
            rest: 0.0,
        },
    ));
    Ok(())
}

pub(crate) fn hike(time: Res<FixedTime>, mut hikers: Query<(&mut NavAgent, &NavWalk, &mut Hiker)>) {
    for (mut agent, walk, mut hiker) in &mut hikers {
        if walk.destination != agent.destination {
            continue;
        }
        match walk.status {
            NavStatus::Unreachable if hiker.rest < HIKER_REST => {
                hiker.rest += time.delta_seconds;
                continue;
            }
            NavStatus::Arrived | NavStatus::Unreachable => {}
            NavStatus::Idle | NavStatus::Searching | NavStatus::Walking => continue,
        }
        hiker.rest = 0.0;
        hiker.toward = 1 - hiker.toward;
        agent.destination = Some(hiker.ends[hiker.toward]);
    }
}
