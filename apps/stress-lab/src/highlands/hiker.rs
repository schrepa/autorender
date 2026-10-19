//! The highlands' hiker: a figure in a red jacket walking a road there and
//! back, turning back when it arrives or, after a rest, when it finds no way.

use engine::prelude::*;

/// A hiker walking between two points: the ends (world x
/// and z), the one it walks toward, and the seconds it rests before
/// setting off again after finding no way.
#[derive(Component)]
pub(super) struct Hiker {
    ends: [[f32; 2]; 2],
    toward: usize,
    rest: f32,
}

/// Seconds a hiker that found no way waits before turning back: searches
/// are not free, so a walk that cannot be found is not searched again and
/// again.
const HIKER_REST: f32 = 10.0;

/// A hiker in a red jacket walking from `road`'s start (terrain-local, the
/// terrain's corner at `-half`) to the first of its points 350 meters away
/// and back.
pub(super) fn hiker(world: &mut World, road: &[[f32; 2]], half: f32) -> engine::Result<()> {
    let world_point = |[x, z]: [f32; 2]| [x - half, z - half];
    let start = world_point(road[0]);
    let far = road
        .iter()
        .map(|&point| world_point(point))
        .find(|point| Vec2::from(*point).distance(Vec2::from(start)) > 350.0)
        .unwrap_or(world_point(road[road.len() - 1]));
    let mut assets = world.resource_mut::<AssetLibrary>();
    let mesh = assets.insert_mesh(Mesh::cube())?;
    let material = assets.insert_material(Material::new([0.62, 0.1, 0.07, 1.0])?)?;
    // A body 1.75 meters tall, its origin at its middle.
    let offset = 0.875;
    world.spawn((
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
        InterpolateTransform,
        Hiker {
            ends: [start, far],
            toward: 1,
            rest: 0.0,
        },
    ));
    Ok(())
}

/// Turns each hiker back when it arrives, or after a rest
/// when it finds no way.
pub(super) fn hike(time: Res<FixedTime>, mut hikers: Query<(&mut NavAgent, &NavWalk, &mut Hiker)>) {
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
            _ => continue,
        }
        hiker.rest = 0.0;
        hiker.toward = 1 - hiker.toward;
        agent.destination = Some(hiker.ends[hiker.toward]);
    }
}
