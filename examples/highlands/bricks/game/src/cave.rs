use bevy_ecs::prelude::*;
use engine_core::{AssetLibrary, MaterialHandle, Mesh, Mesh3d, MeshHandle, ProvidedAssets, Spawner, Transform, math};
use engine_places::ground::TerrainTiles;
use engine_places::lights::{LightRegion, LocalLight};
use engine_places::probes::{Interior, ReflectionProbe};
use glam::{Quat, Vec2, Vec3};

use crate::{CAVE_WALL, HighlandsBrick, HighlandsSettings, WALL_ROCK, caves};

const CAVE_RADIUS: [f32; 2] = [6.0, 4.5];

const CAVE_MARGIN: f32 = 1.5;

struct CaveRock {
    boulders: Vec<MeshHandle>,
    material: MaterialHandle,
    wall: MaterialHandle,
}

impl CaveRock {
    fn of(provided: &ProvidedAssets) -> Option<Self> {
        let boulders: Vec<MeshHandle> = (0_u32..)
            .map_while(|seed| provided.mesh(&format!("boulder{seed}")))
            .collect();
        Some(Self {
            boulders,
            material: provided.material(WALL_ROCK)?,
            wall: provided.material(CAVE_WALL)?,
        })
    }
}

pub(crate) fn open(
    settings: Res<HighlandsSettings>,
    tiles: Res<TerrainTiles>,
    provided: Res<ProvidedAssets>,
    mut assets: ResMut<AssetLibrary>,
    mut opened: Local<bool>,
    mut spawner: Spawner<HighlandsBrick>,
) {
    if *opened {
        return;
    }
    let (Some([mouth_x, mouth_z, radius]), Some(rock)) = (settings.cave, CaveRock::of(&provided))
    else {
        return;
    };
    let mouth = Vec2::new(mouth_x, mouth_z);
    let Some((terrain, layers)) = tiles.terrains.iter().find_map(|terrain| {
        let layers = terrain.baked.as_ref().filter(|_| !terrain.surroundings)?;
        Some((terrain, layers))
    }) else {
        return;
    };
    if layers.hydrology().is_none() {
        return;
    }
    let height = |point: Vec2| {
        terrain.origin.y
            + layers.height_at(
                &terrain.pyramid,
                Vec2::new(point.x - terrain.origin.x, point.y - terrain.origin.z),
            )
    };
    let gradient = Vec2::new(
        height(mouth + Vec2::X * 20.0) - height(mouth - Vec2::X * 20.0),
        height(mouth + Vec2::Y * 20.0) - height(mouth - Vec2::Y * 20.0),
    ) / 40.0;
    let slope = gradient.length();
    let uphill = if slope > 0.02 {
        gradient / slope
    } else {
        Vec2::X
    };
    let side = Vec2::new(-uphill.y, uphill.x);
    let start = mouth - uphill * (radius + 1.0);
    let center = height(start) - 0.5 + 0.55 * CAVE_RADIUS[1] * 1.15;
    let fall = math::max(0.75 - slope, 0.25);
    let at = |along: f32, across: f32, drop: f32| {
        let point = mouth + uphill * along + side * across;
        Vec3::new(point.x, center - drop, point.y)
    };
    let points = [
        at(-(radius + 1.0), 0.0, 0.0),
        at(12.0, 0.0, fall * 18.0),
        at(28.0, 6.0, fall * 18.0 + fall * 8.0),
        at(44.0, -5.0, fall * 18.0 + fall * 16.0),
        at(58.0, 0.0, fall * 18.0 + fall * 22.0),
    ];
    let facing = Quat::from_rotation_y((-uphill.x).atan2(-uphill.y));
    let framing: Vec<(Transform, MeshHandle)> = [
        (4.0, 0.0, 1.5, 10.0, 0),
        (-1.0, 10.0, -1.5, 7.0, 1),
        (-1.0, -10.0, -1.5, 7.0, 2),
    ]
    .into_iter()
    .filter_map(
        |(along, across, rise, scale, index): (f32, f32, f32, f32, usize)| {
            let point = mouth + uphill * along + side * across;
            let mesh = *rock.boulders.get(index % rock.boulders.len().max(1))?;
            Some((
                Transform {
                    translation: Vec3::new(point.x, height(point) + rise, point.y),
                    rotation: facing,
                    scale: Vec3::splat(scale),
                },
                mesh,
            ))
        },
    )
    .collect();
    let inward = Vec3::new(uphill.x, 0.0, uphill.y);
    *opened = true;
    let built = match caves::cave(&points, CAVE_RADIUS, 79) {
        Ok(built) => built,
        Err(error) => {
            crate::report_failure("highlands_cave_failed", &error);
            return;
        }
    };
    let (low, high) = cave_interior(&built, &points, inward);
    let mesh = match assets.insert_mesh(built) {
        Ok(mesh) => mesh,
        Err(error) => {
            crate::report_failure("highlands_cave_failed", &error);
            return;
        }
    };
    spawn_cave(&mut spawner, mesh, (&points, inward, low, high), (&rock, framing));
}

fn spawn_cave(
    spawner: &mut Spawner<HighlandsBrick>,
    mesh: MeshHandle,
    (points, inward, low, high): (&[Vec3; 5], Vec3, Vec3, Vec3),
    (rock, framing): (&CaveRock, Vec<(Transform, MeshHandle)>),
) {
    spawner.spawn((
        Transform {
            translation: points[0],
            ..Transform::default()
        },
        Mesh3d {
            mesh,
            material: rock.wall,
        },
    ));
    let chamber = points[4] - inward * 6.0 + Vec3::Y * 2.0;
    spawner.spawn((
        Transform {
            translation: chamber,
            ..Transform::default()
        },
        LocalLight::point(Vec3::new(1.0, 0.62, 0.3), 40.0, 26.0).with_region(LightRegion::Interior),
    ));
    spawner.spawn((
        Transform {
            translation: (low + high) * 0.5,
            ..Transform::default()
        },
        ReflectionProbe::new((high - low) * 0.5)
            .with_blend(CAVE_MARGIN)
            .with_interior(Interior { sky: 1.0, sun: 1.0 }),
    ));
    for (transform, mesh) in framing {
        spawner.spawn((
            transform,
            Mesh3d {
                mesh,
                material: rock.material,
            },
        ));
    }
}

fn cave_interior(mesh: &Mesh, points: &[Vec3], inward: Vec3) -> (Vec3, Vec3) {
    let bend = (points[1] - points[0]).dot(inward);
    let (low, high) = mesh
        .vertices()
        .iter()
        .map(|vertex| points[0] + Vec3::from(vertex.position))
        .filter(|&point| (point - points[0]).dot(inward) >= bend)
        .fold(
            (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
            |(low, high), point| (low.min(point), high.max(point)),
        );
    (
        low - Vec3::splat(CAVE_MARGIN),
        high + Vec3::splat(CAVE_MARGIN),
    )
}
