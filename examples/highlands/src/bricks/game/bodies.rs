use std::cmp::Reverse;

use bevy_ecs::prelude::*;
use engine_core::{
    AssetLibrary, Camera, Error, Material, Mesh, Mesh3d, MeshHandle, Spawner, Transform, math,
};
use engine_heightfield::RIVER_STEP;
use engine_places::ground::TerrainTiles;
use engine_places::physics::{Collider, RigidBody};
use engine_places::streaming::StreamingFocus;
use engine_places::water::{Hydrology, WaterBody, WaterSample};
use glam::{Quat, Vec2, Vec3};

use crate::bricks::game::HighlandsBrick;
use crate::bricks::game::content::SEED;
use crate::bricks::game::random::Rng;

const FARTHEST: f32 = 2500.0;
const NEAR: f32 = 800.0;
const STEEP: f32 = 0.004;
const DEEP: f32 = 0.45;
const WIDE: f32 = 2.0;
const UPSTREAM: usize = 24;
const DAM_LOGS: usize = 4;
const DAM_LOG: f32 = 0.5;
const CREST: f32 = 1.0;
const BOULDERS: u32 = 3;

#[derive(Resource, Default)]
pub(crate) struct Bodies {
    pub(crate) bank: Option<(MeshHandle, u32)>,
    dammed: bool,
}

struct Site {
    origin: Vec3,
    points: Vec<[f32; 2]>,
    half: Vec<f32>,
    bed: Vec<f32>,
    point: usize,
}

fn run(points: &[[f32; 2]], index: usize) -> Vec2 {
    let next = points[(index + 1).min(points.len() - 1)];
    let previous = points[index.saturating_sub(1)];
    (Vec2::from(next) - Vec2::from(previous)).normalize_or(Vec2::X)
}

fn site(hydrology: &Hydrology, tiles: &TerrainTiles, eye: Vec3) -> Option<Site> {
    let mut best: Option<(f32, Site)> = None;
    for (terrain, bake) in &hydrology.bakes {
        let Some(origin) = tiles
            .terrains
            .iter()
            .find(|entry| entry.terrain == *terrain && !entry.surroundings)
            .map(|entry| entry.origin)
        else {
            continue;
        };
        for course in bake.rivers().courses() {
            let points = course.points();
            let count = points.len();
            if count < 2 * UPSTREAM {
                continue;
            }
            for index in UPSTREAM..count - UPSTREAM / 2 {
                let at = Vec2::new(points[index][0] + origin.x, points[index][1] + origin.z);
                let distance = at.distance(Vec2::new(eye.x, eye.z));
                let running = (index - UPSTREAM..=index + 2).all(|point| !course.falls_at(point));
                if !running
                    || distance > FARTHEST
                    || course.channel(index) < DEEP
                    || course.half(index) < WIDE
                {
                    continue;
                }
                let run = UPSTREAM as f32 * RIVER_STEP;
                let slope = (course.bed(index - UPSTREAM) - course.bed(index)) / run;
                if slope < STEEP {
                    continue;
                }
                let held = 2.0 * course.half(index) * (CREST / slope) * CREST * 0.5;
                let filling = held / math::max(course.discharge(index), 1.0e-3);
                let pull = 1.0 / (filling * (1.0 + (distance / NEAR) * (distance / NEAR)));
                if best.as_ref().is_some_and(|(best, _)| *best >= pull) {
                    continue;
                }
                best = Some((
                    pull,
                    Site {
                        origin,
                        points: points.to_vec(),
                        half: (0..count).map(|point| course.half(point)).collect(),
                        bed: (0..count).map(|point| course.bed(point)).collect(),
                        point: index,
                    },
                ));
            }
        }
    }
    best.map(|(_, site)| site)
}

fn water_in(waters: &Query<&WaterBody>, point: Vec3) -> Option<WaterSample> {
    let mut held: Vec<&WaterBody> = waters.iter().collect();
    held.sort_by_key(|body| Reverse(body.precedence));
    held.iter().find_map(|body| body.shape.water(point))
}

pub(crate) fn place(
    (camera, hydrology, tiles): (Res<Camera>, Res<Hydrology>, Res<TerrainTiles>),
    waters: Query<&WaterBody>,
    mut bodies: ResMut<Bodies>,
    mut assets: ResMut<AssetLibrary>,
    mut spawner: Spawner<HighlandsBrick>,
) {
    if bodies.dammed {
        return;
    }
    let Some((boulder, count)) = bodies.bank else {
        return;
    };
    let Some(site) = site(&hydrology, &tiles, camera.position) else {
        return;
    };
    let [x, z] = site.points[site.point];
    let bed = site.bed[site.point];
    let at = site.origin + Vec3::new(x, bed + 0.05, z);
    let Some(sample) = water_in(&waters, at) else {
        return;
    };
    let water = sample.surface - site.origin.y;
    bodies.dammed = true;
    if let Err(error) = build(
        (&mut spawner, &mut *assets),
        &site,
        water,
        (count, SEED),
        boulder,
    ) {
        crate::bricks::game::report_failure("highlands_bodies_failed", &error);
    }
}

fn build(
    (spawner, assets): (&mut Spawner<HighlandsBrick>, &mut AssetLibrary),
    site: &Site,
    water: f32,
    (count, seed): (u32, u32),
    boulder: MeshHandle,
) -> Result<(), Error> {
    let cube = assets.insert_mesh(Mesh::cube())?;
    let bark = assets.insert_material(Material::new([0.2, 0.13, 0.08, 1.0])?)?;
    let pale = assets.insert_material(Material::new([0.42, 0.3, 0.19, 1.0])?)?;
    let stone = assets.insert_material(Material::new([0.38, 0.37, 0.35, 1.0])?)?;
    let origin = site.origin;
    let at = site.point;
    let world_point = |point: [f32; 2], height: f32| {
        Vec3::new(point[0] + origin.x, height + origin.y, point[1] + origin.z)
    };
    let along = run(&site.points, at);
    let side = Vec2::new(-along.y, along.x);
    let water_at = |point: usize| water + site.bed[point] - site.bed[at];
    let bed = site.bed[at];
    let span = 2.0 * site.half[at] + 4.0;
    let yaw = Quat::from_rotation_y((-side.y).atan2(side.x));
    for log in 0..DAM_LOGS {
        let height = bed + DAM_LOG * (log as f32 + 0.5);
        let top = water + CREST;
        let height = math::min(height, top - DAM_LOG * 0.5);
        spawner.spawn((
            Transform {
                translation: world_point(site.points[at], height),
                rotation: yaw,
                scale: Vec3::new(span, DAM_LOG, DAM_LOG),
            },
            Mesh3d {
                mesh: cube,
                material: if log % 2 == 0 { bark } else { pale },
            },
            Collider::cuboid(Vec3::splat(0.5))?,
        ));
    }
    spawner.spawn((
        Transform::from_translation(world_point(site.points[at], water)),
        StreamingFocus,
    ));
    let mut rng = Rng::new(seed, 0x10c5, at as i32);
    for log in 0..count {
        let point = at - 4 - (log as usize * (UPSTREAM - 6)) / count.max(1) as usize;
        let across = rng.range(-0.6, 0.6) * site.half[point];
        let flat = Vec2::from(site.points[point]) + side * across;
        let length = rng.range(2.8, 4.4);
        spawner.spawn((
            Transform {
                translation: world_point(flat.to_array(), water_at(point) + 0.8),
                rotation: Quat::from_rotation_y(rng.range(0.0, std::f32::consts::TAU)),
                scale: Vec3::new(length, 0.42, 0.42),
            },
            Mesh3d {
                mesh: cube,
                material: if log % 2 == 0 { pale } else { bark },
            },
            Collider::cuboid(Vec3::splat(0.5))?,
            RigidBody {
                density: rng.range(560.0, 720.0),
                ..RigidBody::default()
            },
        ));
    }
    for stone_index in 0..BOULDERS {
        let point = at - 6 - stone_index as usize * 5;
        let bank = side * (site.half[point] + 2.5) * if stone_index % 2 == 0 { 1.0 } else { -1.0 };
        let flat = Vec2::from(site.points[point]) + bank;
        let size = rng.range(0.9, 1.4);
        spawner.spawn((
            Transform {
                translation: world_point(flat.to_array(), water_at(point) + 3.0),
                rotation: Quat::from_rotation_y(rng.range(0.0, std::f32::consts::TAU)),
                scale: Vec3::splat(size),
            },
            Mesh3d {
                mesh: boulder,
                material: stone,
            },
            Collider::sphere(0.45)?,
            RigidBody {
                density: 2600.0,
                friction: 0.8,
                ..RigidBody::default()
            },
        ));
    }
    Ok(())
}
