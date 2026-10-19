//! Bodies in the highlands' rivers, once the terrain's hydrology bake is
//! done: a dam of stacked logs across a river near the camera, where its
//! pool fills soonest, logs dropped upstream that its current carries down
//! against the dam, and boulders on its bank that roll in. The dam holds
//! the river back: a pool fills behind it while the river below runs dry,
//! until the water pours over it, and logs floating up on the pool may
//! spill over. As many drifting logs as the workload's `bodies`; the
//! ground about the dam stays loaded while they drift. Rivers' courses say
//! where the dam fits; the water the terrain's tiles hold says where its
//! water stands.

use engine::prelude::*;

use super::rocks;
use crate::kit::StressConfig;
use crate::kit::random::Rng;

/// The rivers have their bodies.
#[derive(Resource)]
struct Dammed;

/// Meters across the ground from under the camera the dam stands within,
/// and over which a site's pull halves.
const FARTHEST: f32 = 2500.0;
const NEAR: f32 = 800.0;
/// The gentlest slope of the river's water above the dam: on gentler
/// reaches its pool spreads too far to fill while anyone watches.
const STEEP: f32 = 0.004;
/// The shallowest and narrowest river channel the dam spans (meters deep,
/// meters across half its width).
const DEEP: f32 = 0.45;
const WIDE: f32 = 2.0;
/// River points above the dam the logs drop over, and the dam's own
/// points from either end of its river.
const UPSTREAM: usize = 24;
/// Logs stacked in the dam, each this thick (meters), and how far their
/// top stands above the river's water.
const DAM_LOGS: usize = 4;
const DAM_LOG: f32 = 0.5;
const CREST: f32 = 1.0;
/// Boulders on the bank.
const BOULDERS: u32 = 3;

/// A river point fit for the dam: its terrain's origin, its river's points
/// and at each half its width and its bed, and the point.
struct Site {
    origin: Vec3,
    points: Vec<[f32; 2]>,
    half: Vec<f32>,
    bed: Vec<f32>,
    point: usize,
}

/// The way a river's points `points` run at point `index` (a unit vector
/// along x and z).
fn run(points: &[[f32; 2]], index: usize) -> Vec2 {
    let next = points[(index + 1).min(points.len() - 1)];
    let previous = points[index.saturating_sub(1)];
    (Vec2::from(next) - Vec2::from(previous)).normalize_or(Vec2::X)
}

/// The river point within reach of `eye` whose pool would fill soonest,
/// sooner the nearer it stands across the ground, whose river runs in its
/// bed and steeply enough for [`UPSTREAM`] points above it: its pool holds
/// about the river's width times its reach up the slope (the crest over the
/// slope) times half the crest, filled by the discharge.
fn site(terrains: &Terrains, eye: Vec3) -> Option<Site> {
    let mut best: Option<(f32, Site)> = None;
    for (_, origin, layers) in terrains.waters() {
        let Some(hydrology) = layers.hydrology() else {
            continue;
        };
        for course in hydrology.rivers().courses() {
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
                let run = UPSTREAM as f32 * engine::terrain::data::RIVER_STEP;
                let slope = (course.bed(index - UPSTREAM) - course.bed(index)) / run;
                if slope < STEEP {
                    continue;
                }
                let held = 2.0 * course.half(index) * (CREST / slope) * CREST * 0.5;
                let filling = held / engine::math::max(course.discharge(index), 1.0e-3);
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

/// Builds the dam, drops the logs and sets the boulders once the bake is
/// done, a river fits and the tiles there hold its water.
pub(super) fn place(world: &mut World) {
    let Some(config) = world.get_resource::<StressConfig>() else {
        return;
    };
    let (count, seed) = (config.bodies, config.seed);
    if count == 0 || world.contains_resource::<Dammed>() {
        return;
    }
    let eye = world.resource::<Camera>().position;
    let Some((site, water)) = world.get_resource::<Terrains>().and_then(|terrains| {
        let site = site(terrains, eye)?;
        let [x, z] = site.points[site.point];
        let bed = site.bed[site.point];
        let at = site.origin + Vec3::new(x, bed + 0.05, z);
        let water = terrains.water_at(at)?.surface - site.origin.y;
        Some((site, water))
    }) else {
        return;
    };
    world.insert_resource(Dammed);
    if let Err(error) = build(world, &site, water, (count, seed)) {
        eprintln!("The highlands' rivers hold no bodies: {}", error.message);
    }
}

/// The dam, the logs and the boulders at `site`, its water standing at
/// `water` (terrain-local) at the dam.
fn build(
    world: &mut World,
    site: &Site,
    water: f32,
    (count, seed): (u32, u32),
) -> engine::Result<()> {
    let mut assets = world.resource_mut::<AssetLibrary>();
    let cube = assets.insert_mesh(Mesh::cube())?;
    let bark = assets.insert_material(Material::new([0.2, 0.13, 0.08, 1.0])?)?;
    let pale = assets.insert_material(Material::new([0.42, 0.3, 0.19, 1.0])?)?;
    let stone = assets.insert_material(Material::new([0.38, 0.37, 0.35, 1.0])?)?;
    let (boulder, _) = rocks::boulder(seed ^ 0x5eed, 0)?;
    let boulder = assets.insert_mesh(boulder)?;
    let origin = site.origin;
    let at = site.point;
    let world_point = |point: [f32; 2], height: f32| {
        Vec3::new(point[0] + origin.x, height + origin.y, point[1] + origin.z)
    };
    let along = run(&site.points, at);
    let side = Vec2::new(-along.y, along.x);
    // The water up the river stands about as deep over its bed.
    let water_at = |point: usize| water + site.bed[point] - site.bed[at];
    // Logs across the river from bed to crest, into either bank.
    let bed = site.bed[at];
    let span = 2.0 * site.half[at] + 4.0;
    let yaw = Quat::from_rotation_y((-side.y).atan2(side.x));
    for log in 0..DAM_LOGS {
        let height = bed + DAM_LOG * (log as f32 + 0.5);
        let top = water + CREST;
        let height = engine::math::min(height, top - DAM_LOG * 0.5);
        world.spawn((
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
    // The ground about the dam stays loaded, so what drifts to it moves.
    world.spawn((
        Transform::from_translation(world_point(site.points[at], water)),
        StreamingFocus,
    ));
    // Logs dropped over the river above the dam.
    let mut rng = Rng::new(seed, 0x10c5, at as i32);
    for log in 0..count {
        let point = at - 4 - (log as usize * (UPSTREAM - 6)) / count.max(1) as usize;
        let across = rng.range(-0.6, 0.6) * site.half[point];
        let flat = Vec2::from(site.points[point]) + side * across;
        let length = rng.range(2.8, 4.4);
        world.spawn((
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
    // Boulders on the bank, a little above the water, rolling in.
    for stone_index in 0..BOULDERS {
        let point = at - 6 - stone_index as usize * 5;
        let bank = side * (site.half[point] + 2.5) * if stone_index % 2 == 0 { 1.0 } else { -1.0 };
        let flat = Vec2::from(site.points[point]) + bank;
        let size = rng.range(0.9, 1.4);
        world.spawn((
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
