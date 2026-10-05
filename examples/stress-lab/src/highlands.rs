//! Highlands (decision 051): an 8 km terrain the engine generates from a
//! layer stack (eroded mountains, then hand-placed stamps: a plateau
//! leveled for a hamlet, a flat valley floor and a cave mouth cut as a
//! hole), streamed around the camera, with the scene's renderables standing
//! on the plateau as the tower field does.

use engine::prelude::*;

use crate::StressConfig;

/// Level-0 tiles along each axis, and their sample spacing: 8,192 meters.
const TILES: u32 = 32;
const SPACING: f32 = 1.0;
const SIZE: f32 = TILES as f32 * 256.0 * SPACING;
/// The plateau's center, where the hamlet stands.
const PLATEAU: [f32; 2] = [0.0, 0.0];

/// The height the hamlet's plateau is leveled to.
#[derive(Resource, Clone, Copy)]
pub(crate) struct Plateau(pub f32);

/// Spawns the terrain; returns the region shadows cover and the height
/// the hamlet stands at.
pub(crate) fn populate(
    world: &mut World,
    config: &StressConfig,
    hamlet: f32,
) -> Result<((Vec3, Vec3), f32), String> {
    let half = SIZE * 0.5;
    // Terrain-local: the terrain's first corner is at -half.
    let local = |point: [f32; 2]| [point[0] + half, point[1] + half];
    let plateau = local(PLATEAU);
    let pyramid = Pyramid::new([TILES, TILES], SPACING).map_err(|error| error.to_string())?;
    let mountains = Layer::Noise(NoiseLayer {
        seed: config.seed,
        height: 1500.0,
        wavelength: 3200.0,
        ridges: 0.5,
        warp: 0.4,
        erosion: 0.9,
        smoothing: 0.6,
        valleys: 0.7,
        roughness: 0.3,
        ..NoiseLayer::default()
    });
    // Leveled where the mountains already stand there.
    let level = Layers::new(vec![mountains.clone()])
        .map_err(|error| error.to_string())?
        .height_at(&pyramid, Vec2::from(plateau));
    world.insert_resource(Plateau(level));
    let layers = Layers::new(vec![
        mountains,
        // The hamlet's plateau, wide enough for the field's grid.
        Layer::Stamp(Stamp {
            center: plateau,
            radius: hamlet,
            falloff: 220.0,
            op: StampOp::Flatten(level),
        }),
        // A wide valley floor to the north-east, leveled low.
        Layer::Stamp(Stamp {
            center: local([1500.0, -1300.0]),
            radius: 520.0,
            falloff: 700.0,
            op: StampOp::Flatten(60.0),
        }),
        // A cave mouth in the plateau's slope.
        Layer::Stamp(Stamp {
            center: [plateau[0] + hamlet + 120.0, plateau[1]],
            radius: 7.0,
            falloff: 0.0,
            op: StampOp::Hole,
        }),
    ])
    .map_err(|error| error.to_string())?;
    // Snow on the peaks only.
    world.spawn(
        Terrain::generated(Vec3::new(-half, 0.0, -half), pyramid, layers).with_surface(
            TerrainSurface::new(
                &TerrainSurface::default_rules(800.0),
                TerrainSurface::default_materials(),
            )
            .map_err(|error| error.to_string())?,
        ),
    );
    Ok((
        (
            Vec3::new(-half, -50.0, -half),
            Vec3::new(half, 1600.0, half),
        ),
        level,
    ))
}

/// The plateau's radius for the hamlet of `config`'s renderables, the tower
/// field's grid with a margin.
pub(crate) fn hamlet_radius(config: &StressConfig) -> f32 {
    let side = (config.renderables as f32).sqrt().ceil() * crate::scene::SPACING;
    side * 0.75 + 30.0
}

/// The ground's world height under world `(x, z)` of `world`'s terrain,
/// from its layers wherever tiles are resident or not.
fn ground(world: &World, x: f32, z: f32) -> f32 {
    let mut terrains = world
        .try_query::<&Terrain>()
        .expect("terrain is registered");
    terrains
        .iter(world)
        .find_map(|terrain| match &terrain.source {
            TerrainSource::Generated(layers) => Some(
                terrain.origin.y
                    + layers.height_at(
                        &terrain.pyramid,
                        Vec2::new(x - terrain.origin.x, z - terrain.origin.z),
                    ),
            ),
            TerrainSource::File { .. } => None,
        })
        .unwrap_or_else(|| world.resource::<Plateau>().0)
}

/// Horizontal unit directions around the plateau, every 45 degrees.
fn directions() -> impl Iterator<Item = Vec2> {
    (0..8).map(|step| {
        let (sin, cos) = math::sin_cos(step as f32 * std::f32::consts::FRAC_PI_4);
        Vec2::new(cos, sin)
    })
}

/// The direction from the plateau whose ground `reach` meters out is
/// lowest: the widest view over the valleys.
fn lowest(world: &World, reach: f32) -> Vec2 {
    directions()
        .map(|direction| {
            let point = direction * reach;
            (ground(world, point.x, point.y), direction)
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map_or(Vec2::Y, |(_, direction)| direction)
}

/// A camera `distance` meters from `target` along horizontal `direction`,
/// `rise` meters up per meter out, raised until its line of sight clears
/// the ground by 30 meters.
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

/// The overview: high over the plateau from its lowest side, looking back
/// across the hamlet into the mountains.
pub(crate) fn overview(world: &World, config: &StressConfig) -> (Vec3, Vec3) {
    let target = Vec3::new(0.0, world.resource::<Plateau>().0, 0.0);
    let direction = lowest(world, 1200.0);
    let distance = hamlet_radius(config) + 700.0;
    (clear_view(world, target, direction, distance, 0.45), target)
}

/// Inside the plateau's edge, looking out over the lowest valley beyond it.
pub(crate) fn close_view(world: &World, config: &StressConfig) -> (Vec3, Vec3) {
    let direction = lowest(world, 1500.0);
    let edge = direction * hamlet_radius(config) * 0.85;
    let far = direction * 1500.0;
    (
        Vec3::new(edge.x, ground(world, edge.x, edge.y) + 10.0, edge.y),
        Vec3::new(far.x, ground(world, far.x, far.y) + 60.0, far.y),
    )
}

/// From high above the plateau across the whole terrain to the north.
pub(crate) fn away_view(world: &World) -> (Vec3, Vec3) {
    let height = ground(world, 0.0, 0.0) + 700.0;
    (
        Vec3::new(0.0, height, 600.0),
        Vec3::new(0.0, height - 500.0, -3000.0),
    )
}
