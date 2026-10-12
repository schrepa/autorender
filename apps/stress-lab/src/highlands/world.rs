//! The highlands as a Stress Lab world: the tower field on a plateau of a
//! streamed terrain under golden hour, wind and clouds, framed from where
//! the valleys open out.

use engine::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

use engine::prelude::*;
use engine::terrain::TerrainSource;
#[cfg(not(target_arch = "wasm32"))]
use engine::{Error, ErrorCode};
#[cfg(not(target_arch = "wasm32"))]
use engine_native::files;

#[cfg(not(target_arch = "wasm32"))]
use super::{FILES, ROCK_FILES};
use super::{Plateau, add_systems, populate};
use crate::kit::field::{SPACING, animate, field};
use crate::kit::motion::Animation;
use crate::kit::{StressConfig, StressFile, StressWorld};

pub(crate) struct HighlandsWorld;

impl StressWorld for HighlandsWorld {
    fn name(&self) -> &'static str {
        "highlands"
    }

    fn label(&self) -> &'static str {
        "Highlands"
    }

    fn look(&self) -> &'static str {
        "golden"
    }

    fn populate(
        &self,
        world: &mut World,
        config: &StressConfig,
        files: &HashMap<String, StressFile>,
    ) -> Result<(u32, (Vec3, Vec3)), String> {
        let (bounds, plateau) = populate(world, config, hamlet_radius(config), files)?;
        Ok((field(world, config, plateau)?, bounds))
    }

    fn environment(&self, world: &mut World) {
        // A brisk wind sways the highlands' grass and spruces,
        // and drives clouds over them twice as fast aloft; their blades and far
        // trees, thinner than pixels, resolve over frames.
        world.insert_resource(TemporalAntialiasing { enabled: true });
        world.insert_resource(Wind {
            velocity: [5.5, 2.0],
            gusts: 0.55,
        });
        world.insert_resource(Clouds {
            coverage: 0.38,
            base: 1500.0,
            thickness: 1500.0,
            wind: [11.0, 4.0],
            shadows: 0.6,
            ..Clouds::default()
        });
    }

    fn shadows(&self, bounds: (Vec3, Vec3)) -> Shadows {
        // Shadows reach across the valleys, and two kept levels keep still
        // casters sharper within a kilometer of the camera: a quarter meter
        // a texel within 256 meters, one beyond.
        Shadows {
            bounds: Some(bounds),
            distance: 6000.0,
            near_distance: 160.0,
            resolution: 4096,
            levels: 2,
            level_extent: 512.0,
            ..Shadows::default()
        }
    }

    fn reflections(&self) -> Reflections {
        // The lakes mirror shores kilometers away.
        Reflections {
            distance: 4000.0,
            ..Reflections::default()
        }
    }

    fn add_systems(&self, app: &mut App, moving_nodes: u32) {
        // The cave opens once the terrain's bakes are done; the tower
        // field's objects move as the field's do.
        add_systems(app);
        if moving_nodes > 0 {
            app.add_systems(FixedSimulation, animate.in_set(Animation));
        }
    }

    fn camera(&self, world: &World, config: &StressConfig, view: &str) -> Option<Camera> {
        match view {
            "overview" => {
                let (position, target) = overview(world, config);
                Some(Camera {
                    position,
                    target,
                    near: 0.5,
                    far: 20_000.0,
                    vertical_fov_radians: 55.0_f32.to_radians(),
                })
            }
            "close" => {
                let (position, target) = close_view(world, config);
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
            _ => None,
        }
    }

    fn camera_speed(&self, _span: f32) -> f32 {
        // The camera crosses kilometers.
        160.0
    }

    /// The terrain material maps, without which its ground shows plain
    /// colors, and with `rocks` the optional scanned rock maps.
    #[cfg(not(target_arch = "wasm32"))]
    fn files(&self, assets: &Path, rocks: bool) -> engine::Result<HashMap<String, StressFile>> {
        let mut found = HashMap::default();
        let image = |path: &Path| -> engine::Result<Option<files::Image>> {
            let bytes = std::fs::read(path).map_err(|error| {
                Error::new(
                    ErrorCode::MissingFile,
                    "$",
                    format!("cannot read {}: {error}", path.display()),
                )
            })?;
            if files::image_format(&bytes).map(|format| format.name) != Some("ktx2-rgba8") {
                return Ok(None);
            }
            files::decode_image(&path.display().to_string(), &bytes)
                .map(Some)
                .map_err(|message| Error::new(ErrorCode::InvalidTexture, "$", message))
        };
        let mut add = |key: String, image: files::Image| {
            found.insert(
                key,
                StressFile {
                    bytes: image.pixels,
                    url: None,
                    size: Some([image.width, image.height]),
                },
            );
        };
        let materials = FILES.iter().try_for_each(|name| {
            let path = assets.join("stress/terrain").join(name);
            let decoded = image(&path)?.ok_or_else(|| {
                Error::new(
                    ErrorCode::InvalidTexture,
                    "$",
                    format!("{} is not an uncompressed RGBA8 KTX2 image", path.display()),
                )
            })?;
            add((*name).to_owned(), decoded);
            Ok::<_, Error>(())
        });
        if let Err(error) = materials {
            eprintln!(
                "The ground shows plain colors: {}. Build its materials with \
                 `cargo run -p engine_assetc --release --bin terrain_materials`.",
                error.message
            );
            return Ok(HashMap::default());
        }
        // Scanned rock maps, kept locally and optional: without them
        // rocks wear the terrain rock's maps.
        for name in ROCK_FILES.iter().filter(|_| rocks) {
            if let Ok(Some(decoded)) = image(&assets.join("stress/rocks").join(name)) {
                add(format!("rocks/{name}"), decoded);
            }
        }
        Ok(found)
    }
}

/// The plateau's radius for the hamlet of `config`'s renderables, the tower
/// field's grid with a margin.
fn hamlet_radius(config: &StressConfig) -> f32 {
    let side = (config.renderables as f32).sqrt().ceil() * SPACING;
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
fn overview(world: &World, config: &StressConfig) -> (Vec3, Vec3) {
    let target = Vec3::new(0.0, world.resource::<Plateau>().0, 0.0);
    let direction = lowest(world, 1200.0);
    let distance = hamlet_radius(config) + 700.0;
    (clear_view(world, target, direction, distance, 0.45), target)
}

/// Inside the plateau's edge, looking out over the lowest valley beyond it.
fn close_view(world: &World, config: &StressConfig) -> (Vec3, Vec3) {
    let direction = lowest(world, 1500.0);
    let edge = direction * hamlet_radius(config) * 0.85;
    let far = direction * 1500.0;
    (
        Vec3::new(edge.x, ground(world, edge.x, edge.y) + 10.0, edge.y),
        Vec3::new(far.x, ground(world, far.x, far.y) + 60.0, far.y),
    )
}

/// From high above the plateau across the whole terrain to the north.
fn away_view(world: &World) -> (Vec3, Vec3) {
    let height = ground(world, 0.0, 0.0) + 700.0;
    (
        Vec3::new(0.0, height, 600.0),
        Vec3::new(0.0, height - 500.0, -3000.0),
    )
}
