use engine::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

use engine::prelude::*;
#[cfg(not(target_arch = "wasm32"))]
use engine::{Error, ErrorCode};
#[cfg(not(target_arch = "wasm32"))]
use engine_native::files;

#[cfg(not(target_arch = "wasm32"))]
use super::ROCK_FILES;
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
    ) -> engine::Result<u32> {
        let plateau = populate(world, config, hamlet_radius(config), files)?;
        field(world, config, plateau)
    }

    fn environment(&self, world: &mut World) {
        world.insert_resource(TemporalAntialiasing {
            enabled: true,
            ..TemporalAntialiasing::default()
        });
        world.insert_resource(Wind {
            velocity: (Vec2::new(5.5, 2.0).normalize() * 2.5).to_array(),
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
        world.insert_resource(AutoExposure::default());
        world.insert_resource(Water {
            color: [0.003, 0.02, 0.024],
            clarity: 4.5,
            sediment: 0.01,
            sediment_color: [0.04, 0.16, 0.15],
            simulation_cell: 1.0,
            ..Water::default()
        });
    }

    fn reflections(&self) -> Reflections {
        Reflections {
            distance: 4000.0,
            ..Reflections::default()
        }
    }

    fn add_systems(&self, app: &mut App, moving_nodes: u32) {
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
            _ => WATER_VIEWS
                .iter()
                .find(|(name, ..)| *name == view)
                .map(|&(_, position, yaw, pitch)| water_view(position, yaw, pitch)),
        }
    }

    fn camera_speed(&self, _span: f32) -> f32 {
        160.0
    }

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
        for name in ROCK_FILES.iter().filter(|_| rocks) {
            if let Ok(Some(decoded)) = image(&assets.join("stress/rocks").join(name)) {
                add(format!("rocks/{name}"), decoded);
            }
        }
        Ok(found)
    }
}

fn hamlet_radius(config: &StressConfig) -> f32 {
    let side = (config.renderables as f32).sqrt().ceil() * SPACING;
    side * 0.75 + 30.0
}

fn ground(world: &World, x: f32, z: f32) -> f32 {
    let mut terrains = world
        .try_query::<&Terrain>()
        .expect("terrain is registered");
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

fn overview(world: &World, config: &StressConfig) -> (Vec3, Vec3) {
    let target = Vec3::new(0.0, world.resource::<Plateau>().0, 0.0);
    let direction = lowest(world, 1200.0);
    let distance = hamlet_radius(config) + 700.0;
    (clear_view(world, target, direction, distance, 0.45), target)
}

fn close_view(world: &World, config: &StressConfig) -> (Vec3, Vec3) {
    let direction = lowest(world, 1500.0);
    let edge = direction * hamlet_radius(config) * 0.85;
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
