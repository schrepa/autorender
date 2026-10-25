//! The highlands as a Stress Lab world: the tower field on a plateau of a
//! streamed terrain under golden hour, wind and clouds, framed from where
//! the valleys open out, and views of its water near and far.

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
        // A brisk wind sways the highlands' grass and spruces,
        // and drives clouds over them twice as fast aloft; their blades and far
        // trees, thinner than pixels, resolve over frames.
        world.insert_resource(TemporalAntialiasing {
            enabled: true,
            ..TemporalAntialiasing::default()
        });
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
        // The eye adapts: the cave brightens by day, the night keeps dark.
        world.insert_resource(AutoExposure::default());
        // Rivers settle in meter cells, the ground's own samples, which its
        // rivers are traced along: the cook settles every watered tile at
        // that size, and the settled water stands everywhere.
        //
        // The water runs as glacial streams do, milky with the rock flour
        // the ice grinds: its grains, microns across, scatter light strongly
        // and send back blue-green, which turns mountain water turquoise.
        // The water itself is a deep blue-green halving the light in a
        // meter, and the flour scatters 0.8 a meter on top, a little more
        // than the water takes, so its teal leads the body's color. Seen 30
        // degrees down through 30 centimeters, the light runs about 0.9
        // meters in the water: a third of the stones under it shows through
        // a body two thirds of the way to its deep teal.
        world.insert_resource(Water {
            color: [0.005, 0.035, 0.04],
            clarity: 1.0,
            sediment: 0.8,
            sediment_color: [0.04, 0.16, 0.15],
            simulation_cell: 1.0,
            // Over the 256 meters around the camera, as far as its finest
            // tiles stand about an eye near the ground, the rivers run live
            // in the settled water's meter cells, and what stands in them
            // pushes them.
            simulated: 256.0,
            live_cell: 1.0,
            ..Water::default()
        });
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
            _ => WATER_VIEWS
                .iter()
                .find(|(name, ..)| *name == view)
                .map(|&(_, position, yaw, pitch)| water_view(position, yaw, pitch)),
        }
    }

    fn camera_speed(&self, _span: f32) -> f32 {
        // The camera crosses kilometers.
        160.0
    }

    /// With `rocks`, the optional scanned rock maps: the ground's maps load
    /// as texture records.
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

/// The views of the water: down onto running water (`torrent`, `rapids`,
/// `meadow_river`), far over water, which shows whether its waves band at a
/// distance (`band`), and close along a stream (`close_stream`); each one's
/// name, the camera's world position, and the yaw (degrees from -z toward
/// +x) and pitch (degrees from level, negative down) it looks along.
const WATER_VIEWS: [(&str, [f32; 3], f32, f32); 5] = [
    ("torrent", [-1584.0, 675.0, -2973.0], 0.0, -34.0),
    ("rapids", [-3008.0, 228.0, -168.0], 0.0, -30.0),
    ("meadow_river", [-266.0, 671.0, -2986.0], 0.0, -32.0),
    ("band", [-300.0, 730.0, -3180.0], -10.0, -38.0),
    ("close_stream", [1474.5, 605.9, 743.3], 112.0, -18.0),
];

/// A camera at world `position` looking along `yaw` and `pitch` (degrees,
/// as [`WATER_VIEWS`] gives them), its near plane a fifth of a meter out and
/// its field of view 60 degrees high.
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

/// From high above the plateau across the whole terrain to the north.
fn away_view(world: &World) -> (Vec3, Vec3) {
    let height = ground(world, 0.0, 0.0) + 700.0;
    (
        Vec3::new(0.0, height, 600.0),
        Vec3::new(0.0, height - 500.0, -3000.0),
    )
}
