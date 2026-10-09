//! Read-only terrain observations: heights at points, a
//! sweep comparing what terrain queries, collision rays and generated
//! terrains' layers say the ground is, routes for paths, and
//! the sea's surface and tracks; temporary
//! edits of generated terrains, with undo and redo, and
//! tracks pressed.

use engine_collision::{CollisionRay, CollisionWorld, QueryFilter, refresh_collision_world};
use std::sync::Arc;

use engine_core::{
    Deformation, Engine, FixedTime, Ocean, PresentedTime, RouteOptions, Terrain, TerrainEdit,
    TerrainSource, Terrains, Track, Tracks,
    bevy_ecs::prelude::{Entity, World},
    glam::{Vec2, Vec3},
    terrain_data::{Stamp, StampOp},
};
use engine_scene::{SceneEntity, SceneInstance};
use serde_json::{Value, json};

use crate::{DevError, DevRequest};

/// The most points a height request names, and samples along a sweep's
/// side.
pub(crate) const MAX_POINTS: usize = 256;
pub(crate) const MAX_SWEEP_SAMPLES: u32 = 64;
/// The largest coordinate a request names, as collision queries bound them.
const LIMIT: f64 = 1.0e6;
/// Hits a sweep ray may meet: placed objects above the
/// ground, and the ground itself.
const PLACED_RAY_HITS: usize = 16;

/// A track an agent presses.
pub struct Press {
    pub from: [f64; 2],
    pub to: Option<[f64; 2]>,
    pub radius: f64,
    pub depth: f64,
    pub flatness: f64,
    pub clear: bool,
}

/// Presses `press` into the log as a deformer would at the fixed clock's
/// time; its result.
/// Sends navigation agent `id` in `scene` to `destination` (world x and
/// z), or stops it.
pub fn nav_go(
    engine: &mut Engine,
    scene: Option<&SceneInstance>,
    id: &str,
    destination: Option<[f32; 2]>,
) -> Result<Value, DevError> {
    if let Some(destination) = destination
        && !destination
            .iter()
            .all(|value| value.is_finite() && value.abs() <= 1.0e6)
    {
        return Err(DevError::new(
            "invalid_request",
            "$.destination",
            "must be finite and within 1000000 meters",
        ));
    }
    let world = engine.app.world_mut();
    let entity = scene
        .and_then(|scene| crate::inspect::authored_entity(world, scene, id))
        .filter(|&entity| world.get::<engine_core::NavAgent>(entity).is_some())
        .ok_or_else(|| {
            DevError::new(
                "not_found",
                "$.entity",
                format!("no navigation agent {id} in the scene"),
            )
        })?;
    let mut agent = world
        .get_mut::<engine_core::NavAgent>(entity)
        .expect("checked above");
    agent.destination = destination;
    Ok(json!({ "temporary": true, "destination": destination }))
}

/// Removes and restores streamed items: `remove` and
/// `restore` name item IDs (at most 256 each, each 1 to 192 bytes);
/// `clear` restores every item first.
pub fn change_world(
    engine: &mut Engine,
    remove: &[String],
    restore: &[String],
    clear: bool,
) -> Result<Value, DevError> {
    for (ids, path) in [(remove, "$.remove"), (restore, "$.restore")] {
        if ids.len() > 256 {
            return Err(DevError::new("invalid_request", path, "at most 256 IDs"));
        }
        if let Some(index) = ids.iter().position(|id| id.is_empty() || id.len() > 192) {
            return Err(DevError::new(
                "invalid_request",
                format!("{path}[{index}]"),
                "IDs are 1 to 192 bytes",
            ));
        }
    }
    let mut changes = engine
        .app
        .world_mut()
        .resource_mut::<engine_core::WorldChanges>();
    if clear {
        changes.clear();
    }
    let removed = remove.iter().filter(|id| changes.remove(id)).count();
    let restored = restore.iter().filter(|id| changes.restore(id)).count();
    Ok(json!({
        "temporary": true,
        "newly_removed": removed,
        "restored": restored,
        "removed": changes.removed().len(),
    }))
}

pub fn press_track(engine: &mut Engine, press: Press) -> Result<Value, DevError> {
    let from = [
        coordinate(press.from[0], "$.from[0]")?,
        coordinate(press.from[1], "$.from[1]")?,
    ];
    let to = match press.to {
        Some(to) => [coordinate(to[0], "$.to[0]")?, coordinate(to[1], "$.to[1]")?],
        None => from,
    };
    for (value, range, path) in [
        (press.radius, 0.01..=50.0, "$.radius"),
        (press.depth, 0.0..=10.0, "$.depth"),
        (press.flatness, 0.0..=1.0, "$.flatness"),
    ] {
        if !(value.is_finite() && range.contains(&value)) {
            return Err(DevError::new(
                "invalid_request",
                path,
                format!("must be in [{}, {}]", range.start(), range.end()),
            ));
        }
    }
    let world = engine.app.world_mut();
    let seconds = world.resource::<FixedTime>().elapsed_seconds;
    let resource = world.get_resource::<Deformation>().copied();
    let settings = resource.unwrap_or_default().sanitized();
    let track = Track {
        from,
        to,
        radius: press.radius as f32,
        depth: press.depth as f32,
        flatness: press.flatness as f32,
        seconds,
    };
    let mut tracks = world.resource_mut::<Tracks>();
    if press.clear {
        tracks.clear();
    }
    tracks.press(track, &settings);
    Ok(json!({
        "temporary": true,
        "track": track,
        "stamps": tracks.len(),
        "dropped": tracks.dropped(),
        "enabled": resource.is_some_and(|settings| settings.enabled),
    }))
}

/// A terrain edit control.
pub enum TerrainEditing {
    Edit {
        id: String,
        center: [f64; 2],
        radius: f64,
        falloff: f64,
        change: StampOp,
    },
    Undo,
    Redo,
}

/// The longest edit ID.
const MAX_EDIT_ID: usize = 64;
/// Samples across an edit's reach along each side for the most the ground
/// moved.
const CHANGE_SAMPLES: usize = 9;

/// Runs `editing` on the terrain `name` names in `scene` (the first terrain
/// when `None`), taking edits undo removed from and putting them back on
/// `redo`; its result.
pub fn edit_terrain(
    engine: &mut Engine,
    scene: Option<&SceneInstance>,
    name: Option<&str>,
    redo: &mut Vec<(Entity, TerrainEdit)>,
    editing: TerrainEditing,
) -> Result<Value, DevError> {
    let world = engine.app.world_mut();
    let entity = match name {
        Some(name) => scene
            .and_then(|scene| scene.entity(name))
            .filter(|&entity| world.get::<Terrain>(entity).is_some())
            .ok_or_else(|| {
                DevError::new(
                    "not_found",
                    "$.terrain",
                    format!("no terrain entity {name} in the scene"),
                )
            })?,
        None => world
            .query::<(Entity, &Terrain)>()
            .iter(world)
            .map(|(entity, _)| entity)
            .next()
            .ok_or_else(|| DevError::new("not_found", "$.terrain", "no terrain"))?,
    };
    let terrain = world.get::<Terrain>(entity).cloned().expect("found above");
    if !matches!(terrain.source, TerrainSource::Generated(_)) {
        return Err(DevError::new(
            "not_generated",
            "$.terrain",
            "only terrains generated from layers take edits",
        ));
    }
    let origin = terrain.origin;
    let mut edits = terrain.edits.to_vec();
    // Where an edit reaches, world.
    let reach = |stamp: &Stamp| {
        let center = Vec2::from(stamp.center) + Vec2::new(origin.x, origin.z);
        let radius = stamp.radius + stamp.falloff;
        json!({
            "min": [center.x - radius, center.y - radius],
            "max": [center.x + radius, center.y + radius],
        })
    };
    let result = match editing {
        TerrainEditing::Edit {
            id,
            center,
            radius,
            falloff,
            change,
        } => {
            if id.is_empty()
                || id.len() > MAX_EDIT_ID
                || !id
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || b"_-.".contains(&byte))
            {
                return Err(DevError::new(
                    "invalid_request",
                    "$.id",
                    format!("id must be 1 to {MAX_EDIT_ID} letters, digits, '_', '-' and '.'"),
                ));
            }
            let x = coordinate(center[0], "$.center[0]")?;
            let z = coordinate(center[1], "$.center[1]")?;
            let (radius, falloff) = (radius as f32, falloff as f32);
            if edits.iter().any(|edit| *edit.id == *id) {
                let standing = edits.iter().find(|edit| *edit.id == *id).expect("found");
                return Ok(json!({
                    "temporary": true,
                    "applied": false,
                    "id": id,
                    "edits": edits.len(),
                    "dirty": reach(&standing.stamp),
                    "largest_change": 0.0,
                }));
            }
            // Levels to a world height, as the terrain's stamps hold it.
            let op = match change {
                StampOp::Flatten(height) => StampOp::Flatten(height - origin.y),
                other => other,
            };
            let stamp = Stamp {
                center: [x - origin.x, z - origin.z],
                radius,
                falloff,
                op,
            };
            // The stamp's own rules, named as the request names its fields.
            stamp.validate().map_err(|invalid| {
                let field = match invalid.field.strip_prefix("op") {
                    Some(rest) => format!("change{rest}"),
                    None => invalid.field.into_owned(),
                };
                DevError::new("invalid_request", format!("$.{field}"), invalid.rule)
            })?;
            let terrains = world.resource::<Terrains>();
            let before = terrains.layers(entity).ok_or_else(|| {
                DevError::new("not_generated", "$.terrain", "the terrain holds no layers")
            })?;
            let after = before
                .with_stamps(&[stamp])
                .map_err(|error| DevError::new("invalid_request", "$.change", error.to_string()))?;
            // The most the ground moved across the edit's reach.
            let span = radius + falloff;
            let mut largest = 0.0_f32;
            for row in 0..CHANGE_SAMPLES {
                for column in 0..CHANGE_SAMPLES {
                    let at = |index: usize| index as f32 / (CHANGE_SAMPLES - 1) as f32 * 2.0 - 1.0;
                    let point = Vec2::from(stamp.center) + Vec2::new(at(column), at(row)) * span;
                    let change = after.height_at(&terrain.pyramid, point)
                        - before.height_at(&terrain.pyramid, point);
                    largest = engine_core::math::max(largest, change.abs());
                }
            }
            edits.push(TerrainEdit {
                id: Arc::from(id.as_str()),
                stamp,
            });
            redo.retain(|(terrain, _)| *terrain != entity);
            json!({
                "temporary": true,
                "applied": true,
                "id": id,
                "edits": edits.len(),
                "dirty": reach(&stamp),
                "largest_change": largest,
            })
        }
        TerrainEditing::Undo => match edits.pop() {
            Some(edit) => {
                let result = json!({
                    "temporary": true,
                    "undone": &*edit.id,
                    "edits": edits.len(),
                    "dirty": reach(&edit.stamp),
                });
                redo.push((entity, edit));
                result
            }
            None => json!({"temporary": true, "undone": null, "edits": 0}),
        },
        TerrainEditing::Redo => {
            match redo
                .iter()
                .rposition(|(terrain, _)| *terrain == entity)
                .map(|index| redo.remove(index).1)
            {
                Some(edit) => {
                    let result = json!({
                        "temporary": true,
                        "redone": &*edit.id,
                        "edits": edits.len() + 1,
                        "dirty": reach(&edit.stamp),
                    });
                    edits.push(edit);
                    result
                }
                None => json!({"temporary": true, "redone": null, "edits": edits.len()}),
            }
        }
    };
    if let Some(mut terrain) = world.get_mut::<Terrain>(entity) {
        terrain.edits = Arc::from(edits);
    }
    Ok(result)
}

fn coordinate(value: f64, path: &str) -> Result<f32, DevError> {
    if value.is_finite() && value.abs() <= LIMIT {
        Ok(value as f32)
    } else {
        Err(DevError::new(
            "invalid_request",
            path,
            "coordinates must be finite and within +/-1e6",
        ))
    }
}

/// A terrain's authored ID, or its entity bits outside scenes.
fn terrain_id(world: &World, entity: engine_core::bevy_ecs::prelude::Entity) -> Value {
    world.get::<SceneEntity>(entity).map_or_else(
        || json!(entity.to_bits()),
        |source| json!(source.authored_id),
    )
}

pub(crate) fn execute(
    engine: &mut Engine,
    _scene: Option<&SceneInstance>,
    request: DevRequest,
) -> Result<Value, DevError> {
    match request {
        DevRequest::TerrainHeight { points, .. } => {
            if points.is_empty() || points.len() > MAX_POINTS {
                return Err(DevError::new(
                    "invalid_request",
                    "$.points",
                    format!("name 1 to {MAX_POINTS} points"),
                ));
            }
            let world = engine.app.world();
            let terrains = world.resource::<Terrains>();
            let samples = points
                .iter()
                .enumerate()
                .map(|(index, point)| {
                    let x = coordinate(point[0], &format!("$.points[{index}][0]"))?;
                    let z = coordinate(point[1], &format!("$.points[{index}][1]"))?;
                    Ok(terrains.sample(x, z).map_or(Value::Null, |sample| {
                        // What its hydrology bake found there,
                        // as the finest level sees it, and the water standing
                        // there: its surface (world height)
                        // and depth.
                        let bake = terrains.layers(sample.terrain).and_then(|layers| {
                            let terrain = terrains.terrain(sample.terrain)?;
                            let water = layers.hydrology()?;
                            let local = (x - terrain.origin.x, z - terrain.origin.z);
                            // Above the ground as tiles hold it, which layers
                            // after the bake may raise over the water.
                            let standing = water
                                .water_at(local.0, local.1)
                                .map(|(surface, _)| terrain.origin.y + surface)
                                .filter(|&surface| surface > sample.height)
                                .map(|surface| {
                                    json!({"surface": surface, "depth": surface - sample.height})
                                });
                            Some((
                                water.at(local.0, local.1, terrain.pyramid.spacing(0)),
                                standing,
                            ))
                        });
                        // Meters from the nearest road's edge,
                        // where a path's masks reach.
                        let path = terrains.layers(sample.terrain).and_then(|layers| {
                            let terrain = terrains.terrain(sample.terrain)?;
                            let road = layers.path_at(x - terrain.origin.x, z - terrain.origin.z);
                            (road.distance < engine_core::terrain_data::PATH_FAR)
                                .then_some(road.edge())
                        });
                        json!({
                            "terrain": terrain_id(world, sample.terrain),
                            "height": sample.height,
                            "normal": sample.normal.to_array(),
                            "hole": sample.hole,
                            "level": sample.level,
                            "water": bake.as_ref().map(|(found, _)| found),
                            "standing_water": bake.and_then(|(_, standing)| standing),
                            "path": path,
                        })
                    }))
                })
                .collect::<Result<Vec<_>, DevError>>()?;
            Ok(json!({"samples": samples, "stats": terrains.stats()}))
        }
        DevRequest::NavPath {
            from, to, options, ..
        } => {
            let from = Vec2::new(
                coordinate(from[0], "$.from[0]")?,
                coordinate(from[1], "$.from[1]")?,
            );
            let to = Vec2::new(coordinate(to[0], "$.to[0]")?, coordinate(to[1], "$.to[1]")?);
            let options = options.unwrap_or_default();
            options.validate().map_err(|invalid| {
                DevError::new(
                    "invalid_request",
                    format!("$.options.{}", invalid.field),
                    invalid.rule,
                )
            })?;
            let world = engine.app.world_mut();
            let entity = world
                .query::<(Entity, &Terrain)>()
                .iter(world)
                .find(|(_, terrain)| matches!(terrain.source, TerrainSource::Generated(_)))
                .map(|(entity, _)| entity)
                .ok_or_else(|| DevError::new("not_found", "$", "no generated terrain"))?;
            let walk = world
                .resource::<Terrains>()
                .walk(entity, from, to, &options);
            let round = |value: f32| (f64::from(value) * 100.0).round() / 100.0;
            Ok(json!({
                "terrain": terrain_id(world, entity),
                "options": options,
                "walk": walk.map(|walk| json!({
                    "points": walk.points.iter().map(|point| point.map(round)).collect::<Vec<_>>(),
                    "length": walk.length,
                    "level": walk.level,
                    "searched": walk.searched,
                })),
            }))
        }
        DevRequest::DeformationQuery {
            points, seconds, ..
        } => {
            if points.is_empty() || points.len() > MAX_POINTS {
                return Err(DevError::new(
                    "invalid_request",
                    "$.points",
                    format!("name 1 to {MAX_POINTS} points"),
                ));
            }
            let world = engine.app.world();
            let seconds = match seconds {
                Some(seconds) if seconds.is_finite() && seconds.abs() <= 1.0e9 => seconds,
                Some(_) => {
                    return Err(DevError::new(
                        "invalid_request",
                        "$.seconds",
                        "seconds must be finite and within +/-1e9",
                    ));
                }
                None => world
                    .get_resource::<PresentedTime>()
                    .map_or(0.0, |time| time.seconds),
            };
            let settings = world
                .get_resource::<Deformation>()
                .copied()
                .unwrap_or_default()
                .sanitized();
            let tracks = world.resource::<Tracks>();
            let terrains = world.resource::<Terrains>();
            // Tenths of millimeters keep responses small.
            let round = |value: f32| (f64::from(value) * 1.0e4).round() / 1.0e4;
            let samples = points
                .iter()
                .enumerate()
                .map(|(index, point)| {
                    let x = coordinate(point[0], &format!("$.points[{index}][0]"))?;
                    let z = coordinate(point[1], &format!("$.points[{index}][1]"))?;
                    let depth = tracks.at(Vec2::new(x, z), seconds, &settings);
                    let deform = terrains.deform_at(x, z);
                    Ok(json!({
                        "track": round(depth.track),
                        "rim": round(depth.rim),
                        "deform": deform.map(round),
                        "depth": deform.map(|deform| round(engine_core::math::min(depth.track, deform))),
                        "ground": terrains.height_at(x, z).map(round),
                    }))
                })
                .collect::<Result<Vec<_>, DevError>>()?;
            Ok(json!({
                "seconds": seconds,
                "samples": samples,
                "stamps": tracks.len(),
                "dropped": tracks.dropped(),
            }))
        }
        DevRequest::WaterQuery {
            points, seconds, ..
        } => {
            if points.is_empty() || points.len() > MAX_POINTS {
                return Err(DevError::new(
                    "invalid_request",
                    "$.points",
                    format!("name 1 to {MAX_POINTS} points"),
                ));
            }
            let world = engine.app.world();
            // Frames draw the sea at the simulation time they present.
            let seconds = match seconds {
                Some(seconds) if seconds.is_finite() && seconds.abs() <= 1.0e9 => seconds,
                Some(_) => {
                    return Err(DevError::new(
                        "invalid_request",
                        "$.seconds",
                        "seconds must be finite and within +/-1e9",
                    ));
                }
                None => world
                    .get_resource::<PresentedTime>()
                    .map_or(0.0, |time| time.seconds),
            };
            let waves = world
                .get_resource::<Ocean>()
                .filter(|ocean| ocean.enabled)
                .map(|ocean| ocean.spectrum());
            let terrains = world.resource::<Terrains>();
            let samples = points
                .iter()
                .enumerate()
                .map(|(index, point)| {
                    let x = coordinate(point[0], &format!("$.points[{index}][0]"))?;
                    let z = coordinate(point[1], &format!("$.points[{index}][1]"))?;
                    let surface = waves
                        .as_ref()
                        .map(|waves| waves.surface(point[0], point[1], seconds));
                    let ground = terrains.sample(x, z).map(|sample| sample.height);
                    // Millimeters and ten-thousandths keep responses small.
                    let round = |value: f64, step: f64| (value / step).round() * step;
                    Ok(json!({
                        "ocean": surface.map(|surface| json!({
                            "height": round(f64::from(surface.height), 1.0e-3),
                            "normal": surface.normal.map(|value| round(f64::from(value), 1.0e-4)),
                            "compression": round(f64::from(surface.compression), 1.0e-4),
                            "still": surface.still.map(|value| round(value, 1.0e-3)),
                        })),
                        "ground": ground.map(|ground| round(f64::from(ground), 1.0e-3)),
                        "depth": surface
                            .zip(ground)
                            .map(|(surface, ground)| round(f64::from(surface.height - ground), 1.0e-3)),
                    }))
                })
                .collect::<Result<Vec<_>, DevError>>()?;
            Ok(json!({
                "seconds": seconds,
                "samples": samples,
                "level": waves.as_ref().map(|waves| waves.level),
                "amplitude": waves.as_ref().map(|waves| waves.amplitude()),
                "waves": waves.as_ref().map(|waves| waves.waves().to_vec()),
            }))
        }
        DevRequest::TerrainSweep {
            center,
            half_extent,
            samples,
            ..
        } => {
            let cx = coordinate(center[0], "$.center[0]")?;
            let cz = coordinate(center[1], "$.center[1]")?;
            if !(half_extent.is_finite() && (0.0..=4096.0).contains(&half_extent)) {
                return Err(DevError::new(
                    "invalid_request",
                    "$.half_extent",
                    "half_extent must be in [0, 4096] meters",
                ));
            }
            if !(2..=MAX_SWEEP_SAMPLES).contains(&samples) {
                return Err(DevError::new(
                    "invalid_request",
                    "$.samples",
                    format!("samples must be 2 to {MAX_SWEEP_SAMPLES} along each side"),
                ));
            }
            sweep(engine, Vec2::new(cx, cz), half_extent as f32, samples)
        }
        DevRequest::TerrainRoute {
            from,
            to,
            level,
            max_grade,
            grade_cost,
            water_cost,
            turn_cost,
            ..
        } => {
            let from = Vec2::new(
                coordinate(from[0], "$.from[0]")?,
                coordinate(from[1], "$.from[1]")?,
            );
            let to = Vec2::new(coordinate(to[0], "$.to[0]")?, coordinate(to[1], "$.to[1]")?);
            let defaults = RouteOptions::default();
            let options = RouteOptions {
                level: level.unwrap_or(defaults.level),
                max_grade: max_grade.map_or(defaults.max_grade, |value| value as f32),
                grade_cost: grade_cost.map_or(defaults.grade_cost, |value| value as f32),
                water_cost: water_cost.map_or(defaults.water_cost, |value| value as f32),
                turn_cost: turn_cost.map_or(defaults.turn_cost, |value| value as f32),
            };
            options.validate().map_err(|invalid| {
                DevError::new(
                    "invalid_request",
                    format!("$.{}", invalid.field),
                    invalid.rule,
                )
            })?;
            let world = engine.app.world();
            let terrains = world.resource::<Terrains>();
            let Some(sample) = terrains.sample(from.x, from.y) else {
                return Err(DevError::new(
                    "invalid_request",
                    "$.from",
                    "no resident terrain lies under from",
                ));
            };
            let Some(terrain) = terrains.terrain(sample.terrain) else {
                return Err(DevError::new(
                    "invalid_request",
                    "$.from",
                    "no terrain lies under from",
                ));
            };
            let origin = Vec2::new(terrain.origin.x, terrain.origin.z);
            let route = terrains.route(sample.terrain, from, to, &options);
            Ok(json!({
                "terrain": terrain_id(world, sample.terrain),
                "options": options,
                "origin": origin.to_array(),
                // Centimeters are plenty, and keep long routes within the
                // response limit.
                "route": route.map(|route| json!({
                    "points": route
                        .points
                        .iter()
                        .map(|point| point.map(|value| (f64::from(value) * 100.0).round() / 100.0))
                        .collect::<Vec<_>>(),
                    "length": route.length,
                    "steepest": route.steepest,
                    "hairpin_steps": route.hairpin_steps,
                })),
            }))
        }
        _ => unreachable!("only terrain requests reach here"),
    }
}

/// The largest difference so far and where.
#[derive(Default)]
struct Worst {
    difference: f32,
    at: Option<[f32; 2]>,
}

impl Worst {
    fn add(&mut self, difference: f32, point: Vec2) {
        if difference > self.difference || self.at.is_none() {
            self.difference = engine_core::math::max(self.difference, difference);
            self.at = Some(point.to_array());
        }
    }

    fn value(&self) -> Value {
        json!({"max_difference": self.difference, "at": self.at})
    }
}

/// The ground under a grid of `samples` by `samples` points across
/// `half_extent` meters of `center`, by terrain queries (the resident
/// tiles renderers draw), collision rays and generated terrains' layers.
fn sweep(
    engine: &mut Engine,
    center: Vec2,
    half_extent: f32,
    samples: u32,
) -> Result<Value, DevError> {
    let collision = engine.app.world().contains_resource::<CollisionWorld>();
    if collision {
        refresh_collision_world(engine.app.world_mut()).map_err(crate::collision::native_error)?;
    }
    let world = engine.app.world();
    let terrains = world.resource::<Terrains>();
    let mut queried = 0;
    let mut level_zero = 0;
    let mut holes = 0;
    let mut rays = Worst::default();
    let mut layers = Worst::default();
    let (mut ray_checks, mut ray_misses, mut layer_checks, mut placed_first) = (0, 0, 0, 0);
    let mut collected = Vec::new();
    for row in 0..samples {
        for column in 0..samples {
            let unit = Vec2::new(column as f32, row as f32) / (samples - 1) as f32 * 2.0 - 1.0;
            let point = center + unit * half_extent;
            let Some(sample) = terrains.sample(point.x, point.y) else {
                continue;
            };
            queried += 1;
            if sample.hole {
                holes += 1;
                continue;
            }
            level_zero += usize::from(sample.level == 0);
            if sample.level != 0 {
                continue;
            }
            // Collision holds level-0 tiles around focus points only. Rays
            // meeting placed objects first count, and the
            // ground beneath them compares.
            if collision {
                let mut hits = Vec::new();
                world
                    .resource::<CollisionWorld>()
                    .raycast_all(
                        CollisionRay {
                            origin: Vec3::new(point.x, sample.height + 1000.0, point.y),
                            direction: Vec3::NEG_Y,
                            max_distance: 2000.0,
                        },
                        QueryFilter::default(),
                        &mut hits,
                        PLACED_RAY_HITS,
                    )
                    .map_err(crate::collision::native_error)?;
                placed_first += usize::from(hits.first().is_some_and(|hit| hit.placed.is_some()));
                let hit = hits.into_iter().find(|hit| hit.placed.is_none());
                match hit.filter(|hit| hit.entity == sample.terrain) {
                    Some(hit) => {
                        ray_checks += 1;
                        rays.add((hit.point.y - sample.height).abs(), point);
                    }
                    None => ray_misses += 1,
                }
            }
            // The layers tiles generate from, their hydrology baked.
            if let Some(terrain) = world.get::<Terrain>(sample.terrain)
                && let Some(stack) = world.resource::<Terrains>().layers(sample.terrain)
            {
                let local = point - Vec2::new(terrain.origin.x, terrain.origin.z);
                let height = terrain.origin.y + stack.height_at(&terrain.pyramid, local);
                layer_checks += 1;
                layers.add((height - sample.height).abs(), point);
            }
            if collected.len() < 8 {
                collected.push(json!({"at": point.to_array(), "height": sample.height}));
            }
        }
    }
    Ok(json!({
        "points": samples * samples,
        "resident": queried,
        "level_zero": level_zero,
        "holes": holes,
        "collision": collision.then(|| json!({
            "compared": ray_checks,
            "missed": ray_misses,
            "placed_first": placed_first,
            "worst": rays.value(),
        })),
        "layers": json!({"compared": layer_checks, "worst": layers.value()}),
        "examples": collected,
    }))
}
