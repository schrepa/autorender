//! Controls of terrain and what stands on it: temporary edits of generated
//! terrains, with undo and redo, tracks pressed, navigation agents sent and
//! streamed items removed and restored.

use std::sync::Arc;

use engine_core::{
    Deformation, Engine, Feature, FixedTime, Terrain, TerrainEdit, TerrainSource, Terrains, Track,
    Tracks,
    bevy_ecs::prelude::Entity,
    glam::Vec2,
    has_feature,
    terrain_data::{Stamp, StampOp},
};
use engine_scene::SceneInstance;
use engine_settings::ErrorCode;
use serde_json::{Value, json};

use super::{coordinate, feature, feature_mut, missing};
use crate::protocol::bounds::{self, MAX_EDIT_ID};
use crate::{DevError, bounded};

/// A track an agent presses.
pub struct Press {
    pub from: [f64; 2],
    pub to: Option<[f64; 2]>,
    pub radius: f64,
    pub depth: f64,
    pub flatness: f64,
    pub clear: bool,
}

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
            ErrorCode::OutOfRange,
            "$.destination",
            "must be finite and within 1000000 meters",
        ));
    }
    let world = engine.app.world_mut();
    if !has_feature(world, Feature::Navigation) {
        return Err(missing(Feature::Navigation));
    }
    let entity = scene
        .and_then(|scene| crate::inspect::authored_entity(world, scene, id))
        .filter(|&entity| world.get::<engine_core::NavAgent>(entity).is_some())
        .ok_or_else(|| {
            DevError::new(
                ErrorCode::UnknownEntity,
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
        if ids.len() > bounds::MAX_CHANGES {
            return Err(DevError::new(
                ErrorCode::OutOfRange,
                path,
                format!("at most {} IDs", bounds::MAX_CHANGES),
            ));
        }
        if let Some(index) = ids.iter().position(|id| id.is_empty() || id.len() > 192) {
            return Err(DevError::new(
                ErrorCode::OutOfRange,
                format!("{path}[{index}]"),
                "IDs are 1 to 192 bytes",
            ));
        }
    }
    let mut changes =
        feature_mut::<engine_core::WorldChanges>(engine.app.world_mut(), Feature::WorldStreaming)?;
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

/// Presses `press` into the log as a deformer would at the fixed clock's
/// time; its result.
pub fn press_track(engine: &mut Engine, press: Press) -> Result<Value, DevError> {
    let from = [
        coordinate(press.from[0], "$.from[0]")?,
        coordinate(press.from[1], "$.from[1]")?,
    ];
    let to = match press.to {
        Some(to) => [coordinate(to[0], "$.to[0]")?, coordinate(to[1], "$.to[1]")?],
        None => from,
    };
    // A press keeps a deformer's rules.
    for (value, field) in [
        (press.radius, "radius"),
        (press.depth, "depth"),
        (press.flatness, "flatness"),
    ] {
        let rule = engine_core::Deformer::rule_of(field).expect("deformers declare the field");
        bounded(value, rule, ErrorCode::OutOfRange, &format!("$.{field}"))?;
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
    let mut tracks = feature_mut::<Tracks>(world, Feature::Deformation)?;
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

/// Samples across an edit's reach along each side for the most the ground
/// moved.
const CHANGE_SAMPLES: usize = 9;

/// Runs `editing` on the terrain `name` names in `scene` (the first terrain
/// when `None`), taking edits undo removed from and putting them back on
/// `redo`; its result.
#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
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
                    ErrorCode::UnknownTerrain,
                    "$.terrain",
                    format!("no terrain entity {name} in the scene"),
                )
            })?,
        None => world
            .query::<(Entity, &Terrain)>()
            .iter(world)
            .map(|(entity, _)| entity)
            .next()
            .ok_or_else(|| DevError::new(ErrorCode::UnknownTerrain, "$.terrain", "no terrain"))?,
    };
    let terrain = world.get::<Terrain>(entity).cloned().expect("found above");
    if !matches!(terrain.source, TerrainSource::Generated(_)) {
        return Err(DevError::new(
            ErrorCode::NotGenerated,
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
                    ErrorCode::InvalidRequest,
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
                other @ (StampOp::Raise(_) | StampOp::Lower(_) | StampOp::Hole) => other,
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
                DevError::new(ErrorCode::OutOfRange, format!("$.{field}"), invalid.rule)
            })?;
            let terrains = feature::<Terrains>(world, Feature::Terrain)?;
            let before = terrains.layers(entity).ok_or_else(|| {
                DevError::new(
                    ErrorCode::NotGenerated,
                    "$.terrain",
                    "the terrain holds no layers",
                )
            })?;
            let after = before.with_stamps(&[stamp]).map_err(|error| {
                DevError::new(ErrorCode::InvalidRequest, "$.change", error.to_string())
            })?;
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
