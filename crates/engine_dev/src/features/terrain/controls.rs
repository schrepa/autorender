//! Temporary edits of generated terrains, with undo and redo.

use std::sync::Arc;

use engine_core::{
    Engine, Feature, Terrain, TerrainEdit, TerrainSource, Terrains,
    bevy_ecs::prelude::Entity,
    glam::Vec2,
    terrain_data::{Stamp, StampOp},
};
use engine_scene::SceneInstance;
use engine_settings::ErrorCode;

use super::results::{Reach, TerrainEdited};
use crate::DevError;
use crate::coordinate;
use crate::features::feature;

/// A terrain edit control.
pub(super) enum TerrainEditing {
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
pub(super) fn edit_terrain(
    engine: &mut Engine,
    scene: Option<&SceneInstance>,
    name: Option<&str>,
    redo: &mut Vec<(Entity, TerrainEdit)>,
    editing: TerrainEditing,
) -> Result<TerrainEdited, DevError> {
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
        Reach {
            min: [center.x - radius, center.y - radius],
            max: [center.x + radius, center.y + radius],
        }
    };
    let result = match editing {
        TerrainEditing::Edit {
            id,
            center,
            radius,
            falloff,
            change,
        } => {
            if !engine_settings::is_id(&id) {
                return Err(DevError::new(
                    ErrorCode::InvalidRequest,
                    "$.id",
                    format!(
                        "id must be 1 to {} letters, digits, '_', '-' and '.'",
                        engine_settings::MAX_ID_BYTES
                    ),
                ));
            }
            let x = coordinate(center[0], "$.center[0]")?;
            let z = coordinate(center[1], "$.center[1]")?;
            let (radius, falloff) = (radius as f32, falloff as f32);
            if edits.iter().any(|edit| *edit.id == *id) {
                let standing = edits.iter().find(|edit| *edit.id == *id).expect("found");
                return Ok(TerrainEdited::Edit {
                    temporary: true,
                    applied: false,
                    id,
                    edits: edits.len(),
                    dirty: reach(&standing.stamp),
                    largest_change: 0.0,
                });
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
            TerrainEdited::Edit {
                temporary: true,
                applied: true,
                id,
                edits: edits.len(),
                dirty: reach(&stamp),
                largest_change: largest,
            }
        }
        TerrainEditing::Undo => match edits.pop() {
            Some(edit) => {
                let result = TerrainEdited::Undo {
                    temporary: true,
                    undone: Some(edit.id.to_string()),
                    edits: edits.len(),
                    dirty: Some(reach(&edit.stamp)),
                };
                redo.push((entity, edit));
                result
            }
            None => TerrainEdited::Undo {
                temporary: true,
                undone: None,
                edits: 0,
                dirty: None,
            },
        },
        TerrainEditing::Redo => {
            match redo
                .iter()
                .rposition(|(terrain, _)| *terrain == entity)
                .map(|index| redo.remove(index).1)
            {
                Some(edit) => {
                    let result = TerrainEdited::Redo {
                        temporary: true,
                        redone: Some(edit.id.to_string()),
                        edits: edits.len() + 1,
                        dirty: Some(reach(&edit.stamp)),
                    };
                    edits.push(edit);
                    result
                }
                None => TerrainEdited::Redo {
                    temporary: true,
                    redone: None,
                    edits: edits.len(),
                    dirty: None,
                },
            }
        }
    };
    if let Some(mut terrain) = world.get_mut::<Terrain>(entity) {
        terrain.edits = Arc::from(edits);
    }
    Ok(result)
}
