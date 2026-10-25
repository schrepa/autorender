//! Temporary edits of generated terrains, with undo and redo, and rivers
//! drawn over them.

use std::sync::Arc;

use engine_core::{
    Engine, Feature, GivenRiver, Terrain, TerrainEdit, TerrainRiver, Terrains,
    bevy_ecs::prelude::{Entity, World},
    glam::Vec2,
    terrain_data::{Stamp, StampOp},
};
use engine_scene::SceneInstance;
use engine_settings::ErrorCode;

use super::TerrainRiverControl;
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

/// The terrain entity `name` names in `scene` (the first terrain when
/// `None`).
fn terrain_entity(
    world: &mut World,
    scene: Option<&SceneInstance>,
    name: Option<&str>,
) -> Result<Entity, DevError> {
    match name {
        Some(name) => scene
            .and_then(|scene| scene.entity(name))
            .filter(|&entity| world.get::<Terrain>(entity).is_some())
            .ok_or_else(|| {
                DevError::new(
                    ErrorCode::UnknownTerrain,
                    "$.terrain",
                    format!("no terrain entity {name} in the scene"),
                )
            }),
        None => world
            .query::<(Entity, &Terrain)>()
            .iter(world)
            .map(|(entity, _)| entity)
            .next()
            .ok_or_else(|| DevError::new(ErrorCode::UnknownTerrain, "$.terrain", "no terrain")),
    }
}

/// Whether `id` is an authored ID, which edits and drawn rivers go by.
fn valid_id(id: &str) -> Result<(), DevError> {
    if engine_settings::is_id(id) {
        Ok(())
    } else {
        Err(DevError::new(
            ErrorCode::InvalidRequest,
            "$.id",
            format!(
                "id must be 1 to {} letters, digits, '_', '-' and '.'",
                engine_settings::MAX_ID_BYTES
            ),
        ))
    }
}

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
    let entity = terrain_entity(world, scene, name)?;
    let terrain = world.get::<Terrain>(entity).cloned().expect("found above");
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
            valid_id(&id)?;
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
                    ErrorCode::UnknownTerrain,
                    "$.terrain",
                    "the terrain does not stream yet",
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

/// Draws `drawing`'s river (world points, which the terrain holds
/// terrain-local) under its id on the terrain its `terrain` names in
/// `scene` (the first terrain when `None`), replacing the one drawn under
/// that id, or takes that one away where it names no points; its result.
pub(super) fn draw_river(
    engine: &mut Engine,
    scene: Option<&SceneInstance>,
    drawing: &TerrainRiverControl,
) -> Result<TerrainEdited, DevError> {
    let world = engine.app.world_mut();
    let entity = terrain_entity(world, scene, drawing.terrain.as_deref())?;
    feature::<Terrains>(world, Feature::Terrain)?;
    let terrain = world.get::<Terrain>(entity).cloned().expect("found above");
    if !terrain.layers.has_hydrology() {
        return Err(DevError::new(
            ErrorCode::InvalidRequest,
            "$.terrain",
            "the terrain has no hydrology layer to give rivers to",
        ));
    }
    valid_id(&drawing.id)?;
    let origin = Vec2::new(terrain.origin.x, terrain.origin.z);
    // Where rivers lie, world: their points' bounds out past the widest
    // half each gives its channel.
    let reach = |rivers: &[&GivenRiver]| -> Option<Reach> {
        let mut bounds: Option<(Vec2, Vec2)> = None;
        for river in rivers {
            let half = river.width.iter().fold(0.0_f32, |widest, &width| {
                engine_core::math::max(widest, 0.5 * width)
            });
            for &point in &river.points {
                let point = Vec2::from(point) + origin;
                let (low, high) = (point - Vec2::splat(half), point + Vec2::splat(half));
                bounds = Some(bounds.map_or((low, high), |(from, to)| {
                    (from.min(low), to.max(high))
                }));
            }
        }
        bounds.map(|(low, high)| Reach {
            min: low.to_array(),
            max: high.to_array(),
        })
    };
    let mut rivers = terrain.rivers.to_vec();
    let standing = rivers.iter().position(|drawn| *drawn.id == *drawing.id);
    let unchanged = |rivers: &[TerrainRiver], dirty: Option<Reach>| TerrainEdited::River {
        temporary: true,
        applied: false,
        id: drawing.id.clone(),
        rivers: rivers.len(),
        dirty,
    };
    let dirty = if drawing.points.is_empty() {
        // No points take the river drawn under the id away.
        let Some(at) = standing else {
            return Ok(unchanged(&rivers, None));
        };
        let gone = rivers.remove(at);
        reach(&[&gone.river])
    } else {
        let points = drawing
            .points
            .iter()
            .enumerate()
            .map(|(index, &[x, z])| -> Result<[f32; 2], DevError> {
                let x = coordinate(x, &format!("$.points[{index}][0]"))?;
                let z = coordinate(z, &format!("$.points[{index}][1]"))?;
                Ok([x - origin.x, z - origin.y])
            })
            .collect::<Result<Vec<_>, DevError>>()?;
        let values =
            |values: &[f64]| -> Vec<f32> { values.iter().map(|&value| value as f32).collect() };
        let river = GivenRiver {
            name: drawing.name.clone(),
            points,
            width: values(&drawing.width),
            depth: values(&drawing.depth),
            discharge: values(&drawing.discharge),
        };
        // The river's own rules, named as the request names its fields.
        river.validate().map_err(|invalid| {
            DevError::new(
                ErrorCode::OutOfRange,
                format!("$.{}", invalid.field),
                invalid.rule,
            )
        })?;
        if standing.is_some_and(|at| rivers[at].river == river) {
            return Ok(unchanged(&rivers, reach(&[&river])));
        }
        let most = engine_core::terrain_data::MAX_GIVEN_RIVERS;
        if standing.is_none() && rivers.len() >= most {
            return Err(DevError::new(
                ErrorCode::OutOfRange,
                "$.id",
                format!("a terrain holds at most {most} drawn rivers"),
            ));
        }
        let dirty = match standing {
            Some(at) => reach(&[&rivers[at].river, &river]),
            None => reach(&[&river]),
        };
        let drawn = TerrainRiver {
            id: Arc::from(drawing.id.as_str()),
            river,
        };
        match standing {
            Some(at) => rivers[at] = drawn,
            None => rivers.push(drawn),
        }
        dirty
    };
    let result = TerrainEdited::River {
        temporary: true,
        applied: true,
        id: drawing.id.clone(),
        rivers: rivers.len(),
        dirty,
    };
    if let Some(mut terrain) = world.get_mut::<Terrain>(entity) {
        terrain.rivers = Arc::from(rivers);
    }
    Ok(result)
}
