use super::operations::{coordinate, reply, terrains};
use super::*;
use crate::{MAX_GIVEN_RIVERS, StampOp};
use engine_core::{Error, ErrorCode, Operation, authored_entity};
#[cfg(feature = "schema")]
use engine_settings::{rule::same, stated};
use serde_json::Value;

const CHANGE_SAMPLES: usize = 9;

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct TerrainEditControl {
    #[serde(default)]
    pub terrain: Option<String>,
    #[cfg_attr(
        feature = "schema",
        schemars(transform = stated(engine_settings::rule::id()))
    )]
    pub id: String,
    pub center: [f64; 2],
    #[cfg_attr(feature = "schema", schemars(transform = stamp_rule("radius")))]
    pub radius: f64,
    #[serde(default)]
    #[cfg_attr(feature = "schema", schemars(transform = stamp_rule("falloff")))]
    pub falloff: f64,
    pub change: StampOp,
}

impl Operation for TerrainEditControl {
    const NAME: &'static str = "terrain_edit";
    const CONTROL: bool = true;
    const SHOWS: &'static str = "Terrain";
    type Request = Self;

    fn run(world: &mut World, request: Self) -> Result<Value, Error> {
        let Self {
            terrain,
            id,
            center,
            radius,
            falloff,
            change,
        } = request;
        editing(
            world,
            terrain.as_deref(),
            TerrainEditing::Edit {
                id,
                center,
                radius,
                falloff,
                change,
            },
        )
    }

    #[cfg(feature = "schema")]
    fn schema() -> schemars::Schema {
        schemars::schema_for!(Self)
    }

    #[cfg(feature = "schema")]
    fn result_schema() -> Option<schemars::Schema> {
        Some(schemars::schema_for!(TerrainEdited))
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct TerrainRiverControl {
    #[serde(default)]
    pub terrain: Option<String>,
    #[cfg_attr(
        feature = "schema",
        schemars(transform = stated(engine_settings::rule::id()))
    )]
    pub id: String,
    #[serde(default)]
    pub points: Vec<[f64; 2]>,
    #[serde(default)]
    #[cfg_attr(feature = "schema", schemars(transform = stated(same(GivenRiver::rule_of("width")))))]
    pub width: Vec<f64>,
    #[serde(default)]
    #[cfg_attr(feature = "schema", schemars(transform = stated(same(GivenRiver::rule_of("depth")))))]
    pub depth: Vec<f64>,
    #[serde(default)]
    #[cfg_attr(feature = "schema", schemars(transform = stated(same(GivenRiver::rule_of("discharge")))))]
    pub discharge: Vec<f64>,
    #[serde(default)]
    pub name: Option<String>,
}

impl Operation for TerrainRiverControl {
    const NAME: &'static str = "terrain_river";
    const CONTROL: bool = true;
    const SHOWS: &'static str = "Terrain";
    type Request = Self;

    fn run(world: &mut World, request: Self) -> Result<Value, Error> {
        draw_river(world, &request).map(reply)
    }

    #[cfg(feature = "schema")]
    fn schema() -> schemars::Schema {
        schemars::schema_for!(Self)
    }

    #[cfg(feature = "schema")]
    fn result_schema() -> Option<schemars::Schema> {
        Some(schemars::schema_for!(TerrainEdited))
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct TerrainUndoControl {
    #[serde(default)]
    pub terrain: Option<String>,
}

impl Operation for TerrainUndoControl {
    const NAME: &'static str = "terrain_undo";
    const CONTROL: bool = true;
    const SHOWS: &'static str = "Terrain";
    type Request = Self;

    fn run(world: &mut World, request: Self) -> Result<Value, Error> {
        editing(world, request.terrain.as_deref(), TerrainEditing::Undo)
    }

    #[cfg(feature = "schema")]
    fn schema() -> schemars::Schema {
        schemars::schema_for!(Self)
    }

    #[cfg(feature = "schema")]
    fn result_schema() -> Option<schemars::Schema> {
        Some(schemars::schema_for!(TerrainEdited))
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct TerrainRedoControl {
    #[serde(default)]
    pub terrain: Option<String>,
}

impl Operation for TerrainRedoControl {
    const NAME: &'static str = "terrain_redo";
    const CONTROL: bool = true;
    const SHOWS: &'static str = "Terrain";
    type Request = Self;

    fn run(world: &mut World, request: Self) -> Result<Value, Error> {
        editing(world, request.terrain.as_deref(), TerrainEditing::Redo)
    }

    #[cfg(feature = "schema")]
    fn schema() -> schemars::Schema {
        schemars::schema_for!(Self)
    }

    #[cfg(feature = "schema")]
    fn result_schema() -> Option<schemars::Schema> {
        Some(schemars::schema_for!(TerrainEdited))
    }
}

#[cfg(feature = "schema")]
fn stamp_rule(field: &'static str) -> impl Fn(&mut schemars::Schema) {
    stated(Stamp::rule_of(field).expect("stamps declare the field"))
}

#[derive(Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(untagged)]
pub enum TerrainEdited {
    River {
        temporary: bool,
        applied: bool,
        id: String,
        rivers: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        dirty: Option<Reach>,
    },
    Edit {
        temporary: bool,
        applied: bool,
        id: String,
        edits: usize,
        dirty: Reach,
        largest_change: f32,
    },
    Undo {
        temporary: bool,
        undone: Option<String>,
        edits: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        dirty: Option<Reach>,
    },
    Redo {
        temporary: bool,
        redone: Option<String>,
        edits: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        dirty: Option<Reach>,
    },
}

#[derive(Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Reach {
    pub min: [f32; 2],
    pub max: [f32; 2],
}

enum TerrainEditing {
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

#[derive(Resource, Default)]
struct TerrainRedo(Vec<(Entity, TerrainEdit)>);

fn editing(world: &mut World, name: Option<&str>, editing: TerrainEditing) -> Result<Value, Error> {
    let mut redo = world.remove_resource::<TerrainRedo>().unwrap_or_default();
    let result = edit_terrain(world, name, &mut redo.0, editing);
    world.insert_resource(redo);
    result.map(reply)
}

fn terrain_entity(world: &mut World, name: Option<&str>) -> Result<Entity, Error> {
    match name {
        Some(name) => authored_entity(world, name)
            .filter(|&entity| world.get::<Terrain>(entity).is_some())
            .ok_or_else(|| {
                Error::new(
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
            .ok_or_else(|| Error::new(ErrorCode::UnknownTerrain, "$.terrain", "no terrain")),
    }
}

fn valid_id(id: &str) -> Result<(), Error> {
    if engine_settings::is_id(id) {
        Ok(())
    } else {
        Err(Error::new(
            ErrorCode::InvalidRequest,
            "$.id",
            format!(
                "id must be 1 to {} letters, digits, '_', '-' and '.'",
                engine_settings::MAX_ID_BYTES
            ),
        ))
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
fn edit_terrain(
    world: &mut World,
    name: Option<&str>,
    redo: &mut Vec<(Entity, TerrainEdit)>,
    editing: TerrainEditing,
) -> Result<TerrainEdited, Error> {
    let entity = terrain_entity(world, name)?;
    let terrain = world.get::<Terrain>(entity).cloned().expect("found above");
    let origin = terrain.origin;
    let mut edits = terrain.edits.to_vec();
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
            stamp.validate().map_err(|invalid| {
                let field = match invalid.field.strip_prefix("op") {
                    Some(rest) => format!("change{rest}"),
                    None => invalid.field.into_owned(),
                };
                Error::new(ErrorCode::OutOfRange, format!("$.{field}"), invalid.rule)
            })?;
            let before = terrains(world)?.layers(entity).ok_or_else(|| {
                Error::new(
                    ErrorCode::UnknownTerrain,
                    "$.terrain",
                    "the terrain does not stream yet",
                )
            })?;
            let after = before.with_stamps(&[stamp]).map_err(|error| {
                Error::new(ErrorCode::InvalidRequest, "$.change", error.to_string())
            })?;
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

fn draw_river(world: &mut World, drawing: &TerrainRiverControl) -> Result<TerrainEdited, Error> {
    let entity = terrain_entity(world, drawing.terrain.as_deref())?;
    terrains(world)?;
    let terrain = world.get::<Terrain>(entity).cloned().expect("found above");
    if !terrain.layers.has_hydrology() {
        return Err(Error::new(
            ErrorCode::InvalidRequest,
            "$.terrain",
            "the terrain has no hydrology layer to give rivers to",
        ));
    }
    valid_id(&drawing.id)?;
    let origin = Vec2::new(terrain.origin.x, terrain.origin.z);
    let reach = |rivers: &[&GivenRiver]| -> Option<Reach> {
        let mut bounds: Option<(Vec2, Vec2)> = None;
        for river in rivers {
            let half = river.width.iter().fold(0.0_f32, |widest, &width| {
                engine_core::math::max(widest, 0.5 * width)
            });
            for &point in &river.points {
                let point = Vec2::from(point) + origin;
                let (low, high) = (point - Vec2::splat(half), point + Vec2::splat(half));
                bounds =
                    Some(bounds.map_or((low, high), |(from, to)| (from.min(low), to.max(high))));
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
            .map(|(index, &[x, z])| -> Result<[f32; 2], Error> {
                let x = coordinate(x, &format!("$.points[{index}][0]"))?;
                let z = coordinate(z, &format!("$.points[{index}][1]"))?;
                Ok([x - origin.x, z - origin.y])
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let values =
            |values: &[f64]| -> Vec<f32> { values.iter().map(|&value| value as f32).collect() };
        let river = GivenRiver {
            name: drawing.name.clone(),
            points,
            width: values(&drawing.width),
            depth: values(&drawing.depth),
            discharge: values(&drawing.discharge),
        };
        river.validate().map_err(|invalid| {
            Error::new(
                ErrorCode::OutOfRange,
                format!("$.{}", invalid.field),
                invalid.rule,
            )
        })?;
        if standing.is_some_and(|at| rivers[at].river == river) {
            return Ok(unchanged(&rivers, reach(&[&river])));
        }
        let most = MAX_GIVEN_RIVERS;
        if standing.is_none() && rivers.len() >= most {
            return Err(Error::new(
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
