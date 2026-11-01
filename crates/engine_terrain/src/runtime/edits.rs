use super::operations::{coordinate, reply, terrain_entity};
use super::*;
use crate::{MAX_GIVEN_RIVERS, StampOp};
use engine_heightfield::given::{MAX_DEPTH, MAX_DISCHARGE, MAX_WIDTH};
use engine_core::{Joined, Operation, SceneIds};
use engine_places::terrain::TerrainLayers;
use serde_json::Value;

const CHANGE_SAMPLES: usize = 9;

type Scenes = Query<'static, 'static, (Entity, &'static mut TerrainScene)>;

type Editing = (
    Scenes,
    SceneIds<'static, 'static>,
    Joined<'static, TerrainLayers>,
    Res<'static, Terrains>,
    ResMut<'static, TerrainRedo>,
);

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TerrainEditControl {
    #[serde(default)]
    pub terrain: Option<String>,
    #[schemars(transform = engine_core::fields::stated(engine_core::Rule::id()))]
    pub id: String,
    pub center: [f64; 2],
    #[schemars(transform = stamp_rule("radius"))]
    pub radius: f64,
    #[serde(default)]
    #[schemars(transform = stamp_rule("falloff"))]
    pub falloff: f64,
    pub change: StampOp,
}

impl Operation for TerrainEditControl {
    const NAME: &'static str = "terrain_edit";
    type Params = Editing;
    type Request = Self;

    fn run(
        (mut scenes, ids, kinds, terrains, mut redo): SystemParamItem<'_, '_, Editing>,
        request: Self,
    ) -> Result<Value, Error> {
        let Self {
            terrain,
            id,
            center,
            radius,
            falloff,
            change,
        } = request;
        let kinds: Vec<&'static LayerKind> = kinds.iter().copied().collect();
        let entity = terrain_entity(&scenes, &ids, terrain.as_deref())?;
        edit_terrain(
            (&mut scenes, entity),
            (&terrains, &kinds),
            &mut redo.0,
            TerrainEditing::Edit {
                id,
                center,
                radius,
                falloff,
                change,
            },
        )
        .map(reply)
    }
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TerrainRiverControl {
    #[serde(default)]
    pub terrain: Option<String>,
    #[schemars(transform = engine_core::fields::stated(engine_core::Rule::id()))]
    pub id: String,
    #[serde(default)]
    pub points: Vec<[f64; 2]>,
    #[serde(default)]
    #[schemars(transform = river_rule(MAX_WIDTH))]
    pub width: Vec<f64>,
    #[serde(default)]
    #[schemars(transform = river_rule(MAX_DEPTH))]
    pub depth: Vec<f64>,
    #[serde(default)]
    #[schemars(transform = river_rule(MAX_DISCHARGE))]
    pub discharge: Vec<f64>,
    #[serde(default)]
    pub name: Option<String>,
}

impl Operation for TerrainRiverControl {
    const NAME: &'static str = "terrain_river";
    type Params = (
        Scenes,
        SceneIds<'static, 'static>,
        Joined<'static, TerrainLayers>,
    );
    type Request = Self;

    fn run(
        (mut scenes, ids, kinds): SystemParamItem<'_, '_, Self::Params>,
        request: Self,
    ) -> Result<Value, Error> {
        let entity = terrain_entity(&scenes, &ids, request.terrain.as_deref())?;
        let rivers = kinds.iter().any(|kind| kind.name == RIVERS);
        draw_river((&mut scenes, entity), rivers, &request).map(reply)
    }
}

const RIVERS: &str = "rivers";

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TerrainUndoControl {
    #[serde(default)]
    pub terrain: Option<String>,
}

impl Operation for TerrainUndoControl {
    const NAME: &'static str = "terrain_undo";
    type Params = Editing;
    type Request = Self;

    fn run(
        (mut scenes, ids, kinds, terrains, mut redo): SystemParamItem<'_, '_, Editing>,
        request: Self,
    ) -> Result<Value, Error> {
        let kinds: Vec<&'static LayerKind> = kinds.iter().copied().collect();
        let entity = terrain_entity(&scenes, &ids, request.terrain.as_deref())?;
        edit_terrain(
            (&mut scenes, entity),
            (&terrains, &kinds),
            &mut redo.0,
            TerrainEditing::Undo,
        )
        .map(reply)
    }
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TerrainRedoControl {
    #[serde(default)]
    pub terrain: Option<String>,
}

impl Operation for TerrainRedoControl {
    const NAME: &'static str = "terrain_redo";
    type Params = Editing;
    type Request = Self;

    fn run(
        (mut scenes, ids, kinds, terrains, mut redo): SystemParamItem<'_, '_, Editing>,
        request: Self,
    ) -> Result<Value, Error> {
        let kinds: Vec<&'static LayerKind> = kinds.iter().copied().collect();
        let entity = terrain_entity(&scenes, &ids, request.terrain.as_deref())?;
        edit_terrain(
            (&mut scenes, entity),
            (&terrains, &kinds),
            &mut redo.0,
            TerrainEditing::Redo,
        )
        .map(reply)
    }
}

const STAMP_RULES: [(&str, Rule); 2] = [
    ("radius", Rule::range(0.0, Layers::MAX_COORDINATE as f64)),
    ("falloff", Rule::range(0.0, Layers::MAX_COORDINATE as f64)),
];

fn river_rule(most: f32) -> impl Fn(&mut schemars::Schema) {
    engine_core::fields::stated(engine_core::fields::above_to(0.0, most))
}

fn stamp_rule(field: &'static str) -> impl Fn(&mut schemars::Schema) {
    let rule = STAMP_RULES
        .iter()
        .find(|(name, _)| *name == field)
        .map(|&(_, rule)| rule);
    engine_core::fields::stated(engine_core::fields::same(rule))
}

#[derive(Serialize, schemars::JsonSchema)]
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

#[derive(Serialize, schemars::JsonSchema)]
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
pub struct TerrainRedo(Vec<(Entity, TerrainEdit)>);

fn valid_id(id: &str) -> Result<(), Error> {
    if engine_core::is_id(id) {
        Ok(())
    } else {
        Err(Error::new(ErrorCode::InvalidRequest, "$.id")
            .expected(Expected::Id)
            .got(id))
    }
}

fn built(scenes: &Query<'_, '_, (Entity, &mut TerrainScene)>, entity: Entity) -> Terrain {
    scenes
        .get(entity)
        .ok()
        .and_then(|(_, scene)| scene.built.clone())
        .expect("terrain_found")
}

#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
fn edit_terrain(
    (scenes, entity): (&mut Query<'_, '_, (Entity, &mut TerrainScene)>, Entity),
    (terrains, kinds): (&Terrains, &[&'static LayerKind]),
    redo: &mut Vec<(Entity, TerrainEdit)>,
    editing: TerrainEditing,
) -> Result<TerrainEdited, Error> {
    let terrain = built(scenes, entity);
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
                let standing = edits
                    .iter()
                    .find(|edit| *edit.id == *id)
                    .expect("edit_found");
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
            stamp.validate().map_err(|error| {
                let path = match error.path.strip_prefix("$.op") {
                    Some(rest) => format!("$.change{rest}"),
                    None => error.path.clone(),
                };
                Error {
                    code: ErrorCode::OutOfRange,
                    path,
                    ..error
                }
            })?;
            let before = terrains.layers(entity, kinds).ok_or_else(|| {
                Error::new(ErrorCode::UnknownTerrain, "$.terrain").expected(Expected::Present)
            })?;
            let after = before.with_stamps(&[stamp], kinds).map_err(|error| Error {
                code: ErrorCode::InvalidRequest,
                path: "$.change".to_owned(),
                ..error
            })?;
            let span = radius + falloff;
            let mut largest = 0.0_f32;
            for row in 0..CHANGE_SAMPLES {
                for column in 0..CHANGE_SAMPLES {
                    let at = |index: usize| index as f32 / (CHANGE_SAMPLES - 1) as f32 * 2.0 - 1.0;
                    let point = Vec2::from(stamp.center) + Vec2::new(at(column), at(row)) * span;
                    let change = after.height_at(&terrain.pyramid, point)
                        - before.height_at(&terrain.pyramid, point);
                    largest = engine_math::max(largest, change.abs());
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
    if let Ok((_, mut scene)) = scenes.get_mut(entity)
        && let Some(terrain) = scene.built.as_mut()
    {
        terrain.edits = Arc::from(edits);
    }
    Ok(result)
}

fn draw_river(
    (scenes, entity): (&mut Query<'_, '_, (Entity, &mut TerrainScene)>, Entity),
    rivers_listed: bool,
    drawing: &TerrainRiverControl,
) -> Result<TerrainEdited, Error> {
    let terrain = built(scenes, entity);
    if !terrain.layers.has_hydrology() {
        return Err(Error::new(ErrorCode::MissingLayer, "$.terrain").variants(["hydrology"]));
    }
    if !rivers_listed {
        return Err(Error::new(ErrorCode::MissingLayer, "$.terrain").variants([RIVERS]));
    }
    valid_id(&drawing.id)?;
    let origin = Vec2::new(terrain.origin.x, terrain.origin.z);
    let reach = |rivers: &[&GivenRiver]| -> Option<Reach> {
        let mut bounds: Option<(Vec2, Vec2)> = None;
        for river in rivers {
            let half = river.width.iter().fold(0.0_f32, |widest, &width| {
                engine_math::max(widest, 0.5 * width)
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
        river.validate().map_err(|error| Error {
            code: ErrorCode::OutOfRange,
            ..error
        })?;
        if standing.is_some_and(|at| rivers[at].river == river) {
            return Ok(unchanged(&rivers, reach(&[&river])));
        }
        let most = MAX_GIVEN_RIVERS;
        if standing.is_none() && rivers.len() >= most {
            return Err(Error::new(ErrorCode::OutOfRange, "$.id")
                .at_most(most as u64)
                .got(rivers.len()));
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
    if let Ok((_, mut scene)) = scenes.get_mut(entity)
        && let Some(terrain) = scene.built.as_mut()
    {
        terrain.rivers = Arc::from(rivers);
    }
    Ok(result)
}
