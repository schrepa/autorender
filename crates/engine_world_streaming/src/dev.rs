use bevy_ecs::prelude::World;
use engine_core::{Error, ErrorCode, Operation, missing_brick};
use serde::{Deserialize, Serialize};

use crate::{STREAMING_GRIDS, WorldCells, WorldChanges};

pub const MAX_CHANGES: usize = 256;

#[derive(Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct WorldCellsView {
    pub cells: u32,
    pub loaded: u32,
    pub pending: u32,
    pub items: u32,
    pub entities: u32,
    pub spawned: u32,
    pub despawned: u32,
    pub velocity: [f32; 3],
    pub grids: Vec<StreamingGridView>,
    pub nearest: Vec<StreamedCellView>,
}

#[derive(Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct StreamingGridView {
    pub cell: f32,
    pub range: f32,
    pub reach: Option<f32>,
    pub cells: u32,
    pub loaded: u32,
}

#[derive(Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct StreamedCellView {
    pub grid: u8,
    pub key: [i32; 2],
    pub items: u32,
    pub entities: u32,
    pub loaded: bool,
    pub distance: f32,
}

#[derive(Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct WorldChangesView {
    pub removed: usize,
    pub ids: Vec<String>,
}

pub(crate) fn world_cells(cells: &WorldCells) -> WorldCellsView {
    let stats = cells.stats();
    WorldCellsView {
        cells: stats.cells,
        loaded: stats.loaded,
        pending: stats.pending,
        items: stats.items,
        entities: stats.entities,
        spawned: stats.spawned,
        despawned: stats.despawned,
        velocity: cells.velocity().to_array(),
        grids: STREAMING_GRIDS
            .iter()
            .zip(cells.grids())
            .map(|(grid, [count, loaded])| StreamingGridView {
                cell: grid.cell,
                range: grid.range,
                reach: grid.reach.is_finite().then_some(grid.reach),
                cells: count,
                loaded,
            })
            .collect(),
        nearest: cells
            .nearest(32)
            .into_iter()
            .map(|cell| StreamedCellView {
                grid: cell.grid,
                key: cell.key,
                items: cell.items,
                entities: cell.entities,
                loaded: cell.loaded,
                distance: cell.distance,
            })
            .collect(),
    }
}

pub(crate) fn world_changes(changes: &WorldChanges) -> WorldChangesView {
    WorldChangesView {
        removed: changes.removed().len(),
        ids: changes.removed().take(64).map(str::to_owned).collect(),
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct WorldChangesControl {
    #[serde(default)]
    #[cfg_attr(feature = "schema", schemars(length(max = MAX_CHANGES)))]
    pub remove: Vec<String>,
    #[serde(default)]
    #[cfg_attr(feature = "schema", schemars(length(max = MAX_CHANGES)))]
    pub restore: Vec<String>,
    #[serde(default)]
    pub clear: bool,
}

impl Operation for WorldChangesControl {
    const NAME: &'static str = "world_changes";
    const CONTROL: bool = true;
    type Request = Self;

    fn run(world: &mut World, request: Self) -> Result<serde_json::Value, Error> {
        change_world(world, &request.remove, &request.restore, request.clear)
    }

    #[cfg(feature = "schema")]
    fn schema() -> schemars::Schema {
        schemars::schema_for!(WorldChangesControl)
    }
}

fn change_world(
    world: &mut World,
    remove: &[String],
    restore: &[String],
    clear: bool,
) -> Result<serde_json::Value, Error> {
    for (ids, path) in [(remove, "$.remove"), (restore, "$.restore")] {
        if ids.len() > MAX_CHANGES {
            return Err(Error::new(ErrorCode::OutOfRange, path)
                .at_most(MAX_CHANGES as u64)
                .got(ids.len()));
        }
        if let Some(index) = ids.iter().position(|id| id.is_empty() || id.len() > 192) {
            return Err(
                Error::new(ErrorCode::OutOfRange, format!("{path}[{index}]"))
                    .range(1.0, 192.0)
                    .got(ids[index].len()),
            );
        }
    }
    let mut changes = world
        .get_resource_mut::<WorldChanges>()
        .ok_or_else(|| missing_brick("world_streaming", "$.op"))?;
    if clear {
        changes.clear();
    }
    let removed = remove.iter().filter(|id| changes.remove(id)).count();
    let restored = restore.iter().filter(|id| changes.restore(id)).count();
    Ok(serde_json::to_value(WorldChangesResult {
        temporary: true,
        newly_removed: removed,
        restored,
        removed: changes.removed().len(),
    })
    .expect("results_serialize"))
}

#[derive(Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct WorldChangesResult {
    pub temporary: bool,
    pub newly_removed: usize,
    pub restored: usize,
    pub removed: usize,
}
