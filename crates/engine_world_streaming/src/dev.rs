use bevy_ecs::prelude::*;
use engine_core::{Error, ErrorCode, Operation};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{STREAMING_GRIDS, WorldCells, WorldChanges};

pub const MAX_CHANGES: usize = 256;

#[derive(Serialize)]
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
pub struct StreamingGridView {
    pub cell: f32,
    pub range: f32,
    pub reach: Option<f32>,
    pub cells: u32,
    pub loaded: u32,
}

#[derive(Serialize)]
pub struct StreamedCellView {
    pub grid: u8,
    pub key: [i32; 2],
    pub items: u32,
    pub entities: u32,
    pub loaded: bool,
    pub distance: f32,
}

#[derive(Serialize)]
pub struct WorldChangesView {
    pub removed: usize,
    pub ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorldCellsQuery {}

impl Operation for WorldCellsQuery {
    const NAME: &'static str = "world_cells";
    type Params = Res<'static, WorldCells>;
    type Request = Self;

    fn run(cells: SystemParamItem<'_, '_, Self::Params>, _: Self) -> Result<Value, Error> {
        Ok(serde_json::to_value(world_cells(&cells)).expect("results_serialize"))
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorldChangesQuery {}

impl Operation for WorldChangesQuery {
    const NAME: &'static str = "world_removals";
    type Params = Res<'static, WorldChanges>;
    type Request = Self;

    fn run(changes: SystemParamItem<'_, '_, Self::Params>, _: Self) -> Result<Value, Error> {
        Ok(serde_json::to_value(WorldChangesView {
            removed: changes.removed().len(),
            ids: changes.removed().take(64).map(str::to_owned).collect(),
        })
        .expect("results_serialize"))
    }
}

fn world_cells(cells: &WorldCells) -> WorldCellsView {
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

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorldChangesControl {
    #[serde(default)]
    pub remove: Vec<String>,
    #[serde(default)]
    pub restore: Vec<String>,
    #[serde(default)]
    pub clear: bool,
}

impl Operation for WorldChangesControl {
    const NAME: &'static str = "world_changes";
    type Params = ResMut<'static, WorldChanges>;
    type Request = Self;

    fn run(
        mut changes: SystemParamItem<'_, '_, Self::Params>,
        request: Self,
    ) -> Result<Value, Error> {
        for (ids, path) in [
            (&request.remove, "$.remove"),
            (&request.restore, "$.restore"),
        ] {
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
        let held = changes.bypass_change_detection();
        let cleared = request.clear && held.clear();
        let removed = request.remove.iter().filter(|id| held.remove(id)).count();
        let restored = request.restore.iter().filter(|id| held.restore(id)).count();
        let total = held.removed().len();
        if cleared || removed > 0 || restored > 0 {
            changes.set_changed();
        }
        Ok(serde_json::to_value(WorldChangesResult {
            temporary: true,
            newly_removed: removed,
            restored,
            removed: total,
        })
        .expect("results_serialize"))
    }
}

#[derive(Serialize)]
pub struct WorldChangesResult {
    pub temporary: bool,
    pub newly_removed: usize,
    pub restored: usize,
    pub removed: usize,
}
