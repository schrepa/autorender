//! World streaming's development tools: the inspectors of streamed cells,
//! the changes to them and streamed entities' IDs.

use serde_json::{Value, json};

use crate::{DevTools, InspectorDescriptor};

/// Registers world streaming's inspectors.
pub(crate) fn register(tools: &mut DevTools, schemas: &Value) {
    tools.register_resource::<engine_core::WorldCells>(
        InspectorDescriptor::generated(schemas, "WorldCells"),
        |cells| {
            let stats = cells.stats();
            let grids = cells.grids();
            json!(crate::views::WorldCellsView {
                cells: stats.cells,
                loaded: stats.loaded,
                pending: stats.pending,
                items: stats.items,
                entities: stats.entities,
                spawned: stats.spawned,
                despawned: stats.despawned,
                velocity: cells.velocity().to_array(),
                grids: engine_core::STREAMING_GRIDS
                    .iter()
                    .zip(grids)
                    .map(|(grid, [count, loaded])| crate::views::StreamingGridView {
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
                    .map(|cell| crate::views::StreamedCellView {
                        grid: cell.grid,
                        key: cell.key,
                        items: cell.items,
                        entities: cell.entities,
                        loaded: cell.loaded,
                        distance: cell.distance,
                    })
                    .collect(),
            })
        },
    )
    .expect("built-in descriptor is bounded");
    tools.register_resource::<engine_core::WorldChanges>(
        InspectorDescriptor::generated(schemas, "WorldChanges"),
        |changes| {
            json!(crate::views::WorldChangesView {
                removed: changes.removed().len(),
                ids: changes.removed().take(64).map(str::to_owned).collect(),
            })
        },
    )
    .expect("built-in descriptor is bounded");
    tools.register_component::<engine_core::StreamedId>(
        InspectorDescriptor::generated(schemas, "StreamedId"),
        |id| {
            json!(crate::views::StreamedIdView {
                id: id.0.to_string()
            })
        },
    )
    .expect("built-in descriptor is bounded");
}

/// Adds the schemas of world streaming's inspectors.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn schemas(schemas: &mut serde_json::Map<String, Value>) {
    schemas.insert("WorldCells".into(), super::schema::<crate::views::WorldCellsView>());
    schemas.insert("WorldChanges".into(), super::schema::<crate::views::WorldChangesView>());
    schemas.insert("StreamedId".into(), super::schema::<crate::views::StreamedIdView>());
}
