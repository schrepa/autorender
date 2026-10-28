use engine_core::{Engine, Feature};
use engine_scene::SceneInstance;
use engine_settings::ErrorCode;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::{EngineControl, feature_mut};
use crate::protocol::bounds;
use crate::results::result;
use crate::{DevError, DevTools, InspectorDescriptor};

pub(crate) fn register(tools: &mut DevTools, schemas: &Value) {
    tools
        .register_resource::<engine_core::WorldCells>(
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
    tools
        .register_resource::<engine_core::WorldChanges>(
            InspectorDescriptor::generated(schemas, "WorldChanges"),
            |changes| {
                json!(crate::views::WorldChangesView {
                    removed: changes.removed().len(),
                    ids: changes.removed().take(64).map(str::to_owned).collect(),
                })
            },
        )
        .expect("built-in descriptor is bounded");
    tools
        .register_component::<engine_core::StreamedId>(
            InspectorDescriptor::generated(schemas, "StreamedId"),
            |id| {
                json!(crate::views::StreamedIdView {
                    id: id.0.to_string()
                })
            },
        )
        .expect("built-in descriptor is bounded");
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn schemas(schemas: &mut serde_json::Map<String, Value>) {
    schemas.insert(
        "WorldCells".into(),
        super::schema::<crate::views::WorldCellsView>(),
    );
    schemas.insert(
        "WorldChanges".into(),
        super::schema::<crate::views::WorldChangesView>(),
    );
    schemas.insert(
        "StreamedId".into(),
        super::schema::<crate::views::StreamedIdView>(),
    );
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct WorldChangesControl {
    #[serde(default)]
    #[cfg_attr(not(target_arch = "wasm32"), schemars(length(max = bounds::MAX_CHANGES)))]
    pub remove: Vec<String>,
    #[serde(default)]
    #[cfg_attr(not(target_arch = "wasm32"), schemars(length(max = bounds::MAX_CHANGES)))]
    pub restore: Vec<String>,
    #[serde(default)]
    pub clear: bool,
}

impl EngineControl for WorldChangesControl {
    fn changed(&self) -> &'static str {
        "world changed"
    }

    fn apply(&self, engine: &mut Engine, _: Option<&SceneInstance>) -> Result<Value, DevError> {
        change_world(engine, &self.remove, &self.restore, self.clear)
    }
}

fn change_world(
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
    Ok(result::world_changes(WorldChangesResult {
        temporary: true,
        newly_removed: removed,
        restored,
        removed: changes.removed().len(),
    }))
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct WorldChangesResult {
    pub temporary: bool,
    pub newly_removed: usize,
    pub restored: usize,
    pub removed: usize,
}
