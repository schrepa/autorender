use engine_core::smoke::{SmokeResult, SmokeSession, ask};
use engine_core::smoke_check;
use serde_json::json;

use crate::layer::{LayerKind, NoFiles, TYPE_KEY, TerrainLayerKinds, layer_from_json};

const PATH: &str = "$.layers[0]";

const DEFINITION: &str = "SceneTerrainLayer";

fn built(kind: &'static LayerKind) -> SmokeResult {
    let named = format!("{PATH}.{TYPE_KEY}");
    match layer_from_json(&json!({ (TYPE_KEY): kind.name }), &NoFiles, PATH) {
        Ok(layer) => {
            smoke_check!(
                layer.kind().name == kind.name,
                "the_kind_builds_its_own_layer",
                &layer.kind().name,
                &kind.name
            );
        }
        Err(error) => {
            smoke_check!(
                error.path.starts_with(PATH) && error.path != named,
                "the_kind_refuses_with_a_structured_error",
                &error,
                &json!({ "kind": kind.name, "path": { "under": PATH, "not": named } })
            );
        }
    }
    Ok(())
}

pub(crate) fn layers(session: &mut dyn SmokeSession) -> SmokeResult {
    let kinds: Vec<&'static LayerKind> = session
        .app()
        .world()
        .get_resource::<TerrainLayerKinds>()
        .map(|known| known.kinds().to_vec())
        .unwrap_or_default();
    if kinds.is_empty() {
        return Ok(());
    }
    let described = ask(
        session,
        "describe",
        json!({ "name": "SceneDocument", "definition": DEFINITION }),
    )?;
    let listed: Vec<&str> = described["schema"]["oneOf"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|variant| variant["properties"][TYPE_KEY]["const"].as_str())
        .collect();
    for kind in kinds {
        if cfg!(feature = "schema") {
            let schema = (kind.schema)();
            smoke_check!(
                !schema.is_null(),
                "the_kind_has_a_schema",
                &kind.name,
                &json!({ "schema": "not null" })
            );
        }
        smoke_check!(
            listed.contains(&kind.name),
            "the_scene_schema_lists_the_kind",
            &kind.name,
            &listed
        );
        built(kind)?;
    }
    Ok(())
}
