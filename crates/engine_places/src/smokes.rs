use std::any::{Any, TypeId};

use engine_heightfield::layer::{LayerKind, NoFiles, TYPE_KEY, layer_from_json};
use engine_place::{Runs, Smoke, SmokeFailure, SmokeSession, ask, smoke_check, smoke_eq};
use serde_json::json;

use crate::terrain::TerrainLayers;

const fn frames(name: &'static str) -> Smoke {
    Smoke {
        name,
        runs: Runs::Gpu,
        run: drawn,
    }
}

fn drawn(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    smoke_eq!(session.frame(), true, "frame_drawn");
    Ok(())
}

pub const LIGHTING: Smoke = frames("slot_lighting");
pub const SUN: Smoke = frames("slot_sun");
pub const SHADOWS: Smoke = frames("slot_shadows");
pub const LIGHTS: Smoke = frames("slot_lights");
pub const SKY: Smoke = frames("slot_sky");
pub const EXPOSURE: Smoke = frames("slot_exposure");
pub const TONEMAP: Smoke = frames("slot_tonemap");
pub const BLOOM: Smoke = frames("slot_bloom");
pub const DITHER: Smoke = frames("slot_dither");
pub const REFLECTIONS: Smoke = frames("slot_reflections");
pub const PROBE_LIGHT: Smoke = frames("slot_probe_light");
pub const OCCUPANCY: Smoke = frames("slot_occupancy");
pub const PLACED_TILES: Smoke = frames("slot_placed_tiles");
pub const COVER: Smoke = frames("slot_cover");
pub const HORIZONS: Smoke = frames("slot_horizons");
pub const HEIGHTFIELD: Smoke = frames("slot_heightfield");
pub const GROUND_TILES: Smoke = frames("slot_ground_tiles");
pub const WATER: Smoke = frames("slot_water");
pub const INSTANCES: Smoke = frames("slot_instances");
pub const TRANSPARENCY: Smoke = frames("slot_transparency");

pub const TERRAIN_LAYERS: Smoke = Smoke {
    name: "list_terrain_layers",
    runs: Runs::Headless,
    run: layers,
};

const LAYER_PATH: &str = "$.layers[0]";

const LAYER_DEFINITION: &str = "SceneTerrainLayer";

fn joined(session: &mut dyn SmokeSession) -> Vec<&'static LayerKind> {
    let mut kinds = Vec::new();
    session.joined(TypeId::of::<TerrainLayers>(), &mut |member: &dyn Any| {
        if let Some(&kind) = member.downcast_ref::<&'static LayerKind>() {
            kinds.push(kind);
        }
    });
    kinds
}

fn built(kind: &'static LayerKind, kinds: &[&'static LayerKind]) -> Result<(), SmokeFailure> {
    let named = format!("{LAYER_PATH}.{TYPE_KEY}");
    match layer_from_json(
        &json!({ (TYPE_KEY): kind.name }),
        &NoFiles,
        LAYER_PATH,
        kinds,
    ) {
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
                error.path.starts_with(LAYER_PATH) && error.path != named,
                "the_kind_refuses_with_a_structured_error",
                &error,
                &json!({ "kind": kind.name, "path": { "under": LAYER_PATH, "not": named } })
            );
        }
    }
    Ok(())
}

fn layers(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    let kinds = joined(session);
    if kinds.is_empty() {
        return Ok(());
    }
    let described = ask(
        session,
        "describe",
        json!({ "name": "SceneDocument", "definition": LAYER_DEFINITION }),
    )?;
    let listed: Vec<&str> = described["schema"]["oneOf"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|variant| variant["properties"][TYPE_KEY]["const"].as_str())
        .collect();
    for &kind in &kinds {
        let schema = (kind.schema)();
        smoke_check!(
            !schema.is_null(),
            "the_kind_has_a_schema",
            &kind.name,
            &json!({ "schema": "not null" })
        );
        smoke_check!(
            listed.contains(&kind.name),
            "the_scene_schema_lists_the_kind",
            &kind.name,
            &listed
        );
        built(kind, &kinds)?;
    }
    Ok(())
}
