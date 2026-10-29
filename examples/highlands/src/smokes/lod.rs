use engine::smoke::{SmokeResult, SmokeSession};
use engine::terrain::data::smokes::{
    COMPILE_FRAMES, Checked, above, ask, compiling, environment, number, present, refuse,
};
use engine::{smoke_check, smoke_eq};
use serde_json::{Value, json};

use super::{aim, open};

const STEADY_FRAMES: u32 = 600;

pub(super) fn run(session: &mut dyn SmokeSession) -> SmokeResult {
    open(session, false)?;
    environment(session, json!({"reflection_probes": {"enabled": false}}))?;
    described(session)?;
    aim(session, "overview")?;
    let finest = levels(session)?;
    budgets(session, &finest)
}

fn described(session: &mut dyn SmokeSession) -> Checked<()> {
    let listing = ask(session, "describe", json!({}))?;
    let listed = listing["types"]
        .as_array()
        .is_some_and(|types| types.iter().any(|kind| kind["name"] == "LevelOfDetail"));
    smoke_check!(
        listed,
        "the_level_of_detail_settings_are_discoverable",
        &listed,
        &true
    );
    let look = ask(session, "inspect", json!({"component": "Environment"}))?;
    let detail = &look["value"]["level_of_detail"];
    smoke_check!(
        detail["enabled"] == true
            && detail["pixel_error"] == 1.0
            && detail["impostors"] == true
            && detail["fade_seconds"] == 0.5,
        "the_game_keeps_the_default_detail",
        detail,
        &json!({"enabled": true, "pixel_error": 1.0, "impostors": true, "fade_seconds": 0.5})
    );
    Ok(())
}

fn levels(session: &mut dyn SmokeSession) -> Checked<Value> {
    let full = steady(session)?;
    let detail = &full["level_of_detail"];
    smoke_check!(
        coarser(&full) > 0.0,
        "coarser_levels_draw",
        detail,
        &json!({"lod1_instances": above(0)})
    );
    smoke_check!(
        number(&detail["impostors"]) > 0.0 && number(&detail["impostor_instances"]) > 0.0,
        "impostors_draw",
        detail,
        &json!({"impostors": above(0), "impostor_instances": above(0)})
    );
    smoke_eq!(
        &detail["dropped_impostors"],
        &json!(0),
        "no_impostor_dropped"
    );
    smoke_eq!(
        &detail["meshes_without_levels"],
        &json!(0),
        "every_mesh_has_levels"
    );
    environment(session, json!({"level_of_detail": {"impostors": false}}))?;
    let meshes = steady(session)?;
    smoke_check!(
        meshes["level_of_detail"]["impostor_instances"]
            .as_u64()
            .unwrap_or(0)
            == 0,
        "without_impostors_none_draw",
        &meshes["level_of_detail"],
        &json!({"impostor_instances": 0})
    );
    smoke_check!(
        coarser(&meshes) > 0.0,
        "mesh_levels_draw_alone",
        &meshes["level_of_detail"],
        &json!({"lod1_instances": above(0)})
    );
    let disabled = environment(session, json!({"level_of_detail": {"enabled": false}}))?;
    smoke_eq!(
        &disabled["environment_changed"],
        &json!(["level_of_detail"]),
        "the_level_of_detail_section_changes"
    );
    let exact = steady(session)?;
    smoke_check!(
        coarser(&exact) <= 0.0,
        "disabled_detail_draws_meshes_exactly",
        &exact["level_of_detail"],
        &json!({"lod1_instances": 0, "lod2_instances": 0, "impostor_instances": 0})
    );
    Ok(full)
}

fn budgets(session: &mut dyn SmokeSession, finest: &Value) -> Checked<()> {
    environment(session, json!({"level_of_detail": {"max_impostors": 2}}))?;
    let budget = steady(session)?;
    let detail = &budget["level_of_detail"];
    let impostors = number(&detail["impostors"]);
    smoke_check!(
        impostors > 0.0 && impostors <= 2.0 && number(&detail["dropped_impostors"]) > 0.0,
        "the_impostor_budget_holds",
        detail,
        &json!({"impostors": {"above": 0, "at_most": 2}, "dropped_impostors": above(0)})
    );
    environment(
        session,
        json!({"level_of_detail": {"enabled": true, "pixel_error": 4.0}}),
    )?;
    let coarse = steady(session)?;
    smoke_check!(
        coarser(&coarse) > coarser(finest),
        "pixel_error_4_draws_more_coarsely",
        &coarser(&coarse),
        &above(coarser(finest))
    );
    let invalid = refuse(
        session,
        "control",
        "environment",
        json!({"level_of_detail": {"pixel_error": 0.1}}),
    )?;
    smoke_eq!(
        &invalid["code"],
        &json!("invalid_environment"),
        "a_tiny_pixel_error_is_refused"
    );
    smoke_eq!(
        &invalid["path"],
        &json!("$.args.level_of_detail.pixel_error"),
        "a_tiny_pixel_error_is_refused_at"
    );
    environment(
        session,
        json!({"level_of_detail": {"enabled": true, "pixel_error": 1.0}}),
    )?;
    Ok(())
}

fn coarser(rendering: &Value) -> f64 {
    let detail = &rendering["level_of_detail"];
    ["lod1_instances", "lod2_instances", "impostor_instances"]
        .iter()
        .map(|field| detail[*field].as_f64().unwrap_or(0.0))
        .sum()
}

fn steady(session: &mut dyn SmokeSession) -> Checked<Value> {
    let mut rendering = present(session, 0.0)?;
    for _ in 0..COMPILE_FRAMES {
        if !compiling(&rendering) {
            break;
        }
        rendering = present(session, 0.0)?;
    }
    let (mut previous, mut stable) = (Value::Null, 0);
    for frame in 1..=STEADY_FRAMES {
        rendering = present(session, 0.0)?;
        let detail = &rendering["level_of_detail"];
        let instances = &rendering["occlusion_culling"]["instances"];
        let counts = json!([
            instances,
            detail["lod0_instances"],
            detail["lod1_instances"],
            detail["lod2_instances"],
            detail["impostor_instances"],
        ]);
        let age = rendering["read_back_age"].as_f64().unwrap_or(0.0);
        let current = !instances.is_null() && age < f64::from(frame);
        stable = if current && counts == previous {
            stable + 1
        } else {
            0
        };
        previous = if current { counts } else { Value::Null };
        if frame >= 8 && stable >= 5 {
            return Ok(rendering);
        }
    }
    smoke_check!(
        false,
        "read_back_counts_settle",
        &rendering["level_of_detail"],
        &json!({ "steady_frames": 5 })
    );
    Ok(rendering)
}
