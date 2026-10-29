use engine_core::smoke::{SmokeResult, SmokeSession};
use engine_core::{smoke_check, smoke_eq};
use serde_json::{Value, json};

use super::{
    COMPILE_FRAMES, Checked, SETTLE_FRAMES, WAIT_FRAMES, above, camera, compiling, environment,
    fast, inspect, not, number, open, overview, present, refuse, render, renders_until, settle,
    shot, view, with,
};

pub(super) fn run(session: &mut dyn SmokeSession) -> SmokeResult {
    let (_, settled) = open(session)?;
    materials(session)?;
    cover(session)?;
    wind(session)?;
    occupancy(session, &settled)?;
    clipmap(session)
}

fn materials(session: &mut dyn SmokeSession) -> Checked<()> {
    environment(
        session,
        json!({"terrain": with(&fast(), json!({"gpu_materials": false}))}),
    )?;
    settle(session)?;
    environment(session, json!({ "terrain": fast() }))?;
    settle(session)?;
    camera(
        session,
        view([380.0, 90.0, -420.0], [300.0, 30.0, -300.0]),
    )?;
    let mut controls = 0.0;
    for _ in 0..SETTLE_FRAMES {
        let rendering = render(session)?;
        let terrain = &rendering["terrain"];
        controls += number(&terrain["gpu_controls"]);
        if controls > 0.0 && terrain["loading_tiles"] == 0 && terrain["fallback_tiles"] == 0 {
            break;
        }
    }
    smoke_check!(
        controls > 0.0,
        "the_gpu_gives_new_tiles_their_control_values",
        &controls,
        &above(0)
    );
    camera(session, overview())?;
    settle(session)?;
    Ok(())
}

fn cover(session: &mut dyn SmokeSession) -> Checked<()> {
    camera(
        session,
        json!({
            "position": [-12.0, 32.0, -8.0],
            "target": [10.0, 30.0, 6.0],
            "vertical_fov_radians": 1.0,
            "near": 0.1,
            "far": 5000.0,
        }),
    )?;
    let (grew, grown) = renders_until(session, WAIT_FRAMES, |rendering| {
        number(&rendering["ground_cover"]["blades"]) > 0.0
    })?;
    smoke_check!(
        grew,
        "grass_grows",
        &grown["ground_cover"],
        &json!({"blades": above(0)})
    );
    smoke_eq!(
        &grown["ground_cover"]["dropped"],
        &json!(0),
        "no_blade_dropped"
    );
    let cover = inspect(session, "GroundCover")?;
    environment(
        session,
        json!({"ground_cover": with(&cover, json!({"enabled": false}))}),
    )?;
    let mut bare = Value::Null;
    for _ in 0..8 {
        bare = render(session)?;
    }
    smoke_eq!(
        &bare["ground_cover"]["blades"],
        &json!(0),
        "no_blades_once_cover_stops"
    );
    environment(session, json!({ "ground_cover": cover }))?;
    Ok(())
}

fn wind(session: &mut dyn SmokeSession) -> Checked<()> {
    let calm = inspect(session, "Wind")?;
    let breeze = json!({"velocity": [8.0, -2.0], "gusts": 0.25});
    environment(session, json!({ "wind": breeze }))?;
    let read = inspect(session, "Wind")?;
    smoke_eq!(&read, &breeze, "the_wind_reads_back");
    let gale = refuse(
        session,
        "control",
        "environment",
        json!({"wind": {"velocity": [80.0, 0.0], "gusts": 0.25}}),
    )?;
    smoke_eq!(
        &gale["code"],
        &json!("invalid_environment"),
        "winds_past_40_m_s_are_refused"
    );
    smoke_eq!(
        &gale["path"],
        &json!("$.args.wind.velocity"),
        "winds_past_40_m_s_are_refused_at"
    );
    environment(session, json!({ "wind": calm }))?;
    Ok(())
}

fn occupancy(session: &mut dyn SmokeSession, settled: &Value) -> Checked<()> {
    smoke_check!(
        number(&settled["occupancy"]["redraws"]) > 0.0,
        "the_occupancy_map_draws",
        &settled["occupancy"],
        &json!({"redraws": above(0)})
    );
    let occupancy = inspect(session, "Occupancy")?;
    environment(
        session,
        json!({"occupancy": with(&occupancy, json!({"strength": 0.5}))}),
    )?;
    let read = inspect(session, "Occupancy")?;
    smoke_eq!(&read["strength"], &json!(0.5), "occupancy_reads_back");
    let coarse = refuse(
        session,
        "control",
        "environment",
        json!({"occupancy": with(&occupancy, json!({"resolution": 16}))}),
    )?;
    smoke_eq!(
        &coarse["path"],
        &json!("$.args.occupancy.resolution"),
        "a_coarse_occupancy_map_is_refused"
    );
    environment(session, json!({ "occupancy": occupancy }))?;
    Ok(())
}

fn clipmap(session: &mut dyn SmokeSession) -> Checked<()> {
    let clipmap = inspect(session, "Shadows")?;
    let (sharp, sharp_view) = shadowed(session, clipmap.clone())?;
    smoke_eq!(
        &sharp["shadows"]["tiles_missing"],
        &json!(0),
        "shadow_tiles_draw"
    );
    let (_, blurred_view) = shadowed(session, with(&clipmap, json!({"finest_texel": 0.5})))?;
    smoke_check!(
        sharp_view != blurred_view,
        "finer_texels_change_the_view",
        &blurred_view,
        &not(&sharp_view)
    );
    let deep = refuse(
        session,
        "control",
        "environment",
        json!({"shadows": with(&clipmap, json!({"levels": 64}))}),
    )?;
    smoke_eq!(
        &deep["path"],
        &json!("$.args.shadows.levels"),
        "too_many_shadow_levels_are_refused"
    );
    environment(session, json!({ "shadows": clipmap }))?;
    camera(session, overview())?;
    Ok(())
}

fn shadowed(session: &mut dyn SmokeSession, shadows: Value) -> Checked<(Value, Value)> {
    environment(session, json!({ "shadows": shadows }))?;
    let mut rendering = Value::Null;
    let mut ready = false;
    for frame in 0..COMPILE_FRAMES {
        rendering = present(session, 1.0)?;
        let drawn = number(&rendering["shadows"]["tiles_marked"]) > 0.0
            && rendering["shadows"]["tiles_missing"] == 0;
        ready = frame > 8 && drawn && !compiling(&rendering);
        if ready {
            break;
        }
    }
    smoke_check!(
        ready,
        "shadow_tiles_and_pipelines_finish",
        &rendering["shadows"],
        &json!({"tiles_marked": above(0), "tiles_missing": 0})
    );
    let digest = shot(session, json!({"alpha": 1.0}))?;
    Ok((rendering, digest))
}
