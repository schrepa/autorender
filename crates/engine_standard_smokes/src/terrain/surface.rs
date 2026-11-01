use engine_core::smoke::{
    COMPILE_FRAMES, Checked, above, camera, compiling, environment, inspect, not, number, present,
    refuse, render, renders_until, shot, with,
};
use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use serde_json::{Value, json};

use super::{SETTLE_FRAMES, WAIT_FRAMES, fast, open, overview, settle, view};
use crate::requests::seen;
use crate::stats::terrain;

pub(crate) fn run(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    let settled = open(session, false)?;
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
    camera(session, view([380.0, 90.0, -420.0], [300.0, 30.0, -300.0]))?;
    let mut controls = 0.0;
    for _ in 0..SETTLE_FRAMES {
        let rendering = render(session)?;
        let drawn = terrain(&rendering);
        controls += number(&drawn["gpu_controls"]);
        if controls > 0.0 && drawn["loading_tiles"] == 0 && drawn["fallback_tiles"] == 0 {
            break;
        }
    }
    let wanted = above(0);
    smoke_eq!(
        seen(controls > 0.0, json!(controls), &wanted),
        wanted,
        "the_gpu_gives_new_tiles_their_control_values"
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
        number(&rendering["ground_cover_stats"]["blade_cut"]["kept"]) > 0.0
    })?;
    let wanted = json!({"blade_cut": {"kept": above(0)}});
    smoke_eq!(
        seen(grew, grown["ground_cover_stats"].clone(), &wanted),
        wanted,
        "grass_grows"
    );
    smoke_eq!(
        &grown["ground_cover_stats"]["blade_cut"]["cut"],
        &json!(0),
        "no_blade_dropped"
    );
    let cover = inspect(session, "ground_cover")?;
    environment(
        session,
        json!({"ground_cover": with(&cover, json!({"enabled": false}))}),
    )?;
    let mut bare = Value::Null;
    for _ in 0..8 {
        bare = render(session)?;
    }
    smoke_eq!(
        &bare["ground_cover_stats"]["blade_cut"]["kept"],
        &json!(0),
        "no_blades_once_cover_stops"
    );
    environment(session, json!({ "ground_cover": cover }))?;
    Ok(())
}

fn wind(session: &mut dyn SmokeSession) -> Checked<()> {
    let calm = inspect(session, "wind")?;
    let breeze = json!({"velocity": [8.0, -2.0], "gusts": 0.25});
    environment(session, json!({ "wind": breeze }))?;
    let read = inspect(session, "wind")?;
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
        &json!("$.args.wind.velocity[0]"),
        "winds_past_40_m_s_are_refused_at"
    );
    environment(session, json!({ "wind": calm }))?;
    Ok(())
}

fn occupancy(session: &mut dyn SmokeSession, settled: &Value) -> Checked<()> {
    let wanted = json!({"redraws": above(0)});
    smoke_eq!(
        seen(
            number(&settled["scatter_stats"]["occupancy_redraws"]) > 0.0,
            json!({
                "redraws": settled["scatter_stats"]["occupancy_redraws"],
                "averages": settled["scatter_stats"]["occupancy_averages"],
            }),
            &wanted,
        ),
        wanted,
        "the_occupancy_map_draws"
    );
    let occupancy = inspect(session, "occupancy")?;
    environment(
        session,
        json!({"occupancy": with(&occupancy, json!({"strength": 0.5}))}),
    )?;
    let read = inspect(session, "occupancy")?;
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
    let clipmap = inspect(session, "shadows")?;
    let (sharp, sharp_view) = shadowed(session, clipmap.clone())?;
    smoke_eq!(
        &sharp["shadow_stats"]["tile_cut"]["cut"],
        &json!(0),
        "shadow_tiles_draw"
    );
    let (_, blurred_view) = shadowed(session, with(&clipmap, json!({"finest_texel": 0.5})))?;
    let wanted = not(&sharp_view);
    smoke_eq!(
        seen(sharp_view != blurred_view, blurred_view.clone(), &wanted),
        wanted,
        "finer_texels_change_the_view"
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
        let drawn = number(&rendering["shadow_stats"]["tiles_marked"]) > 0.0
            && rendering["shadow_stats"]["tile_cut"]["cut"] == 0;
        ready = frame > 8 && drawn && !compiling(&rendering);
        if ready {
            break;
        }
    }
    let wanted = json!({"tiles_marked": above(0), "tile_cut": {"cut": 0}});
    smoke_eq!(
        seen(ready, rendering["shadow_stats"].clone(), &wanted),
        wanted,
        "shadow_tiles_and_pipelines_finish"
    );
    let digest = shot(session, json!({"alpha": 1.0}))?;
    Ok((rendering, digest))
}
