use engine_core::smoke::{
    Checked, above, ask, at_most, below, camera, capture, component, control, environment, inspect,
    not, number, present, refuse, render, renders_until, with,
};
use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use serde_json::{Value, json};

use super::{
    FAST, SETTLE_FRAMES, WAIT_FRAMES, fast, fixture, heights, layer_sweep, open, opened, overview,
    settle, subdivided, view, viewed,
};
use crate::requests::seen;
use crate::stats::terrain;

pub(crate) fn run(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    let settled = open(session, false)?;
    streams(session, &settled)?;
    listed(session)?;
    surroundings(session)?;
    budget(session)?;
    detail(session)?;
    motion(session)?;
    memory(session)?;
    debug_views(session)?;
    freeze(session)?;
    prefetch(session)
}

fn streams(session: &mut dyn SmokeSession, settled: &Value) -> Checked<()> {
    let drawn = terrain(settled);
    let wanted = json!({"leaf_cut": {"kept": above(0)}, "triangles": above(0)});
    smoke_eq!(
        seen(
            number(&drawn["leaf_cut"]["kept"]) > 0.0 && number(&drawn["triangles"]) > 0.0,
            drawn.clone(),
            &wanted,
        ),
        wanted,
        "terrain_draws"
    );
    let shadows = inspect(session, "shadows")?;
    let finer = number(&shadows["finest_texel"]) / 2.0;
    environment(
        session,
        json!({"shadows": with(&shadows, json!({"finest_texel": finer}))}),
    )?;
    let (cast, casting) = renders_until(session, WAIT_FRAMES, |rendering| {
        number(&terrain(rendering)["shadow_patches"]) > 0.0
    })?;
    let wanted = json!({"shadow_patches": above(0)});
    smoke_eq!(
        seen(cast, terrain(&casting), &wanted),
        wanted,
        "the_terrain_casts_shadows"
    );
    environment(session, json!({ "shadows": shadows }))?;
    settle(session)?;
    smoke_eq!(&drawn["dropped_tiles"], &json!(0), "no_tile_dropped");
    smoke_eq!(&drawn["tile_cut"]["cut"], &json!(0), "no_tile_cut");
    smoke_eq!(
        &drawn["dropped_materials"],
        &json!(0),
        "no_material_dropped"
    );
    let placed = number(&drawn["placed_objects"]);
    let far = number(&drawn["placed_far_objects"]);
    let wanted = above(0);
    smoke_eq!(
        seen(placed > 0.0, json!(placed), &wanted),
        wanted,
        "crates_stand_on_the_meadow"
    );
    let wanted = json!({"above": 0, "below": placed});
    smoke_eq!(
        seen(far > 0.0 && far < placed, json!(far), &wanted),
        wanted,
        "far_crates_place"
    );
    let mut presented = Value::Null;
    for _ in 0..60 {
        presented = present(session, 0.0)?;
        if number(&terrain(&presented)["placed_impostors"]) > 0.0 {
            break;
        }
    }
    smoke_eq!(
        terrain(&presented)["placed_impostors"].clone(),
        json!(1),
        "the_ball_impostor_comes"
    );
    smoke_eq!(
        &presented["meshes_stats"]["level_of_detail"]["impostor_cut"]["cut"],
        &json!(0),
        "no_impostor_dropped"
    );
    Ok(())
}

fn listed(session: &mut dyn SmokeSession) -> Checked<()> {
    let listing = ask(session, "describe", json!({}))?;
    let names: Vec<&str> = listing["types"]
        .as_array()
        .map(|types| {
            types
                .iter()
                .filter_map(|kind| kind["name"].as_str())
                .collect()
        })
        .unwrap_or_default();
    for wanted in ["terrain", "terrain_stats", "terrain_shading"] {
        let shown = json!(wanted);
        smoke_eq!(
            seen(names.contains(&wanted), json!(names), &shown),
            shown,
            "describe_lists_terrain_types"
        );
    }
    let look = inspect(session, "Environment")?;
    smoke_eq!(
        &look["terrain"]["samples_per_frame"],
        &json!(FAST),
        "environment_reads_back"
    );
    let stepless = refuse(
        session,
        "control",
        "environment",
        json!({"terrain_shading": {"relief_steps": 1}}),
    )?;
    smoke_eq!(
        &stepless["path"],
        &json!("$.args.terrain_shading.relief_steps"),
        "one_relief_step_refused"
    );
    Ok(())
}

fn surroundings(session: &mut dyn SmokeSession) -> Checked<()> {
    let around = viewed(session)?["surroundings"].take();
    let wanted = json!({"reach": 3000, "spacing": 4, "resident_tiles": above(0)});
    smoke_eq!(
        seen(
            around["reach"].as_array().is_some_and(|reach| {
                !reach.is_empty() && reach.iter().all(|reach| number(reach) >= 3000.0)
            }) && number(&around["spacing"]) == 4.0
                && number(&around["resident_tiles"]) > 0.0,
            around.clone(),
            &wanted,
        ),
        wanted,
        "the_surroundings_stream"
    );
    let status = inspect(session, "terrain_stats")?;
    let wanted = json!({"terrains": 1, "surroundings_tiles": above(0)});
    smoke_eq!(
        seen(
            status["terrains"] == 1 && number(&status["surroundings_tiles"]) > 0.0,
            status.clone(),
            &wanted,
        ),
        wanted,
        "the_surroundings_draw"
    );
    edges(session)?;
    camera(
        session,
        with(
            &view([0.0, 1500.0, 1800.0], [0.0, 0.0, -2000.0]),
            json!({"far": 20000.0}),
        ),
    )?;
    let surrounded_frame = settle(session)?;
    let surrounded = capture(session)?;
    let edged = environment(
        session,
        json!({"terrain": with(&fast(), json!({"surroundings": false}))}),
    )?;
    smoke_eq!(
        &edged["environment_changed"],
        &json!(["terrain"]),
        "surroundings_setting_changes"
    );
    let alone = settle(session)?;
    let off = inspect(session, "terrain_stats")?;
    smoke_eq!(
        &off["surroundings_tiles"],
        &json!(0),
        "settings_turn_the_surroundings_off"
    );
    smoke_eq!(
        terrain(&alone)["placed_objects"].clone(),
        terrain(&surrounded_frame)["placed_objects"].clone(),
        "the_surroundings_place_nothing"
    );
    let bare = capture(session)?;
    let wanted = not(&surrounded);
    smoke_eq!(
        seen(surrounded != bare, bare.clone(), &wanted),
        wanted,
        "the_surroundings_fill_the_view"
    );
    environment(session, json!({ "terrain": fast() }))?;
    camera(session, overview())?;
    settle(session)?;
    Ok(())
}

fn edges(session: &mut dyn SmokeSession) -> Checked<()> {
    let mut pairs = Vec::new();
    for along in [-300.0, 0.0, 300.0] {
        pairs.extend([
            [[511.5, along], [512.5, along]],
            [[-511.5, along], [-512.5, along]],
            [[along, 511.5], [along, 512.5]],
            [[along, -511.5], [along, -512.5]],
        ]);
    }
    let points: Vec<[f64; 2]> = pairs.iter().flatten().copied().collect();
    let sampled = heights(session, &points)?;
    for (pair, sampled) in pairs.iter().zip(sampled.chunks(2)) {
        let met = sampled.iter().all(Value::is_object)
            && (number(&sampled[0]["height"]) - number(&sampled[1]["height"])).abs() < 3.0;
        let wanted = json!(pair);
        smoke_eq!(seen(met, json!(sampled), &wanted), wanted, "the_edge_meets");
    }
    let beyond = heights(session, &[[2500.0, 2500.0], [-3000.0, 1000.0]])?;
    let wanted = json!({"height": "finite"});
    smoke_eq!(
        seen(
            beyond
                .iter()
                .all(|sample| number(&sample["height"]).is_finite()),
            json!(beyond),
            &wanted,
        ),
        wanted,
        "queries_reach_past_the_edges"
    );
    Ok(())
}

fn budget(session: &mut dyn SmokeSession) -> Checked<()> {
    let limited = environment(
        session,
        json!({"terrain": with(&fast(), json!({"max_tiles": 3}))}),
    )?;
    let tiles = terrain(&limited["rendering"])["tiles"].clone();
    let wanted = at_most(3);
    smoke_eq!(
        seen(number(&tiles) <= 3.0, tiles, &wanted),
        wanted,
        "tiles_keep_within_their_budget"
    );
    let budgeted = render(session)?;
    let drawn = terrain(&budgeted);
    let wanted = json!({"tiles": at_most(3), "tile_cut": {"cut": above(0)}});
    smoke_eq!(
        seen(
            number(&drawn["tiles"]) <= 3.0 && number(&drawn["tile_cut"]["cut"]) > 0.0,
            drawn.clone(),
            &wanted,
        ),
        wanted,
        "the_budget_drops_tiles"
    );
    environment(session, json!({ "terrain": fast() }))?;
    settle(session)?;
    Ok(())
}

fn detail(session: &mut dyn SmokeSession) -> Checked<()> {
    let before = viewed(session)?["level_ranges"].take();
    let coarse = subdivided(session)?;
    let fine = environment(
        session,
        json!({
            "terrain": with(&fast(), json!({"pixel_error": 2})),
            "terrain_shading": {"subdivision_error": 0.5},
        }),
    )?;
    settle(session)?;
    let finer = subdivided(session)?;
    let after = viewed(session)?["level_ranges"].take();
    let wanted = above(&before[0]);
    smoke_eq!(
        seen(
            number(&after[0]) > number(&before[0]),
            after.clone(),
            &wanted
        ),
        wanted,
        "level_0_reaches_further"
    );
    let wanted = above(&terrain(&coarse)["triangles"]);
    smoke_eq!(
        seen(
            number(&terrain(&finer)["triangles"]) > number(&terrain(&coarse)["triangles"]),
            terrain(&finer)["triangles"].clone(),
            &wanted,
        ),
        wanted,
        "a_finer_error_draws_more_triangles"
    );
    smoke_eq!(
        &fine["environment_changed"],
        &json!(["terrain", "terrain_shading"]),
        "detail_changes_the_terrain_section"
    );
    let invalid = refuse(
        session,
        "control",
        "environment",
        json!({"terrain": {"pixel_error": 0.5}}),
    )?;
    smoke_eq!(
        &invalid["code"],
        &json!("invalid_environment"),
        "small_pixel_error_refused"
    );
    smoke_eq!(
        &invalid["path"],
        &json!("$.args.terrain.pixel_error"),
        "small_pixel_error_refused_at"
    );
    let shaded = capture(session)?;
    let off = environment(session, json!({"terrain": {"enabled": false}}))?;
    smoke_eq!(
        terrain(&off["rendering"])["leaf_cut"]["kept"].clone(),
        json!(0),
        "disabled_terrain_draws_nothing"
    );
    let empty = capture(session)?;
    let wanted = not(&shaded);
    smoke_eq!(
        seen(shaded != empty, empty.clone(), &wanted),
        wanted,
        "terrain_covers_the_view"
    );
    environment(session, json!({ "terrain": fast(), "terrain_shading": {} }))?;
    settle(session)?;
    Ok(())
}

fn motion(session: &mut dyn SmokeSession) -> Checked<()> {
    let moving = with(
        &fast(),
        json!({
            "collision_radius": 0,
            "motion_detail": {"speed": 1, "fastest": 2, "levels": 8, "settle": 0.25},
        }),
    );
    environment(session, json!({ "terrain": moving }))?;
    let mut waited = 0.0_f64;
    for step in 0..12_i32 {
        let along = -300.0 + f64::from(step) * 50.0;
        camera(
            session,
            view([along, 120.0, along], [along + 100.0, 30.0, along + 100.0]),
        )?;
        session.step(1);
        waited = waited.max(number(&inspect(session, "terrain_stats")?["waiting_tiles"]));
    }
    let wanted = above(0);
    smoke_eq!(
        seen(waited > 0.0, json!(waited), &wanted),
        wanted,
        "tiles_wait_while_the_camera_races"
    );
    for _ in 0..80 {
        session.step(1);
    }
    let rested = inspect(session, "terrain_stats")?;
    smoke_eq!(
        &rested["waiting_tiles"],
        &json!(0),
        "nothing_waits_once_the_camera_rests"
    );
    settle(session)?;
    let too_many = refuse(
        session,
        "control",
        "environment",
        json!({"terrain": with(&moving, json!({"motion_detail": {"levels": 9}}))}),
    )?;
    smoke_eq!(
        &too_many["code"],
        &json!("invalid_environment"),
        "nine_motion_levels_refused"
    );
    smoke_eq!(
        &too_many["path"],
        &json!("$.args.terrain.motion_detail.levels"),
        "nine_motion_levels_refused_at"
    );
    environment(session, json!({ "terrain": fast() }))?;
    camera(session, overview())?;
    settle(session)?;
    Ok(())
}

fn memory(session: &mut dyn SmokeSession) -> Checked<()> {
    let starved = json!({"terrain": with(&fast(), json!({"cache_bytes": 1}))});
    environment(session, starved.clone())?;
    render(session)?;
    environment(session, json!({ "terrain": fast() }))?;
    let home = inspect(session, "terrain_stats")?;
    camera(
        session,
        view([-420.0, 90.0, -420.0], [-300.0, 30.0, -300.0]),
    )?;
    settle(session)?;
    let layers = layer_sweep(session, [-380.0, -380.0], 100.0, 24)?;
    let wanted = above(100);
    smoke_eq!(
        seen(
            number(&layers["compared"]) > 100.0,
            layers["compared"].clone(),
            &wanted,
        ),
        wanted,
        "the_sweep_compares"
    );
    let wanted = json!({"max_difference": below(5e-3)});
    smoke_eq!(
        seen(
            number(&layers["worst"]["max_difference"]) < 5e-3,
            layers["worst"].clone(),
            &wanted,
        ),
        wanted,
        "tiles_match_their_layers"
    );
    camera(session, overview())?;
    settle(session)?;
    let back = inspect(session, "terrain_stats")?;
    let wanted = json!({"restored_tiles": above(&home["restored_tiles"])});
    smoke_eq!(
        seen(
            number(&back["restored_tiles"]) > number(&home["restored_tiles"]),
            back.clone(),
            &wanted,
        ),
        wanted,
        "tiles_return_from_memory"
    );
    environment(session, starved)?;
    render(session)?;
    let emptied = inspect(session, "terrain_stats")?;
    smoke_eq!(&emptied["cached_tiles"], &json!(0), "a_byte_keeps_no_tile");
    environment(session, json!({ "terrain": fast() }))?;
    settle(session)?;
    Ok(())
}

fn debug_views(session: &mut dyn SmokeSession) -> Checked<()> {
    let baseline = capture(session)?;
    let levels = environment(session, json!({"debug_view": {"view": "levels"}}))?;
    smoke_eq!(
        &levels["environment_changed"],
        &json!(["debug_view"]),
        "the_debug_view_changes"
    );
    let level_view = capture(session)?;
    let wanted = not(&baseline);
    smoke_eq!(
        seen(level_view != baseline, level_view.clone(), &wanted),
        wanted,
        "the_levels_view_differs"
    );
    environment(session, json!({"debug_view": {"view": "tiles"}}))?;
    capture(session)?;
    environment(session, json!({"debug_view": {"view": "materials"}}))?;
    let material_view = capture(session)?;
    let wanted = not(&baseline);
    smoke_eq!(
        seen(material_view != baseline, material_view.clone(), &wanted),
        wanted,
        "the_materials_view_differs"
    );
    let (arrived, mapped) = renders_until(session, SETTLE_FRAMES, |rendering| {
        terrain(rendering)["sky_maps"] == 2
    })?;
    let wanted = json!(2);
    smoke_eq!(
        seen(arrived, terrain(&mapped)["sky_maps"].clone(), &wanted),
        wanted,
        "the_sky_maps_arrive"
    );
    environment(session, json!({"debug_view": {"view": "sky"}}))?;
    let sky_view = capture(session)?;
    let wanted = not(&baseline);
    smoke_eq!(
        seen(sky_view != baseline, sky_view.clone(), &wanted),
        wanted,
        "the_sky_view_differs"
    );
    environment(session, json!({"debug_view": {"view": "off"}}))?;
    Ok(())
}

fn freeze(session: &mut dyn SmokeSession) -> Checked<()> {
    let frozen = control(session, "view", json!({"freeze": true}))?;
    let wanted = json!("array");
    smoke_eq!(
        seen(
            frozen["frozen_at"].is_array(),
            frozen["frozen_at"].clone(),
            &wanted,
        ),
        wanted,
        "freezing_holds_an_eye"
    );
    let read = control(session, "view", json!({}))?;
    smoke_eq!(
        &read["frozen_at"],
        &frozen["frozen_at"],
        "the_debug_section_reads_the_freeze"
    );
    camera(session, view([0.0, 900.0, 1400.0], [0.0, 30.0, 0.0]))?;
    let afar = subdivided(session)?;
    control(session, "view", json!({"freeze": false}))?;
    let moved = subdivided(session)?;
    let wanted = below(&terrain(&afar)["leaf_cut"]["kept"]);
    smoke_eq!(
        seen(
            number(&terrain(&moved)["leaf_cut"]["kept"])
                < number(&terrain(&afar)["leaf_cut"]["kept"]),
            terrain(&moved)["leaf_cut"]["kept"].clone(),
            &wanted,
        ),
        wanted,
        "unfrozen_the_subdivision_coarsens"
    );
    camera(session, overview())?;
    settle(session)?;
    Ok(())
}

fn prefetch(session: &mut dyn SmokeSession) -> Checked<()> {
    let spawn = [380.0, 90.0, 380.0];
    session.without(&[]);
    fixture::prefetched(session, spawn);
    opened(session)?;
    let reach = component(session, "spawn-point", "prefetch")?;
    let described = ask(session, "describe", json!({"name": "prefetch"}))?;
    let default = described["descriptor"]["schema"]["properties"]["reach"]["default"].clone();
    smoke_eq!(
        seen(
            default.is_number() && reach["reach"] == default,
            reach["reach"].clone(),
            &default,
        ),
        default,
        "the_place_inspects_its_reach"
    );
    let mut stored = inspect(session, "terrain_stats")?;
    for _ in 0..SETTLE_FRAMES {
        if number(&stored["cached_tiles"]) > 0.0 && stored["prefetching_tiles"] == 0 {
            break;
        }
        render(session)?;
        stored = inspect(session, "terrain_stats")?;
    }
    let wanted = json!({"cached_tiles": above(0), "prefetching_tiles": 0});
    smoke_eq!(
        seen(
            number(&stored["cached_tiles"]) > 0.0 && stored["prefetching_tiles"] == 0,
            stored.clone(),
            &wanted,
        ),
        wanted,
        "the_place_loads_into_memory"
    );
    camera(session, view(spawn, [300.0, 30.0, 300.0]))?;
    settle(session)?;
    let sent = inspect(session, "terrain_stats")?;
    let wanted = json!({"restored_tiles": above(&stored["restored_tiles"])});
    smoke_eq!(
        seen(
            number(&sent["restored_tiles"]) > number(&stored["restored_tiles"]),
            sent.clone(),
            &wanted,
        ),
        wanted,
        "the_place_returns_from_memory"
    );
    camera(session, overview())?;
    settle(session)?;
    Ok(())
}
