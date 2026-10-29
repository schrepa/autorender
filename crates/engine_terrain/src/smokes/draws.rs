use bevy_ecs::prelude::Entity;
use engine_core::smoke::{SmokeResult, SmokeSession};
use engine_core::{Transform, smoke_check, smoke_eq};
use glam::{Vec2, Vec3};
use serde_json::{Value, json};

use super::{
    Checked, FAST, SETTLE_FRAMES, WAIT_FRAMES, above, ask, at_most, below, camera, capture,
    component, control, environment, fast, heights, inspect, layer_sweep, name, not, number, open,
    overview, present, refuse, render, renders_until, seen, settle, subdivided, view, with,
};
use crate::TerrainPrefetch;

pub(super) fn run(session: &mut dyn SmokeSession) -> SmokeResult {
    let (fixture, settled) = open(session)?;
    streams(session, &settled)?;
    listed(session)?;
    surroundings(session, fixture.ground)?;
    budget(session)?;
    detail(session, fixture.ground)?;
    motion(session)?;
    memory(session)?;
    debug_views(session)?;
    freeze(session)?;
    prefetch(session)
}

fn streams(session: &mut dyn SmokeSession, settled: &Value) -> Checked<()> {
    let terrain = &settled["terrain"];
    smoke_check!(
        number(&terrain["leaves"]) > 0.0 && number(&terrain["triangles"]) > 0.0,
        "terrain_draws",
        terrain,
        &json!({"leaves": above(0), "triangles": above(0)})
    );
    let shadows = inspect(session, "Shadows")?;
    let finer = number(&shadows["finest_texel"]) / 2.0;
    environment(
        session,
        json!({"shadows": with(&shadows, json!({"finest_texel": finer}))}),
    )?;
    let (cast, casting) = renders_until(session, WAIT_FRAMES, |rendering| {
        number(&rendering["terrain"]["shadow_patches"]) > 0.0
    })?;
    smoke_check!(
        cast,
        "the_terrain_casts_shadows",
        &casting["terrain"],
        &json!({"shadow_patches": above(0)})
    );
    environment(session, json!({ "shadows": shadows }))?;
    settle(session)?;
    smoke_eq!(&terrain["dropped_tiles"], &json!(0), "no_tile_dropped");
    smoke_eq!(
        &terrain["dropped_materials"],
        &json!(0),
        "no_material_dropped"
    );
    let placed = number(&terrain["placed_objects"]);
    let far = number(&terrain["placed_far_objects"]);
    smoke_check!(
        placed > 0.0,
        "crates_stand_on_the_meadow",
        &placed,
        &above(0)
    );
    smoke_check!(
        far > 0.0 && far < placed,
        "far_crates_place",
        &far,
        &json!({"above": 0, "below": placed})
    );
    let mut presented = Value::Null;
    for _ in 0..60 {
        presented = present(session, 0.0)?;
        if number(&presented["terrain"]["placed_impostors"]) > 0.0 {
            break;
        }
    }
    smoke_eq!(
        &presented["terrain"]["placed_impostors"],
        &json!(1),
        "the_ball_impostor_comes"
    );
    smoke_eq!(
        &presented["level_of_detail"]["dropped_impostors"],
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
    for wanted in ["Terrain", "Terrains", "TerrainSettings"] {
        smoke_check!(
            names.contains(&wanted),
            "describe_lists_terrain_types",
            &names,
            &wanted
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
        json!({"terrain": {"relief_steps": 1}}),
    )?;
    smoke_eq!(
        &stepless["path"],
        &json!("$.args.terrain.relief_steps"),
        "one_relief_step_refused"
    );
    Ok(())
}

fn surroundings(session: &mut dyn SmokeSession, ground: Entity) -> Checked<()> {
    let around = seen(session, ground).surroundings;
    smoke_check!(
        around.is_some_and(|around| {
            around.reach.iter().all(|&reach| reach >= 3000.0)
                && around.spacing == 4.0
                && around.resident_tiles > 0
        }),
        "the_surroundings_stream",
        &around,
        &json!({"reach": 3000, "spacing": 4, "resident_tiles": above(0)})
    );
    let status = inspect(session, "Terrains")?;
    smoke_check!(
        status["terrains"] == 1 && number(&status["surroundings_tiles"]) > 0.0,
        "the_surroundings_draw",
        &status,
        &json!({"terrains": 1, "surroundings_tiles": above(0)})
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
    let off = inspect(session, "Terrains")?;
    smoke_eq!(
        &off["surroundings_tiles"],
        &json!(0),
        "settings_turn_the_surroundings_off"
    );
    smoke_eq!(
        &alone["terrain"]["placed_objects"],
        &surrounded_frame["terrain"]["placed_objects"],
        "the_surroundings_place_nothing"
    );
    let bare = capture(session)?;
    smoke_check!(
        surrounded != bare,
        "the_surroundings_fill_the_view",
        &bare,
        &not(&surrounded)
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
        smoke_check!(met, "the_edge_meets", &sampled, pair);
    }
    let beyond = heights(session, &[[2500.0, 2500.0], [-3000.0, 1000.0]])?;
    smoke_check!(
        beyond
            .iter()
            .all(|sample| number(&sample["height"]).is_finite()),
        "queries_reach_past_the_edges",
        &beyond,
        &json!({"height": "finite"})
    );
    Ok(())
}

fn budget(session: &mut dyn SmokeSession) -> Checked<()> {
    let limited = environment(
        session,
        json!({"terrain": with(&fast(), json!({"max_tiles": 3}))}),
    )?;
    let tiles = &limited["rendering"]["terrain"]["tiles"];
    smoke_check!(
        number(tiles) <= 3.0,
        "tiles_keep_within_their_budget",
        tiles,
        &at_most(3)
    );
    let budgeted = render(session)?;
    let terrain = &budgeted["terrain"];
    smoke_check!(
        number(&terrain["tiles"]) <= 3.0 && number(&terrain["dropped_tiles"]) > 0.0,
        "the_budget_drops_tiles",
        terrain,
        &json!({"tiles": at_most(3), "dropped_tiles": above(0)})
    );
    environment(session, json!({ "terrain": fast() }))?;
    settle(session)?;
    Ok(())
}

fn detail(session: &mut dyn SmokeSession, ground: Entity) -> Checked<()> {
    let before = seen(session, ground).ranges;
    let coarse = subdivided(session)?;
    let fine = environment(
        session,
        json!({"terrain": with(&fast(), json!({"pixel_error": 2, "subdivision_error": 0.5}))}),
    )?;
    settle(session)?;
    let finer = subdivided(session)?;
    let after = seen(session, ground).ranges;
    smoke_check!(
        after.first() > before.first(),
        "level_0_reaches_further",
        &after,
        &above(before.first())
    );
    smoke_check!(
        number(&finer["terrain"]["triangles"]) > number(&coarse["terrain"]["triangles"]),
        "a_finer_error_draws_more_triangles",
        &finer["terrain"]["triangles"],
        &above(&coarse["terrain"]["triangles"])
    );
    smoke_eq!(
        &fine["environment_changed"],
        &json!(["terrain"]),
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
        &off["rendering"]["terrain"]["leaves"],
        &json!(0),
        "disabled_terrain_draws_nothing"
    );
    let empty = capture(session)?;
    smoke_check!(
        shaded != empty,
        "terrain_covers_the_view",
        &empty,
        &not(&shaded)
    );
    environment(session, json!({ "terrain": fast() }))?;
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
        waited = waited.max(number(&inspect(session, "Terrains")?["waiting_tiles"]));
    }
    smoke_check!(
        waited > 0.0,
        "tiles_wait_while_the_camera_races",
        &waited,
        &above(0)
    );
    for _ in 0..80 {
        session.step(1);
    }
    let rested = inspect(session, "Terrains")?;
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
    let home = inspect(session, "Terrains")?;
    camera(
        session,
        view([-420.0, 90.0, -420.0], [-300.0, 30.0, -300.0]),
    )?;
    settle(session)?;
    let layers = &layer_sweep(session, Vec2::new(-380.0, -380.0), 100.0, 24);
    smoke_check!(
        number(&layers["compared"]) > 100.0,
        "the_sweep_compares",
        &layers["compared"],
        &above(100)
    );
    smoke_check!(
        number(&layers["worst"]["max_difference"]) < 5e-3,
        "tiles_match_their_layers",
        &layers["worst"],
        &json!({"max_difference": below(5e-3)})
    );
    camera(session, overview())?;
    settle(session)?;
    let back = inspect(session, "Terrains")?;
    smoke_check!(
        number(&back["restored_tiles"]) > number(&home["restored_tiles"]),
        "tiles_return_from_memory",
        &back,
        &json!({"restored_tiles": above(&home["restored_tiles"])})
    );
    environment(session, starved)?;
    render(session)?;
    let emptied = inspect(session, "Terrains")?;
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
    smoke_check!(
        level_view != baseline,
        "the_levels_view_differs",
        &level_view,
        &not(&baseline)
    );
    environment(session, json!({"debug_view": {"view": "tiles"}}))?;
    capture(session)?;
    environment(session, json!({"debug_view": {"view": "materials"}}))?;
    let material_view = capture(session)?;
    smoke_check!(
        material_view != baseline,
        "the_materials_view_differs",
        &material_view,
        &not(&baseline)
    );
    let (arrived, mapped) = renders_until(session, SETTLE_FRAMES, |rendering| {
        rendering["terrain"]["sky_maps"] == 2
    })?;
    smoke_check!(
        arrived,
        "the_sky_maps_arrive",
        &mapped["terrain"]["sky_maps"],
        &2
    );
    environment(session, json!({"debug_view": {"view": "sky"}}))?;
    let sky_view = capture(session)?;
    smoke_check!(
        sky_view != baseline,
        "the_sky_view_differs",
        &sky_view,
        &not(&baseline)
    );
    environment(session, json!({"debug_view": {"view": "off"}}))?;
    Ok(())
}

fn freeze(session: &mut dyn SmokeSession) -> Checked<()> {
    let frozen = control(session, "terrain", json!({"freeze": true}))?;
    smoke_check!(
        frozen["frozen_at"].is_array(),
        "freezing_holds_an_eye",
        &frozen["frozen_at"],
        &json!("array")
    );
    let debug = inspect(session, "TerrainDebug")?;
    smoke_eq!(
        &debug,
        &json!({"freeze": frozen["frozen_at"]}),
        "the_debug_section_reads_the_freeze"
    );
    camera(session, view([0.0, 900.0, 1400.0], [0.0, 30.0, 0.0]))?;
    let afar = subdivided(session)?;
    control(session, "terrain", json!({"freeze": false}))?;
    let moved = subdivided(session)?;
    smoke_check!(
        number(&moved["terrain"]["leaves"]) < number(&afar["terrain"]["leaves"]),
        "unfrozen_the_subdivision_coarsens",
        &moved["terrain"]["leaves"],
        &below(&afar["terrain"]["leaves"])
    );
    camera(session, overview())?;
    settle(session)?;
    Ok(())
}

fn prefetch(session: &mut dyn SmokeSession) -> Checked<()> {
    let spawn = [380.0, 90.0, 380.0];
    session.restart(&[]);
    open(session)?;
    {
        let world = session.app().world_mut();
        let place = world
            .spawn((
                Transform {
                    translation: Vec3::from_array(spawn.map(|meters| meters as f32)),
                    ..Transform::IDENTITY
                },
                TerrainPrefetch::default(),
            ))
            .id();
        name(world, "spawn-point", place);
    }
    let reach = component(session, "spawn-point", "TerrainPrefetch")?;
    smoke_eq!(
        &reach["reach"],
        &json!(TerrainPrefetch::default().reach),
        "the_place_inspects_its_reach"
    );
    let mut stored = inspect(session, "Terrains")?;
    for _ in 0..SETTLE_FRAMES {
        if number(&stored["cached_tiles"]) > 0.0 && stored["prefetching_tiles"] == 0 {
            break;
        }
        render(session)?;
        stored = inspect(session, "Terrains")?;
    }
    smoke_check!(
        number(&stored["cached_tiles"]) > 0.0 && stored["prefetching_tiles"] == 0,
        "the_place_loads_into_memory",
        &stored,
        &json!({"cached_tiles": above(0), "prefetching_tiles": 0})
    );
    camera(session, view(spawn, [300.0, 30.0, 300.0]))?;
    settle(session)?;
    let sent = inspect(session, "Terrains")?;
    smoke_check!(
        number(&sent["restored_tiles"]) > number(&stored["restored_tiles"]),
        "the_place_returns_from_memory",
        &sent,
        &json!({"restored_tiles": above(&stored["restored_tiles"])})
    );
    camera(session, overview())?;
    settle(session)?;
    Ok(())
}
