use engine_core::smoke::{
    Checked, above, at_most, below, control, inspect, near, number, query, refuse, render, with,
};
use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use serde_json::{Value, json};

use super::{SETTLE_FRAMES, height_at, heights, layer_sweep, open, raycast, viewed};
use crate::requests::seen;

const ROAD: [[f64; 2]; 4] = [
    [100.0, 300.0],
    [250.0, 260.0],
    [400.0, 150.0],
    [450.0, 60.0],
];

const SIDE: usize = 513;

pub(crate) fn run(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    open(session, false)?;
    inspection(session)?;
    erosion(session)?;
    path(session)?;
    bridge(session)?;
    tunnel(session)?;
    route(session)?;
    samples(session)?;
    collision(session)?;
    edits(session)?;
    rivers(session)?;
    authored(session)
}

fn ids(listed: &Value) -> Vec<String> {
    listed
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|entry| entry["id"].as_str().map(str::to_owned))
        .collect()
}

fn inspection(session: &mut dyn SmokeSession) -> Checked<()> {
    let state = viewed(session)?;
    let layers = state["terrain"]["layers"].as_array().map_or(0, Vec::len);
    smoke_eq!(&layers, &10_usize, "the_terrain_keeps_its_layers");
    smoke_eq!(
        &state["extent"],
        &json!([1024.0, 1024.0]),
        "the_tiles_span_the_extent"
    );
    let resident: f64 = state["resident_tiles_by_level"]
        .as_array()
        .into_iter()
        .flatten()
        .map(number)
        .sum();
    let wanted = above(0);
    smoke_eq!(
        seen(resident > 0.0, json!(resident), &wanted),
        wanted,
        "tiles_reside"
    );
    let errors: Vec<f64> = state["level_errors"]
        .as_array()
        .into_iter()
        .flatten()
        .map(number)
        .collect();
    let wanted = above(0);
    smoke_eq!(
        seen(
            errors
                .split_last()
                .is_some_and(|(_, rest)| rest.iter().all(|&error| error > 0.0)),
            json!(errors),
            &wanted,
        ),
        wanted,
        "levels_measure_their_error"
    );
    let ranges: Vec<f64> = state["level_ranges"]
        .as_array()
        .into_iter()
        .flatten()
        .map(number)
        .collect();
    let wanted = json!("rising");
    smoke_eq!(
        seen(
            ranges.windows(2).all(|pair| pair[1] > pair[0]),
            json!(ranges),
            &wanted,
        ),
        wanted,
        "level_ranges_rise"
    );
    Ok(())
}

fn erosion(session: &mut dyn SmokeSession) -> Checked<()> {
    let bakes = viewed(session)?;
    smoke_eq!(
        &bakes["baking"],
        &json!(false),
        "erosion_is_done_once_tiles_load"
    );
    let worn = &bakes["erosion"];
    let wanted = json!({ "samples": SIDE * SIDE });
    smoke_eq!(
        seen(
            number(&worn["samples"]) == (SIDE * SIDE) as f64,
            worn.clone(),
            &wanted,
        ),
        wanted,
        "erosion_reads_its_level"
    );
    let wanted = json!({"rounds": 6, "deepest": above(0)});
    smoke_eq!(
        seen(
            number(&worn["rounds"]) == 6.0 && number(&worn["deepest"]) > 0.0,
            worn.clone(),
            &wanted,
        ),
        wanted,
        "erosion_wears_ground_away"
    );
    Ok(())
}

fn path(session: &mut dyn SmokeSession) -> Checked<()> {
    let paths = viewed(session)?["paths"].take();
    let road = &paths[0];
    let wanted = json!({"layer": 4, "baked": true});
    smoke_eq!(
        seen(
            road["layer"] == 4 && road["baked"] == true,
            paths.clone(),
            &wanted,
        ),
        wanted,
        "the_path_bakes"
    );
    let length = ROAD
        .windows(2)
        .map(|pair| (pair[1][0] - pair[0][0]).hypot(pair[1][1] - pair[0][1]))
        .sum::<f64>();
    let samples = (length / 4.0).ceil() as usize + 1;
    let wanted = json!({"length": length, "samples": samples});
    smoke_eq!(
        seen(
            (number(&road["length"]) - length).abs() < 1e-2
                && road["samples"].as_u64() == Some(samples as u64),
            road.clone(),
            &wanted,
        ),
        wanted,
        "the_profile_reads_its_line"
    );
    let steepest = number(&road["steepest"]);
    let wanted = json!({"above": 0, "below": 0.6});
    smoke_eq!(
        seen(steepest > 0.0 && steepest < 0.6, json!(steepest), &wanted),
        wanted,
        "the_road_climbs_gently"
    );
    let mut across = Vec::new();
    for along in [0.25, 0.5, 0.75] {
        for pair in ROAD[..3].windows(2) {
            let (dx, dz) = (pair[1][0] - pair[0][0], pair[1][1] - pair[0][1]);
            let run = dx.hypot(dz);
            let (nx, nz) = (-dz / run, dx / run);
            let center = [
                pair[0][0] + dx * along - 512.0,
                pair[0][1] + dz * along - 512.0,
            ];
            across.extend([
                center,
                [center[0] + 2.0 * nx, center[1] + 2.0 * nz],
                [center[0] - 2.0 * nx, center[1] - 2.0 * nz],
            ]);
        }
    }
    let sections = heights(session, &across)?;
    let (mut tilt, mut level) = (0.0_f64, 0);
    for section in sections.chunks(3) {
        let [center, left, right] = section else {
            continue;
        };
        if center["path"].is_null() {
            continue;
        }
        let wanted = near(-3, 0.05);
        smoke_eq!(
            seen(
                (number(&center["path"]) + 3.0).abs() < 0.05,
                center.clone(),
                &wanted,
            ),
            wanted,
            "the_centerline_lies_3_m_inside_the_edges"
        );
        tilt = tilt.max((number(&left["height"]) - number(&right["height"])).abs());
        level += 1;
    }
    let wanted = json!({"tilt": below(0.15), "sections": {"at_least": 3}});
    smoke_eq!(
        seen(
            level >= 3 && tilt < 0.15,
            json!({"tilt": tilt, "sections": level}),
            &wanted,
        ),
        wanted,
        "the_road_lies_level_across"
    );
    let away = height_at(session, [400.0, -400.0])?;
    let wanted = json!({ "path": null });
    smoke_eq!(
        seen(away["path"].is_null(), away.clone(), &wanted),
        wanted,
        "far_from_roads_queries_report_none"
    );
    Ok(())
}

fn bridge(session: &mut dyn SmokeSession) -> Checked<()> {
    let paths = viewed(session)?["paths"].take();
    let baked: Vec<Value> = paths
        .as_array()
        .into_iter()
        .flatten()
        .filter(|path| path["baked"] == true)
        .cloned()
        .collect();
    let road = baked.first().cloned().unwrap_or(Value::Null);
    let wanted = json!({"bridges": {"at_least": 1}, "bridged": above(15)});
    smoke_eq!(
        seen(
            road["bridges"].as_u64().is_some_and(|bridges| bridges >= 1)
                && number(&road["bridged"]) > 15.0,
            road.clone(),
            &wanted,
        ),
        wanted,
        "the_road_bridges_the_trench"
    );
    let bridges: u64 = baked
        .iter()
        .filter_map(|stats| stats["bridges"].as_u64())
        .sum();
    let terrains = inspect(session, "terrain_stats")?;
    smoke_eq!(
        &terrains["bridges"],
        &json!(bridges),
        "the_terrains_count_their_bridges"
    );
    let trench = [430.0 - 512.0, 96.0 - 512.0];
    let floor = height_at(session, trench)?;
    let wanted = json!({ "path": null });
    smoke_eq!(
        seen(floor["path"].is_null(), floor.clone(), &wanted),
        wanted,
        "under_its_bridge_the_ground_is_no_road"
    );
    let landing = raycast(session, [trench[0], 300.0, trench[1]], 600.0)?;
    let hit = &landing["hit"];
    let wanted = json!({"placed": {"layer": 65535}, "point": above(&floor["height"])});
    smoke_eq!(
        seen(
            hit["placed"]["layer"] == 65535
                && number(&hit["point"][1]) > number(&floor["height"]) + 5.0,
            hit.clone(),
            &wanted,
        ),
        wanted,
        "rays_land_on_the_deck"
    );
    Ok(())
}

fn tunnel(session: &mut dyn SmokeSession) -> Checked<()> {
    let paths = viewed(session)?["paths"].take();
    let bore = &paths[1];
    let wanted = json!({"layer": 6, "tunnels": 1, "tunneled": above(40)});
    smoke_eq!(
        seen(
            bore["layer"] == 6
                && bore["baked"] == true
                && bore["tunnels"] == 1
                && number(&bore["tunneled"]) > 40.0,
            bore.clone(),
            &wanted,
        ),
        wanted,
        "the_road_tunnels_the_mound"
    );
    let terrains = inspect(session, "terrain_stats")?;
    smoke_eq!(
        &terrains["tunnels"],
        &json!(1),
        "the_terrains_count_the_tunnel"
    );
    let summit = [750.0 - 512.0, 200.0 - 512.0];
    let mound = height_at(session, summit)?;
    let wanted = json!({"path": null, "hole": false});
    smoke_eq!(
        seen(
            mound["path"].is_null() && mound["hole"] == false,
            mound.clone(),
            &wanted,
        ),
        wanted,
        "above_the_tunnel_the_ground_is_no_road"
    );
    let top = raycast(session, [summit[0], 400.0, summit[1]], 600.0)?;
    let hit = &top["hit"];
    let height = number(&mound["height"]);
    let landed = if hit["placed"].is_object() {
        hit["placed"]["layer"] == 65535 && number(&hit["point"][1]) < height - 10.0
    } else {
        (number(&hit["point"][1]) - height).abs() < 0.05
    };
    let wanted = json!({ "height": height });
    smoke_eq!(
        seen(landed, hit.clone(), &wanted),
        wanted,
        "rays_land_on_the_mound_or_the_floor_under_it"
    );
    let bored: Vec<[f64; 2]> = (0..=100)
        .map(|step| [f64::from(600 + 3 * step) - 512.0, summit[1]])
        .collect();
    let portals = heights(session, &bored)?
        .iter()
        .filter(|sample| sample["hole"] == true)
        .count();
    let wanted = above(0);
    smoke_eq!(
        seen(portals > 0, json!(portals), &wanted),
        wanted,
        "the_ground_opens_at_its_portals"
    );
    Ok(())
}

fn route(session: &mut dyn SmokeSession) -> Checked<()> {
    let ends = json!({"from": [-400.0, 100.0], "to": [300.0, -250.0]});
    let routed = query(
        session,
        "terrain_route",
        with(&ends, json!({"max_grade": 0.2})),
    )?;
    smoke_eq!(
        &routed["terrain"],
        &json!("ground"),
        "the_route_names_its_terrain"
    );
    let found = &routed["route"];
    let points = found["points"].as_array().cloned().unwrap_or_default();
    let wanted = json!({"steepest": at_most(0.2), "points": {"at_least": 2}});
    smoke_eq!(
        seen(
            number(&found["steepest"]) <= 0.2 + 1e-6 && points.len() >= 2,
            found.clone(),
            &wanted,
        ),
        wanted,
        "a_route_keeps_within_its_grade"
    );
    smoke_eq!(
        &routed["origin"],
        &json!([-512.0, -512.0]),
        "the_route_reads_from_the_origin"
    );
    let world = |point: &Value| [number(&point[0]) - 512.0, number(&point[1]) - 512.0];
    let (first, last) = (points.first().map(world), points.last().map(world));
    let joined = first.is_some_and(|[x, z]| (x + 400.0).hypot(z - 100.0) < 3.0)
        && last.is_some_and(|[x, z]| (x - 300.0).hypot(z + 250.0) < 3.0);
    let wanted = json!([ends["from"], ends["to"]]);
    smoke_eq!(
        seen(joined, json!([first, last]), &wanted),
        wanted,
        "the_route_joins_its_ends"
    );
    let shortest = 700.0_f64.hypot(350.0) - 8.0;
    let wanted = json!({ "at_least": shortest });
    smoke_eq!(
        seen(
            number(&found["length"]) >= shortest,
            found["length"].clone(),
            &wanted,
        ),
        wanted,
        "a_route_is_no_shorter_than_its_ends_distance"
    );
    let unrouted = refuse(
        session,
        "query",
        "terrain_route",
        with(&ends, json!({"level": 99})),
    )?;
    smoke_eq!(
        &unrouted["code"],
        &json!("out_of_range"),
        "a_level_past_the_pyramid_is_refused"
    );
    Ok(())
}

fn samples(session: &mut dyn SmokeSession) -> Checked<()> {
    let found = heights(session, &[[0.0, 0.0], [48.0, 0.0], [5000.0, 5000.0]])?;
    let (plateau, hole, outside) = (&found[0], &found[1], &found[2]);
    smoke_eq!(
        &plateau["terrain"],
        &json!("ground"),
        "samples_name_their_terrain"
    );
    let wanted = near(30, 1e-3);
    smoke_eq!(
        seen(
            (number(&plateau["height"]) - 30.0).abs() < 1e-3,
            plateau["height"].clone(),
            &wanted,
        ),
        wanted,
        "the_plateau_stands_30_m_high"
    );
    let wanted = json!({"y": above(0.999)});
    smoke_eq!(
        seen(
            number(&plateau["normal"][1]) > 0.999,
            plateau["normal"].clone(),
            &wanted,
        ),
        wanted,
        "the_plateau_lies_flat"
    );
    smoke_eq!(&plateau["level"], &json!(0), "the_plateau_reads_level_0");
    smoke_eq!(&hole["hole"], &json!(true), "the_stamp_opens_a_hole");
    smoke_eq!(outside, &Value::Null, "outside_the_terrain_no_ground");
    let peak = height_at(session, [252.0, 252.0])?;
    let wanted = near(200, 1e-3);
    smoke_eq!(
        seen(
            (number(&peak["height"]) - 200.0).abs() < 1e-3,
            peak["height"].clone(),
            &wanted,
        ),
        wanted,
        "the_heightmap_peak_stands_200_m_high"
    );
    Ok(())
}

fn collision(session: &mut dyn SmokeSession) -> Checked<()> {
    let ray = raycast(session, [10.0, 200.0, -7.0], 400.0)?;
    let under = height_at(session, [10.0, -7.0])?;
    smoke_eq!(
        &ray["hit"]["entity"]["id"],
        &json!("ground"),
        "rays_hit_the_ground"
    );
    let wanted = near(&under["height"], 2e-3);
    smoke_eq!(
        seen(
            (number(&ray["hit"]["point"][1]) - number(&under["height"])).abs() < 2e-3,
            ray["hit"]["point"].clone(),
            &wanted,
        ),
        wanted,
        "ray_and_query_agree"
    );
    let wanted = json!({"placed_colliders": above(0)});
    smoke_eq!(
        seen(
            number(&ray["snapshot"]["placed_colliders"]) > 0.0,
            ray["snapshot"].clone(),
            &wanted,
        ),
        wanted,
        "placed_objects_collide"
    );
    let through = raycast(session, [48.0, 200.0, 0.0], 400.0)?;
    smoke_eq!(&through["hit"], &Value::Null, "rays_fall_through_holes");
    let sweep = query(
        session,
        "terrain_sweep",
        json!({"center": [0.0, 0.0], "half_extent": 120.0, "samples": 24}),
    )?;
    let collided = &sweep["collision"];
    let layered = layer_sweep(session, [0.0, 0.0], 120.0, 24)?;
    let wanted = json!({"compared": above(100), "worst": {"max_difference": below(5e-3)}});
    smoke_eq!(
        seen(
            number(&collided["compared"]) > 100.0
                && number(&collided["worst"]["max_difference"]) < 5e-3,
            collided.clone(),
            &wanted,
        ),
        wanted,
        "collision_matches_the_drawn_ground"
    );
    let wanted = above(0);
    smoke_eq!(
        seen(
            number(&collided["placed_first"]) > 0.0,
            collided["placed_first"].clone(),
            &wanted,
        ),
        wanted,
        "rays_meet_placed_objects_first"
    );
    let wanted = json!({"compared": above(100), "worst": {"max_difference": below(5e-3)}});
    smoke_eq!(
        seen(
            number(&layered["compared"]) > 100.0
                && number(&layered["worst"]["max_difference"]) < 5e-3,
            layered.clone(),
            &wanted,
        ),
        wanted,
        "layers_match_the_drawn_ground"
    );
    Ok(())
}

fn edits(session: &mut dyn SmokeSession) -> Checked<()> {
    let spot = [-200.0, 150.0];
    let authored = number(&height_at(session, spot)?["height"]);
    let mound = json!({
        "id": "smoke-mound",
        "center": spot,
        "radius": 20.0,
        "falloff": 10.0,
        "change": {"raise": 8.0},
    });
    let raised = control(session, "terrain_edit", mound.clone())?;
    let wanted = json!({"applied": true, "edits": 1, "largest_change": 8.0});
    smoke_eq!(
        seen(
            raised["applied"] == true
                && raised["edits"] == 1
                && (number(&raised["largest_change"]) - 8.0).abs() < 1e-3,
            raised.clone(),
            &wanted,
        ),
        wanted,
        "the_edit_applies"
    );
    let height = reaches(session, spot, authored + 8.0)?;
    let wanted = near(authored + 8.0, 0.05);
    smoke_eq!(
        seen(
            (height - (authored + 8.0)).abs() < 0.05,
            json!(height),
            &wanted,
        ),
        wanted,
        "the_edit_raises"
    );
    let repeated = control(session, "terrain_edit", mound.clone())?;
    smoke_eq!(
        &repeated["applied"],
        &json!(false),
        "repeating_an_edit_leaves_it_alone"
    );
    let listed = ids(&viewed(session)?["edits"]);
    smoke_eq!(
        &listed,
        &vec!["smoke-mound".to_owned()],
        "the_terrain_lists_its_edits"
    );
    history(session, spot, authored, &mound)
}

fn history(
    session: &mut dyn SmokeSession,
    spot: [f64; 2],
    authored: f64,
    mound: &Value,
) -> Checked<()> {
    let undone = control(session, "terrain_undo", json!({}))?;
    smoke_eq!(
        &undone["undone"],
        &json!("smoke-mound"),
        "undo_names_the_edit"
    );
    let height = reaches(session, spot, authored)?;
    let wanted = near(authored, 0.05);
    smoke_eq!(
        seen((height - authored).abs() < 0.05, json!(height), &wanted),
        wanted,
        "undo_restores"
    );
    let redone = control(session, "terrain_redo", json!({}))?;
    smoke_eq!(
        &redone["redone"],
        &json!("smoke-mound"),
        "redo_names_the_edit"
    );
    let height = reaches(session, spot, authored + 8.0)?;
    let wanted = near(authored + 8.0, 0.05);
    smoke_eq!(
        seen(
            (height - (authored + 8.0)).abs() < 0.05,
            json!(height),
            &wanted,
        ),
        wanted,
        "redo_raises_again"
    );
    let refused = refuse(
        session,
        "control",
        "terrain_edit",
        with(mound, json!({"id": "no spaces"})),
    )?;
    smoke_eq!(
        &refused["path"],
        &json!("$.args.id"),
        "spaced_ids_are_refused"
    );
    let last = control(session, "terrain_undo", json!({}))?;
    smoke_eq!(
        &last["undone"],
        &json!("smoke-mound"),
        "undo_takes_the_edit_back"
    );
    let height = reaches(session, spot, authored)?;
    let wanted = near(authored, 0.05);
    smoke_eq!(
        seen((height - authored).abs() < 0.05, json!(height), &wanted),
        wanted,
        "the_fixture_is_left_as_authored"
    );
    Ok(())
}

fn reaches(session: &mut dyn SmokeSession, spot: [f64; 2], wanted: f64) -> Checked<f64> {
    let mut height = f64::NAN;
    for _ in 0..SETTLE_FRAMES {
        height = number(&height_at(session, spot)?["height"]);
        if (height - wanted).abs() < 0.05 {
            break;
        }
        render(session)?;
    }
    Ok(height)
}

fn rivers(session: &mut dyn SmokeSession) -> Checked<()> {
    let stream = json!({
        "id": "smoke-river",
        "points": [[-260.0, 120.0], [-200.0, 150.0], [-140.0, 190.0]],
        "width": [3.0],
        "discharge": [2.0],
    });
    let drawn = control(session, "terrain_river", stream.clone())?;
    let wanted = json!({"applied": true, "rivers": 1, "dirty": {"not": null}});
    smoke_eq!(
        seen(
            drawn["applied"] == true && drawn["rivers"] == 1 && !drawn["dirty"].is_null(),
            drawn.clone(),
            &wanted,
        ),
        wanted,
        "the_river_draws"
    );
    let again = control(session, "terrain_river", stream)?;
    smoke_eq!(
        &again["applied"],
        &json!(false),
        "drawing_the_same_river_again_leaves_it_alone"
    );
    let listed = ids(&viewed(session)?["rivers"]);
    smoke_eq!(
        &listed,
        &vec!["smoke-river".to_owned()],
        "the_terrain_lists_its_drawn_rivers"
    );
    let taken = control(session, "terrain_river", json!({"id": "smoke-river"}))?;
    let wanted = json!({"applied": true, "rivers": 0});
    smoke_eq!(
        seen(
            taken["applied"] == true && taken["rivers"] == 0,
            taken.clone(),
            &wanted,
        ),
        wanted,
        "no_points_take_the_river_away"
    );
    Ok(())
}

fn authored(session: &mut dyn SmokeSession) -> Checked<()> {
    let spot = [18.0, -12.0];
    let floor = number(&height_at(session, spot)?["height"]);
    let top = raycast(session, [spot[0], floor + 50.0, spot[1]], 100.0)?;
    let hit = &top["hit"];
    let wanted = json!({"placed": {"layer": 65534}, "point": near(floor + 1.5, 0.3)});
    smoke_eq!(
        seen(
            hit["placed"]["layer"] == 65534
                && (number(&hit["point"][1]) - (floor + 1.5)).abs() < 0.3,
            hit.clone(),
            &wanted,
        ),
        wanted,
        "the_authored_crate_stands_on_the_plateau"
    );
    let placed = viewed(session)?["terrain"]["surface"]["placed"]
        .as_array()
        .map_or(0, Vec::len);
    smoke_eq!(&placed, &2_usize, "the_terrain_lists_its_authored_objects");
    Ok(())
}
