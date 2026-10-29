use bevy_ecs::prelude::Entity;
use engine_core::smoke::{SmokeResult, SmokeSession};
use engine_core::{smoke_check, smoke_eq};
use glam::Vec2;
use serde_json::{Value, json};

use super::{
    Checked, SETTLE_FRAMES, above, at_most, below, control, height_at, heights, inspect,
    layer_sweep, near, number, open, query, raycast, refuse, render, seen, with,
};

const ROAD: [[f64; 2]; 4] = [
    [100.0, 300.0],
    [250.0, 260.0],
    [400.0, 150.0],
    [450.0, 60.0],
];

const SIDE: usize = 513;

pub(super) fn run(session: &mut dyn SmokeSession) -> SmokeResult {
    let (fixture, _) = open(session)?;
    let ground = fixture.ground;
    inspection(session, ground)?;
    erosion(session, ground)?;
    path(session, ground)?;
    bridge(session, ground)?;
    tunnel(session, ground)?;
    route(session)?;
    samples(session)?;
    collision(session)?;
    edits(session, ground)?;
    rivers(session, ground)?;
    authored(session, ground)
}

fn inspection(session: &mut dyn SmokeSession, ground: Entity) -> Checked<()> {
    let state = seen(session, ground);
    let layers = state
        .terrain
        .as_ref()
        .map_or(0, |terrain| terrain.layers.layers().len());
    smoke_eq!(&layers, &10_usize, "the_terrain_keeps_its_layers");
    let extent = state
        .terrain
        .as_ref()
        .map(|terrain| terrain.pyramid.extent().to_array());
    smoke_eq!(
        &extent,
        &Some([1024.0, 1024.0]),
        "the_tiles_span_the_extent"
    );
    smoke_check!(
        state.resident > 0,
        "tiles_reside",
        &state.resident,
        &above(0)
    );
    smoke_check!(
        state
            .errors
            .split_last()
            .is_some_and(|(_, rest)| rest.iter().all(|&error| error > 0.0)),
        "levels_measure_their_error",
        &state.errors,
        &above(0)
    );
    smoke_check!(
        state.ranges.windows(2).all(|pair| pair[1] > pair[0]),
        "level_ranges_rise",
        &state.ranges,
        &json!("rising")
    );
    Ok(())
}

fn erosion(session: &mut dyn SmokeSession, ground: Entity) -> Checked<()> {
    let bakes = seen(session, ground).bakes;
    smoke_check!(
        !bakes.baking,
        "erosion_is_done_once_tiles_load",
        &bakes.baking,
        &false
    );
    let worn = bakes
        .layers
        .iter()
        .find(|stats| stats.layer == "erosion")
        .map(|stats| stats.stats.clone());
    let read = |name: &str| worn.as_ref().map_or(f64::NAN, |worn| number(&worn[name]));
    smoke_check!(
        read("samples") == (SIDE * SIDE) as f64,
        "erosion_reads_its_level",
        &worn,
        &json!({ "samples": SIDE * SIDE })
    );
    smoke_check!(
        read("rounds") == 6.0 && read("deepest") > 0.0,
        "erosion_wears_ground_away",
        &worn,
        &json!({"rounds": 6, "deepest": above(0)})
    );
    Ok(())
}

fn path(session: &mut dyn SmokeSession, ground: Entity) -> Checked<()> {
    let paths = seen(session, ground).paths;
    let road = paths.first().cloned();
    smoke_check!(
        road.as_ref().is_some_and(|(layer, stats)| *layer == 4 && stats.is_some()),
        "the_path_bakes",
        &paths,
        &json!({"layer": 4, "baked": true})
    );
    let stats = road.and_then(|(_, stats)| stats).unwrap_or(Value::Null);
    let length = ROAD
        .windows(2)
        .map(|pair| (pair[1][0] - pair[0][0]).hypot(pair[1][1] - pair[0][1]))
        .sum::<f64>();
    let samples = (length / 4.0).ceil() as usize + 1;
    smoke_check!(
        (number(&stats["length"]) - length).abs() < 1e-2
            && stats["samples"].as_u64() == Some(samples as u64),
        "the_profile_reads_its_line",
        &stats,
        &json!({"length": length, "samples": samples})
    );
    let steepest = number(&stats["steepest"]);
    smoke_check!(
        steepest > 0.0 && steepest < 0.6,
        "the_road_climbs_gently",
        &steepest,
        &json!({"above": 0, "below": 0.6})
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
        smoke_check!(
            (number(&center["path"]) + 3.0).abs() < 0.05,
            "the_centerline_lies_3_m_inside_the_edges",
            center,
            &near(-3, 0.05)
        );
        tilt = tilt.max((number(&left["height"]) - number(&right["height"])).abs());
        level += 1;
    }
    smoke_check!(
        level >= 3 && tilt < 0.15,
        "the_road_lies_level_across",
        &json!({"tilt": tilt, "sections": level}),
        &json!({"tilt": below(0.15), "sections": {"at_least": 3}})
    );
    let away = height_at(session, [400.0, -400.0])?;
    smoke_check!(
        away["path"].is_null(),
        "far_from_roads_queries_report_none",
        &away,
        &json!({ "path": null })
    );
    Ok(())
}

fn bridge(session: &mut dyn SmokeSession, ground: Entity) -> Checked<()> {
    let baked: Vec<Value> = seen(session, ground)
        .paths
        .into_iter()
        .filter_map(|(_, stats)| stats)
        .collect();
    let road = baked.first().cloned().unwrap_or(Value::Null);
    smoke_check!(
        road["bridges"].as_u64().is_some_and(|bridges| bridges >= 1)
            && number(&road["bridged"]) > 15.0,
        "the_road_bridges_the_trench",
        &road,
        &json!({"bridges": {"at_least": 1}, "bridged": above(15)})
    );
    let bridges: u64 = baked
        .iter()
        .filter_map(|stats| stats["bridges"].as_u64())
        .sum();
    let terrains = inspect(session, "Terrains")?;
    smoke_eq!(
        &terrains["bridges"],
        &json!(bridges),
        "the_terrains_count_their_bridges"
    );
    let trench = [430.0 - 512.0, 96.0 - 512.0];
    let floor = height_at(session, trench)?;
    smoke_check!(
        floor["path"].is_null(),
        "under_its_bridge_the_ground_is_no_road",
        &floor,
        &json!({ "path": null })
    );
    let landing = raycast(session, [trench[0], 300.0, trench[1]], 600.0)?;
    let hit = &landing["hit"];
    smoke_check!(
        hit["placed"]["layer"] == 65535
            && number(&hit["point"][1]) > number(&floor["height"]) + 5.0,
        "rays_land_on_the_deck",
        hit,
        &json!({"placed": {"layer": 65535}, "point": above(&floor["height"])})
    );
    Ok(())
}

fn tunnel(session: &mut dyn SmokeSession, ground: Entity) -> Checked<()> {
    let paths = seen(session, ground).paths;
    let bore = paths.get(1).cloned();
    smoke_check!(
        bore.as_ref().is_some_and(|(layer, stats)| {
            *layer == 6
                && stats.as_ref().is_some_and(|stats| {
                    stats["tunnels"] == 1 && number(&stats["tunneled"]) > 40.0
                })
        }),
        "the_road_tunnels_the_mound",
        &bore,
        &json!({"layer": 6, "tunnels": 1, "tunneled": above(40)})
    );
    let terrains = inspect(session, "Terrains")?;
    smoke_eq!(
        &terrains["tunnels"],
        &json!(1),
        "the_terrains_count_the_tunnel"
    );
    let summit = [750.0 - 512.0, 200.0 - 512.0];
    let mound = height_at(session, summit)?;
    smoke_check!(
        mound["path"].is_null() && mound["hole"] == false,
        "above_the_tunnel_the_ground_is_no_road",
        &mound,
        &json!({"path": null, "hole": false})
    );
    let top = raycast(session, [summit[0], 400.0, summit[1]], 600.0)?;
    let hit = &top["hit"];
    let height = number(&mound["height"]);
    let landed = if hit["placed"].is_object() {
        hit["placed"]["layer"] == 65535 && number(&hit["point"][1]) < height - 10.0
    } else {
        (number(&hit["point"][1]) - height).abs() < 0.05
    };
    smoke_check!(
        landed,
        "rays_land_on_the_mound_or_the_floor_under_it",
        hit,
        &json!({ "height": height })
    );
    let bored: Vec<[f64; 2]> = (0..=100)
        .map(|step| [f64::from(600 + 3 * step) - 512.0, summit[1]])
        .collect();
    let portals = heights(session, &bored)?
        .iter()
        .filter(|sample| sample["hole"] == true)
        .count();
    smoke_check!(
        portals > 0,
        "the_ground_opens_at_its_portals",
        &portals,
        &above(0)
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
    smoke_check!(
        number(&found["steepest"]) <= 0.2 + 1e-6 && points.len() >= 2,
        "a_route_keeps_within_its_grade",
        found,
        &json!({"steepest": at_most(0.2), "points": {"at_least": 2}})
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
    smoke_check!(
        joined,
        "the_route_joins_its_ends",
        &json!([first, last]),
        &json!([ends["from"], ends["to"]])
    );
    let shortest = 700.0_f64.hypot(350.0) - 8.0;
    smoke_check!(
        number(&found["length"]) >= shortest,
        "a_route_is_no_shorter_than_its_ends_distance",
        &found["length"],
        &json!({ "at_least": shortest })
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
    smoke_check!(
        (number(&plateau["height"]) - 30.0).abs() < 1e-3,
        "the_plateau_stands_30_m_high",
        &plateau["height"],
        &near(30, 1e-3)
    );
    smoke_check!(
        number(&plateau["normal"][1]) > 0.999,
        "the_plateau_lies_flat",
        &plateau["normal"],
        &json!({"y": above(0.999)})
    );
    smoke_eq!(&plateau["level"], &json!(0), "the_plateau_reads_level_0");
    smoke_eq!(&hole["hole"], &json!(true), "the_stamp_opens_a_hole");
    smoke_eq!(outside, &Value::Null, "outside_the_terrain_no_ground");
    let peak = height_at(session, [252.0, 252.0])?;
    smoke_check!(
        (number(&peak["height"]) - 200.0).abs() < 1e-3,
        "the_heightmap_peak_stands_200_m_high",
        &peak["height"],
        &near(200, 1e-3)
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
    smoke_check!(
        (number(&ray["hit"]["point"][1]) - number(&under["height"])).abs() < 2e-3,
        "ray_and_query_agree",
        &ray["hit"]["point"],
        &near(&under["height"], 2e-3)
    );
    smoke_check!(
        number(&ray["snapshot"]["placed_colliders"]) > 0.0,
        "placed_objects_collide",
        &ray["snapshot"],
        &json!({"placed_colliders": above(0)})
    );
    let through = raycast(session, [48.0, 200.0, 0.0], 400.0)?;
    smoke_eq!(&through["hit"], &Value::Null, "rays_fall_through_holes");
    let sweep = query(
        session,
        "terrain_sweep",
        json!({"center": [0.0, 0.0], "half_extent": 120.0, "samples": 24}),
    )?;
    let collided = &sweep["collision"];
    let layered = &layer_sweep(session, Vec2::ZERO, 120.0, 24);
    smoke_check!(
        number(&collided["compared"]) > 100.0
            && number(&collided["worst"]["max_difference"]) < 5e-3,
        "collision_matches_the_drawn_ground",
        collided,
        &json!({"compared": above(100), "worst": {"max_difference": below(5e-3)}})
    );
    smoke_check!(
        number(&collided["placed_first"]) > 0.0,
        "rays_meet_placed_objects_first",
        &collided["placed_first"],
        &above(0)
    );
    smoke_check!(
        number(&layered["compared"]) > 100.0 && number(&layered["worst"]["max_difference"]) < 5e-3,
        "layers_match_the_drawn_ground",
        layered,
        &json!({"compared": above(100), "worst": {"max_difference": below(5e-3)}})
    );
    Ok(())
}

fn edits(session: &mut dyn SmokeSession, ground: Entity) -> Checked<()> {
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
    smoke_check!(
        raised["applied"] == true
            && raised["edits"] == 1
            && (number(&raised["largest_change"]) - 8.0).abs() < 1e-3,
        "the_edit_applies",
        &raised,
        &json!({"applied": true, "edits": 1, "largest_change": 8.0})
    );
    let height = reaches(session, spot, authored + 8.0)?;
    smoke_check!(
        (height - (authored + 8.0)).abs() < 0.05,
        "the_edit_raises",
        &height,
        &near(authored + 8.0, 0.05)
    );
    let repeated = control(session, "terrain_edit", mound.clone())?;
    smoke_eq!(
        &repeated["applied"],
        &json!(false),
        "repeating_an_edit_leaves_it_alone"
    );
    let listed: Vec<String> = seen(session, ground)
        .terrain
        .map_or_else(Vec::new, |terrain| {
            terrain
                .edits
                .iter()
                .map(|edit| edit.id.to_string())
                .collect()
        });
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
    smoke_check!(
        (height - authored).abs() < 0.05,
        "undo_restores",
        &height,
        &near(authored, 0.05)
    );
    let redone = control(session, "terrain_redo", json!({}))?;
    smoke_eq!(
        &redone["redone"],
        &json!("smoke-mound"),
        "redo_names_the_edit"
    );
    let height = reaches(session, spot, authored + 8.0)?;
    smoke_check!(
        (height - (authored + 8.0)).abs() < 0.05,
        "redo_raises_again",
        &height,
        &near(authored + 8.0, 0.05)
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
    smoke_check!(
        (height - authored).abs() < 0.05,
        "the_fixture_is_left_as_authored",
        &height,
        &near(authored, 0.05)
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

fn rivers(session: &mut dyn SmokeSession, ground: Entity) -> Checked<()> {
    let stream = json!({
        "id": "smoke-river",
        "points": [[-260.0, 120.0], [-200.0, 150.0], [-140.0, 190.0]],
        "width": [3.0],
        "discharge": [2.0],
    });
    let drawn = control(session, "terrain_river", stream.clone())?;
    smoke_check!(
        drawn["applied"] == true && drawn["rivers"] == 1 && !drawn["dirty"].is_null(),
        "the_river_draws",
        &drawn,
        &json!({"applied": true, "rivers": 1, "dirty": {"not": null}})
    );
    let again = control(session, "terrain_river", stream)?;
    smoke_eq!(
        &again["applied"],
        &json!(false),
        "drawing_the_same_river_again_leaves_it_alone"
    );
    let listed: Vec<String> = seen(session, ground)
        .terrain
        .map_or_else(Vec::new, |terrain| {
            terrain
                .rivers
                .iter()
                .map(|drawn| drawn.id.to_string())
                .collect()
        });
    smoke_eq!(
        &listed,
        &vec!["smoke-river".to_owned()],
        "the_terrain_lists_its_drawn_rivers"
    );
    let taken = control(session, "terrain_river", json!({"id": "smoke-river"}))?;
    smoke_check!(
        taken["applied"] == true && taken["rivers"] == 0,
        "no_points_take_the_river_away",
        &taken,
        &json!({"applied": true, "rivers": 0})
    );
    Ok(())
}

fn authored(session: &mut dyn SmokeSession, ground: Entity) -> Checked<()> {
    let spot = [18.0, -12.0];
    let floor = number(&height_at(session, spot)?["height"]);
    let top = raycast(session, [spot[0], floor + 50.0, spot[1]], 100.0)?;
    let hit = &top["hit"];
    smoke_check!(
        hit["placed"]["layer"] == 65534 && (number(&hit["point"][1]) - (floor + 1.5)).abs() < 0.3,
        "the_authored_crate_stands_on_the_plateau",
        hit,
        &json!({"placed": {"layer": 65534}, "point": near(floor + 1.5, 0.3)})
    );
    let placed = seen(session, ground)
        .terrain
        .map_or(0, |terrain| terrain.surface.rules.authored().placed.len());
    smoke_eq!(&placed, &2_usize, "the_terrain_lists_its_authored_objects");
    Ok(())
}
