use engine_core::smoke::{Checked, above, component, control, near, number, query, refuse, render};
use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use serde_json::{Value, json};

use super::requests::seen;
use super::terrain::{SETTLE_FRAMES, height_at, open, viewed};

const GOAL: [f64; 2] = [-25.0, 12.0];

const HIKER: &str = "hiker";

pub(crate) fn walks(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    open(session, true)?;
    around_a_hole(session)?;
    hikes(session)
}

fn around_a_hole(session: &mut dyn SmokeSession) -> Checked<()> {
    let (start, end): ([f64; 2], [f64; 2]) = ([-300.0, -250.0], [-100.0, -250.0]);
    let (center, radius): ([f64; 2], f64) = ([-200.0, -250.0], 18.0);
    control(
        session,
        "terrain_edit",
        json!({"id": "smoke-pit", "center": center, "radius": radius, "change": "hole"}),
    )?;
    let mut found = Value::Null;
    for _ in 0..SETTLE_FRAMES {
        found = query(session, "nav_path", json!({"from": start, "to": end}))?;
        if found["searching"] == false {
            break;
        }
        render(session)?;
    }
    smoke_eq!(&found["searching"], &json!(false), "the_walk_search_ends");
    let walk = &found["walk"];
    let points = walk["points"].as_array().cloned().unwrap_or_default();
    let wanted = json!({"points": above(2)});
    smoke_eq!(
        seen(points.len() > 2, walk["points"].clone(), &wanted),
        wanted,
        "a_walk_reaches"
    );
    let straight = (end[0] - start[0]).hypot(end[1] - start[1]);
    let level = walk["level"].as_u64().unwrap_or_default() as usize;
    let spacing = number(&viewed(session)?["level_spacings"][level]);
    let rim = radius - spacing;
    let away = |point: &Value| {
        let (x, z) = (number(&point[0]), number(&point[2]));
        (x - center[0]).hypot(z - center[1])
    };
    let clear = points.iter().all(|point| away(point) > rim);
    let nearest = points.iter().map(away).fold(f64::INFINITY, f64::min);
    let wanted = json!({"length": above(straight), "nearest": above(rim)});
    smoke_eq!(
        seen(
            clear && number(&walk["length"]) > straight,
            json!({"length": walk["length"], "nearest": nearest}),
            &wanted,
        ),
        wanted,
        "the_walk_goes_around_the_hole"
    );
    let steep = refuse(
        session,
        "query",
        "nav_path",
        json!({"from": start, "to": end, "options": {"max_slope": 0.0}}),
    )?;
    smoke_eq!(
        &steep["path"],
        &json!("$.args.options.max_slope"),
        "flat_only_walks_are_refused"
    );
    let undone = control(session, "terrain_undo", json!({}))?;
    smoke_eq!(&undone["undone"], &json!("smoke-pit"), "the_pit_is_undone");
    Ok(())
}

fn walk(session: &mut dyn SmokeSession) -> Checked<Value> {
    query(session, "nav_agent", json!({ "entity": HIKER }))
}

fn hikes(session: &mut dyn SmokeSession) -> Checked<()> {
    let sent = control(
        session,
        "nav_go",
        json!({"entity": HIKER, "destination": GOAL}),
    )?;
    smoke_eq!(
        &sent["destination"],
        &json!(GOAL),
        "the_hiker_takes_its_destination"
    );
    for _ in 0..SETTLE_FRAMES {
        session.step(1);
        if walk(session)?["status"] != "searching" {
            break;
        }
    }
    let mut walked = walk(session)?;
    for _ in 0..30 {
        session.step(60);
        walked = walk(session)?;
        if walked["status"] == "arrived" {
            break;
        }
    }
    smoke_eq!(&walked["status"], &json!("arrived"), "the_hiker_arrives");
    let wanted = json!({"searched": above(0), "length": above(40)});
    smoke_eq!(
        seen(
            number(&walked["searched"]) > 0.0 && number(&walked["length"]) > 40.0,
            json!({"searched": walked["searched"], "length": walked["length"]}),
            &wanted,
        ),
        wanted,
        "its_walk_was_searched"
    );
    let seen_status = walk(session)?;
    smoke_eq!(
        &seen_status["status"],
        &json!("arrived"),
        "the_hiker_inspects_as_arrived"
    );
    let at = component(session, HIKER, "Transform")?["translation"].take();
    let wanted = json!(GOAL);
    smoke_eq!(
        seen(
            (number(&at[0]) - GOAL[0]).hypot(number(&at[2]) - GOAL[1]) < 1.5,
            at.clone(),
            &wanted,
        ),
        wanted,
        "it_stands_at_its_goal"
    );
    let floor = number(&height_at(session, [number(&at[0]), number(&at[2])])?["height"]);
    let wanted = near(floor + 0.5, 0.25);
    smoke_eq!(
        seen(
            (number(&at[1]) - (floor + 0.5)).abs() < 0.25,
            at[1].clone(),
            &wanted,
        ),
        wanted,
        "it_stands_on_the_ground"
    );
    stops(session)
}

fn stops(session: &mut dyn SmokeSession) -> Checked<()> {
    control(
        session,
        "nav_go",
        json!({"entity": HIKER, "destination": [5000.0, 5000.0]}),
    )?;
    session.step(2);
    let lost = walk(session)?["status"].take();
    smoke_eq!(
        &lost,
        &json!("unreachable"),
        "off_the_terrain_nothing_reaches"
    );
    let agents = query(session, "nav_agents", json!({}))?;
    smoke_eq!(&agents["agents"], &json!(1), "the_tick_counts_its_agents");
    let starved = refuse(
        session,
        "control",
        "environment",
        json!({"navigation": {"samples_per_tick": 10}}),
    )?;
    smoke_eq!(
        &starved["path"],
        &json!("$.args.navigation.samples_per_tick"),
        "a_starved_search_budget_is_refused"
    );
    control(
        session,
        "nav_go",
        json!({"entity": HIKER, "destination": null}),
    )?;
    Ok(())
}
