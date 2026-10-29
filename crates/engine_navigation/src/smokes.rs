use bevy_ecs::prelude::Entity;
use engine_core::smoke::{SmokeResult, SmokeSession};
use engine_core::{Bricks, Mesh3d, Transform, smoke_check, smoke_eq};
use engine_terrain::smokes::{
    Checked, Fixture, SETTLE_FRAMES, above, component, control, height_at, inspect, name, near,
    number, open, query, refuse, render,
};
use glam::Vec3;
use serde_json::{Value, json};

use crate::{NavAgent, NavStatus, NavWalk};

const GOAL: [f32; 2] = [-25.0, 12.0];

pub(crate) fn register(bricks: &mut Bricks<'_>) {
    bricks.smoke("navigation_walks", true, walks);
}

fn walks(session: &mut dyn SmokeSession) -> SmokeResult {
    let (fixture, _) = open(session)?;
    around_a_hole(session)?;
    let hiker = hiker(session, &fixture);
    hikes(session, hiker)
}

fn around_a_hole(session: &mut dyn SmokeSession) -> Checked<()> {
    let (start, end): ([f64; 2], [f64; 2]) = ([-300.0, -250.0], [-100.0, -250.0]);
    control(
        session,
        "terrain_edit",
        json!({"id": "smoke-pit", "center": [-200.0, -250.0], "radius": 18.0, "change": "hole"}),
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
    smoke_check!(
        points.len() > 2,
        "a_walk_reaches",
        &walk["points"],
        &json!({"points": above(2)})
    );
    let straight = (end[0] - start[0]).hypot(end[1] - start[1]);
    let clear = points
        .iter()
        .all(|point| (number(&point[0]) + 200.0).hypot(number(&point[2]) + 250.0) > 18.0);
    smoke_check!(
        clear && number(&walk["length"]) > straight,
        "the_walk_goes_around_the_hole",
        &walk["length"],
        &above(straight)
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

fn hiker(session: &mut dyn SmokeSession, fixture: &Fixture) -> Entity {
    let world = session.app().world_mut();
    let hiker = world
        .spawn((
            Transform {
                translation: Vec3::new(20.0, 31.0, 20.0),
                ..Transform::IDENTITY
            },
            Mesh3d {
                mesh: fixture.cube,
                material: fixture.plain,
            },
            NavAgent {
                speed: 6.0,
                offset: 0.5,
                ..NavAgent::default()
            },
        ))
        .id();
    name(world, "hiker", hiker);
    hiker
}

fn walk(session: &mut dyn SmokeSession, hiker: Entity) -> NavWalk {
    session
        .app()
        .world()
        .get::<NavWalk>(hiker)
        .cloned()
        .unwrap_or_default()
}

fn hikes(session: &mut dyn SmokeSession, hiker: Entity) -> Checked<()> {
    let sent = control(
        session,
        "nav_go",
        json!({"entity": "hiker", "destination": GOAL}),
    )?;
    smoke_eq!(
        &sent["destination"],
        &json!(GOAL),
        "the_hiker_takes_its_destination"
    );
    for _ in 0..SETTLE_FRAMES {
        session.step(1);
        if walk(session, hiker).status != NavStatus::Searching {
            break;
        }
    }
    let mut walked = walk(session, hiker);
    for _ in 0..30 {
        session.step(60);
        walked = walk(session, hiker);
        if walked.status == NavStatus::Arrived {
            break;
        }
    }
    smoke_check!(
        walked.status == NavStatus::Arrived,
        "the_hiker_arrives",
        &walked.status,
        &NavStatus::Arrived
    );
    smoke_check!(
        walked.searched > 0 && walked.length > 40.0,
        "its_walk_was_searched",
        &json!({"searched": walked.searched, "length": walked.length}),
        &json!({"searched": above(0), "length": above(40)})
    );
    let seen = component(session, "hiker", "NavAgent")?;
    smoke_eq!(
        &seen["status"],
        &json!(NavStatus::Arrived),
        "the_hiker_inspects_as_arrived"
    );
    let at = session
        .app()
        .world()
        .get::<Transform>(hiker)
        .map_or(Vec3::NAN, |transform| transform.translation);
    smoke_check!(
        (at.x - GOAL[0]).hypot(at.z - GOAL[1]) < 1.5,
        "it_stands_at_its_goal",
        &at.to_array(),
        &GOAL
    );
    let floor = number(&height_at(session, [f64::from(at.x), f64::from(at.z)])?["height"]);
    smoke_check!(
        (f64::from(at.y) - (floor + 0.5)).abs() < 0.25,
        "it_stands_on_the_ground",
        &at.y,
        &near(floor + 0.5, 0.25)
    );
    stops(session, hiker)
}

fn stops(session: &mut dyn SmokeSession, hiker: Entity) -> Checked<()> {
    control(
        session,
        "nav_go",
        json!({"entity": "hiker", "destination": [5000.0, 5000.0]}),
    )?;
    session.step(2);
    let lost = walk(session, hiker).status;
    smoke_check!(
        lost == NavStatus::Unreachable,
        "off_the_terrain_nothing_reaches",
        &lost,
        &NavStatus::Unreachable
    );
    let agents = inspect(session, "NavAgents")?;
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
        json!({"entity": "hiker", "destination": null}),
    )?;
    Ok(())
}
