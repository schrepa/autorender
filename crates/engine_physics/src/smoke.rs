use engine_core::smoke::{
    Checked, ask, component, control, environment, inspect, not, number, query,
};
use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use serde_json::{Value, json};

const SCENE: &str = "physics-fixture";
const ENTITIES: [&str; 4] = ["floor", "crate", "seabed", "stone"];
const DESCRIBED: [&str; 5] = [
    "rigid_body",
    "physics_body",
    "physics_stats",
    "physics",
    "body_push",
];
const IMPULSE: f64 = 1800.0;
const FORCE: f64 = 12000.0;
const FLOOR_TOP: f64 = 10.0;
const HALF: f64 = 0.5;
const RESTS_WITHIN: f64 = 0.05;
const SECOND: u32 = 60;
const STEP: usize = 120;

pub(crate) fn run(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    session.source(&scene().to_string());
    let built: Vec<bool> = ENTITIES
        .iter()
        .map(|id| {
            session
                .request("inspect", json!({"entity": id, "component": "Transform"}))
                .is_ok()
        })
        .collect();
    let every = [true; ENTITIES.len()];
    smoke_eq!(json!(built), json!(every), "fixture_builds");
    control(session, "pause", json!({}))?;
    environment(
        session,
        json!({"merge": true, "physics": {"water_density": 1000}}),
    )?;
    for name in DESCRIBED {
        ask(session, "describe", json!({"name": name}))?;
    }
    let settings = inspect(session, "physics")?;
    let expected = json!({"enabled": true, "gravity": [0.0, -9.81, 0.0]});
    let gravity = (number(&settings["gravity"][1]) + 9.81).abs() < 1e-4;
    smoke_eq!(
        shown(settings["enabled"] == true && gravity, &settings, &expected),
        expected,
        "settings_inspect"
    );
    rests(session)?;
    pushed(session)?;
    toggles(session)?;
    parented(session)?;
    session.source(&empty().to_string());
    session.without(&["engine_collision", "engine_physics"]);
    let missing = session.request("inspect", json!({"component": "physics"}));
    smoke_eq!(
        json!(
            missing
                .err()
                .map(|refused| refused["error"]["code"].clone())
        ),
        json!("missing_plugin"),
        "left_out_without_collision"
    );
    Ok(())
}

fn rests(session: &mut dyn SmokeSession) -> Checked<()> {
    seconds(session, 3);
    let crate_at = component(session, "crate", "Transform")?;
    let height = number(&crate_at["translation"][1]);
    let expected = json!({"translation": [0.0, FLOOR_TOP + HALF, 0.0]});
    smoke_eq!(
        shown(
            (height - (FLOOR_TOP + HALF)).abs() < RESTS_WITHIN,
            &crate_at,
            &expected
        ),
        expected,
        "falls_and_rests"
    );
    let body = query(session, "physics_body", json!({"entity": "crate"}))?;
    let expected = json!({"simulated": true, "problem": null, "mass": 600.0});
    smoke_eq!(
        shown(
            body["simulated"] == true
                && body["problem"].is_null()
                && (number(&body["mass"]) - 600.0).abs() < 1.0,
            &body,
            &expected
        ),
        expected,
        "body_inspects"
    );
    let stone = component(session, "stone", "Transform")?;
    let expected = json!({"translation": [32.0, -2.0, 0.0]});
    smoke_eq!(
        shown(number(&stone["translation"][1]) < -2.0, &stone, &expected),
        expected,
        "sinks"
    );
    let stats = inspect(session, "physics_stats")?;
    let expected = json!({"bodies": 2, "held": 0, "solids": 2});
    smoke_eq!(
        shown(
            stats["bodies"] == 2 && stats["held"] == 0 && number(&stats["solids"]) >= 2.0,
            &stats,
            &expected
        ),
        expected,
        "world_inspects"
    );
    let status = ask(session, "query", json!({"name": "status"}))?;
    smoke_eq!(
        json!(status["rendering"]["physics_stats"]["bodies"]),
        json!(2),
        "frames_observe"
    );
    Ok(())
}

fn crate_speed(session: &mut dyn SmokeSession) -> Checked<f64> {
    let body = query(session, "physics_body", json!({"entity": "crate"}))?;
    Ok(number(&body["velocity"][0]))
}

fn pushed(session: &mut dyn SmokeSession) -> Checked<()> {
    control(
        session,
        "body_push",
        json!({"entity": "crate", "push": {"impulse": [IMPULSE, 0.0, 0.0]}}),
    )?;
    session.step(1);
    let hit = crate_speed(session)?;
    let expected = json!({"above": 1.0});
    smoke_eq!(
        shown(hit > 1.0, &json!(hit), &expected),
        expected,
        "impulse_gains_speed"
    );
    session.step(1);
    let after = crate_speed(session)?;
    let expected = json!({"at_most": hit});
    smoke_eq!(
        shown(after <= hit, &json!(after), &expected),
        expected,
        "impulse_applies_once"
    );
    control(
        session,
        "body_push",
        json!({"entity": "crate", "push": {"force": [FORCE, 0.0, 0.0]}}),
    )?;
    session.step(10);
    let forced = crate_speed(session)?;
    let expected = json!({"above": after});
    smoke_eq!(
        shown(forced > after, &json!(forced), &expected),
        expected,
        "force_keeps_pushing"
    );
    control(session, "body_push", json!({"entity": "crate", "push": {}}))?;
    Ok(())
}

fn toggles(session: &mut dyn SmokeSession) -> Checked<()> {
    environment(
        session,
        json!({"merge": true, "physics": {"enabled": false}}),
    )?;
    let held = component(session, "stone", "Transform")?;
    seconds(session, 1);
    smoke_eq!(
        component(session, "stone", "Transform")?,
        held,
        "physics_off_holds"
    );
    environment(
        session,
        json!({"merge": true, "physics": {"enabled": true}}),
    )?;
    seconds(session, 1);
    let moved = component(session, "stone", "Transform")?;
    let expected = not(&held);
    smoke_eq!(
        shown(moved != held, &moved, &expected),
        expected,
        "physics_on_moves"
    );
    Ok(())
}

fn parented(session: &mut dyn SmokeSession) -> Checked<()> {
    let mut hung = scene();
    hung["id"] = json!("physics-hung");
    if let Some(entities) = hung["entities"].as_array_mut() {
        entities.push(json!({
            "id": "hung",
            "collider": {"shape": {"kind": "sphere", "radius": HALF}},
            "rigid_body": {},
            "transform": {"translation": [5.0, FLOOR_TOP + HALF, 0.0]},
        }));
    }
    session.source(&hung.to_string());
    seconds(session, 3);
    let before = component(session, "crate", "Transform")?;
    let parent = session.request(
        "control",
        json!({"name": "parent", "args": {"entity": "hung", "parent": "floor"}}),
    );
    smoke_eq!(
        json!(parent.err().map(|refused| refused["error"].clone())),
        json!(null),
        "parented_body_builds"
    );
    session.step(1);
    let parented = query(session, "physics_body", json!({"entity": "hung"}))?;
    let expected = json!({"problem": "not_root", "simulated": false});
    smoke_eq!(
        shown(
            parented["problem"] == "not_root" && parented["simulated"] == false,
            &parented,
            &expected
        ),
        expected,
        "parented_body_refused"
    );
    smoke_eq!(
        component(session, "crate", "Transform")?,
        before,
        "parented_body_leaves_world"
    );
    Ok(())
}

fn shown(passed: bool, got: &Value, expected: &Value) -> Value {
    if passed {
        expected.clone()
    } else {
        got.clone()
    }
}

fn seconds(session: &mut dyn SmokeSession, count: u32) {
    for _ in (0..count * SECOND).step_by(STEP) {
        session.step(STEP as u32);
    }
}

fn empty() -> Value {
    json!({
        "id": "physics-empty",
        "revision": 1,
        "materials": [],
        "meshes": [],
        "entities": [],
    })
}

fn scene() -> Value {
    let cuboid = |half: [f64; 3]| json!({"shape": {"kind": "cuboid", "half_extents": half}});
    let sphere = json!({"shape": {"kind": "sphere", "radius": HALF}});
    json!({
        "id": SCENE,
        "revision": 1,
        "materials": [],
        "meshes": [],
        "entities": [
            {
                "id": "floor",
                "collider": cuboid([10.0, HALF, 10.0]),
                "transform": {"translation": [0.0, FLOOR_TOP - HALF, 0.0]},
            },
            {
                "id": "crate",
                "collider": cuboid([HALF; 3]),
                "rigid_body": {},
                "transform": {"translation": [0.0, 14.0, 0.0]},
            },
            {
                "id": "seabed",
                "collider": cuboid([20.0, HALF, 20.0]),
                "transform": {"translation": [30.0, -12.5, 0.0]},
            },
            {
                "id": "stone",
                "collider": sphere,
                "rigid_body": {"density": 2600.0},
                "transform": {"translation": [32.0, 40.0, 0.0]},
            },
        ],
    })
}
