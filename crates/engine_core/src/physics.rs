use serde_json::{Value, json};

use crate::smoke::{Checked, SmokeEntry, SmokeResult, SmokeSession, component, control, number};
use crate::{BrickList, BrickRegistry, Slot};

const FLOOR_TOP: f64 = 0.0;
const HALF: f64 = 0.5;
const START: f64 = 4.0;
const TICKS: u32 = 180;
const RESTS_WITHIN: f64 = 0.05;
const HELD_WITHIN: f64 = 1.0e-4;

pub struct PhysicsSlot;

impl Slot for PhysicsSlot {
    const NAME: &'static str = "physics";
    const SMOKES: &'static [SmokeEntry] = &[SmokeEntry {
        name: "slot_physics",
        brick: Self::NAME,
        gpu: false,
        run: physics,
    }];
}

fn physics(session: &mut dyn SmokeSession) -> SmokeResult {
    let members = session
        .app()
        .world()
        .get_resource::<BrickRegistry>()
        .map(|registry| registry.members(PhysicsSlot::NAME))
        .unwrap_or_default();
    if members.is_empty() {
        return Ok(());
    }
    let bricks = session.bricks();
    let rests = height(session, bricks.clone(), true)?;
    let rested = (number(&rests) - (FLOOR_TOP + HALF)).abs() <= RESTS_WITHIN;
    crate::smoke_check!(
        rested,
        "box_rests_on_floor",
        json!({ "members": members, "height": rests }),
        json!({ "height": FLOOR_TOP + HALF, "within": RESTS_WITHIN })
    );
    let left_out = bricks.clone().without(members.iter().copied());
    let held = height(session, left_out, false)?;
    let stayed = (number(&held) - START).abs() <= HELD_WITHIN;
    crate::smoke_check!(
        stayed,
        "box_holds_without_filler",
        json!({ "members": members, "height": held }),
        json!({ "height": START })
    );
    session.restart(bricks);
    Ok(())
}

fn height(session: &mut dyn SmokeSession, bricks: BrickList, body: bool) -> Checked<Value> {
    session.source(&scene(body).to_string());
    session.restart(bricks);
    control(session, "pause", json!({}))?;
    session.step(TICKS);
    let mut transform = component(session, "box", "Transform")?;
    Ok(transform["translation"][1].take())
}

fn scene(body: bool) -> Value {
    let cuboid = |half: [f64; 3]| json!({ "shape": { "kind": "cuboid", "half_extents": half } });
    let mut falling = json!({
        "id": "box",
        "collider": cuboid([HALF; 3]),
        "transform": { "translation": [0.0, START, 0.0] },
    });
    if body {
        falling["rigid_body"] = json!({});
    }
    json!({
        "id": "slot-physics",
        "revision": 1,
        "materials": [],
        "meshes": [],
        "entities": [
            {
                "id": "floor",
                "collider": cuboid([10.0, HALF, 10.0]),
                "transform": { "translation": [0.0, FLOOR_TOP - HALF, 0.0] },
            },
            falling,
        ],
    })
}
