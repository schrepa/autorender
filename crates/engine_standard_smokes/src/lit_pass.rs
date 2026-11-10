use bevy_ecs::prelude::*;
use engine_core::{
    AttachPoint, KindDecl, Pass, PassName, PassPurpose, PassViews, Section, SlotSet, SmokeFailure,
    SmokeSession, Writes, settings, smoke_check,
};
use engine_smoke::{above, control, joined_passes, number, render, settled};
use serde_json::{Map, Value, json};

const GROUND: &str = "terrain";

const FRAMES: u32 = 8;

settings! {
    #[section("foreign_lit")]
    pub struct ForeignLit {
        pub enabled: bool = false => any,
    }
}

pub struct ForeignLitPass;

impl PassName for ForeignLitPass {
    const NAME: &'static str = "foreign_lit";
    const PASS: Pass = Pass {
        writes: Writes::LIT,
        reads: SlotSet::ALL,
        views: PassViews::Main,
        at: AttachPoint::Opaque,
        purpose: PassPurpose::Scene,
        takes: ground,
    };
}

fn ground(kind: &KindDecl) -> bool {
    kind.kind.name() == GROUND
}

pub(super) fn foreign_lit_on(lit: Res<ForeignLit>) -> bool {
    lit.enabled
}

fn scene() -> Value {
    let mut environment = Map::new();
    environment.insert(ForeignLit::KEY.to_owned(), json!({ "enabled": true }));
    json!({
        "id": "foreign-lit", "revision": 1,
        "materials": [],
        "meshes": [],
        "environment": environment,
        "entities": [
            {"id": "ground", "terrain": {
                "origin": [-128.0, 0.0, -128.0], "tiles": [1, 1], "spacing": 1.0,
                "layers": [{"type": "noise", "seed": 3, "height": 140.0, "wavelength": 600.0}],
            }},
            {"id": "camera", "transform": {"translation": [0.0, 80.0, 140.0], "rotation_degrees": [-19.6538, 0.0, 0.0]},
             "node": {"clock": "real", "mobility": "moving"},
             "camera": {"near": 0.5, "far": 4000.0, "projection": {"Perspective": {"vertical_fov_radians": std::f64::consts::FRAC_PI_3}}}, "view": {}},
        ],
    })
}

fn foreign_cell(rendering: &Value) -> Option<Value> {
    rendering["kind_passes"]
        .as_array()?
        .iter()
        .find(|cell| cell["kind"] == GROUND && cell["pass"] == ForeignLitPass::NAME)
        .cloned()
}

fn drew(cell: &Value) -> bool {
    cell["state"]["state"] == "drew" && number(&cell["state"]["draws"]) > 0.0
}

pub(super) fn run(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    session.source(&scene().to_string());
    control(session, "pause", json!({}))?;
    settled(session)?;
    let joined = joined_passes(session, GROUND)?;
    smoke_check!(
        joined.iter().any(|pass| pass == ForeignLitPass::NAME),
        "terrain_joins_a_lit_pass_it_did_not_write",
        json!({ "joins": joined }),
        json!({ "joins": [ForeignLitPass::NAME] })
    );
    let mut cell = Value::Null;
    for _ in 0..FRAMES {
        let rendering = render(session)?;
        if let Some(found) = foreign_cell(&rendering) {
            cell = found;
        }
        if drew(&cell) {
            break;
        }
    }
    smoke_check!(
        drew(&cell),
        "terrain_draws_in_a_lit_pass_it_did_not_write",
        cell,
        json!({ "pass": ForeignLitPass::NAME, "state": { "state": "drew", "draws": above(0) } })
    );
    Ok(())
}
