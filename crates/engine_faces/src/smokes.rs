use engine_core::smoke::{Checked, SmokeResult, SmokeSession, ask, control};
use engine_core::{Absent, BrickRegistry, PlaceKind, smoke_check};
use serde_json::{Value, json};

const FRAMES: u32 = 8;

const FRAME_MS: f64 = 1000.0 / 60.0;

const STEPS: u32 = 3;

const UNLINKED: [&str; 4] = [
    "shader_link",
    "face_missing",
    "slot_unfilled",
    "render_failed",
];

const LEFT_OUT: [&str; 2] = ["slot_unfilled", "face_missing"];

const NAMED_AT: [&str; 2] = ["$.slots.", "$.faces."];

struct Members {
    kind: PlaceKind,
    bricks: Vec<&'static str>,
    required: Vec<&'static str>,
}

fn members_of(session: &mut dyn SmokeSession, place: &'static str) -> Option<Members> {
    let registry = session.app().world().get_resource::<BrickRegistry>()?;
    let kind = registry.place(place)?.kind;
    let bricks = registry.members(place);
    let mut required: Vec<&'static str> = Vec::new();
    for filled in bricks
        .iter()
        .filter_map(|&name| registry.brick(name))
        .flat_map(|brick| &brick.fills)
    {
        let unanswered = registry
            .place(filled.place)
            .is_some_and(|known| known.face.absent == Absent::Required);
        if unanswered && !required.contains(&filled.place) {
            required.push(filled.place);
        }
    }
    Some(Members {
        kind,
        bricks,
        required,
    })
}

fn scene() -> String {
    json!({
        "id": "place-frames", "revision": 1,
        "camera": { "position": [0, 4, 9], "target": [0, 1, 0], "fov_degrees": 60, "near": 0.1, "far": 1000 },
        "materials": [
            { "id": "stone", "color": [0.7, 0.6, 0.5, 1] },
            { "id": "paint", "color": [0.25, 0.6, 0.85, 1] },
        ],
        "meshes": [{ "id": "box", "source": { "kind": "cube" } }],
        "entities": [
            { "id": "floor", "mesh": "box", "material": "stone", "transform": { "translation": [0, -0.5, 0], "scale": [12, 1, 12] } },
            { "id": "block", "mesh": "box", "material": "paint", "transform": { "translation": [0, 1, 0], "scale": [2, 2, 2] } },
            { "id": "lamp", "light": { "color": [1, 0.8, 0.6], "intensity": 10, "range": 10 }, "transform": { "translation": [0, 3, 2] } },
            { "id": "ground", "terrain": { "origin": [-128, -20, -128], "tiles": [2, 2], "spacing": 1, "layers": [{ "type": "noise", "seed": 3, "height": 8, "wavelength": 120 }] } },
        ],
    })
    .to_string()
}

fn shown() -> Value {
    json!({ "name": "frames", "args": { "count": FRAMES, "interval_ms": FRAME_MS } })
}

fn described(
    session: &mut dyn SmokeSession,
    place: &'static str,
    members: &Members,
) -> Checked<()> {
    let listing = ask(session, "describe", json!({}))?;
    let bricks = listing["bricks"].as_array();
    for &name in &members.bricks {
        let described = bricks
            .into_iter()
            .flatten()
            .find(|brick| brick["name"] == name);
        let placed = described.is_some_and(|brick| {
            let entries = match members.kind {
                PlaceKind::Slot => &brick["fills"],
                PlaceKind::List => &brick["joins"],
            };
            entries
                .as_array()
                .is_some_and(|entries| entries.iter().any(|entry| names(entry, place)))
        });
        smoke_check!(
            placed,
            "describe_lists_the_members_in_their_place",
            &described,
            &json!({ "name": name, "place": place })
        );
    }
    Ok(())
}

fn names(entry: &Value, place: &str) -> bool {
    entry.as_str() == Some(place) || entry["place"] == place
}

fn healthy(rendering: &Value) -> Checked<()> {
    let disabled = &rendering["disabled_features"];
    smoke_check!(
        disabled.as_array().is_none_or(Vec::is_empty),
        "no_part_is_disabled",
        disabled,
        &json!([])
    );
    Ok(())
}

fn presented(session: &mut dyn SmokeSession) -> Checked<()> {
    let reply = session.request("control", shown());
    let refused = reply
        .as_ref()
        .err()
        .map(|refused| refused["error"].clone())
        .unwrap_or_default();
    smoke_check!(
        !UNLINKED.iter().any(|code| refused["code"] == *code),
        "the_place_links_with_its_faces",
        &refused,
        &json!({ "none_of": UNLINKED })
    );
    smoke_check!(
        reply.is_ok(),
        "frames_present",
        &refused,
        &json!({ "frames": FRAMES })
    );
    let rendering = reply
        .map(|mut answered| answered["result"]["rendering"].take())
        .unwrap_or_default();
    healthy(&rendering)?;
    smoke_check!(
        rendering["render_passes"]
            .as_array()
            .is_some_and(|passes| !passes.is_empty()),
        "render_passes_run",
        &rendering["render_passes"],
        &json!({ "not": [] })
    );
    Ok(())
}

fn left_out(session: &mut dyn SmokeSession, members: &Members) -> Checked<()> {
    let reply = match session.request("control", json!({ "name": "pause", "args": {} })) {
        Ok(_) => {
            session.step(STEPS);
            session.request("control", shown())
        }
        Err(refused) => Err(refused),
    };
    let refused = match reply {
        Ok(mut answered) => return healthy(&answered["result"]["rendering"].take()),
        Err(mut refused) => refused["error"].take(),
    };
    let code = refused["code"].as_str().unwrap_or_default();
    let named = refused["path"]
        .as_str()
        .and_then(|path| NAMED_AT.iter().find_map(|at| path.strip_prefix(at)))
        .unwrap_or_default();
    smoke_check!(
        LEFT_OUT.contains(&code) && members.required.contains(&named),
        "left_out_renders_or_names_the_place",
        &refused,
        &json!({ "frames": FRAMES, "or_refused": { "code": LEFT_OUT, "naming": members.required } })
    );
    Ok(())
}

fn frames(session: &mut dyn SmokeSession, place: &'static str) -> SmokeResult {
    let Some(members) = members_of(session, place) else {
        return Ok(());
    };
    let original = session.bricks();
    session.source(&scene());
    session.restart(original.clone());
    control(session, "pause", json!({}))?;
    session.step(STEPS);
    described(session, place, &members)?;
    presented(session)?;
    session.restart(original.clone().without(members.bricks.iter().copied()));
    left_out(session, &members)?;
    session.restart(original);
    control(session, "pause", json!({}))?;
    Ok(())
}

pub fn lighting(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "lighting")
}

pub fn meshes(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "meshes")
}

pub fn sun(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "sun")
}

pub fn shadows(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "shadows")
}

pub fn lights(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "lights")
}

pub fn sky(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "sky")
}

pub fn antialiasing(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "antialiasing")
}

pub fn exposure(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "exposure")
}

pub fn tonemap(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "tonemap")
}

pub fn bloom(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "bloom")
}

pub fn dither(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "dither")
}

pub fn reflections(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "reflections")
}

pub fn probe_light(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "probe_light")
}

pub fn occupancy(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "occupancy")
}

pub fn placed(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "placed")
}

pub fn cover(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "cover")
}

pub fn horizons(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "horizons")
}

pub fn heightfield(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "heightfield")
}

pub fn ground_tiles(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "ground_tiles")
}

pub fn water(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "water")
}

pub fn ground_cover(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "ground_cover")
}

pub fn scatter(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "scatter")
}

pub fn terrain_shading(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "terrain_shading")
}

pub fn probes(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "probes")
}

pub fn particles(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "particles")
}

pub fn debug_view(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "debug_view")
}

pub fn post_effects(session: &mut dyn SmokeSession) -> SmokeResult {
    frames(session, "post_effects")
}
