use engine_core::{SmokeFailure, SmokeSession, smoke_check, smoke_eq};
use engine_smoke::{Checked, control, render, settled};
use serde_json::{Value, json};

const HOLDER: &str = "holder";
const HELD: &str = "held";
const SPEAKER: &str = "speaker";
const REMOVED: &str = "entity_removed";
const LOADING: &str = "sound_loading";
const DEVICE_OFF: &str = "audio_failed";
const SAYS_WHY: [&str; 4] = [LOADING, "missing_file", DEVICE_OFF, "place_not_filled"];
const FRAMES: usize = 240;

fn empty() -> Value {
    json!({"id": "entities", "revision": 1, "materials": [], "meshes": [], "entities": []})
}

fn spawned() -> Value {
    json!({"entities": [
        {"id": HOLDER, "transform": {"translation": [0.0, 0.0, 0.0]}},
        {"id": HELD, "parent": HOLDER, "transform": {"translation": [0.0, 1.0, 0.0]}},
        {"id": SPEAKER, "transform": {"translation": [4.0, 0.0, 0.0]},
         "sound": {"file": "silent.ogg", "autoplay": false}},
    ]})
}

fn answer(session: &mut dyn SmokeSession, verb: &str, name: &str, args: Value) -> Value {
    match session.request(verb, json!({"name": name, "args": args})) {
        Ok(reply) | Err(reply) => reply,
    }
}

fn code(reply: &Value) -> Value {
    reply["error"]["code"].clone()
}

fn reason(reply: &Value) -> Value {
    reply["result"]["rows"][0]["refused"]["reason"].clone()
}

fn play(session: &mut dyn SmokeSession) -> Value {
    answer(session, "control", "sound_play", json!({"entity": SPEAKER}))
}

fn plays_or_says_why(reply: &Value) -> bool {
    if reply["ok"] == true {
        reply["result"]["voice"].is_u64()
    } else {
        SAYS_WHY.iter().any(|why| reply["error"]["code"] == *why)
    }
}

pub(super) fn run(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    session.source(&empty().to_string());
    settled(session)?;
    let spawn = control(session, "spawn", spawned())?;
    smoke_eq!(
        json!({"done": spawn["done"], "refused": spawn["refused"]}),
        json!({"done": 3, "refused": 0}),
        "spawn_lands_every_row"
    );
    let first = play(session);
    smoke_check!(
        plays_or_says_why(&first),
        "sound_play_right_after_spawn_plays_or_says_why",
        first,
        json!({"voice": "a number", "or": {"error": {"code": SAYS_WHY}}})
    );
    parent_takes_children(session)?;
    removed_answers_one_code(session)?;
    sound_says_why(session)
}

fn parent_takes_children(session: &mut dyn SmokeSession) -> Checked<()> {
    let removal = control(session, "remove", json!({"entities": [HOLDER]}))?;
    smoke_eq!(
        json!({"done": removal["done"], "removed": removal["rows"][0]["done"]["removed"]}),
        json!({"done": 1, "removed": 2}),
        "remove_takes_the_subtree"
    );
    let why = answer(session, "query", "why", json!({"entity": HELD}));
    smoke_eq!(code(&why), json!(REMOVED), "a_removed_parent_takes_its_children");
    Ok(())
}

fn removed_answers_one_code(session: &mut dyn SmokeSession) -> Checked<()> {
    let remove = answer(session, "control", "remove", json!({"entities": [HELD]}));
    let change = answer(
        session,
        "control",
        "change",
        json!({"changes": [{"entity": HOLDER, "component": "node", "data": {}}]}),
    );
    let why = answer(session, "query", "why", json!({"entity": HOLDER}));
    smoke_eq!(
        json!({"remove": reason(&remove), "change": reason(&change), "why": code(&why)}),
        json!({"remove": REMOVED, "change": REMOVED, "why": REMOVED}),
        "a_removed_id_answers_entity_removed_everywhere"
    );
    Ok(())
}

fn sound_says_why(session: &mut dyn SmokeSession) -> Checked<()> {
    let mut reply = play(session);
    for _ in 0..FRAMES {
        if code(&reply) != LOADING {
            break;
        }
        render(session)?;
        reply = play(session);
    }
    smoke_check!(
        code(&reply) != LOADING && plays_or_says_why(&reply),
        "sound_play_answers_once_the_sound_is_read",
        reply,
        json!({"voice": "a number", "or": {"error": {"code": SAYS_WHY}}})
    );
    control(session, "audio_device", json!({"available": false}))?;
    render(session)?;
    let off = play(session);
    control(session, "audio_device", json!({"available": true}))?;
    smoke_eq!(code(&off), json!(DEVICE_OFF), "sound_play_says_the_device_is_off");
    Ok(())
}
