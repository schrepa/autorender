use engine_core::{Error, ErrorCode};
use engine_places::animation::{Animator, AnimatorScene, Attachment};
use serde_json::{Value, json};

pub(crate) fn check_animator(animator: &Animator, path: &str) -> Result<(), Error> {
    let rig = animator.rig();
    rig.check().map_err(|error| error.within(path))?;
    match animator.start() {
        Some(start) if rig.clip(&start.clip).is_none() => Err(Error::new(
            ErrorCode::InvalidValue,
            format!("{path}.play.clip"),
        )
        .variants(rig.clips.iter().map(|clip| clip.name.clone()))
        .got(&start.clip)),
        Some(_) | None => Ok(()),
    }
}

pub(crate) fn animator_schema() -> Value {
    serde_json::to_value(schemars::schema_for!(AnimatorScene)).unwrap_or(Value::Null)
}

pub(crate) fn animator_example() -> Value {
    json!({
        "rig": {
            "joints": [
                { "name": "root" },
                { "name": "tip", "parent": 0, "translation": [0.0, 1.0, 0.0] },
            ],
            "palette": [{
                "joint": 0,
                "inverse_bind": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }],
            "clips": [{
                "name": "wave",
                "duration": 1.0,
                "tracks": [{
                    "joint": 1,
                    "rotation": {
                        "times": [0.0, 1.0],
                        "values": [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]],
                    },
                }],
            }],
            "parts": [{
                "mesh": "mesh",
                "material": "material",
                "skin": [{ "joints": [0, 0, 0, 0], "weights": [1.0, 0.0, 0.0, 0.0] }],
            }],
        },
        "play": { "clip": "wave" },
    })
}

pub(crate) fn attachment_schema() -> Value {
    serde_json::to_value(schemars::schema_for!(Attachment)).unwrap_or(Value::Null)
}

pub(crate) fn attachment_example() -> Value {
    json!({ "joint": "tip" })
}
