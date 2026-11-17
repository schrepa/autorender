use engine_core::{Error, ErrorCode};
use engine_places_rendering::animation::{Animator, AnimatorScene, Attachment};
use serde_json::Value;

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

pub(crate) fn attachment_schema() -> Value {
    serde_json::to_value(schemars::schema_for!(Attachment)).unwrap_or(Value::Null)
}
