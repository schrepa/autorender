//! The environment control: sections of a scene environment, named ones
//! replacing their resources, merged over their current values with
//! `merge`.

use serde_json::{Value, json};

use super::rendered;
use crate::app_host::AppHost;
use engine_dev::{DevError, EnvironmentControl};

/// Applies `control`, sent as `source`, then renders: the sections before
/// and after.
pub(super) fn set(
    app: &mut AppHost,
    source: &str,
    control: EnvironmentControl,
    (width, height): (u32, u32),
) -> Result<Value, DevError> {
    // The sections of a scene environment, validated alike; the
    // named ones replace their resources, the others stay.
    let current = |app: &AppHost| {
        serde_json::to_value(engine_scene::SceneEnvironment::current(
            app.engine.app.world(),
        ))
        .expect("environments serialize")
    };
    let before = current(app);
    let environment = if control.merge {
        // Each named section over its current values.
        let mut request: Value = serde_json::from_str(source).expect("the control parsed as JSON");
        let mut merged = serde_json::Map::new();
        for (key, value) in request.as_object_mut().into_iter().flatten() {
            if ["op", "version", "merge"].contains(&key.as_str()) {
                continue;
            }
            let mut section = before[key.as_str()].clone();
            overlay(&mut section, value.take());
            merged.insert(key.clone(), section);
        }
        engine_scene::parse_json::<engine_scene::SceneEnvironment>(
            &Value::Object(merged).to_string(),
        )?
    } else {
        control.sections()
    };
    environment.validate("$")?;
    environment.check_features(app.engine.app.world(), "$")?;
    environment.apply_sections(app.engine.app.world_mut());
    rendered(app.dev_render(width, height), "environment changed")?;
    let after = current(app);
    let named = environment.sections();
    let pick = |all: &Value| {
        named
            .iter()
            .map(|&key| (key.to_owned(), all[key].clone()))
            .collect::<serde_json::Map<_, _>>()
    };
    Ok(json!({
        "temporary": true,
        "environment_changed": named,
        "before": pick(&before),
        "after": pick(&after),
    }))
}

/// `patch`'s fields over `base`'s, objects merged field by field.
fn overlay(base: &mut Value, patch: Value) {
    match (base, patch) {
        (Value::Object(base), Value::Object(patch)) => {
            for (key, value) in patch {
                overlay(base.entry(key).or_insert(Value::Null), value);
            }
        }
        (base, patch) => *base = patch,
    }
}
