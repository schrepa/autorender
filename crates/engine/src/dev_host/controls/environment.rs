use serde_json::Value;

use super::rendered;
use crate::app_host::AppHost;
use engine_dev::results::{EnvironmentResult, result};
use engine_core::Error;
use engine_dev::EnvironmentControl;

pub(super) fn set(
    app: &mut AppHost,
    args: Value,
    control: EnvironmentControl,
    (width, height): (u32, u32),
) -> Result<Value, Error> {
    let current = |app: &AppHost| {
        serde_json::to_value(engine_scene::SceneEnvironment::current(
            app.engine.app.world(),
        ))
        .expect("environment_serializes")
    };
    let before = current(app);
    let environment = if control.merge {
        let mut merged = serde_json::Map::new();
        for (key, value) in args.as_object().into_iter().flatten() {
            if key == "merge" {
                continue;
            }
            let mut section = before[key.as_str()].clone();
            overlay(&mut section, value.clone());
            merged.insert(key.clone(), section);
        }
        engine_scene::parse_json::<engine_scene::SceneEnvironment>(
            &Value::Object(merged).to_string(),
        )?
    } else {
        control.sections()
    };
    environment.apply_sections(app.engine.app.world_mut(), "$")?;
    rendered(app.dev_render(width, height))?;
    let after = current(app);
    let named = environment.ordered_keys(app.engine.app.world());
    let pick = |all: &Value| {
        named
            .iter()
            .map(|key| (key.clone(), all[key.as_str()].clone()))
            .collect::<serde_json::Map<_, _>>()
    };
    Ok(result::environment(EnvironmentResult {
        temporary: true,
        environment_changed: named.clone(),
        before: pick(&before),
        after: pick(&after),
    }))
}

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
