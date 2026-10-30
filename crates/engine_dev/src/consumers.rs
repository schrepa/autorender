use engine_core::BrickRegistry;
use serde_json::Value;

const CUT_FIELDS: [&str; 3] = ["asked", "kept", "cut"];

pub(crate) fn check(registry: &BrickRegistry, stats: &Value, problems: &mut Vec<String>) {
    let mut names: Vec<&str> = Vec::new();
    for (brick, consumer) in registry.consumers() {
        let name = consumer.name;
        if names.contains(&name) {
            problems.push(format!(
                "detail consumer {name} ({brick}) is registered twice"
            ));
        }
        names.push(name);
        let named = consumer
            .setting
            .split_once('.')
            .and_then(|(key, field)| registry.section(key).map(|section| (section, field)));
        let known = named.is_some_and(|(section, field)| {
            let schema = serde_json::to_value(section.schema()).unwrap_or_default();
            schema["properties"].get(field).is_some()
        });
        if !known {
            problems.push(format!(
                "detail consumer {name} ({brick}) names setting {}: no registered section has \
                 that field",
                consumer.setting
            ));
        }
        if !is_cut(stat_at(stats, consumer.stat), stats) {
            problems.push(format!(
                "detail consumer {name} ({brick}) names stat {}: the frame statistics hold no \
                 Cut {{asked, kept, cut}} there",
                consumer.stat
            ));
        }
    }
}

fn resolved<'a>(node: &'a Value, root: &'a Value) -> &'a Value {
    if let Some(reference) = node["$ref"].as_str() {
        let name = reference.rsplit('/').next().unwrap_or_default();
        return resolved(&root["$defs"][name], root);
    }
    for key in ["allOf", "anyOf", "oneOf"] {
        if let Some(options) = node[key].as_array() {
            let options: Vec<&Value> = options
                .iter()
                .filter(|option| option["type"] != "null")
                .collect();
            if let [only] = options[..] {
                return resolved(only, root);
            }
        }
    }
    node
}

fn stat_at<'a>(stats: &'a Value, path: &str) -> Option<&'a Value> {
    let mut node = stats;
    for field in path.split('.') {
        node = resolved(node, stats)["properties"].get(field)?;
    }
    Some(node)
}

fn is_cut(node: Option<&Value>, stats: &Value) -> bool {
    let Some(fields) = node.and_then(|node| resolved(node, stats)["properties"].as_object()) else {
        return false;
    };
    let integer = |field: &str| {
        fields
            .get(field)
            .is_some_and(|field| field["type"] == "integer")
    };
    fields.len() == CUT_FIELDS.len() && CUT_FIELDS.into_iter().all(integer)
}
