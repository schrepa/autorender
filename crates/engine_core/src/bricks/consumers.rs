use bevy_ecs::prelude::World;
use serde::Serialize;
use serde_json::{Value, json};

use super::{BrickRegistry, List};
use crate::smoke::{SmokeEntry, SmokeResult, SmokeSession, control, render};

pub struct DetailConsumersList;

impl List for DetailConsumersList {
    const NAME: &'static str = "detail_consumers";
    const SMOKES: &'static [SmokeEntry] = &[SmokeEntry {
        name: "list_detail_consumers",
        brick: Self::NAME,
        gpu: true,
        run: detail_consumers,
    }];
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Cut {
    pub asked: u32,
    pub kept: u32,
    pub cut: u32,
}

impl Cut {
    pub fn of(asked: u32, kept: u32) -> Self {
        let kept = kept.min(asked);
        Self {
            asked,
            kept,
            cut: asked - kept,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DetailConsumer {
    pub name: &'static str,
    pub setting: &'static str,
    pub stat: &'static str,
}

const FRAMES: u32 = 8;

fn scene() -> Value {
    json!({
        "id": "detail-consumers", "revision": 1,
        "camera": { "position": [0, 4, 9], "target": [0, 1, 0], "fov_degrees": 60, "near": 0.1, "far": 1000 },
        "materials": [{ "id": "stone", "color": [0.7, 0.6, 0.5, 1] }],
        "meshes": [{ "id": "box", "source": { "kind": "cube" } }],
        "entities": [
            { "id": "floor", "mesh": "box", "material": "stone", "transform": { "translation": [0, -0.5, 0], "scale": [12, 1, 12] } },
            { "id": "block", "mesh": "box", "material": "stone", "transform": { "translation": [0, 1, 0], "scale": [2, 2, 2] } },
            { "id": "lamp", "light": { "color": [1, 0.8, 0.6], "intensity": 10, "range": 10 }, "transform": { "translation": [0, 3, 2] } },
        ],
    })
}

fn registered(world: &World) -> Vec<(&'static str, DetailConsumer)> {
    world
        .get_resource::<BrickRegistry>()
        .map(|registry| {
            registry
                .consumers()
                .map(|(brick, consumer)| (brick, *consumer))
                .collect()
        })
        .unwrap_or_default()
}

fn named(world: &World, setting: &str) -> bool {
    let Some(registry) = world.get_resource::<BrickRegistry>() else {
        return false;
    };
    let Some((section, field)) = setting
        .split_once('.')
        .and_then(|(key, field)| registry.section(key).map(|section| (section, field)))
    else {
        return false;
    };
    let mut current = section.current(world);
    let Some(fields) = current.as_object_mut() else {
        return false;
    };
    if fields.contains_key(field) {
        return true;
    }
    fields.insert(field.to_owned(), Value::Null);
    section.check(current, "$").is_ok()
}

fn at<'a>(stats: &'a Value, path: &str) -> &'a Value {
    path.split('.').fold(stats, |node, field| &node[field])
}

fn counted(reported: &Value) -> Option<[u64; 3]> {
    let [asked, kept, cut] = ["asked", "kept", "cut"].map(|field| reported[field].as_u64());
    Some([asked?, kept?, cut?])
}

fn detail_consumers(session: &mut dyn SmokeSession) -> SmokeResult {
    session.source(&scene().to_string());
    session.restart(session.bricks());
    control(session, "pause", json!({}))?;
    control(session, "step", json!({ "ticks": 3 }))?;
    let consumers = registered(session.app().world());
    for (brick, consumer) in &consumers {
        crate::smoke_check!(
            named(session.app().world(), consumer.setting),
            "consumer_setting_is_a_section_field",
            json!({ "brick": brick, "consumer": consumer.name, "setting": consumer.setting }),
            json!("<registered section key>.<field of its current JSON>")
        );
    }
    let mut rendering = Value::Null;
    for _ in 0..FRAMES {
        rendering = render(session)?;
    }
    for (brick, consumer) in &consumers {
        let reported = at(&rendering, consumer.stat);
        crate::smoke_check!(
            counted(reported).is_some_and(|[asked, kept, cut]| asked == kept + cut),
            "consumer_stat_is_a_cut",
            json!({ "brick": brick, "consumer": consumer.name, "stat": consumer.stat, "reported": reported }),
            json!({ "asked": "kept + cut", "kept": "integer", "cut": "integer" })
        );
    }
    let Ok(explained) = session.request("explain", json!({ "about": "frame" })) else {
        return Ok(());
    };
    let stats = &explained["result"]["rendering"];
    let records = explained["result"]["records"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    for (brick, consumer) in &consumers {
        if at(stats, consumer.stat)["cut"].as_u64().unwrap_or(0) == 0 {
            continue;
        }
        crate::smoke_check!(
            records.iter().any(|record| record["decided"] == "cut"
                && record["brick"] == *brick
                && record["consumer"] == consumer.name
                && record["setting"] == consumer.setting),
            "cut_is_explained",
            json!({ "brick": brick, "consumer": consumer.name, "records": records }),
            json!({ "decided": "cut", "brick": brick, "consumer": consumer.name, "setting": consumer.setting })
        );
    }
    Ok(())
}
