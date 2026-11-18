use std::collections::BTreeMap;
use std::num::NonZeroU64;

use bevy_ecs::resource::Resource;
use bevy_ecs::world::World;
use engine_core::{Error, ErrorCode};
use engine_registry::{Kind, Lands, NameEntry, ReachMut, register};
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::graph::{Bits, Life, ResourceEntry, StaticGraph};
use crate::machine::Machine;

const CHANGES: &str = "skips each listed pass on frames whose number its every does not divide; \
                       the pass's outputs keep their previous result";

#[derive(Deserialize, engine_core::__private::schemars::JsonSchema)]
#[schemars(crate = "engine_core::__private::schemars")]
#[serde(deny_unknown_fields)]
struct Every {
    every: u32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NoArgs {}

#[derive(Resource, Default)]
pub(crate) struct WhatIf {
    asked: BTreeMap<String, NonZeroU64>,
    passes: Vec<(&'static str, Option<String>)>,
    skips: Vec<(usize, NonZeroU64)>,
}

impl WhatIf {
    fn refused(&self, name: &str) -> Option<&str> {
        let mut named = self
            .passes
            .iter()
            .filter(|(known, _)| *known == name)
            .peekable();
        if named.peek().is_none() {
            return Some("unknown_pass");
        }
        named.find_map(|(_, refused)| refused.as_deref())
    }

    fn settle(&mut self) {
        let mut skips = Vec::new();
        for (name, &every) in &self.asked {
            if self.refused(name).is_some() {
                continue;
            }
            skips.extend(
                self.passes
                    .iter()
                    .enumerate()
                    .filter(|(_, (known, _))| known == name)
                    .map(|(pass, _)| (pass, every)),
            );
        }
        self.skips = skips;
    }

    fn value(&self) -> Value {
        let entries: Map<String, Value> = self
            .asked
            .iter()
            .map(|(name, every)| {
                let entry = match self.refused(name) {
                    Some(why) => json!({ "refused": why }),
                    None => json!({ "every": every }),
                };
                (name.clone(), entry)
            })
            .collect();
        Value::Object(entries)
    }
}

fn refusal(graph: &StaticGraph, pass: usize) -> Option<String> {
    let written: Vec<&ResourceEntry> = graph
        .writes(pass)
        .iter()
        .filter_map(|index| graph.resource(index))
        .collect();
    if let Some(target) = written.iter().find(|resource| resource.view) {
        return Some(format!("writes_shared_target {}", target.label));
    }
    written
        .iter()
        .find(|resource| resource.life != Life::Kept)
        .map(|resource| format!("writes_frame_resource {}", resource.label))
}

pub(crate) fn install(world: &mut World, machine: &Machine) {
    let passes = machine
        .declared
        .passes
        .iter()
        .enumerate()
        .map(|(index, pass)| (pass.name, refusal(&machine.graph, index)))
        .collect();
    let mut whatif = world.get_resource_or_init::<WhatIf>();
    whatif.passes = passes;
    whatif.settle();
}

pub(crate) fn skip(world: &World, frame: u64, runs: &mut Bits) -> u64 {
    let Some(whatif) = world.get_resource::<WhatIf>() else {
        return 0;
    };
    let mut skips = 0;
    for (at, &(pass, every)) in whatif.skips.iter().enumerate() {
        if frame % every != 0 {
            runs.remove(pass);
            skips |= 1_u64.checked_shl(at as u32).unwrap_or(1 << 63);
        }
    }
    skips
}

pub(crate) fn changed(world: &World, seen: &mut Vec<(usize, NonZeroU64)>) -> bool {
    let skips = (world.get_resource::<WhatIf>()).map_or(&[][..], |whatif| &whatif.skips);
    if seen.as_slice() == skips {
        return false;
    }
    seen.clear();
    seen.extend_from_slice(skips);
    true
}

fn set(reach: &mut ReachMut<'_>, given: &Value) -> Result<Value, Error> {
    let asked: BTreeMap<String, Every> = engine_core::parse_value_at(given.clone(), "$.args")?;
    let mut changes = Vec::with_capacity(asked.len());
    for (name, Every { every }) in asked {
        let Some(every) = NonZeroU64::new(u64::from(every)) else {
            return Err(
                Error::new(ErrorCode::OutOfRange, format!("$.args.{name}.every"))
                    .at_least(1)
                    .got(every),
            );
        };
        changes.push((name, every));
    }
    let mut whatif = reach.world.get_resource_or_init::<WhatIf>();
    let before = whatif.value();
    for (name, every) in &changes {
        if every.get() == 1 {
            whatif.asked.remove(name);
        } else {
            whatif.asked.insert(name.clone(), *every);
        }
    }
    whatif.settle();
    let skipped: Vec<Value> = changes
        .iter()
        .filter(|(_, every)| every.get() != 1)
        .filter_map(|(name, _)| {
            whatif
                .refused(name)
                .map(|why| json!({ "field": name, "why": why }))
        })
        .collect();
    let mut reply = json!({ "before": before, "after": whatif.value() });
    if !skipped.is_empty() {
        reply["skipped"] = Value::Array(skipped);
    }
    Ok(reply)
}

pub(crate) fn names(world: &mut World) {
    let entry = NameEntry::read("gpu.whatif", Kind::Setting, |reach, given| {
        let NoArgs {} = engine_core::parse_value_at(given.clone(), "$.args")?;
        Ok(reach
            .world
            .get_resource::<WhatIf>()
            .map_or_else(|| json!({}), WhatIf::value))
    })
    .args(
        engine_core::schema_of::<BTreeMap<String, Every>>(),
        json!({ "drawn_pose": { "every": 2 } }),
    )
    .set(Lands::NextFrame, set)
    .changes_frame(CHANGES);
    register(world, entry);
}
