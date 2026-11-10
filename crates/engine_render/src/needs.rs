use engine_core::{AttachPoint, Error, ErrorCode, ScenePasses, TargetName};
use engine_registry::{Read, ReadKind, Registry};

use crate::machine::{Machine, Target, final_target};

fn target_name(machine: &Machine, target: Target) -> Option<&'static str> {
    match target {
        Target::Scene => Some(TargetName::COLOR.name()),
        Target::Depth => Some(TargetName::DEPTH.name()),
        Target::Output(index) => machine
            .outputs
            .get(index as usize)
            .map(|output| output.name),
        Target::Copy(_, view) => target_name(machine, final_target(view)),
        Target::Canvas => None,
    }
}

fn named(machine: &Machine, resource: usize) -> Option<&'static str> {
    machine
        .ids
        .targets
        .iter()
        .find(|(_, id)| id.index() == resource)
        .and_then(|&(target, _)| target_name(machine, target))
}

fn brick_of(machine: &Machine, scenes: &ScenePasses, pass: usize) -> &'static str {
    let scene = machine
        .declared
        .passes
        .get(pass)
        .and_then(|declared| declared.scene.as_ref())
        .filter(|scene| scene.core)
        .and_then(|scene| scenes.get(scene.name));
    match scene {
        Some(listed) => listed.brick,
        None => machine.graph.entry(pass).map_or("", |entry| entry.brick),
    }
}

fn writers_of(machine: &Machine, scenes: &ScenePasses, name: &str) -> Vec<&'static str> {
    let mut found = Vec::new();
    for &(target, id) in &machine.ids.targets {
        if matches!(target, Target::Copy(..)) || target_name(machine, target) != Some(name) {
            continue;
        }
        for pass in machine.graph.writers(id.index()) {
            let brick = brick_of(machine, scenes, pass);
            if !brick.is_empty() && !found.contains(&brick) {
                found.push(brick);
            }
        }
    }
    found
}

fn push(reads: &mut Vec<Read>, read: Read) {
    if !reads.iter().any(|known| {
        known.reader == read.reader && known.kind == read.kind && known.place == read.place
    }) {
        reads.push(read);
    }
}

fn reads_of(machine: &Machine, scenes: &ScenePasses, registry: &Registry) -> Vec<Read> {
    let mut reads = Vec::new();
    for (reader, face) in machine.linker.face_readers() {
        let providers = registry.place_members(&face);
        push(
            &mut reads,
            Read {
                reader,
                kind: ReadKind::Face,
                place: face,
                providers,
            },
        );
    }
    for pass in 0..machine.graph.passes() {
        let reader = brick_of(machine, scenes, pass);
        if reader.is_empty() {
            continue;
        }
        for resource in machine.graph.reads(pass).iter() {
            let Some(name) = named(machine, resource) else {
                continue;
            };
            push(
                &mut reads,
                Read {
                    reader,
                    kind: ReadKind::Target,
                    place: name.to_owned(),
                    providers: writers_of(machine, scenes, name),
                },
            );
        }
    }
    reads
}

fn readers_of(machine: &Machine, pass: usize) -> Vec<usize> {
    let copied = machine.graph.writes(pass);
    (0..machine.graph.passes())
        .filter(|&other| {
            let read = machine.graph.reads(other);
            copied.iter().any(|resource| read.contains(resource))
        })
        .collect()
}

fn unmet_of(machine: &Machine, scenes: &ScenePasses, registry: &Registry) -> Vec<Error> {
    let mut unmet = Vec::new();
    for (pass, resource) in machine.graph.unmet() {
        let target = named(machine, resource);
        let label = target.unwrap_or_else(|| machine.graph.label(resource));
        let readers = if brick_of(machine, scenes, pass).is_empty() {
            readers_of(machine, pass)
        } else {
            vec![pass]
        };
        let listed = match target {
            Some(name) => writers_of(machine, scenes, name),
            None => machine
                .graph
                .writers(resource)
                .map(|writer| brick_of(machine, scenes, writer))
                .filter(|brick| !brick.is_empty())
                .collect(),
        };
        for reading in readers {
            let Some(entry) = machine.graph.entry(reading) else {
                continue;
            };
            let reader = brick_of(machine, scenes, reading);
            let could = target
                .map(|name| registry.could_write(name, entry.layer))
                .unwrap_or_default();
            unmet.push(
                Error::new(
                    ErrorCode::ReadBeforeWrite,
                    format!("$.bricks.{reader}.passes.{}.reads.{label}", entry.label),
                )
                .got(serde_json::json!({
                    "before": entry.layer.map(AttachPoint::name),
                    "listed_writers": listed,
                }))
                .nearest(could),
            );
        }
    }
    unmet
}

pub(crate) fn report(machine: &Machine, scenes: &ScenePasses, registry: &mut Registry) {
    let reads = reads_of(machine, scenes, registry);
    registry.note_reads(&[ReadKind::Face, ReadKind::Target], reads);
    for error in unmet_of(machine, scenes, registry) {
        registry.report_unmet(error);
    }
}
