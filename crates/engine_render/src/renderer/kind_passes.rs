use bevy_ecs::world::World;
use engine_core::{DrawKind, Error, ErrorCode, KindPass, KindPassState};

use crate::encode::Run;

fn drawn_kinds(world: &World) -> Vec<DrawKind> {
    let mut kinds: Vec<DrawKind> = Vec::new();
    if let Some(table) = world.get_resource::<engine_core::DrawnTable>() {
        for batch in table.batches() {
            if !kinds.contains(&batch.kind) {
                kinds.push(batch.kind);
            }
        }
    }
    kinds
}

fn state_of(run: &Run, decl: &engine_core::KindDecl, pass: &'static str) -> Option<KindPassState> {
    if let Some(skip) = decl.skip(pass) {
        return Some(KindPassState::Skipped {
            reason: skip.reason,
        });
    }
    let tallied = run
        .tallies
        .kinds
        .listed
        .iter()
        .find(|tally| tally.kind == decl.kind && tally.pass == pass);
    match tallied {
        Some(tally) => Some(match (tally.skipped, tally.draws) {
            (Some(reason), _) => KindPassState::Skipped { reason },
            (None, 0) => KindPassState::Silent,
            (None, draws) => KindPassState::Drew { draws },
        }),
        None => {
            let members: Vec<u32> = run
                .installs
                .draw_entries
                .iter()
                .filter(|member| member.kind.kind == decl.kind)
                .map(|member| member.part)
                .collect();
            let off = !members.is_empty() && members.iter().all(|&part| !run.drawable(part));
            (!off).then_some(KindPassState::Silent)
        }
    }
}

pub(super) fn kind_passes(run: &mut Run, world: &World) -> Vec<KindPass> {
    let Some(passes) = world.get_resource::<engine_core::ScenePasses>() else {
        return Vec::new();
    };
    let drawn = drawn_kinds(world);
    let mut listed = Vec::new();
    for scene in passes.listed() {
        if scene.bit.is_none() || !run.tallies.kinds.ran.contains(&scene.name) {
            continue;
        }
        for decl in passes.kinds() {
            if !drawn.contains(&decl.kind) || !(scene.pass.takes)(decl) {
                continue;
            }
            let Some(state) = state_of(run, decl, scene.name) else {
                continue;
            };
            listed.push(KindPass {
                kind: decl.kind.name(),
                pass: scene.name,
                state,
            });
        }
    }
    for pair in &listed {
        let named = (DrawKind(pair.kind), pair.pass);
        let kinds = &mut run.tallies.kinds;
        if pair.state != KindPassState::Silent || kinds.reported.contains(&named) {
            continue;
        }
        kinds.reported.push(named);
        let error = Error::new(
            ErrorCode::PassMismatch,
            format!("$.kind_passes.{}.{}", pair.kind, pair.pass),
        )
        .got("drew_nothing");
        kinds.problems.push(error.clone());
        let owner = run
            .installs
            .draw_entries
            .iter()
            .find(|member| member.kind.kind == named.0)
            .map(|member| member.part);
        if let Some(part) = owner {
            run.installs.faults.push((part, error));
        }
    }
    listed
}
