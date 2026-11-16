use bevy_ecs::world::World;
use engine_core::{
    Asset, AssetResidency, DrawKind, DrawReports, DropReason, DroppedDraws, Error, ErrorCode,
    KindPass, KindPassState, KindRefusal, Request,
};
use engine_registry::CORE_ROW;

use crate::encode::{KindAnswer, Run};

fn brick_of(run: &Run, part: u32) -> &'static str {
    run.machine
        .declared
        .parts
        .get(part as usize)
        .map_or(CORE_ROW, |info| info.package)
}

fn part_of(run: &Run, kind: DrawKind) -> Option<u32> {
    run.installs
        .draw_entries
        .iter()
        .find(|member| member.kind.kind == kind)
        .map(|member| member.part)
}

fn kind_brick(run: &Run, kind: DrawKind) -> &'static str {
    part_of(run, kind).map_or(CORE_ROW, |part| brick_of(run, part))
}

fn dropped_state(run: &Run, kind: DrawKind, pass: &'static str) -> Option<KindPassState> {
    let mut draws = 0;
    let mut most: Option<(DropReason, u32)> = None;
    for dropped in &run.tallies.kinds.dropped {
        if dropped.kind != Some(kind) || dropped.pass != pass {
            continue;
        }
        draws += dropped.count;
        let ahead = most.is_none_or(|(reason, count)| {
            (dropped.reason.faults(), dropped.count) > (reason.faults(), count)
        });
        if ahead {
            most = Some((dropped.reason, dropped.count));
        }
    }
    most.map(|(reason, _)| KindPassState::Dropped { reason, draws })
}

fn state_of(
    run: &Run,
    decl: &engine_core::KindDecl,
    scene: &engine_core::ScenePass,
) -> Option<KindPassState> {
    let pass = scene.name;
    if let Some(skip) = decl.skip(pass, &scene.pass) {
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
        Some(tally) => Some(match (tally.answer, tally.reached) {
            (Some(KindAnswer::Refused(on)), _) => KindPassState::Refused { on },
            (Some(KindAnswer::Waiting(on)), _) => KindPassState::Waiting { on },
            (Some(KindAnswer::Skipped(reason)), _) => KindPassState::Skipped { reason },
            (None, 0) => dropped_state(run, decl.kind, pass).unwrap_or(KindPassState::Silent),
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

pub(super) fn dropped_draws(run: &Run) -> Vec<DroppedDraws> {
    run.tallies
        .kinds
        .dropped
        .iter()
        .map(|dropped| DroppedDraws {
            brick: brick_of(run, dropped.part),
            part: run
                .machine
                .declared
                .parts
                .get(dropped.part as usize)
                .map_or("", |part| part.name),
            kind: dropped.kind.map(DrawKind::name),
            pass: dropped.pass,
            reason: dropped.reason,
            faults: dropped.reason.faults(),
            count: dropped.count,
        })
        .collect()
}

fn kind_fault(pair: &KindPass) -> Option<Error> {
    let error = || {
        Error::new(
            ErrorCode::PassMismatch,
            format!("$.kind_passes.{}.{}", pair.kind, pair.pass),
        )
    };
    match pair.state {
        KindPassState::Silent => Some(error().got("drew_nothing")),
        KindPassState::Dropped { reason, .. } if reason.faults() => Some(error().got(reason)),
        KindPassState::Dropped { .. }
        | KindPassState::Drew { .. }
        | KindPassState::Waiting { .. }
        | KindPassState::Refused { .. }
        | KindPassState::Skipped { .. } => None,
    }
}

fn held_reason(residency: &AssetResidency, dropped: &crate::encode::DroppedTally) -> DropReason {
    let Some(asset) = dropped.asset else {
        return dropped.reason;
    };
    let state = residency.state(asset);
    match asset {
        Asset::Mesh(_) => match state {
            None | Some(Request::Missing) => DropReason::GeometryNotHeld,
            Some(Request::Asked(_)) => DropReason::GeometryWaiting,
            Some(Request::Failed(_)) => DropReason::GeometryFailed,
            Some(Request::Ready(_)) => dropped.reason,
        },
        Asset::Material(_) => match state {
            None | Some(Request::Missing) => DropReason::MaterialNotHeld,
            Some(Request::Asked(_)) => DropReason::MaterialWaiting,
            Some(Request::Failed(_)) => DropReason::MaterialFailed,
            Some(Request::Ready(_)) => dropped.reason,
        },
        Asset::Texture(_) => dropped.reason,
    }
}

fn name_asset_drops(run: &mut Run, world: &World) {
    let Some(residency) = world.get_resource::<AssetResidency>() else {
        return;
    };
    for dropped in &mut run.tallies.kinds.dropped {
        dropped.reason = held_reason(residency, dropped);
    }
}

pub(super) fn kind_passes(run: &mut Run, world: &World) -> Vec<KindPass> {
    name_asset_drops(run, world);
    let Some(passes) = world.get_resource::<engine_core::ScenePasses>() else {
        return Vec::new();
    };
    let mut listed = Vec::new();
    for scene in passes.listed() {
        if scene.bit.is_none() || !run.tallies.kinds.ran.contains(&scene.name) {
            continue;
        }
        for decl in passes.kinds() {
            if !(scene.pass.takes)(decl) {
                continue;
            }
            let Some(state) = state_of(run, decl, scene) else {
                continue;
            };
            listed.push(KindPass {
                brick: kind_brick(run, decl.kind),
                kind: decl.kind.name(),
                pass: scene.name,
                state,
            });
        }
    }
    for pair in &listed {
        let named = (DrawKind(pair.kind), pair.pass);
        let kinds = &mut run.tallies.kinds;
        if kinds.reported.contains(&named) {
            continue;
        }
        let Some(error) = kind_fault(pair) else {
            continue;
        };
        kinds.reported.push(named);
        kinds.problems.push(error.clone());
        if let Some(part) = part_of(run, named.0) {
            run.installs.faults.push((part, error));
        }
    }
    run.tallies.kinds.report = Some(draw_reports(run, &listed));
    listed
}

fn draw_reports(run: &Run, listed: &[KindPass]) -> DrawReports {
    let kinds = listed
        .iter()
        .filter(|pair| match pair.state {
            KindPassState::Waiting { .. }
            | KindPassState::Refused { .. }
            | KindPassState::Dropped { .. }
            | KindPassState::Silent => true,
            KindPassState::Drew { .. } | KindPassState::Skipped { .. } => false,
        })
        .copied()
        .collect();
    let refused = run
        .tallies
        .kinds
        .refused
        .iter()
        .map(|refused| KindRefusal {
            brick: kind_brick(run, refused.kind),
            error: refused.error.clone(),
        })
        .chain(run.machine.over_limit.iter().map(|(part, error)| KindRefusal {
            brick: brick_of(run, *part),
            error: error.clone(),
        }))
        .collect();
    DrawReports {
        kinds,
        refused,
        dropped: dropped_draws(run),
    }
}

pub(super) fn take_reports(run: &mut Run) -> Option<DrawReports> {
    run.tallies.kinds.report.take()
}
