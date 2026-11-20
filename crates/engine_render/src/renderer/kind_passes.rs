use bevy_ecs::world::World;
use engine_core::{
    Assets, DrawReports, DropReason, DroppedDraws, Error, ErrorCode, KindPass, KindPassState,
};
use engine_registry::CORE_ROW;

use crate::encode::{DroppedTally, Run};

fn brick_of(run: &Run, part: u32) -> &'static str {
    run.machine
        .declared
        .parts
        .get(part as usize)
        .map_or(CORE_ROW, |info| info.package)
}

fn part_name(run: &Run, part: u32) -> &'static str {
    run.machine
        .declared
        .parts
        .get(part as usize)
        .map_or("", |info| info.name)
}

fn pass_name(run: &Run, pass: u32) -> &'static str {
    run.machine
        .declared
        .passes
        .get(pass as usize)
        .map_or("", |declared| declared.name)
}

fn kind_name(run: &Run, kind: u32) -> &'static str {
    run.tallies
        .kinds
        .kinds
        .get(kind as usize)
        .map_or("", |kind| kind.name())
}

fn kind_brick(run: &Run, kind: u32) -> &'static str {
    run.installs
        .draw_entries
        .iter()
        .find(|member| member.kind == kind)
        .map_or(CORE_ROW, |member| brick_of(run, member.part))
}

fn dropped_state(dropped: &[DroppedTally], kind: u32, pass: u32) -> Option<KindPassState> {
    let mut draws = 0;
    let mut most: Option<(DropReason, u32)> = None;
    for dropped in dropped {
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

fn state_of(run: &Run, at: usize) -> Option<KindPassState> {
    let kinds = &run.tallies.kinds;
    let cell = kinds.cells.get(at)?;
    let (scene, kind) = kinds.kind_at(at);
    if !cell.joins || !kinds.ran.get(scene as usize).copied().unwrap_or(false) {
        return None;
    }
    if cell.begun {
        let pass = kinds
            .scenes
            .get(scene as usize)
            .map_or(u32::MAX, |scene| scene.1);
        return Some(match cell.reached {
            0 => dropped_state(&kinds.dropped, kind, pass)
                .or(cell.skipped.map(|reason| KindPassState::Skipped { reason }))
                .unwrap_or(KindPassState::Silent),
            draws => KindPassState::Drew { draws },
        });
    }
    let mut parts = run
        .installs
        .draw_entries
        .iter()
        .filter(|member| member.kind == kind)
        .map(|member| member.part)
        .peekable();
    let off = parts.peek().is_some() && parts.all(|part| !run.drawable(part));
    (!off).then_some(KindPassState::Silent)
}

pub(super) fn dropped_draws(run: &Run) -> Vec<DroppedDraws> {
    run.tallies
        .kinds
        .last_dropped
        .iter()
        .map(|dropped| DroppedDraws {
            brick: brick_of(run, dropped.part),
            part: part_name(run, dropped.part),
            kind: dropped.kind.map(|kind| kind_name(run, kind)),
            pass: pass_name(run, dropped.pass),
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
        | KindPassState::Skipped { .. } => None,
    }
}

fn held_reason(assets: &Assets, dropped: &DroppedTally) -> DropReason {
    let Some(asset) = dropped.asset else {
        return dropped.reason;
    };
    let mesh = dropped.reason == DropReason::GeometryNotResident;
    let named = |geometry, material| if mesh { geometry } else { material };
    if assets.name(asset).is_none() {
        named(DropReason::GeometryNotInLoad, DropReason::MaterialNotInLoad)
    } else if assets.refusal(asset).is_some() {
        named(DropReason::GeometryFailed, DropReason::MaterialFailed)
    } else if !assets.ready(asset) {
        named(DropReason::GeometryWaiting, DropReason::MaterialWaiting)
    } else {
        dropped.reason
    }
}

fn name_asset_drops(run: &mut Run, world: &World) {
    let Some(assets) = world.get_resource::<Assets>() else {
        return;
    };
    for dropped in &mut run.tallies.kinds.dropped {
        dropped.reason = held_reason(assets, dropped);
    }
}

fn add_totals(run: &mut Run) {
    let kinds = &mut run.tallies.kinds;
    for dropped in &kinds.dropped {
        let key = DroppedTally {
            asset: None,
            count: 0,
            ..*dropped
        };
        match kinds.totals.iter_mut().find(|(known, _)| *known == key) {
            Some((_, total)) => *total += u64::from(dropped.count),
            None => kinds.totals.push((key, u64::from(dropped.count))),
        }
    }
}

#[cfg(feature = "dev")]
fn dropped_name(run: &Run, dropped: &DroppedTally) -> String {
    let owner = match dropped.kind {
        Some(kind) => kind_name(run, kind),
        None => part_name(run, dropped.part),
    };
    format!("{owner}/{}", pass_name(run, dropped.pass))
}

#[cfg(feature = "dev")]
fn reason_name(reason: DropReason) -> String {
    match serde_json::to_value(reason) {
        Ok(serde_json::Value::String(name)) => name,
        Ok(_) | Err(_) => format!("{reason:?}"),
    }
}

fn uncounted(state: Option<KindPassState>) -> Option<KindPassState> {
    state.map(|state| match state {
        KindPassState::Drew { .. } => KindPassState::Drew { draws: 0 },
        KindPassState::Dropped { reason, .. } => KindPassState::Dropped { reason, draws: 0 },
        KindPassState::Skipped { .. } | KindPassState::Silent => state,
    })
}

fn same_states(now: &[Option<KindPassState>], last: &[Option<KindPassState>]) -> bool {
    now.len() == last.len()
        && (now.iter().zip(last)).all(|(now, last)| uncounted(*now) == uncounted(*last))
}

fn same_drops(now: &[DroppedTally], last: &[DroppedTally]) -> bool {
    now.len() == last.len()
        && (now.iter().zip(last))
            .all(|(now, last)| DroppedTally { count: 0, ..*now } == DroppedTally { count: 0, ..*last })
}

pub(super) fn settle(run: &mut Run, world: &World) {
    name_asset_drops(run, world);
    add_totals(run);
    let mut states = std::mem::take(&mut run.tallies.kinds.states);
    for (at, state) in states.iter_mut().enumerate() {
        *state = state_of(run, at);
    }
    let kinds = &mut run.tallies.kinds;
    kinds.states = states;
    if kinds.settled
        && same_states(&kinds.states, &kinds.last_states)
        && same_drops(&kinds.dropped, &kinds.last_dropped)
    {
        return;
    }
    let kinds = &mut run.tallies.kinds;
    kinds.settled = true;
    kinds.last_states.clone_from(&kinds.states);
    kinds.last_dropped.clone_from(&kinds.dropped);
    let mut listed = Vec::new();
    for (at, state) in run.tallies.kinds.states.iter().enumerate() {
        let Some(state) = *state else {
            continue;
        };
        let (scene, kind) = run.tallies.kinds.kind_at(at);
        let pass = run
            .tallies
            .kinds
            .scenes
            .get(scene as usize)
            .map_or("", |scene| scene.0);
        listed.push((
            at,
            KindPass {
                brick: kind_brick(run, kind),
                kind: kind_name(run, kind),
                pass,
                state,
            },
        ));
    }
    for (at, pair) in &listed {
        let Some(cell) = run.tallies.kinds.cells.get_mut(*at) else {
            continue;
        };
        if cell.reported {
            continue;
        }
        let Some(error) = kind_fault(pair) else {
            continue;
        };
        cell.reported = true;
        run.tallies.kinds.problems.push(error.clone());
        run.installs.faults.push(error);
    }
    let passes: Vec<KindPass> = listed.into_iter().map(|(_, pair)| pair).collect();
    let kinds = passes
        .iter()
        .filter(|pair| match pair.state {
            KindPassState::Dropped { .. } | KindPassState::Silent => true,
            KindPassState::Drew { .. } | KindPassState::Skipped { .. } => false,
        })
        .copied()
        .collect();
    run.tallies.kinds.report = Some(DrawReports {
        kinds,
        dropped: dropped_draws(run),
    });
    run.tallies.kinds.passes = passes;
}

pub(super) fn take_reports(run: &mut Run) -> Option<DrawReports> {
    run.tallies.kinds.report.take()
}

#[cfg(feature = "dev")]
pub(super) fn dropped_totals(run: &Run) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for (dropped, total) in &run.tallies.kinds.totals {
        let key = format!(
            "{}/{}",
            dropped_name(run, dropped),
            reason_name(dropped.reason)
        );
        map.insert(key, serde_json::json!(total));
    }
    serde_json::Value::Object(map)
}
