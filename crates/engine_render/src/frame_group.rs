use engine_core::{Error, ErrorCode};

use crate::bindings::Stages;
use crate::pipeline_layouts::StageLimits;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Class {
    Uniform { dynamic: bool },
    Storage,
    Texture,
    Sampler,
    StorageTexture,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FrameVariable {
    pub(crate) brick: &'static str,
    pub(crate) name: &'static str,
    pub(crate) kind: Class,
}

#[derive(Clone, Debug)]
pub(crate) struct Reach {
    pub(crate) label: &'static str,
    pub(crate) pass: u32,
    pub(crate) stages: Stages,
    pub(crate) own_groups: u32,
    pub(crate) own: Vec<(Class, Stages)>,
    pub(crate) reached: Vec<(u32, Stages)>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Variant {
    pub(crate) entries: Vec<(u32, Stages)>,
    pub(crate) members: Vec<u32>,
}

#[derive(Clone, Debug)]
pub(crate) struct FramePlan {
    pub(crate) variants: Vec<Variant>,
    pub(crate) of_pipeline: Vec<u32>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Counted {
    uniform_buffers: u32,
    storage_buffers: u32,
    sampled_textures: u32,
    samplers: u32,
    storage_textures: u32,
}

impl Counted {
    fn add(&mut self, kind: Class) {
        match kind {
            Class::Uniform { .. } => self.uniform_buffers += 1,
            Class::Storage => self.storage_buffers += 1,
            Class::Texture => self.sampled_textures += 1,
            Class::Sampler => self.samplers += 1,
            Class::StorageTexture => self.storage_textures += 1,
        }
    }

    fn over(&self, limits: &StageLimits) -> Option<&'static str> {
        [
            ("uniform_buffers", self.uniform_buffers, limits.uniform_buffers),
            ("storage_buffers", self.storage_buffers, limits.storage_buffers),
            ("sampled_textures", self.sampled_textures, limits.sampled_textures),
            ("samplers", self.samplers, limits.samplers),
            ("storage_textures", self.storage_textures, limits.storage_textures),
        ]
        .into_iter()
        .find(|&(_, count, limit)| count > limit)
        .map(|(what, _, _)| what)
    }
}

const STAGES: [(&str, Stages); 3] = [
    ("vertex", Stages::VERTEX),
    ("fragment", Stages::FRAGMENT),
    ("compute", Stages::COMPUTE),
];

fn dynamic(kind: Class) -> bool {
    matches!(kind, Class::Uniform { dynamic: true })
}

fn storage(kind: Class) -> bool {
    kind == Class::Storage
}

fn fits(
    reach: &Reach,
    entries: &[(u32, Stages)],
    variables: &[FrameVariable],
    limits: &StageLimits,
) -> Result<(), (&'static str, &'static str)> {
    if 1 + reach.own_groups > limits.bind_groups {
        return Err(("layout", "bind_groups"));
    }
    let dynamics = reach.own.iter().filter(|&&(kind, _)| dynamic(kind)).count()
        + entries
            .iter()
            .filter(|&&(variable, _)| dynamic(variables[variable as usize].kind))
            .count();
    if dynamics as u32 > limits.dynamic_uniform_buffers {
        return Err(("layout", "dynamic_uniform_buffers"));
    }
    for (name, stage) in STAGES {
        if !reach.stages.covers(stage) {
            continue;
        }
        let mut counted = Counted::default();
        for &(kind, visible) in &reach.own {
            if visible.covers(stage) {
                counted.add(kind);
            }
        }
        for &(variable, visible) in entries {
            if visible.covers(stage) {
                counted.add(variables[variable as usize].kind);
            }
        }
        if let Some(what) = counted.over(limits) {
            return Err((name, what));
        }
    }
    Ok(())
}

fn refused(
    code: ErrorCode,
    reach: &Reach,
    variables: &[FrameVariable],
    stage: &str,
    what: &str,
) -> Error {
    let mut bricks: Vec<&'static str> = reach
        .reached
        .iter()
        .map(|&(variable, _)| variables[variable as usize].brick)
        .collect();
    bricks.sort_unstable();
    bricks.dedup();
    let reached: Vec<String> = reach
        .reached
        .iter()
        .map(|&(variable, _)| {
            let variable = &variables[variable as usize];
            format!("{}::{}", variable.brick, variable.name)
        })
        .collect();
    Error::new(
        code,
        format!("{}.{stage}.{what}[{}]", reach.label, bricks.join(",")),
    )
    .got(reached)
}

fn joined(first: &[(u32, Stages)], second: &[(u32, Stages)]) -> Vec<(u32, Stages)> {
    let mut entries: Vec<(u32, Stages)> = first.to_vec();
    for &(variable, stages) in second {
        match entries.iter_mut().find(|(known, _)| *known == variable) {
            Some((_, visible)) => *visible = Stages(visible.0 | stages.0),
            None => entries.push((variable, stages)),
        }
    }
    entries.sort_by_key(|&(variable, _)| variable);
    entries
}

fn adds_storage(reach: &Reach, entries: &[(u32, Stages)], variables: &[FrameVariable]) -> bool {
    entries.iter().any(|&(variable, _)| {
        storage(variables[variable as usize].kind)
            && !reach.reached.iter().any(|&(known, _)| known == variable)
    })
}

pub(crate) fn plan(
    variables: &[FrameVariable],
    reaches: &[Reach],
    limits: &StageLimits,
    conflict: &dyn Fn(u32, u32) -> bool,
) -> Result<FramePlan, Vec<Error>> {
    let mut exact: Vec<Variant> = Vec::new();
    let mut over = Vec::new();
    for (index, reach) in reaches.iter().enumerate() {
        let pipeline = index as u32;
        let entries = joined(&[], &reach.reached);
        if let Some(&(variable, _)) = entries
            .iter()
            .find(|&&(variable, _)| conflict(pipeline, variable))
        {
            over.push(Error::new(
                ErrorCode::UsageConflict,
                format!("{}.{}", reach.label, variables[variable as usize].name),
            ));
            continue;
        }
        if let Err((stage, what)) = fits(reach, &entries, variables, limits) {
            over.push(refused(ErrorCode::BindingLimit, reach, variables, stage, what));
            continue;
        }
        let same = exact.iter_mut().find(|variant| {
            variant.entries.len() == entries.len()
                && variant
                    .entries
                    .iter()
                    .zip(&entries)
                    .all(|(known, entry)| known.0 == entry.0)
        });
        match same {
            Some(variant) => {
                variant.entries = joined(&variant.entries, &entries);
                variant.members.push(pipeline);
            }
            None => exact.push(Variant {
                entries,
                members: vec![pipeline],
            }),
        }
    }
    exact.sort_by(|first, second| {
        second
            .entries
            .len()
            .cmp(&first.entries.len())
            .then_with(|| first.members[0].cmp(&second.members[0]))
    });
    let mut merged: Vec<Variant> = Vec::new();
    for variant in exact {
        let passes: Vec<u32> = variant
            .members
            .iter()
            .map(|&member| reaches[member as usize].pass)
            .collect();
        let shares_pass = |target: &Variant| {
            target
                .members
                .iter()
                .any(|&member| passes.contains(&reaches[member as usize].pass))
        };
        let order: Vec<usize> = (0..merged.len())
            .filter(|&index| shares_pass(&merged[index]))
            .chain((0..merged.len()).filter(|&index| !shares_pass(&merged[index])))
            .collect();
        let target = order.into_iter().find(|&index| {
            let entries = joined(&merged[index].entries, &variant.entries);
            merged[index]
                .members
                .iter()
                .chain(&variant.members)
                .all(|&member| {
                    let reach = &reaches[member as usize];
                    !adds_storage(reach, &entries, variables)
                        && fits(reach, &entries, variables, limits).is_ok()
                        && !entries
                            .iter()
                            .any(|&(variable, _)| conflict(member, variable))
                })
        });
        match target {
            Some(index) => {
                let target = &mut merged[index];
                target.entries = joined(&target.entries, &variant.entries);
                target.members.extend(variant.members);
            }
            None => merged.push(variant),
        }
    }
    let mut of_pipeline = vec![0_u32; reaches.len()];
    for (index, variant) in merged.iter_mut().enumerate() {
        variant.members.sort_unstable();
        for &member in &variant.members {
            of_pipeline[member as usize] = index as u32;
        }
    }
    if !over.is_empty() {
        return Err(over);
    }
    Ok(FramePlan {
        variants: merged,
        of_pipeline,
    })
}
