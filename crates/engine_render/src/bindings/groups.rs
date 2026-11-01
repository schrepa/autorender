use engine_core::{Error, ErrorCode, Expected};

use super::table::{Kind, Stages, Table};
use crate::pipeline_layouts::StageLimits;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResolvedBinding {
    pub group: u32,
    pub number: u32,
    pub name: &'static str,
    pub kind: Kind,
    pub stages: Stages,
    pub owner: &'static str,
}

#[derive(Clone, Debug)]
pub struct ResolvedLayout {
    pub label: &'static str,
    pub groups: u32,
    pub bindings: Vec<ResolvedBinding>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StageCounts {
    pub uniform_buffers: u32,
    pub storage_buffers: u32,
    pub sampled_textures: u32,
    pub samplers: u32,
    pub storage_textures: u32,
}

const STAGES: [(&str, Stages); 3] = [
    ("vertex", Stages::VERTEX),
    ("fragment", Stages::FRAGMENT),
    ("compute", Stages::COMPUTE),
];

impl ResolvedLayout {
    pub fn resolve(
        label: &'static str,
        groups: &[&'static Table],
    ) -> Result<Self, Error> {
        let mut bindings: Vec<ResolvedBinding> = Vec::new();
        for (group, table) in groups.iter().enumerate() {
            let group = group as u32;
            let resolved: Vec<ResolvedBinding> = table
                .bindings
                .iter()
                .enumerate()
                .map(|(number, binding)| ResolvedBinding {
                    group,
                    number: number as u32,
                    name: binding.name,
                    kind: binding.kind,
                    stages: binding.stages,
                    owner: table.label,
                })
                .collect();
            for binding in resolved {
                if let Some(known) = bindings.iter().find(|known| known.name == binding.name) {
                    return Err(Error::new(
                        ErrorCode::DuplicateBrick,
                        format!("$.layouts.{label}.{}", binding.name),
                    )
                    .expected(Expected::Unique)
                    .got(known.owner));
                }
                bindings.push(binding);
            }
        }
        Ok(Self {
            label,
            groups: groups.len() as u32,
            bindings,
        })
    }

    pub fn binding(&self, name: &str) -> Option<&ResolvedBinding> {
        self.bindings.iter().find(|binding| binding.name == name)
    }

    pub fn counts(&self, stage: Stages) -> StageCounts {
        let mut counts = StageCounts::default();
        for binding in &self.bindings {
            if binding.stages.0 & stage.0 == 0 {
                continue;
            }
            match binding.kind {
                Kind::Uniform { .. } => counts.uniform_buffers += 1,
                Kind::Texture { .. } => counts.sampled_textures += 1,
                Kind::Sampler(_) => counts.samplers += 1,
            }
        }
        counts
    }

    pub fn check(&self, limits: &StageLimits) -> Result<(), Error> {
        if self.groups > limits.bind_groups {
            return Err(Error::new(
                ErrorCode::BindingLimit,
                format!("$.layouts.{}.bind_groups", self.label),
            )
            .at_most(u64::from(limits.bind_groups))
            .got(self.groups));
        }
        let dynamic = self
            .bindings
            .iter()
            .filter(|binding| matches!(binding.kind, Kind::Uniform { dynamic: true, .. }))
            .count() as u32;
        if dynamic > limits.dynamic_uniform_buffers {
            return Err(Error::new(
                ErrorCode::BindingLimit,
                format!("$.layouts.{}.dynamic_uniform_buffers", self.label),
            )
            .at_most(u64::from(limits.dynamic_uniform_buffers))
            .got(dynamic));
        }
        for (stage_name, stage) in STAGES {
            let counts = self.counts(stage);
            let checks = [
                (
                    "uniform_buffers",
                    counts.uniform_buffers,
                    limits.uniform_buffers,
                ),
                (
                    "storage_buffers",
                    counts.storage_buffers,
                    limits.storage_buffers,
                ),
                (
                    "sampled_textures",
                    counts.sampled_textures,
                    limits.sampled_textures,
                ),
                ("samplers", counts.samplers, limits.samplers),
                (
                    "storage_textures",
                    counts.storage_textures,
                    limits.storage_textures,
                ),
            ];
            for (what, count, limit) in checks {
                if count <= limit {
                    continue;
                }
                let mut owners: Vec<&'static str> = self
                    .bindings
                    .iter()
                    .filter(|binding| binding.stages.0 & stage.0 != 0)
                    .map(|binding| binding.owner)
                    .collect();
                owners.sort_unstable();
                owners.dedup();
                return Err(Error::new(
                    ErrorCode::BindingLimit,
                    format!("$.layouts.{}.{stage_name}.{what}", self.label),
                )
                .at_most(u64::from(limit))
                .got((count, owners)));
            }
        }
        Ok(())
    }
}
