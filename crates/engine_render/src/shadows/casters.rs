use engine_core::collections::{HashMap, HashSet};

use engine_core::{AlphaMode, MaterialHandle, MeshHandle, RenderFrame};

use super::Source;
use crate::batches::BatchPlan;
use crate::gpu_data::{BatchData, CandidateData};

#[derive(Clone, Copy)]
pub(crate) struct ShadowGroup {
    pub mesh: MeshHandle,
    pub source_index: usize,
    pub material: Option<MaterialHandle>,
    pub start: u32,
    pub capacity: u32,
}

type GroupIndices = HashMap<(MeshHandle, Option<MaterialHandle>), u32>;

fn add_to_group(
    groups: &mut Vec<ShadowGroup>,
    counts: &mut Vec<u32>,
    indices: &mut GroupIndices,
    (mesh, material): (MeshHandle, Option<MaterialHandle>),
    source_index: usize,
    count: u32,
) -> u32 {
    let index = match indices.get(&(mesh, material)) {
        Some(&index) => index,
        None => {
            groups.push(ShadowGroup {
                mesh,
                source_index,
                material,
                start: 0,
                capacity: 0,
            });
            counts.push(0);
            let index = groups.len() as u32 - 1;
            indices.insert((mesh, material), index);
            index
        }
    };
    counts[index as usize] += count;
    index
}

pub(super) struct CasterGroups {
    groups: Vec<ShadowGroup>,
    total: u32,
    batch_groups: Vec<u32>,
    key: Option<(u8, u64, u64)>,
    source: Source,
    pub(super) listed: Vec<CandidateData>,
    pub(super) records: Vec<BatchData>,
}

impl CasterGroups {
    pub(super) fn new() -> Self {
        Self {
            groups: Vec::new(),
            total: 0,
            batch_groups: Vec::new(),
            key: None,
            source: Source::Listed,
            listed: Vec::new(),
            records: Vec::new(),
        }
    }

    pub(super) fn groups(&self) -> &[ShadowGroup] {
        &self.groups
    }

    pub(super) fn total(&self) -> u32 {
        self.total
    }

    pub(super) fn source(&self) -> Source {
        self.source
    }

    pub(super) fn batch_group(&self, index: usize) -> u32 {
        self.batch_groups.get(index).copied().unwrap_or(0)
    }

    pub(super) fn clear(&mut self) {
        self.groups.clear();
        self.batch_groups.clear();
        self.key = None;
    }

    pub(super) fn plan(&mut self, plan: &BatchPlan, frame: RenderFrame<'_>, gpu: bool) -> bool {
        let snapshot = frame.snapshot();
        let (source, key) = if gpu {
            (
                Source::Presented,
                (0, frame.identity(), frame.source_revision()),
            )
        } else {
            (
                Source::Listed,
                (
                    1,
                    frame.identity(),
                    frame.caster_revision().max(frame.source_revision()),
                ),
            )
        };
        self.source = source;
        if self.key == Some(key) {
            return false;
        }
        self.key = Some(key);
        self.groups.clear();
        self.batch_groups.clear();
        let mut indices = GroupIndices::default();
        let mut counts = Vec::new();
        let masked: HashSet<MaterialHandle> = snapshot
            .materials
            .iter()
            .filter(|source| matches!(source.material.alpha_mode(), AlphaMode::Mask { .. }))
            .map(|source| source.handle)
            .collect();
        let key = |mesh, material: Option<MaterialHandle>| {
            (mesh, material.filter(|material| masked.contains(material)))
        };
        if !gpu {
            let sources: HashMap<MeshHandle, usize> = snapshot
                .meshes
                .iter()
                .enumerate()
                .map(|(index, source)| (source.handle, index))
                .collect();
            let instances = &snapshot.instances;
            self.listed.clear();
            for &slot in frame.casters() {
                let mesh = instances[slot].mesh;
                let Some(&source_index) = sources.get(&mesh) else {
                    continue;
                };
                let group = add_to_group(
                    &mut self.groups,
                    &mut counts,
                    &mut indices,
                    key(mesh, instances[slot].material),
                    source_index,
                    1,
                );
                self.listed.push(CandidateData {
                    slot: slot as u32,
                    node: 0,
                    batch: group,
                    position: 0,
                });
            }
        } else {
            for batch in &plan.batches {
                let count = batch.instances.end - batch.instances.start;
                let group = add_to_group(
                    &mut self.groups,
                    &mut counts,
                    &mut indices,
                    key(batch.mesh, batch.material),
                    batch.source_index,
                    count,
                );
                self.batch_groups.push(group);
            }
        }
        let mut start = 0;
        self.records.clear();
        for (group, count) in self.groups.iter_mut().zip(&counts) {
            group.start = start;
            group.capacity = *count;
            start += count;
            let mesh = &snapshot.meshes[group.source_index].mesh;
            let bounds = mesh.bounds();
            self.records.push(BatchData {
                start: group.start,
                count: *count,
                index_count: mesh.indices().len() as u32,
                first_index: 0,
                base_vertex: 0,
                first_instance: 0,
                first_plane: 0,
                lod_ranges: [0; 4],
                lod_errors: [0.0, 0.0, 1.0, 0.0],
                group: self.records.len() as u32,
                bounds_min: bounds.min().extend(0.0).to_array(),
                bounds_max: bounds.max().extend(0.0).to_array(),
            });
        }
        self.total = start;
        true
    }
}
