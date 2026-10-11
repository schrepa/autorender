//! Each frame's caster groups: one per mesh and masked
//! material, of the plan's batches (GPU-presented frames) or of a
//! CPU-prepared frame's casters, each with its ID region in every map.

use std::collections::{HashMap, HashSet};

use engine_core::{AlphaMode, MaterialHandle, MeshHandle, RenderFrame};

use super::Source;
use crate::batches::BatchPlan;
use crate::gpu_data::{BatchData, CandidateData};

/// One mesh drawn into shadow maps, with its casters' ID region per map.
#[derive(Clone, Copy)]
pub(crate) struct ShadowGroup {
    pub mesh: MeshHandle,
    /// Index of its source in the snapshot's meshes.
    pub source_index: usize,
    /// The alpha-masked material whose base color alpha
    /// these casters test; `None` casts the whole mesh.
    pub material: Option<MaterialHandle>,
    /// Whether virtual geometry draws its casters' clusters
    /// instead of the mesh.
    pub virtual_geometry: bool,
    /// Its ID region's start and size in each map.
    pub start: u32,
    pub capacity: u32,
}

/// Group keys: a mesh, and the masked material its casters test.
type GroupIndices = HashMap<(MeshHandle, Option<MaterialHandle>), u32>;

/// Adds `count` casters of `mesh` (masked by `material`) to their group,
/// creating it at the end.
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
                virtual_geometry: false,
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
    // Casters per map: every group's capacity.
    total: u32,
    // Group of each plan batch (GPU-presented frames).
    batch_groups: Vec<u32>,
    // Source kind and stamps, and whether virtual geometry draws.
    key: Option<(u8, u64, u64, bool)>,
    source: Source,
    // The casters of CPU-prepared frames, and a record per group.
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

    /// Casters per map: every group's capacity.
    pub(super) fn total(&self) -> u32 {
        self.total
    }

    /// Where this frame's casters come from.
    pub(super) fn source(&self) -> Source {
        self.source
    }

    /// Group of plan batch `index`, for presentation's batch records.
    pub(super) fn batch_group(&self, index: usize) -> u32 {
        self.batch_groups.get(index).copied().unwrap_or(0)
    }

    /// Forgets the groups while shadows are off.
    pub(super) fn clear(&mut self) {
        self.groups.clear();
        self.batch_groups.clear();
        self.key = None;
    }

    /// Plans this frame's groups, one per mesh and masked material: of the
    /// plan's batches (GPU-presented frames when `gpu`) or of `frame`'s
    /// casters (CPU-prepared frames). Returns whether they changed.
    pub(super) fn plan(&mut self, plan: &BatchPlan, frame: RenderFrame<'_>, gpu: bool) -> bool {
        let snapshot = frame.snapshot();
        // Virtual geometry draws where it is enabled.
        let clustered = snapshot.environment.virtual_geometry.enabled;
        let (source, key) = if gpu {
            (
                Source::Presented,
                (0, frame.identity(), frame.source_revision(), clustered),
            )
        } else {
            // Both stamps only grow: a source change (such as a slot turning
            // transparent) regroups casters too.
            (
                Source::Listed,
                (
                    1,
                    frame.identity(),
                    frame.caster_revision().max(frame.source_revision()),
                    clustered,
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
        let mut indices = GroupIndices::new();
        let mut counts = Vec::new();
        // Casters of alpha-masked materials test their texels.
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
                // Transparent surfaces cast no shadow.
                if crate::batches::is_transparent(&instances[slot], &plan.transparent_materials) {
                    continue;
                }
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
                // Transparent surfaces cast no shadow.
                if batch.transparent {
                    self.batch_groups.push(u32::MAX);
                    continue;
                }
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
            // Meshes with joints cast wherever joints move them;
            // virtual geometry (flag 2) casts clusters.
            let skinned = if mesh.joint_count() != 0 { 1.0 } else { 0.0 };
            group.virtual_geometry = clustered && mesh.virtual_geometry().is_some();
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
                skin_envelopes: [0; 4],
                group: self.records.len() as u32,
                bounds_min: bounds.min().extend(skinned).to_array(),
                bounds_max: bounds
                    .max()
                    .extend(if group.virtual_geometry { 2.0 } else { 0.0 })
                    .to_array(),
            });
        }
        self.total = start;
        true
    }
}
