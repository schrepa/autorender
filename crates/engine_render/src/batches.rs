#![expect(
    clippy::expect_used,
    reason = "materials' textures are validated before they are ranked"
)]

use std::{fmt, ops::Range};

use engine_core::collections::HashMap;

use engine_core::{MaterialHandle, MeshHandle, RenderFrame, RenderSnapshot, TextureHandle};

const UNMAPPED_BUCKET: u32 = u32::MAX;

type GroupIds = HashMap<u64, u32>;

#[derive(Debug, PartialEq)]
pub(crate) struct DrawBatch {
    pub mesh: MeshHandle,
    pub source_index: usize,
    pub material: Option<MaterialHandle>,
    pub instances: Range<u32>,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum SnapshotError {
    InstanceBudget { actual: usize, limit: usize },
    DuplicateMesh { handle: MeshHandle },
    MissingMesh { handle: MeshHandle },
    DuplicateMaterial { handle: MaterialHandle },
    MissingMaterial { handle: MaterialHandle },
    DuplicateTexture { handle: TextureHandle },
    MissingTexture { handle: TextureHandle },
    InvalidCamera,
}

impl fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InstanceBudget { actual, limit } => write!(
                f,
                "Render snapshot has {actual} instances; this GPU's instance buffers hold {limit}. Lower EngineConfig::max_render_instances or the scene workload."
            ),
            Self::DuplicateMesh { handle } => write!(
                f,
                "Render snapshot contains duplicate source mesh {handle:?}. Include each immutable mesh handle once."
            ),
            Self::MissingMesh { handle } => write!(
                f,
                "Render snapshot instance references missing mesh {handle:?}. Include its owned source in snapshot.meshes."
            ),
            Self::DuplicateMaterial { handle } => write!(
                f,
                "Render snapshot contains duplicate source material {handle:?}. Include each material handle once."
            ),
            Self::MissingMaterial { handle } => write!(
                f,
                "Render snapshot instance references missing material {handle:?}. Include it in snapshot.materials."
            ),
            Self::DuplicateTexture { handle } => write!(
                f,
                "Render snapshot contains duplicate source texture {handle:?}. Include each immutable texture handle once."
            ),
            Self::MissingTexture { handle } => write!(
                f,
                "Render snapshot material references missing texture {handle:?}. Include its owned source in snapshot.textures."
            ),
            Self::InvalidCamera => f.write_str(
                "Render snapshot view-projection contains a nonfinite value. Supply a finite camera matrix.",
            ),
        }
    }
}

#[derive(Default)]
pub(crate) struct BatchPlan {
    pub ordered_instances: Vec<usize>,
    pub batches: Vec<DrawBatch>,
    previous_order: Vec<usize>,
    previous_batches: Vec<DrawBatch>,
    generation: u64,
    rebuilds: u64,
    source_indices: Vec<usize>,
    material_source_indices: Vec<usize>,
    texture_source_indices: Vec<usize>,
    pub material_sources: Vec<usize>,
    pub texture_sources: Vec<usize>,
    mesh_handles: Vec<MeshHandle>,
    material_handles: Vec<MaterialHandle>,
    texture_handles: Vec<TextureHandle>,
    instance_buckets: Vec<u32>,
    bucket_cursors: Vec<usize>,
    group_ids: GroupIds,
    group_keys: Vec<u64>,
    group_order: Vec<u32>,
    used_material_sources: Vec<bool>,
    used_texture_sources: Vec<bool>,
    pub instance_limit: Option<usize>,
    valid: bool,
    order_reused: bool,
    retained_structure: Option<(u64, u64)>,
    retained_sources: Option<(u64, u64)>,
    slot_buckets: Vec<u32>,
    slot_mapping_reused: bool,
    key_resolutions: usize,
}

impl BatchPlan {
    pub(crate) fn with_instance_limit(limit: usize) -> Self {
        Self {
            instance_limit: Some(limit),
            ..Self::default()
        }
    }

    pub fn prepare_frame(&mut self, frame: &RenderFrame<'_>) -> Result<(), SnapshotError> {
        let key = (frame.identity(), frame.structure_revision());
        let source_key = (frame.identity(), frame.source_revision());
        let snapshot = frame.snapshot();
        self.key_resolutions = 0;
        let reuse_slots = self.valid
            && self.retained_sources == Some(source_key)
            && self.slot_buckets.len() == snapshot.instances.len();
        self.slot_mapping_reused = reuse_slots;
        if reuse_slots && self.retained_structure == Some(key) {
            if !snapshot.view_projection.is_finite() {
                self.invalidate();
                return Err(SnapshotError::InvalidCamera);
            }
            self.order_reused = true;
        } else {
            self.valid = false;
            if let Err(error) = self.prepare_inner(snapshot, frame.visible_indices(), reuse_slots) {
                self.invalidate();
                return Err(error);
            }
            self.retained_structure = Some(key);
            self.retained_sources = Some(source_key);
        }
        self.record_stats();
        Ok(())
    }

    fn record_stats(&self) {
        if engine_profile::active() {
            engine_profile::counter(
                "render.batch_order_reused",
                if self.order_reused { 1.0 } else { 0.0 },
            );
            engine_profile::counter(
                "render.batch_order_rebuilt",
                if self.order_reused { 0.0 } else { 1.0 },
            );
            engine_profile::counter(
                "render.batch_slot_mapping_reused",
                f64::from(self.slot_mapping_reused),
            );
            engine_profile::counter("render.batch_key_resolutions", self.key_resolutions as f64);
            let group_scratch_bytes = (self.instance_buckets.capacity()
                + self.slot_buckets.capacity()
                + self.group_order.capacity())
                * size_of::<u32>()
                + self.group_keys.capacity() * size_of::<u64>()
                + self.bucket_cursors.capacity() * size_of::<usize>();
            let capacity_bytes = (self.ordered_instances.capacity()
                + self.source_indices.capacity()
                + self.material_source_indices.capacity()
                + self.texture_source_indices.capacity()
                + self.material_sources.capacity()
                + self.texture_sources.capacity())
                * size_of::<usize>()
                + self.batches.capacity() * size_of::<DrawBatch>()
                + self.mesh_handles.capacity() * size_of::<MeshHandle>()
                + self.material_handles.capacity() * size_of::<MaterialHandle>()
                + self.texture_handles.capacity() * size_of::<TextureHandle>()
                + group_scratch_bytes;
            engine_profile::counter("memory.batch_plan_capacity_bytes", capacity_bytes as f64);
            engine_profile::counter(
                "memory.batch_group_scratch_bytes",
                group_scratch_bytes as f64,
            );
        }
    }

    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    pub(crate) fn rebuilds(&self) -> u64 {
        self.rebuilds
    }

    fn invalidate(&mut self) {
        self.generation += 1;
        self.valid = false;
        self.order_reused = false;
        self.retained_structure = None;
        self.retained_sources = None;
        self.slot_mapping_reused = false;
        self.key_resolutions = 0;
        self.slot_buckets.clear();
        self.group_ids.clear();
        self.group_keys.clear();
        self.instance_buckets.clear();
        self.bucket_cursors.clear();
        self.ordered_instances.clear();
        self.batches.clear();
        self.source_indices.clear();
        self.material_source_indices.clear();
        self.texture_source_indices.clear();
        self.material_sources.clear();
        self.texture_sources.clear();
        self.mesh_handles.clear();
        self.material_handles.clear();
        self.texture_handles.clear();
    }

    fn prepare_inner(
        &mut self,
        snapshot: &RenderSnapshot,
        selected: &[usize],
        reuse_slot_mapping: bool,
    ) -> Result<(), SnapshotError> {
        self.rebuilds += 1;
        let instance_count = selected.len();
        let profile_validation = engine_profile::span("render", "batch_source_validation");
        self.source_indices.clear();
        self.material_source_indices.clear();
        self.texture_source_indices.clear();
        self.material_sources.clear();
        self.texture_sources.clear();
        self.mesh_handles.clear();
        self.material_handles.clear();
        self.texture_handles.clear();
        if let Some(limit) = self.instance_limit
            && (snapshot.instances.len() > limit || instance_count > limit)
        {
            return Err(SnapshotError::InstanceBudget {
                actual: snapshot.instances.len().max(instance_count),
                limit,
            });
        }
        if !snapshot.view_projection.is_finite() {
            return Err(SnapshotError::InvalidCamera);
        }
        drop(profile_validation);
        let profile_sources = engine_profile::span("render", "batch_source_index");
        self.source_indices.reserve_exact(snapshot.meshes.len());
        self.source_indices.extend(0..snapshot.meshes.len());
        self.source_indices
            .sort_unstable_by_key(|index| snapshot.meshes[*index].handle);
        for pair in self.source_indices.windows(2) {
            let handle = snapshot.meshes[pair[0]].handle;
            if handle == snapshot.meshes[pair[1]].handle {
                return Err(SnapshotError::DuplicateMesh { handle });
            }
        }
        self.mesh_handles.reserve_exact(snapshot.meshes.len());
        self.mesh_handles.extend(
            self.source_indices
                .iter()
                .map(|index| snapshot.meshes[*index].handle),
        );
        self.material_source_indices
            .reserve_exact(snapshot.materials.len());
        self.material_source_indices
            .extend(0..snapshot.materials.len());
        self.material_source_indices
            .sort_unstable_by_key(|index| snapshot.materials[*index].handle);
        for pair in self.material_source_indices.windows(2) {
            let handle = snapshot.materials[pair[0]].handle;
            if handle == snapshot.materials[pair[1]].handle {
                return Err(SnapshotError::DuplicateMaterial { handle });
            }
        }
        self.texture_source_indices
            .reserve_exact(snapshot.textures.len());
        self.texture_source_indices
            .extend(0..snapshot.textures.len());
        self.texture_source_indices
            .sort_unstable_by_key(|index| snapshot.textures[*index].handle);
        for pair in self.texture_source_indices.windows(2) {
            let handle = snapshot.textures[pair[0]].handle;
            if handle == snapshot.textures[pair[1]].handle {
                return Err(SnapshotError::DuplicateTexture { handle });
            }
        }
        for source in &snapshot.materials {
            for handle in source.material.textures() {
                if self
                    .texture_source_indices
                    .binary_search_by_key(&handle, |&index| snapshot.textures[index].handle)
                    .is_err()
                {
                    return Err(SnapshotError::MissingTexture { handle });
                }
            }
        }
        drop(profile_sources);
        if !reuse_slot_mapping {
            self.group_ids.clear();
            self.group_keys.clear();
            self.slot_buckets.clear();
            self.slot_buckets.reserve_exact(snapshot.instances.len());
            self.slot_buckets
                .resize(snapshot.instances.len(), UNMAPPED_BUCKET);
        }
        self.used_material_sources.clear();
        self.used_material_sources
            .resize(snapshot.materials.len(), false);
        std::mem::swap(&mut self.ordered_instances, &mut self.previous_order);
        std::mem::swap(&mut self.batches, &mut self.previous_batches);
        self.group_instances(snapshot, selected)?;
        if self.ordered_instances != self.previous_order || self.batches != self.previous_batches {
            self.generation += 1;
        }
        self.used_texture_sources.clear();
        self.used_texture_sources
            .resize(snapshot.textures.len(), false);
        for &index in &self.material_source_indices {
            let source = &snapshot.materials[index];
            self.material_handles.push(source.handle);
            if self.used_material_sources[index] {
                self.material_sources.push(index);
                for handle in source.material.textures() {
                    let rank = self
                        .texture_source_indices
                        .binary_search_by_key(&handle, |&index| snapshot.textures[index].handle)
                        .expect("texture_validated");
                    self.used_texture_sources[self.texture_source_indices[rank]] = true;
                }
            }
        }
        for &index in &self.texture_source_indices {
            self.texture_handles.push(snapshot.textures[index].handle);
            if self.used_texture_sources[index] {
                self.texture_sources.push(index);
            }
        }
        self.order_reused = false;
        self.valid = true;
        Ok(())
    }

    fn group_instances(
        &mut self,
        snapshot: &RenderSnapshot,
        selected: &[usize],
    ) -> Result<(), SnapshotError> {
        let _profile = engine_profile::span("render", "batch_instance_group");
        self.instance_buckets.clear();
        self.instance_buckets.reserve_exact(selected.len());
        for &slot in selected {
            let instance = &snapshot.instances[slot];
            let cached = self.slot_buckets[slot];
            let group = if cached != UNMAPPED_BUCKET {
                cached
            } else {
                let (mesh_rank, material_rank) = Self::source_ranks(
                    snapshot,
                    &self.source_indices,
                    &self.material_source_indices,
                    instance.mesh,
                    instance.material,
                )?;
                self.key_resolutions += 1;
                let key =
                    ((mesh_rank as u64) << 31) | material_rank.map_or(0, |rank| rank as u64 + 1);
                let next = self.group_keys.len() as u32;
                let group = *self.group_ids.entry(key).or_insert(next);
                if group == next {
                    self.group_keys.push(key);
                }
                self.slot_buckets[slot] = group;
                group
            };
            self.instance_buckets.push(group);
        }
        self.bucket_cursors.clear();
        self.bucket_cursors.resize(self.group_keys.len(), 0);
        for &group in &self.instance_buckets {
            self.bucket_cursors[group as usize] += 1;
        }
        self.group_order.clear();
        self.group_order.extend(
            (0..self.group_keys.len() as u32)
                .filter(|&group| self.bucket_cursors[group as usize] != 0),
        );
        self.group_order
            .sort_unstable_by_key(|&group| self.group_keys[group as usize]);
        self.batches.clear();
        self.batches.reserve_exact(self.group_order.len());
        let mut start = 0;
        {
            let _profile_ranges = engine_profile::span("render", "batch_draw_ranges");
            for &group in &self.group_order {
                let count = &mut self.bucket_cursors[group as usize];
                let end = start + *count;
                *count = start;
                let key = self.group_keys[group as usize];
                let source_index = self.source_indices[((key >> 31) & 0x7fff_ffff) as usize];
                let material_slot = (key & 0x7fff_ffff) as usize;
                let material = if material_slot == 0 {
                    None
                } else {
                    let index = self.material_source_indices[material_slot - 1];
                    self.used_material_sources[index] = true;
                    Some(snapshot.materials[index].handle)
                };
                self.batches.push(DrawBatch {
                    mesh: snapshot.meshes[source_index].handle,
                    source_index,
                    material,
                    instances: start as u32..end as u32,
                });
                start = end;
            }
        }
        self.ordered_instances.clear();
        self.ordered_instances.reserve_exact(selected.len());
        self.ordered_instances.resize(selected.len(), 0);
        for (&slot, &bucket) in selected.iter().zip(&self.instance_buckets) {
            let cursor = &mut self.bucket_cursors[bucket as usize];
            self.ordered_instances[*cursor] = slot;
            *cursor += 1;
        }
        Ok(())
    }

    fn source_ranks(
        snapshot: &RenderSnapshot,
        mesh_indices: &[usize],
        material_indices: &[usize],
        mesh: MeshHandle,
        material: Option<MaterialHandle>,
    ) -> Result<(usize, Option<usize>), SnapshotError> {
        let mesh_index = mesh_indices
            .binary_search_by_key(&mesh, |&index| snapshot.meshes[index].handle)
            .map_err(|_| SnapshotError::MissingMesh { handle: mesh })?;
        let material_index = material
            .map(|handle| {
                material_indices
                    .binary_search_by_key(&handle, |&index| snapshot.materials[index].handle)
                    .map_err(|_| SnapshotError::MissingMaterial { handle })
            })
            .transpose()?;
        Ok((mesh_index, material_index))
    }

    pub fn contains_mesh(&self, handle: MeshHandle) -> bool {
        self.mesh_handles.binary_search(&handle).is_ok()
    }

    pub fn contains_material(&self, handle: MaterialHandle) -> bool {
        self.material_handles.binary_search(&handle).is_ok()
    }

    pub fn material<'a>(
        &self,
        snapshot: &'a RenderSnapshot,
        handle: MaterialHandle,
    ) -> Option<&'a engine_core::Material> {
        self.material_source_indices
            .binary_search_by_key(&handle, |&index| snapshot.materials[index].handle)
            .ok()
            .map(|rank| &snapshot.materials[self.material_source_indices[rank]].material)
    }

    pub fn contains_texture(&self, handle: TextureHandle) -> bool {
        self.texture_handles.binary_search(&handle).is_ok()
    }
}
