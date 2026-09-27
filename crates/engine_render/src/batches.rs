//! CPU-only draw planning. Storage is reused for opaque mesh/texture batches.

use std::{
    collections::HashMap,
    fmt,
    hash::{BuildHasherDefault, Hasher},
    ops::Range,
};

use engine_core::{MeshHandle, RenderFrame, RenderInstance, RenderSnapshot, TextureHandle};

const UNMAPPED_BUCKET: u32 = u32::MAX;

/// Hashes draw-group keys, which are already distinct integers, with one
/// multiply (the bucket index takes high bits).
#[derive(Default)]
struct KeyHasher(u64);

impl Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.write_u64(self.0 ^ u64::from(byte));
        }
    }

    fn write_u64(&mut self, value: u64) {
        let mixed = (value ^ (value >> 31)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        self.0 = mixed ^ (mixed >> 29);
    }
}

/// Draw group keys to their compact IDs.
type GroupIds = HashMap<u64, u32, BuildHasherDefault<KeyHasher>>;

#[derive(Debug)]
pub(crate) struct DrawBatch {
    pub mesh: MeshHandle,
    pub source_index: usize,
    pub texture: Option<TextureHandle>,
    /// Instances with an opacity below 1 draw in the transparent pass and
    /// cast no shadow (decision 026).
    pub transparent: bool,
    pub instances: Range<u32>,
}

/// Whether `instance` draws in the transparent pass (decision 026).
pub(crate) fn is_transparent(instance: &RenderInstance) -> bool {
    instance.color[3] < 1.0
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum SnapshotError {
    /// More instances than this GPU's instance buffers hold.
    InstanceBudget {
        actual: usize,
        limit: usize,
    },
    DuplicateMesh {
        handle: MeshHandle,
    },
    MissingMesh {
        handle: MeshHandle,
    },
    DuplicateTexture {
        handle: TextureHandle,
    },
    MissingTexture {
        handle: TextureHandle,
    },
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

            Self::DuplicateTexture { handle } => write!(
                f,
                "Render snapshot contains duplicate source texture {handle:?}. Include each immutable texture handle once."
            ),
            Self::MissingTexture { handle } => write!(
                f,
                "Render snapshot instance references missing texture {handle:?}. Include its owned source in snapshot.textures."
            ),
            Self::InvalidCamera => f.write_str(
                "Render snapshot view-projection contains a nonfinite value. Supply a finite camera matrix.",
            ),
        }
    }
}

/// Indices avoid moving gameplay records. Counting scatter groups opaque draws
/// by mesh/texture while preserving source order within each group, including
/// coherent static/moving runs. Exact keys retain ordering across pose/color
/// edits; source validation still runs on every snapshot. Changed keys take
/// O(N * (log M + log T) + M * T + M log M + T log T);
/// unchanged keys take O(N + M log M + T log T + B log(M + T)) for B batches.
/// Certified frames additionally retain one draw-group ID per stable slot;
/// visibility-only changes scatter compact IDs without per-instance key lookup.
/// Draw groups get compact IDs for the mesh, texture and opacity combinations
/// in use, so no mesh or texture count bounds them.
#[derive(Default)]
pub(crate) struct BatchPlan {
    pub ordered_instances: Vec<usize>,
    pub batches: Vec<DrawBatch>,
    source_indices: Vec<usize>,
    texture_source_indices: Vec<usize>,
    /// Texture sources referenced by emitted instances, sorted by handle for upload.
    pub texture_sources: Vec<usize>,
    /// All supplied sources keep existing residency, including culled objects.
    mesh_handles: Vec<MeshHandle>,
    texture_handles: Vec<TextureHandle>,
    instance_keys: Vec<(MeshHandle, Option<TextureHandle>, bool)>,
    // Rebuild scratch: one draw group per admitted instance and one cursor
    // per group in use.
    instance_buckets: Vec<u32>,
    bucket_cursors: Vec<usize>,
    // Compact IDs of draw-group keys (opacity class, mesh rank, texture slot)
    // and each ID's key; they last as long as the slot mapping.
    group_ids: GroupIds,
    group_keys: Vec<u64>,
    group_order: Vec<u32>,
    // Texture sources referenced by this plan's draws.
    used_texture_sources: Vec<bool>,
    /// Most instances the GPU's buffers hold; `None` bounds nothing.
    pub instance_limit: Option<usize>,
    valid: bool,
    /// Last successful decision, also exposed through profiling counters.
    order_reused: bool,
    retained_structure: Option<(u64, u64)>,
    retained_sources: Option<(u64, u64)>,
    // Stable-slot mapping belongs only to a certified core frame/source epoch.
    // Never-visible slots resolve lazily; public snapshots cannot populate it.
    slot_buckets: Vec<u32>,
    slot_mapping_reused: bool,
    key_resolutions: usize,
}

impl BatchPlan {
    /// A plan that rejects snapshots of more than `limit` instances.
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    pub(crate) fn with_instance_limit(limit: usize) -> Self {
        Self {
            instance_limit: Some(limit),
            ..Self::default()
        }
    }

    /// Validate before any GPU cache changes, then produce contiguous draw ranges.
    pub fn prepare(&mut self, snapshot: &RenderSnapshot) -> Result<(), SnapshotError> {
        if self.retained_structure.take().is_some() {
            // Trusted frames do not populate the public per-record key cache.
            // Even an empty public snapshot must rebuild the previous draws.
            self.valid = false;
        }
        self.retained_sources = None;
        self.slot_buckets.clear();
        self.slot_mapping_reused = false;
        self.key_resolutions = 0;
        if let Err(error) = self.prepare_inner(snapshot, None, false) {
            // Never expose a mixture of a previous valid plan and partially
            // prepared source/range data. Capacity remains available for retry.
            self.invalidate();
            return Err(error);
        }
        self.record_stats();
        Ok(())
    }

    /// Only a core-owned frame can certify retained keys, sources and membership.
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
            // A different retained stream/order cannot reuse positional keys.
            self.valid = false;
            if let Err(error) =
                self.prepare_inner(snapshot, Some(frame.visible_indices()), reuse_slots)
            {
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
                + self.texture_source_indices.capacity()
                + self.texture_sources.capacity())
                * size_of::<usize>()
                + self.batches.capacity() * size_of::<DrawBatch>()
                + self.mesh_handles.capacity() * size_of::<MeshHandle>()
                + self.texture_handles.capacity() * size_of::<TextureHandle>()
                + self.instance_keys.capacity()
                    * size_of::<(MeshHandle, Option<TextureHandle>, bool)>()
                + group_scratch_bytes;
            engine_profile::counter("memory.batch_plan_capacity_bytes", capacity_bytes as f64);
            engine_profile::counter(
                "memory.batch_group_scratch_bytes",
                group_scratch_bytes as f64,
            );
        }
    }

    fn invalidate(&mut self) {
        self.valid = false;
        self.order_reused = false;
        self.retained_structure = None;
        self.retained_sources = None;
        self.slot_mapping_reused = false;
        self.key_resolutions = 0;
        self.slot_buckets.clear();
        self.group_ids.clear();
        self.group_keys.clear();
        self.instance_keys.clear();
        self.instance_buckets.clear();
        self.bucket_cursors.clear();
        self.ordered_instances.clear();
        self.batches.clear();
        self.source_indices.clear();
        self.texture_source_indices.clear();
        self.texture_sources.clear();
        self.mesh_handles.clear();
        self.texture_handles.clear();
    }

    fn prepare_inner(
        &mut self,
        snapshot: &RenderSnapshot,
        selected: Option<&[usize]>,
        reuse_slot_mapping: bool,
    ) -> Result<(), SnapshotError> {
        let instance_count = selected.map_or(snapshot.instances.len(), <[usize]>::len);
        let profile_validation = engine_profile::span("render", "batch_source_validation");
        self.source_indices.clear();
        self.texture_source_indices.clear();
        self.texture_sources.clear();
        self.mesh_handles.clear();
        self.texture_handles.clear();
        // Only the GPU's buffers bound instances; meshes and textures are
        // bounded by GPU memory alone.
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
        drop(profile_sources);
        if selected.is_some() && !reuse_slot_mapping {
            // Budget/source validation precedes any per-slot cache growth.
            // Group IDs start over with the slots that refer to them.
            self.group_ids.clear();
            self.group_keys.clear();
            self.slot_buckets.clear();
            self.slot_buckets.reserve_exact(snapshot.instances.len());
            self.slot_buckets
                .resize(snapshot.instances.len(), UNMAPPED_BUCKET);
        }
        let reused = {
            let _profile_keys = engine_profile::span("render", "batch_key_comparison");
            self.valid
                && self.instance_keys.len() == instance_count
                && self
                    .instance_keys
                    .iter()
                    .zip(selected_instances(snapshot, selected))
                    .all(|(&key, (_, instance))| {
                        key == (instance.mesh, instance.texture, is_transparent(instance))
                    })
        };
        self.used_texture_sources.clear();
        self.used_texture_sources
            .resize(snapshot.textures.len(), false);
        if !reused {
            self.group_instances(snapshot, selected)?;
        } else {
            let _profile_references = engine_profile::span("render", "batch_source_references");
            for batch in &mut self.batches {
                // Sources are public mutable snapshot inputs. Unchanged keys
                // still re-resolve after a source list reorder or replacement.
                let (source_index, texture_index) = Self::resolve_sources(
                    snapshot,
                    &self.source_indices,
                    &self.texture_source_indices,
                    batch.mesh,
                    batch.texture,
                )?;
                batch.source_index = source_index;
                if let Some(index) = texture_index {
                    self.used_texture_sources[index] = true;
                }
            }
        }
        for &index in &self.texture_source_indices {
            self.texture_handles.push(snapshot.textures[index].handle);
            if self.used_texture_sources[index] {
                self.texture_sources.push(index);
            }
        }
        if !reused {
            self.instance_keys.clear();
            if selected.is_none() {
                self.instance_keys.reserve_exact(instance_count);
                self.instance_keys
                    .extend(selected_instances(snapshot, selected).map(|(_, instance)| {
                        (instance.mesh, instance.texture, is_transparent(instance))
                    }));
            }
        }
        self.order_reused = reused;
        self.valid = true;
        Ok(())
    }

    fn group_instances(
        &mut self,
        snapshot: &RenderSnapshot,
        selected: Option<&[usize]>,
    ) -> Result<(), SnapshotError> {
        let _profile = engine_profile::span("render", "batch_instance_group");
        if selected.is_none() {
            // Public snapshots keep no slot mapping, so no IDs either.
            self.group_ids.clear();
            self.group_keys.clear();
        }
        self.instance_buckets.clear();
        let instance_count = selected.map_or(snapshot.instances.len(), <[usize]>::len);
        self.instance_buckets.reserve_exact(instance_count);
        for (slot, instance) in selected_instances(snapshot, selected) {
            let cached = if selected.is_some() {
                self.slot_buckets[slot]
            } else {
                UNMAPPED_BUCKET
            };
            let group = if cached != UNMAPPED_BUCKET {
                cached
            } else {
                let (mesh_rank, texture_rank) = Self::source_ranks(
                    snapshot,
                    &self.source_indices,
                    &self.texture_source_indices,
                    instance.mesh,
                    instance.texture,
                )?;
                self.key_resolutions += 1;
                // Opaque groups sort first, then by mesh and texture.
                let key = (u64::from(is_transparent(instance)) << 62)
                    | ((mesh_rank as u64) << 31)
                    | texture_rank.map_or(0, |rank| rank as u64 + 1);
                let next = self.group_keys.len() as u32;
                let group = *self.group_ids.entry(key).or_insert(next);
                if group == next {
                    self.group_keys.push(key);
                }
                if selected.is_some() {
                    self.slot_buckets[slot] = group;
                }
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
                let texture_slot = (key & 0x7fff_ffff) as usize;
                let texture = if texture_slot == 0 {
                    None
                } else {
                    let index = self.texture_source_indices[texture_slot - 1];
                    self.used_texture_sources[index] = true;
                    Some(snapshot.textures[index].handle)
                };
                self.batches.push(DrawBatch {
                    mesh: snapshot.meshes[source_index].handle,
                    source_index,
                    texture,
                    transparent: key >> 62 != 0,
                    instances: start as u32..end as u32,
                });
                start = end;
            }
        }
        self.ordered_instances.clear();
        self.ordered_instances.reserve_exact(instance_count);
        self.ordered_instances.resize(instance_count, 0);
        for (index, &bucket) in self.instance_buckets.iter().enumerate() {
            let cursor = &mut self.bucket_cursors[bucket as usize];
            self.ordered_instances[*cursor] = selected.map_or(index, |indices| indices[index]);
            *cursor += 1;
        }
        Ok(())
    }

    fn resolve_sources(
        snapshot: &RenderSnapshot,
        mesh_indices: &[usize],
        texture_indices: &[usize],
        mesh: MeshHandle,
        texture: Option<TextureHandle>,
    ) -> Result<(usize, Option<usize>), SnapshotError> {
        let (mesh_rank, texture_rank) =
            Self::source_ranks(snapshot, mesh_indices, texture_indices, mesh, texture)?;
        Ok((
            mesh_indices[mesh_rank],
            texture_rank.map(|rank| texture_indices[rank]),
        ))
    }

    fn source_ranks(
        snapshot: &RenderSnapshot,
        mesh_indices: &[usize],
        texture_indices: &[usize],
        mesh: MeshHandle,
        texture: Option<TextureHandle>,
    ) -> Result<(usize, Option<usize>), SnapshotError> {
        let mesh_index = mesh_indices
            .binary_search_by_key(&mesh, |&index| snapshot.meshes[index].handle)
            .map_err(|_| SnapshotError::MissingMesh { handle: mesh })?;
        let texture_index = texture
            .map(|handle| {
                texture_indices
                    .binary_search_by_key(&handle, |&index| snapshot.textures[index].handle)
                    .map_err(|_| SnapshotError::MissingTexture { handle })
            })
            .transpose()?;
        Ok((mesh_index, texture_index))
    }

    /// Supplied sources retain already uploaded assets even when all their
    /// instances are culled. This does not trigger an upload for unused sources.
    /// Used at every valid frame boundary, including suspended/occluded frames.
    /// Dropping a GPU cache entry retires its owned buffers; submitted work keeps
    /// any internal references it needs until the GPU has finished with them.
    pub fn contains_mesh(&self, handle: MeshHandle) -> bool {
        self.mesh_handles.binary_search(&handle).is_ok()
    }

    pub fn contains_texture(&self, handle: TextureHandle) -> bool {
        self.texture_handles.binary_search(&handle).is_ok()
    }
}

fn selected_instances<'a>(
    snapshot: &'a RenderSnapshot,
    selected: Option<&'a [usize]>,
) -> impl Iterator<Item = (usize, &'a RenderInstance)> {
    (0..selected.map_or(snapshot.instances.len(), <[usize]>::len)).map(move |position| {
        let index = selected.map_or(position, |indices| indices[position]);
        (index, &snapshot.instances[index])
    })
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, sync::Arc};

    use engine_core::{AssetLibrary, Mesh, RenderInstance, RenderMesh, RenderTexture, Texture};
    use glam::Mat4;

    use super::*;

    #[test]
    fn retained_visible_slots_and_public_snapshots_do_not_share_revision_trust() {
        use engine_core::{Cube, Engine, PresentationMode, Transform};
        let mut engine = Engine::default();
        engine.set_frustum_culling(false);
        let first = engine
            .app
            .world_mut()
            .spawn((Transform::default(), Cube { color: [1.0; 4] }))
            .id();
        engine.app.world_mut().spawn((
            Transform {
                translation: glam::Vec3::X * 10_000.0,
                ..Default::default()
            },
            Cube { color: [0.5; 4] },
        ));
        let mut plan = BatchPlan::default();
        let source_revision;
        let initial_order;
        {
            let frame = engine
                .extract_frame(1.0, PresentationMode::Current)
                .unwrap();
            plan.prepare_frame(&frame).unwrap();
            assert_eq!(plan.ordered_instances.len(), 2);
            assert_eq!(plan.key_resolutions, 2);
            source_revision = frame.source_revision();
            initial_order = plan.ordered_instances.clone();
            plan.prepare_frame(&frame).unwrap();
            assert!(plan.order_reused);
        }
        // Visibility changes regroup compact IDs without resolving slot keys.
        engine.set_frustum_culling(true);
        {
            let frame = engine
                .extract_frame(1.0, PresentationMode::Current)
                .unwrap();
            assert_eq!(frame.source_revision(), source_revision);
            plan.prepare_frame(&frame).unwrap();
            assert_eq!(plan.ordered_instances, frame.visible_indices());
            assert_eq!(plan.ordered_instances.len(), 1);
            assert!(!plan.order_reused);
            assert!(plan.slot_mapping_reused);
            assert_eq!(plan.key_resolutions, 0);
        }
        engine.set_frustum_culling(false);
        {
            let frame = engine
                .extract_frame(1.0, PresentationMode::Current)
                .unwrap();
            assert_eq!(frame.source_revision(), source_revision);
            plan.prepare_frame(&frame).unwrap();
            assert_eq!(plan.ordered_instances, initial_order);
            assert!(plan.slot_mapping_reused);
            assert_eq!(plan.key_resolutions, 0);
        }
        engine.app.world_mut().despawn(first);
        let frame = engine
            .extract_frame(1.0, PresentationMode::Current)
            .unwrap();
        plan.prepare_frame(&frame).unwrap();
        assert_eq!(plan.ordered_instances, frame.visible_indices());
        assert_eq!(plan.ordered_instances.len(), 1);
        assert!(!plan.slot_mapping_reused);
        assert_eq!(plan.key_resolutions, 1);
        assert_eq!(plan.batches[0].instances, 0..1);
        let empty = RenderSnapshot {
            view_projection: frame.snapshot().view_projection,
            meshes: frame.snapshot().meshes.clone(),
            ..Default::default()
        };
        plan.prepare(&empty).unwrap();
        assert!(plan.ordered_instances.is_empty() && plan.batches.is_empty());
        plan.prepare_frame(&frame).unwrap();
        // A caller-owned snapshot can reuse the exact source records yet is
        // still mutable; it must never acquire the engine frame's certificate.
        let mut public = RenderSnapshot {
            view_projection: frame.snapshot().view_projection,
            instances: vec![frame.snapshot().instances[frame.visible_indices()[0]]],
            meshes: frame.snapshot().meshes.clone(),
            textures: frame.snapshot().textures.clone(),
            ..Default::default()
        };
        plan.prepare(&public).unwrap();
        public.meshes.clear();
        assert!(matches!(
            plan.prepare(&public),
            Err(SnapshotError::MissingMesh { .. })
        ));
        plan.prepare_frame(&frame).unwrap();
        assert!(!plan.order_reused);
        plan.prepare_frame(&frame).unwrap();
        assert!(plan.order_reused);
    }

    fn mesh() -> RenderMesh {
        let mut assets = AssetLibrary::default();
        RenderMesh {
            handle: assets.insert_mesh(Mesh::cube()).unwrap(),
            mesh: Arc::new(Mesh::cube()),
        }
    }

    fn snapshot(meshes: Vec<RenderMesh>, handles: &[MeshHandle]) -> RenderSnapshot {
        RenderSnapshot {
            view_projection: Mat4::IDENTITY,
            instances: handles
                .iter()
                .enumerate()
                .map(|(index, &mesh)| RenderInstance {
                    model: Mat4::from_translation(glam::Vec3::X * index as f32),
                    color: [index as f32 / handles.len() as f32, 0.2, 0.8, 1.0],
                    surface: engine_core::Surface::MATTE,
                    mesh,
                    texture: None,
                })
                .collect(),
            meshes,
            textures: Vec::new(),
            camera_position: glam::Vec3::ZERO,
            atmosphere: Default::default(),
            elapsed_seconds: 0.0,
            shadows: Default::default(),
            ambient_occlusion: Default::default(),
            bloom: Default::default(),
            dropped_instances: 0,
            missing_assets: 0,
            invalid_transforms: 0,
            render_candidates: handles.len(),
            visible_instances: handles.len(),
            culled_instances: 0,
            ..Default::default()
        }
    }

    fn texture() -> RenderTexture {
        let mut assets = AssetLibrary::default();
        let handle = assets
            .insert_texture(Texture::new(1, 1, vec![255; 4]).unwrap())
            .unwrap();
        RenderTexture {
            handle,
            texture: Arc::clone(assets.texture(handle).unwrap()),
        }
    }

    fn assert_complete_plan(plan: &BatchPlan, snapshot: &RenderSnapshot) {
        let mut visited = vec![false; snapshot.instances.len()];
        let mut next = 0;
        for batch in &plan.batches {
            assert_eq!(batch.instances.start, next);
            assert!(batch.instances.end > batch.instances.start);
            next = batch.instances.end;
            assert_eq!(snapshot.meshes[batch.source_index].handle, batch.mesh);
            for offset in batch.instances.clone() {
                let index = plan.ordered_instances[offset as usize];
                let instance = &snapshot.instances[index];
                assert_eq!(
                    (instance.mesh, instance.texture),
                    (batch.mesh, batch.texture)
                );
                assert!(!visited[index]);
                visited[index] = true;
            }
        }
        assert_eq!(next as usize, snapshot.instances.len());
        assert!(visited.into_iter().all(|value| value));
        let mut used: Vec<_> = snapshot
            .instances
            .iter()
            .filter_map(|instance| instance.texture)
            .collect();
        used.sort_unstable();
        used.dedup();
        assert_eq!(
            plan.texture_sources
                .iter()
                .map(|&index| snapshot.textures[index].handle)
                .collect::<Vec<_>>(),
            used
        );
    }

    #[test]
    fn grouping_preserves_source_order_after_membership_changes() {
        let first = mesh();
        let second = mesh();
        let first_texture = texture();
        let second_texture = texture();
        let handles = [
            second.handle,
            first.handle,
            second.handle,
            first.handle,
            second.handle,
            first.handle,
        ];
        let mut scene = snapshot(vec![second, first], &handles);
        scene.instances[0].texture = Some(second_texture.handle);
        scene.instances[2].texture = Some(second_texture.handle);
        scene.instances[3].texture = Some(first_texture.handle);
        scene.instances[5].texture = Some(first_texture.handle);
        scene.textures = vec![second_texture, first_texture];
        let mut plan = BatchPlan::default();
        for edit in 0..3 {
            if edit == 1 {
                scene.instances.remove(1);
            } else if edit == 2 {
                let inserted = scene.instances[0];
                scene.instances.insert(2, inserted);
            }
            plan.prepare(&scene).unwrap();
            assert_complete_plan(&plan, &scene);
            let mut previous_key = None;
            for batch in &plan.batches {
                let key = (batch.mesh, batch.texture);
                assert!(previous_key.is_none_or(|previous| previous < key));
                previous_key = Some(key);
                let expected: Vec<_> = scene
                    .instances
                    .iter()
                    .enumerate()
                    .filter_map(|(index, instance)| {
                        ((instance.mesh, instance.texture) == key).then_some(index)
                    })
                    .collect();
                assert_eq!(
                    &plan.ordered_instances
                        [batch.instances.start as usize..batch.instances.end as usize],
                    expected
                );
            }
        }
    }

    #[test]
    fn static_cache_pose_color_and_camera_edits_retain_batch_order_ranges_and_storage() {
        let first = mesh();
        let second = mesh();
        let source_texture = texture();
        let handles = [second.handle, first.handle, second.handle, first.handle];
        let mut scene = snapshot(vec![first, second], &handles);
        scene.instances[1].texture = Some(source_texture.handle);
        scene.instances[3].texture = Some(source_texture.handle);
        scene.textures.push(source_texture);
        let mut plan = BatchPlan::default();
        plan.prepare(&scene).unwrap();
        assert!(!plan.order_reused);
        let order = plan.ordered_instances.clone();
        let ranges: Vec<_> = plan
            .batches
            .iter()
            .map(|batch| (batch.mesh, batch.texture, batch.instances.clone()))
            .collect();
        let storage = (
            plan.ordered_instances.as_ptr(),
            plan.batches.as_ptr(),
            plan.instance_keys.as_ptr(),
        );
        for step in 1..=3 {
            for (index, instance) in scene.instances.iter_mut().enumerate() {
                instance.model = Mat4::from_translation(glam::Vec3::Y * (index + step) as f32);
                instance.color[0] = step as f32 * 0.1;
            }
            scene.view_projection = Mat4::from_scale(glam::Vec3::splat(step as f32));
            plan.prepare(&scene).unwrap();
            assert!(plan.order_reused);
            assert_eq!(plan.ordered_instances, order);
            assert_eq!(
                plan.batches
                    .iter()
                    .map(|batch| (batch.mesh, batch.texture, batch.instances.clone()))
                    .collect::<Vec<_>>(),
                ranges
            );
            assert_eq!(
                (
                    plan.ordered_instances.as_ptr(),
                    plan.batches.as_ptr(),
                    plan.instance_keys.as_ptr(),
                ),
                storage
            );
            assert_complete_plan(&plan, &scene);
        }
    }

    #[test]
    fn static_cache_source_reorders_key_changes_and_scene_replacement_keep_mappings_current() {
        let first = mesh();
        let second = mesh();
        let first_texture = texture();
        let second_texture = texture();
        let handles = [first.handle, second.handle, first.handle];
        let texture_handles = [first_texture.handle, second_texture.handle];
        let mut scene = snapshot(vec![first, second], &handles);
        scene.instances[0].texture = Some(texture_handles[0]);
        scene.instances[1].texture = Some(texture_handles[1]);
        scene.textures = vec![first_texture, second_texture];
        let mut plan = BatchPlan::default();
        plan.prepare(&scene).unwrap();

        // Source order is independent of instance order and upload residency.
        scene.meshes.swap(0, 1);
        scene.textures.swap(0, 1);
        plan.prepare(&scene).unwrap();
        assert!(plan.order_reused);
        assert_complete_plan(&plan, &scene);
        scene.instances.swap(0, 1);
        plan.prepare(&scene).unwrap();
        assert!(!plan.order_reused);
        assert_complete_plan(&plan, &scene);
        scene.instances[0].mesh = handles[0];
        plan.prepare(&scene).unwrap();
        assert!(!plan.order_reused);
        assert_complete_plan(&plan, &scene);
        scene.instances[0].texture = Some(texture_handles[0]);
        plan.prepare(&scene).unwrap();
        assert!(!plan.order_reused);
        assert_complete_plan(&plan, &scene);

        scene.instances.clear();
        plan.prepare(&scene).unwrap();
        assert!(!plan.order_reused);
        assert_complete_plan(&plan, &scene);
        assert!(plan.contains_mesh(handles[0]));
        assert!(plan.contains_texture(texture_handles[0]));
        scene.meshes.clear();
        scene.textures.clear();
        plan.prepare(&scene).unwrap();
        assert!(plan.order_reused);
        assert!(!plan.contains_mesh(handles[0]));
        assert!(!plan.contains_texture(texture_handles[0]));

        let replacement_mesh = mesh();
        let replacement_texture = texture();
        let replacement_handle = replacement_mesh.handle;
        scene = snapshot(vec![replacement_mesh], &[replacement_handle]);
        scene.instances[0].texture = Some(replacement_texture.handle);
        scene.textures.push(replacement_texture);
        plan.prepare(&scene).unwrap();
        assert!(!plan.order_reused);
        assert!(plan.contains_mesh(replacement_handle));
        assert!(!plan.contains_mesh(handles[0]));
        assert!(!plan.contains_texture(texture_handles[0]));
        assert_complete_plan(&plan, &scene);
    }

    #[test]
    fn static_cache_failed_prepares_clear_partial_plans_and_valid_retries_rebuild() {
        let source = mesh();
        let source_texture = texture();
        let handle = source.handle;
        let texture_handle = source_texture.handle;
        let missing = mesh().handle;
        let mut scene = snapshot(vec![source.clone()], &[handle, handle]);
        scene.instances[1].texture = Some(texture_handle);
        scene.textures.push(source_texture.clone());
        let mut plan = BatchPlan::default();
        plan.prepare(&scene).unwrap();
        let capacity = plan.ordered_instances.capacity();
        for failure in 0..6 {
            let expected = match failure {
                0 => {
                    scene.meshes.clear();
                    SnapshotError::MissingMesh { handle }
                }
                1 => {
                    scene.textures.clear();
                    SnapshotError::MissingTexture {
                        handle: texture_handle,
                    }
                }
                2 => {
                    // Sorting starts over and may already have formed valid
                    // ranges before encountering this missing source.
                    scene.instances[1].mesh = missing;
                    SnapshotError::MissingMesh { handle: missing }
                }
                3 => {
                    scene.view_projection.x_axis.x = f32::NAN;
                    SnapshotError::InvalidCamera
                }
                4 => {
                    scene.textures.push(source_texture.clone());
                    SnapshotError::DuplicateTexture {
                        handle: texture_handle,
                    }
                }
                _ => {
                    plan.instance_limit = Some(1);
                    SnapshotError::InstanceBudget {
                        actual: 2,
                        limit: 1,
                    }
                }
            };
            assert_eq!(plan.prepare(&scene), Err(expected));
            assert!(!plan.valid);
            assert!(!plan.order_reused);
            assert!(plan.ordered_instances.is_empty());
            assert!(plan.batches.is_empty());
            assert!(plan.texture_sources.is_empty());
            assert!(plan.instance_keys.is_empty());
            assert!(!plan.contains_mesh(handle));
            assert!(!plan.contains_texture(texture_handle));
            assert_eq!(plan.ordered_instances.capacity(), capacity);

            scene.view_projection = Mat4::IDENTITY;
            scene.meshes = vec![source.clone()];
            scene.textures = vec![source_texture.clone()];
            scene.instances[1].mesh = handle;
            plan.instance_limit = None;
            plan.prepare(&scene).unwrap();
            assert!(!plan.order_reused);
            assert_complete_plan(&plan, &scene);
            plan.prepare(&scene).unwrap();
            assert!(plan.order_reused);
        }
    }

    #[test]
    fn transparent_instances_draw_in_their_own_batches_after_opaque_ones() {
        let shared = mesh();
        let handles = [shared.handle; 4];
        let mut snapshot = snapshot(vec![shared], &handles);
        snapshot.instances[1].color[3] = 0.5;
        snapshot.instances[3].color[3] = 0.25;
        let mut plan = BatchPlan::default();
        plan.prepare(&snapshot).unwrap();
        assert_eq!(plan.batches.len(), 2);
        assert!(!plan.batches[0].transparent, "opaque buckets come first");
        assert!(plan.batches[1].transparent);
        for batch in &plan.batches {
            for offset in batch.instances.clone() {
                let instance = &snapshot.instances[plan.ordered_instances[offset as usize]];
                assert_eq!(is_transparent(instance), batch.transparent);
            }
        }
        // An opacity change regroups even with the same mesh and texture.
        snapshot.instances[1].color[3] = 1.0;
        plan.prepare(&snapshot).unwrap();
        assert_eq!(plan.batches[0].instances.len(), 3);
        assert_eq!(plan.batches[1].instances.len(), 1);
    }

    #[test]
    fn colors_share_draws_but_different_textures_split_mesh_batches() {
        let first_mesh = mesh();
        let second_mesh = mesh();
        let first_texture = texture();
        let second_texture = texture();
        let handles = [
            first_mesh.handle,
            second_mesh.handle,
            first_mesh.handle,
            first_mesh.handle,
            first_mesh.handle,
        ];
        let textures = [
            Some(first_texture.handle),
            Some(first_texture.handle),
            None,
            Some(second_texture.handle),
            Some(first_texture.handle),
        ];
        let mut snapshot = snapshot(vec![second_mesh, first_mesh], &handles);
        for (instance, texture) in snapshot.instances.iter_mut().zip(textures) {
            instance.texture = texture;
        }
        snapshot.textures = vec![second_texture, first_texture];
        let mut plan = BatchPlan::default();
        plan.prepare(&snapshot).unwrap();
        assert_eq!(plan.batches.len(), 4);
        assert_eq!(plan.texture_sources.len(), 2);
        assert!(plan.batches.iter().any(|batch| batch.instances.len() == 2));
        let mut visited = [false; 5];
        for batch in &plan.batches {
            for offset in batch.instances.clone() {
                let index = plan.ordered_instances[offset as usize];
                let instance = &snapshot.instances[index];
                assert_eq!(
                    (instance.mesh, instance.texture),
                    (batch.mesh, batch.texture)
                );
                assert!(!visited[index]);
                visited[index] = true;
            }
        }
        assert!(visited.into_iter().all(|value| value));
        let capacities = (
            plan.ordered_instances.capacity(),
            plan.batches.capacity(),
            plan.texture_sources.capacity(),
        );
        plan.prepare(&snapshot).unwrap();
        assert_eq!(
            capacities,
            (
                plan.ordered_instances.capacity(),
                plan.batches.capacity(),
                plan.texture_sources.capacity()
            )
        );
    }

    #[test]
    fn missing_and_duplicate_texture_sources_fail_before_residency_changes() {
        let source_mesh = mesh();
        let mesh_handle = source_mesh.handle;
        let source = texture();
        let handle = source.handle;
        let mut snapshot = snapshot(vec![source_mesh], &[mesh_handle]);
        snapshot.instances[0].texture = Some(handle);
        let mut plan = BatchPlan::default();
        assert_eq!(
            plan.prepare(&snapshot),
            Err(SnapshotError::MissingTexture { handle })
        );
        snapshot.textures = vec![source.clone(), source];
        assert_eq!(
            plan.prepare(&snapshot),
            Err(SnapshotError::DuplicateTexture { handle })
        );
    }

    #[test]
    fn many_meshes_and_textures_plan_and_the_instance_limit_rejects_first() {
        let meshes: Vec<_> = (0..300).map(|_| mesh()).collect();
        let textures: Vec<_> = (0..200).map(|_| texture()).collect();
        let handles: Vec<_> = meshes.iter().map(|source| source.handle).collect();
        let mut scene = snapshot(meshes, &handles);
        for (index, instance) in scene.instances.iter_mut().enumerate() {
            instance.texture = Some(textures[index % textures.len()].handle);
        }
        scene.textures = textures;
        let mut plan = BatchPlan::default();
        plan.prepare(&scene).unwrap();
        assert_eq!(plan.batches.len(), 300);
        assert_complete_plan(&plan, &scene);
        let mut fresh = BatchPlan {
            instance_limit: Some(299),
            ..BatchPlan::default()
        };
        assert_eq!(
            fresh.prepare(&scene),
            Err(SnapshotError::InstanceBudget {
                actual: 300,
                limit: 299
            })
        );
        assert_eq!(fresh.ordered_instances.capacity(), 0);
    }

    #[test]
    fn culled_texture_sources_retain_residency_without_uploads() {
        let source_mesh = mesh();
        let mesh_handle = source_mesh.handle;
        let first = texture();
        let second = texture();
        let first_handle = first.handle;
        let second_handle = second.handle;
        let mut snapshot = snapshot(vec![source_mesh], &[mesh_handle, mesh_handle]);
        snapshot
            .instances
            .iter_mut()
            .for_each(|instance| instance.texture = Some(second_handle));
        snapshot.textures = vec![first, second];
        let mut plan = BatchPlan::default();
        let mut resident = BTreeMap::from([(first_handle, 4), (second_handle, 4)]);
        plan.prepare(&snapshot).unwrap();
        assert_eq!(plan.texture_sources.len(), 1);
        assert_eq!(
            snapshot.textures[plan.texture_sources[0]].handle,
            second_handle
        );
        resident.retain(|handle, _| plan.contains_texture(*handle));
        assert_eq!(
            resident,
            BTreeMap::from([(first_handle, 4), (second_handle, 4)])
        );
        snapshot.instances.clear();
        plan.prepare(&snapshot).unwrap();
        resident.retain(|handle, _| plan.contains_texture(*handle));
        assert_eq!(resident.len(), 2);
        assert!(plan.texture_sources.is_empty());
        snapshot.textures.clear();
        plan.prepare(&snapshot).unwrap();
        resident.retain(|handle, _| plan.contains_texture(*handle));
        assert!(resident.is_empty());
    }

    #[test]
    fn interleaved_meshes_and_colors_form_complete_contiguous_draw_ranges() {
        let first = mesh();
        let second = mesh();
        let handles = [
            second.handle,
            first.handle,
            second.handle,
            first.handle,
            first.handle,
        ];
        // Source order must not affect lookup or drawing order.
        let snapshot = snapshot(vec![second, first], &handles);
        let mut plan = BatchPlan::default();
        plan.prepare(&snapshot).unwrap();
        assert_eq!(plan.batches.len(), 2);
        assert_eq!(plan.batches[0].instances, 0..3);
        assert_eq!(plan.batches[1].instances, 3..5);
        let mut visited = [false; 5];
        for batch in &plan.batches {
            assert_eq!(batch.mesh, snapshot.meshes[batch.source_index].handle);
            for offset in batch.instances.clone() {
                let index = plan.ordered_instances[offset as usize];
                assert_eq!(snapshot.instances[index].mesh, batch.mesh);
                assert!(!visited[index]);
                visited[index] = true;
            }
        }
        assert!(visited.into_iter().all(|value| value));
        let capacity = plan.ordered_instances.capacity();
        plan.prepare(&snapshot).unwrap();
        assert_eq!(plan.ordered_instances.capacity(), capacity);
    }

    #[test]
    fn missing_and_duplicate_sources_are_actionable_errors() {
        let source = mesh();
        let handle = source.handle;
        let mut plan = BatchPlan::default();
        assert_eq!(
            plan.prepare(&snapshot(Vec::new(), &[handle])),
            Err(SnapshotError::MissingMesh { handle })
        );
        let duplicate = RenderMesh {
            handle,
            mesh: Arc::clone(&source.mesh),
        };
        assert_eq!(
            plan.prepare(&snapshot(vec![source, duplicate], &[handle])),
            Err(SnapshotError::DuplicateMesh { handle })
        );
    }

    #[test]
    fn culled_mesh_sources_retain_residency_and_source_removal_retires_it() {
        let first = mesh();
        let second = mesh();
        let first_handle = first.handle;
        let second_handle = second.handle;
        let mut resident = BTreeMap::from([(first_handle, 10), (second_handle, 20)]);
        let mut plan = BatchPlan::default();
        let mut snapshot = snapshot(vec![first, second], &[second_handle]);
        plan.prepare(&snapshot).unwrap();
        resident.retain(|handle, _| plan.contains_mesh(*handle));
        assert_eq!(
            resident,
            BTreeMap::from([(first_handle, 10), (second_handle, 20)])
        );
        snapshot.instances.clear();
        plan.prepare(&snapshot).unwrap();
        resident.retain(|handle, _| plan.contains_mesh(*handle));
        assert_eq!(resident.len(), 2);
        assert!(plan.batches.is_empty());
        snapshot.meshes.clear();
        plan.prepare(&snapshot).unwrap();
        resident.retain(|handle, _| plan.contains_mesh(*handle));
        assert!(resident.is_empty());
        assert!(plan.batches.is_empty());
    }

    #[test]
    fn first_culled_appearance_has_no_upload_plan_and_returning_visibility_reuses_handles() {
        let source_mesh = mesh();
        let mesh_handle = source_mesh.handle;
        let source_texture = texture();
        let texture_handle = source_texture.handle;
        let mut snapshot = snapshot(vec![source_mesh], &[]);
        snapshot.textures.push(source_texture);
        let mut plan = BatchPlan::default();
        plan.prepare(&snapshot).unwrap();
        assert!(plan.contains_mesh(mesh_handle));
        assert!(plan.contains_texture(texture_handle));
        assert!(plan.batches.is_empty());
        assert!(plan.texture_sources.is_empty());
        snapshot.instances.push(RenderInstance {
            model: Mat4::IDENTITY,
            color: [1.0; 4],
            surface: engine_core::Surface::MATTE,
            mesh: mesh_handle,
            texture: Some(texture_handle),
        });
        plan.prepare(&snapshot).unwrap();
        assert_eq!(plan.batches.len(), 1);
        assert_eq!(plan.batches[0].mesh, mesh_handle);
        assert_eq!(
            snapshot.textures[plan.texture_sources[0]].handle,
            texture_handle
        );
        let visible = snapshot.instances.pop().unwrap();
        plan.prepare(&snapshot).unwrap();
        assert!(plan.batches.is_empty());
        assert!(plan.texture_sources.is_empty());
        assert!(plan.contains_mesh(mesh_handle));
        assert!(plan.contains_texture(texture_handle));
        snapshot.instances.push(visible);
        plan.prepare(&snapshot).unwrap();
        assert_eq!(plan.batches[0].mesh, mesh_handle);
        assert_eq!(
            snapshot.textures[plan.texture_sources[0]].handle,
            texture_handle
        );
    }

    #[test]
    fn nonfinite_camera_is_rejected() {
        let mut snapshot = snapshot(Vec::new(), &[]);
        snapshot.view_projection.x_axis.x = f32::NAN;
        assert_eq!(
            BatchPlan::default().prepare(&snapshot),
            Err(SnapshotError::InvalidCamera)
        );
    }
}
