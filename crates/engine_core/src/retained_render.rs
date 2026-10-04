//! Engine-owned render records. Structural admission resolves immutable assets;
//! ordinary frames update poses and visibility without rebuilding those records.

use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use bevy_ecs::{
    change_detection::{ContiguousRef, MAX_CHANGE_AGE, Tick},
    entity::{Entity, EntityHashMap},
    prelude::*,
    system::SystemState,
};
use glam::Mat4;

use crate::hierarchy::HierarchyScratch;

use crate::{
    Aabb, AssetLibrary, Camera, Cube, GlobalTransforms, GpuPresentation, MAX_TRANSFORM_NODES,
    MaterialBinding, MaterialHandle, Mesh3d, MeshHandle, PresentedTransforms, RenderInstance,
    RenderJoint, RenderMaterial, RenderMesh, RenderQuery, RenderSnapshot, RenderTexture, Surface,
    TextureHandle, Tint, Transform, TransformIndex,
    culling::{Classification, Frustum},
    decals::DecalExtraction,
    hierarchy::ChangeWindow,
    probes::ProbeExtraction,
    skin::SkinExtraction,
};

// Change stamps of render sources, read in contiguous table slices. GPU-presented
// frames skip the per-renderable pass when nothing structural changed.
type SourceStamps = SystemState<(
    Query<'static, 'static, Ref<'static, Mesh3d>>,
    Query<'static, 'static, Ref<'static, Cube>>,
    Query<'static, 'static, Ref<'static, Tint>>,
)>;

static NEXT_STREAM: AtomicU64 = AtomicU64::new(1);

/// An affine model matrix as its transposed top three rows: row `r` holds the
/// three basis coefficients and the translation of world coordinate `r`. This
/// is the retained upload layout, so renderers copy it without repacking.
pub type PoseRows = [[f32; 4]; 3];

pub(crate) const IDENTITY_ROWS: PoseRows = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
];

/// Exact coefficient copies; the fourth row of an admitted affine matrix is
/// always [0, 0, 0, 1] and is not stored.
#[inline]
pub(crate) fn pose_rows(model: Mat4) -> PoseRows {
    let rows = model.transpose();
    [
        rows.x_axis.to_array(),
        rows.y_axis.to_array(),
        rows.z_axis.to_array(),
    ]
}

/// A synchronous read lease over engine-owned render records.
///
/// Only the engine constructs this value. Revisions certify immutable data for
/// this lease, unlike caller-mutable [`RenderSnapshot`]. Slots are stable across
/// ordinary frames, but may be reused after removal; structure revisions cover
/// reuse. `visible_indices` is the complete bounded submission order. Inactive
/// slots in `snapshot().instances` must never be submitted or validated as draws.
///
/// Poses are read with [`Self::model`] from the prepared presented store; the
/// snapshot's `RenderInstance::model` fields are not maintained for retained
/// frames. Owned snapshots from [`crate::Engine::extract`] still carry models.
#[derive(Clone, Copy)]
pub struct RenderFrame<'a> {
    owner: &'a RetainedRender,
    poses: &'a PresentedTransforms,
    hierarchy: &'a HierarchyScratch,
}

impl<'a> RenderFrame<'a> {
    /// Slot metadata (mesh, material, color), counters, camera and sources.
    pub fn snapshot(self) -> &'a RenderSnapshot {
        &self.owner.snapshot
    }
    /// Displayed model matrix of an active slot, read from the pose prepared
    /// for this frame through the slot's validated store index. Inactive slots
    /// return an arbitrary finite matrix that must never be submitted.
    #[inline]
    pub fn model(self, slot: usize) -> Mat4 {
        self.owner
            .pose_indices
            .get(slot)
            .and_then(|&index| self.poses.prepared_matrix(index as usize))
            .unwrap_or(Mat4::IDENTITY)
    }
    pub fn identity(self) -> u64 {
        self.owner.identity
    }
    pub fn sequence(self) -> u64 {
        self.owner.sequence
    }
    pub fn structure_revision(self) -> u64 {
        self.owner.structure_revision
    }
    /// Slot ownership, mesh/texture keys and source-array revision. Visibility
    /// changes alone preserve this stamp so renderers can retain slot buckets.
    pub fn source_revision(self) -> u64 {
        self.owner.source_revision
    }
    pub fn pose_revision(self) -> u64 {
        self.owner.pose_revision
    }
    /// Advances when any slot's color or surface changes.
    pub fn color_revision(self) -> u64 {
        self.owner.color_revision
    }
    /// Advances whenever the snapshot's lights are rewritten; frames that
    /// keep it keep every light record.
    pub fn lights_revision(self) -> u64 {
        self.owner.lights_revision
    }
    /// Advances whenever the snapshot's joints change; frames that keep it
    /// keep every joint record.
    pub fn joints_revision(self) -> u64 {
        self.owner.joints_revision
    }
    pub fn visible_indices(self) -> &'a [usize] {
        &self.owner.visible
    }
    /// Every slot that may cast a shadow: active, with a valid pose and a
    /// resolved mesh, whether culled or visible, in slot order. GPU-presented
    /// frames list their candidates, as `visible_indices` does; the GPU
    /// hides invalid ones.
    pub fn casters(self) -> &'a [usize] {
        if self.owner.gpu {
            &self.owner.visible
        } else {
            &self.owner.casters
        }
    }
    /// Advances when [`Self::casters`] changes.
    pub fn caster_revision(self) -> u64 {
        if self.owner.gpu {
            self.owner.structure_revision
        } else {
            self.owner.caster_revision
        }
    }
    /// Per-slot stamps; compare within this stream, never across stream identities.
    pub fn pose_changes(self) -> &'a [u64] {
        &self.owner.pose_changes
    }
    /// [`Self::model`] of every slot in upload layout, parallel to the
    /// snapshot instances. A slot's rows are rewritten exactly when its
    /// [`Self::pose_changes`] stamp advances, while extraction already holds
    /// the matrix. Inactive slots hold arbitrary finite rows that must never
    /// be submitted.
    pub fn pose_rows(self) -> &'a [PoseRows] {
        &self.owner.pose_rows
    }
    /// GPU presentation inputs when this frame leaves blending, composition and
    /// culling to the GPU (decision 014). Then `visible_indices` lists every
    /// candidate slot, [`Self::model`] and `pose_rows` are not prepared, and
    /// the renderer culls with [`Self::culling`] and [`Self::instance_budget`].
    pub fn gpu(self) -> Option<GpuPresentation<'a>> {
        self.owner
            .gpu
            .then(|| self.poses.gpu_presentation(self.hierarchy))
    }
    /// Hierarchy node index of each slot (the index space of
    /// [`GpuPresentation`] records); `u32::MAX` for never-mapped slots.
    pub fn slot_nodes(self) -> &'a [u32] {
        &self.owner.pose_indices
    }
    /// Whether frustum culling applies to this frame.
    pub fn culling(self) -> bool {
        self.owner.culling
    }
    /// Maximum submitted instances; later visible instances are dropped.
    pub fn instance_budget(self) -> usize {
        self.owner.budget
    }
    /// Per-slot stamps of color or surface changes.
    pub fn color_changes(self) -> &'a [u64] {
        &self.owner.color_changes
    }

    /// Copies this CPU-prepared frame into `snapshot`, reusing its capacity:
    /// the visible instances in submission order with their displayed
    /// models, local lights in world space (those on slots without a valid
    /// pose count as dropped), and every other record as the frame holds it.
    pub(crate) fn copy_to(self, snapshot: &mut RenderSnapshot) {
        let RenderSnapshot {
            view_projection,
            camera_position,
            elapsed_seconds,
            environment,
            lights,
            dropped_lights,
            emitters,
            dropped_particles,
            decals,
            decal_materials,
            decal_textures,
            dropped_decals,
            reflection_probes,
            dropped_reflection_probes,
            instances,
            joints,
            meshes,
            materials,
            textures,
            render_candidates,
            visible_instances,
            culled_instances,
            dropped_instances,
            missing_assets,
            invalid_transforms,
        } = self.snapshot();
        snapshot.view_projection = *view_projection;
        snapshot.camera_position = *camera_position;
        snapshot.elapsed_seconds = *elapsed_seconds;
        snapshot.environment = *environment;
        snapshot.lights.clear();
        snapshot.dropped_lights = *dropped_lights;
        for light in lights {
            let Some(slot) = light.slot else {
                snapshot.lights.push(*light);
                continue;
            };
            let model = self
                .owner
                .pose_indices
                .get(slot as usize)
                .and_then(|&index| self.poses.prepared_matrix(index as usize));
            let Some(model) = model else {
                snapshot.dropped_lights += 1;
                continue;
            };
            snapshot.lights.push(crate::RenderLight {
                position: model.transform_point3(light.position),
                direction: model.transform_vector3(light.direction).normalize_or_zero(),
                slot: None,
                ..*light
            });
        }
        snapshot.emitters.clone_from(emitters);
        snapshot.dropped_particles = *dropped_particles;
        snapshot.decals.clone_from(decals);
        snapshot.decal_materials.clone_from(decal_materials);
        snapshot.decal_textures.clone_from(decal_textures);
        snapshot.dropped_decals = *dropped_decals;
        snapshot.reflection_probes.clone_from(reflection_probes);
        snapshot.dropped_reflection_probes = *dropped_reflection_probes;
        snapshot.instances.clear();
        snapshot
            .instances
            .extend(self.visible_indices().iter().map(|&slot| RenderInstance {
                model: self.model(slot),
                ..instances[slot]
            }));
        snapshot.joints.clone_from(joints);
        snapshot.meshes.clone_from(meshes);
        snapshot.materials.clone_from(materials);
        snapshot.textures.clone_from(textures);
        snapshot.render_candidates = *render_candidates;
        snapshot.visible_instances = *visible_instances;
        snapshot.culled_instances = *culled_instances;
        snapshot.dropped_instances = *dropped_instances;
        snapshot.missing_assets = *missing_assets;
        snapshot.invalid_transforms = *invalid_transforms;
        engine_profile::counter(
            "extraction.instance_capacity_bytes",
            (snapshot.instances.capacity() * size_of::<RenderInstance>()) as f64,
        );
    }
}

struct ResolvedSource {
    mesh_index: usize,
    material_index: Option<usize>,
    mesh: MeshHandle,
    material: Option<MaterialHandle>,
    color: [f32; 4],
    surface: Surface,
}

const NO_SOURCE: u32 = u32::MAX;

/// Change ticks of the source a slot renders (its Mesh3d, else its Cube)
/// and of its Tint. Absent components are distinguished by `present`;
/// `UNKNOWN` never equals an observed key.
#[derive(Clone, Copy, PartialEq, Eq)]
struct SourceKey {
    ticks: [u32; 2],
    present: u8,
}

impl SourceKey {
    const UNKNOWN: Self = Self {
        ticks: [0; 2],
        present: 0,
    };

    #[inline]
    fn new(cube: Option<Tick>, mesh: Option<Tick>, tint: Option<Tick>) -> Self {
        // A Mesh3d renders in place of a Cube, whose changes then matter not.
        let source = mesh.or(cube);
        Self {
            ticks: [source.map_or(0, Tick::get), tint.map_or(0, Tick::get)],
            present: 0b1000
                | u8::from(cube.is_some())
                | (u8::from(mesh.is_some()) << 1)
                | (u8::from(tint.is_some()) << 2),
        }
    }
}

/// Per-slot state every retained frame reads and updates for a moving slot,
/// in 32 bytes. Ownership stays in `RetainedRender::owners`; the presented
/// store hint is `pose_indices`.
#[derive(Clone, Copy)]
struct SlotFrame {
    // Published presented revisions are never 0.
    pose_revision: u64,
    // Source indices of the resolved mesh/material; NO_SOURCE when unresolved
    // (mesh) or without maps (material).
    mesh: u32,
    material: u32,
    source: SourceKey,
    outside: bool,
}

impl SlotFrame {
    const NEW: Self = Self {
        pose_revision: 0,
        mesh: NO_SOURCE,
        material: NO_SOURCE,
        source: SourceKey::UNKNOWN,
        outside: false,
    };
}

#[derive(Default)]
struct Sources {
    meshes: Vec<RenderMesh>,
    // Local bounds of each mesh source, read by culling through slot indices.
    bounds: Vec<Aabb>,
    // Meshes with joints (decision 036): their joints place them, so their
    // bounds never cull.
    skinned: Vec<bool>,
    // Materials with maps (decision 031) and the textures they read.
    materials: Vec<RenderMaterial>,
    textures: Vec<RenderTexture>,
    mesh_indices: HashMap<MeshHandle, usize>,
    // Materials with maps to their binding's index in `materials`, which
    // holds the first material of each binding (decision 031).
    material_indices: HashMap<MaterialHandle, usize>,
    binding_indices: HashMap<MaterialBinding, usize>,
    texture_indices: HashMap<TextureHandle, usize>,
    mesh_used: Vec<bool>,
    material_used: Vec<bool>,
    previous_mesh_used: Vec<bool>,
    previous_material_used: Vec<bool>,
    // Publishing scratch: textures already listed.
    texture_listed: Vec<bool>,
}

impl Sources {
    fn clear(&mut self) {
        self.meshes.clear();
        self.bounds.clear();
        self.skinned.clear();
        self.materials.clear();
        self.textures.clear();
        self.mesh_indices.clear();
        self.material_indices.clear();
        self.binding_indices.clear();
        self.texture_indices.clear();
        self.mesh_used.clear();
        self.material_used.clear();
        self.previous_mesh_used.clear();
        self.previous_material_used.clear();
    }

    fn resolve(
        &mut self,
        assets: &AssetLibrary,
        cube_mesh: MeshHandle,
        cube: Option<&Cube>,
        mesh3d: Option<&Mesh3d>,
        tint: Option<&Tint>,
    ) -> Option<ResolvedSource> {
        let (handle, color, mapped, surface) = if let Some(mesh3d) = mesh3d {
            let material = assets.material(mesh3d.material)?;
            (
                mesh3d.mesh,
                material.color(),
                material.has_maps().then_some((mesh3d.material, material)),
                material.surface(),
            )
        } else {
            let [r, g, b, _] = cube?.color;
            // Cubes are always opaque.
            (cube_mesh, [r, g, b, 1.0], None, Surface::MATTE)
        };
        let color = tint.map_or(color, |tint| tint.apply(color));
        let mesh = assets.mesh(handle)?;
        // Resolve all weak handles before retaining any source for this record.
        if let Some((_, material)) = mapped {
            for texture in material.textures() {
                assets.texture(texture)?;
            }
        }
        let mesh_index = *self.mesh_indices.entry(handle).or_insert_with(|| {
            let index = self.meshes.len();
            self.meshes.push(RenderMesh {
                handle,
                mesh: Arc::clone(mesh),
            });
            self.bounds.push(mesh.bounds());
            self.skinned.push(mesh.joint_count() != 0);
            self.mesh_used.push(false);
            index
        });
        // Materials with equal bindings draw as the first one resolved.
        let material_index = mapped.map(|(material_handle, material)| {
            *self
                .material_indices
                .entry(material_handle)
                .or_insert_with(|| {
                    let binding = material.binding().expect("materials with maps bind");
                    *self.binding_indices.entry(binding).or_insert_with(|| {
                        for texture in material.textures() {
                            self.texture_indices.entry(texture).or_insert_with(|| {
                                self.textures.push(RenderTexture {
                                    handle: texture,
                                    texture: Arc::clone(
                                        assets.texture(texture).expect("resolved above"),
                                    ),
                                });
                                self.textures.len() - 1
                            });
                        }
                        self.materials.push(RenderMaterial {
                            handle: material_handle,
                            material: *material,
                        });
                        self.material_used.push(false);
                        self.materials.len() - 1
                    })
                })
        });
        Some(ResolvedSource {
            mesh_index,
            material_index,
            mesh: handle,
            material: material_index.map(|index| self.materials[index].handle),
            color,
            surface,
        })
    }

    fn publish(&mut self, snapshot: &mut RenderSnapshot, force: bool) -> bool {
        if !force
            && self.mesh_used == self.previous_mesh_used
            && self.material_used == self.previous_material_used
        {
            return false;
        }
        snapshot.meshes.clear();
        snapshot.materials.clear();
        snapshot.textures.clear();
        snapshot.meshes.extend(
            self.meshes
                .iter()
                .zip(&self.mesh_used)
                .filter(|(_, used)| **used)
                .map(|(mesh, _)| mesh.clone()),
        );
        // Used materials and each of their textures once.
        self.texture_listed.clear();
        self.texture_listed.resize(self.textures.len(), false);
        for (material, _) in self
            .materials
            .iter()
            .zip(&self.material_used)
            .filter(|(_, used)| **used)
        {
            snapshot.materials.push(*material);
            for texture in material.material.textures() {
                let index = self.texture_indices[&texture];
                if !std::mem::replace(&mut self.texture_listed[index], true) {
                    snapshot.textures.push(self.textures[index].clone());
                }
            }
        }
        self.previous_mesh_used.clone_from(&self.mesh_used);
        self.previous_material_used.clone_from(&self.material_used);
        true
    }
}

pub(crate) struct RetainedRender {
    identity: u64,
    sequence: u64,
    structure_revision: u64,
    source_revision: u64,
    pose_revision: u64,
    color_revision: u64,
    caster_revision: u64,
    lights_revision: u64,
    joints_revision: u64,
    // Joints the skins listed, which decals' joints follow, and the decals'
    // joints of the last frame.
    skin_joints: usize,
    decal_joints: Vec<RenderJoint>,
    snapshot: RenderSnapshot,
    // Owner of each slot; None for free slots. Parallel to `frames`.
    owners: Vec<Option<Entity>>,
    frames: Vec<SlotFrame>,
    // Slots visited by the current preparation, one bit each.
    visited: Vec<u64>,
    entities: EntityHashMap<usize>,
    free: Vec<usize>,
    // Slot cached per query row. Freeing a slot clears every entry naming it,
    // so a matching Entity alone proves the cached slot is still its own.
    query_slots: Vec<Option<(Entity, usize)>>,
    visible: Vec<usize>,
    // CPU-prepared frames: every active slot with a valid pose and a mesh.
    casters: Vec<usize>,
    // Presented-store index of each slot's pose, validated for its entity
    // whenever the slot is active. u32 covers the bounded transform store.
    pose_indices: Vec<u32>,
    pose_changes: Vec<u64>,
    pose_rows: Vec<PoseRows>,
    color_changes: Vec<u64>,
    asset_revision: Option<(u64, u64)>,
    sources: Sources,
    last_scan: Option<Tick>,
    view: Option<([u32; 16], bool)>,
    frustum: Option<Frustum>,
    culling: bool,
    budget: usize,
    // Mode of the last preparation, and the hierarchy topology its candidate
    // node indices belong to when it was GPU-presented.
    gpu: bool,
    gpu_topology: Option<u64>,
    source_stamps: Option<SourceStamps>,
    // Tints at the last source scan (removals change no tick).
    tint_count: Option<usize>,
    // Slots given a joint palette by the last skin extraction, and scratch
    // for the next one's (slot, offset) pairs.
    skinned: Vec<usize>,
    skin_scratch: Vec<(usize, u32)>,
    // What the last GPU-presented skin extraction read (each skinned
    // entity's slot and mesh, in query order; the asset, topology and
    // validity revisions) and the change tick it saw: while all match and no
    // skin changed, its joints stand.
    skin_inputs: Vec<(Entity, usize, Option<MeshHandle>)>,
    skin_key: Option<((u64, u64), u64, u64)>,
    skin_tick: Tick,
}

impl Default for RetainedRender {
    fn default() -> Self {
        let identity = NEXT_STREAM
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .expect("render stream identities exhausted");
        Self {
            identity,
            sequence: 0,
            structure_revision: 0,
            source_revision: 0,
            pose_revision: 0,
            color_revision: 0,
            caster_revision: 0,
            lights_revision: 0,
            joints_revision: 0,
            skin_joints: 0,
            decal_joints: Vec::new(),
            snapshot: RenderSnapshot::default(),
            owners: Vec::new(),
            frames: Vec::new(),
            visited: Vec::new(),
            entities: EntityHashMap::default(),
            free: Vec::new(),
            query_slots: Vec::new(),
            visible: Vec::new(),
            casters: Vec::new(),
            pose_indices: Vec::new(),
            pose_changes: Vec::new(),
            pose_rows: Vec::new(),
            color_changes: Vec::new(),
            asset_revision: None,
            sources: Sources::default(),
            last_scan: None,
            view: None,
            frustum: None,
            culling: false,
            budget: 0,
            gpu: false,
            gpu_topology: None,
            source_stamps: None,
            tint_count: None,
            skinned: Vec::new(),
            skin_scratch: Vec::new(),
            skin_inputs: Vec::new(),
            skin_key: None,
            skin_tick: Tick::new(0),
        }
    }
}

impl RetainedRender {
    /// `poses` must be the store read by the latest [`Self::prepare`].
    pub(crate) fn frame<'a>(
        &'a self,
        poses: &'a PresentedTransforms,
        hierarchy: &'a HierarchyScratch,
    ) -> RenderFrame<'a> {
        RenderFrame {
            owner: self,
            poses,
            hierarchy,
        }
    }

    fn slot(
        &mut self,
        entity: Entity,
        row: usize,
        world: &World,
        globals: &GlobalTransforms,
        cube: MeshHandle,
    ) -> Option<usize> {
        if let Some(Some((cached_entity, index))) = self.query_slots.get(row)
            && *cached_entity == entity
        {
            debug_assert_eq!(self.owners[*index], Some(entity));
            return Some(*index);
        }
        let index = if let Some(&index) = self.entities.get(&entity) {
            index
        } else {
            // Canonical transform admission bounds retained render identity too.
            globals.get(entity)?;
            if self.free.is_empty() && self.owners.len() == MAX_TRANSFORM_NODES {
                // A full scene may be replaced between two extractions. Reclaim
                // stale owners before admitting the new scene, not a frame later.
                // This scan happens only on structural admission at capacity.
                self.free
                    .reserve_exact(self.owners.len().saturating_sub(self.free.len()));
                for (index, owner) in self.owners.iter_mut().enumerate() {
                    if let Some(old) = *owner
                        && (world.get::<Transform>(old).is_none()
                            || (world.get::<Cube>(old).is_none()
                                && world.get::<Mesh3d>(old).is_none())
                            || globals.get(old).is_none())
                    {
                        self.entities.remove(&old);
                        *owner = None;
                        self.frames[index].mesh = NO_SOURCE;
                        self.free.push(index);
                    }
                }
                self.forget_freed_rows();
            }
            let index = if let Some(index) = self.free.pop() {
                self.owners[index] = Some(entity);
                self.frames[index] = SlotFrame::NEW;
                // New owner invalidates both retained GPU channels, including
                // an identical color that was never submitted for this owner.
                self.pose_indices[index] = u32::MAX;
                self.pose_changes[index] = 0;
                self.color_changes[index] = 0;
                index
            } else {
                if self.owners.len() == MAX_TRANSFORM_NODES {
                    return None;
                }
                let index = self.owners.len();
                if index == self.owners.capacity() {
                    let capacity = (index.max(16) * 2).min(MAX_TRANSFORM_NODES);
                    self.owners.reserve_exact(capacity - index);
                    self.frames.reserve_exact(capacity - index);
                    self.snapshot.instances.reserve_exact(capacity - index);
                    self.pose_indices.reserve_exact(capacity - index);
                    self.pose_changes.reserve_exact(capacity - index);
                    self.pose_rows.reserve_exact(capacity - index);
                    self.color_changes.reserve_exact(capacity - index);
                }
                self.owners.push(Some(entity));
                self.frames.push(SlotFrame::NEW);
                // Retained frames read poses through `pose_indices`.
                self.snapshot.instances.push(RenderInstance {
                    mesh: cube,
                    material: None,
                    model: Mat4::IDENTITY,
                    color: [1.0; 4],
                    surface: Surface::MATTE,
                    skin: None,
                });
                self.pose_indices.push(u32::MAX);
                self.pose_changes.push(0);
                self.pose_rows.push(IDENTITY_ROWS);
                self.color_changes.push(0);
                index
            };
            self.entities.insert(entity, index);
            self.structure_revision = self.sequence;
            self.source_revision = self.sequence;
            index
        };
        if let Some(cached) = self.query_slots.get_mut(row) {
            *cached = Some((entity, index));
        }
        Some(index)
    }

    /// Fills this frame's local lights and particle emitters (decisions 023
    /// and 026). A light on an entity whose slot draws a mesh stays in that
    /// slot's model space, following the pose the renderer already has.
    pub(crate) fn extract_lights_and_particles(
        &mut self,
        world: &World,
        lights: &mut crate::lights::LightExtraction,
        particles: &mut crate::particles::ParticleExtraction,
        gpu: bool,
    ) {
        let entities = &self.entities;
        let frames = &self.frames;
        let slot_of = |entity: Entity| {
            entities
                .get(&entity)
                .copied()
                .filter(|&index| frames[index].mesh != NO_SOURCE)
                .map(|index| index as u32)
        };
        lights.extract(
            world,
            &mut self.snapshot,
            slot_of,
            self.source_revision,
            gpu,
        );
        self.lights_revision = lights.revision;
        particles.extract(world, &mut self.snapshot, gpu);
    }

    /// Gives skinned slots (decision 036) their joint palettes in
    /// `snapshot.joints`, rebuilt every frame: on `gpu` frames each joint's
    /// node, otherwise its displayed world matrix times its inverse bind
    /// matrix. A slot whose palette offset changes counts as a color change,
    /// since renderers upload the offset with the slot's color.
    pub(crate) fn extract_skins(&mut self, world: &World, skins: &mut SkinExtraction, gpu: bool) {
        let assets = world.resource::<AssetLibrary>();
        let globals = world.resource::<GlobalTransforms>();
        let presented = world.resource::<PresentedTransforms>();
        // GPU-presented joints hold nodes and inverse binds, not poses: they
        // stand while the skins, their slots and meshes, the assets and the
        // hierarchy's topology and validity stand.
        let this_run = world.read_change_tick();
        if gpu {
            let key = (
                assets.revision(),
                globals.topology_revision,
                globals.validity_epoch(),
            );
            let mut inputs = std::mem::take(&mut self.skin_inputs);
            let previous = inputs.len();
            let mut changed = self.skin_key != Some(key);
            let mut count = 0;
            for (entity, skin) in skins.query.iter(world) {
                changed |= skin.last_changed().is_newer_than(self.skin_tick, this_run);
                let input = match self.entities.get(&entity) {
                    Some(&slot) => (entity, slot, Some(self.snapshot.instances[slot].mesh)),
                    None => (entity, usize::MAX, None),
                };
                if !changed && inputs.get(count) != Some(&input) {
                    changed = true;
                }
                if changed {
                    inputs.truncate(count);
                    inputs.push(input);
                }
                count += 1;
            }
            changed |= count != previous;
            inputs.truncate(count);
            self.skin_inputs = inputs;
            self.skin_tick = this_run;
            if !changed {
                // Last frame's decals follow the skins' joints: they list
                // theirs again.
                self.snapshot.joints.truncate(self.skin_joints);
                engine_profile::counter("extraction.skinned", self.skinned.len() as f64);
                engine_profile::counter("extraction.joints", self.snapshot.joints.len() as f64);
                return;
            }
            self.skin_key = Some(key);
        } else {
            // CPU-prepared joints carry this frame's poses.
            self.skin_key = None;
        }
        let joints = &mut self.snapshot.joints;
        joints.clear();
        let mut next = std::mem::take(&mut self.skin_scratch);
        next.clear();
        for (entity, skin) in skins.query.iter(world) {
            let Some(&slot) = self.entities.get(&entity) else {
                continue;
            };
            if self.frames[slot].mesh == NO_SOURCE {
                continue;
            }
            let Some(mesh) = assets.mesh(self.snapshot.instances[slot].mesh) else {
                continue;
            };
            // Too few joints for the mesh's indices: drawn rigidly.
            let needed = mesh.joint_count() as usize;
            if needed == 0 || needed > skin.joints.len() {
                continue;
            }
            let base = joints.len();
            if skins.hints.len() < base + skin.joints.len() {
                skins
                    .hints
                    .resize(base + skin.joints.len(), TransformIndex::default());
            }
            for (index, &joint) in skin.joints.iter().enumerate() {
                let inverse_bind = skin.inverse_bind(index);
                let hint = &mut skins.hints[base + index];
                let (node, matrix) = if gpu {
                    let node = globals
                        .valid_index(joint, hint)
                        .map_or(u32::MAX, |index| index as u32);
                    (node, IDENTITY_ROWS)
                } else {
                    let world = presented
                        .get_indexed(joint, hint)
                        .and_then(|pose| pose.matrix())
                        .unwrap_or(Mat4::IDENTITY);
                    (u32::MAX, pose_rows(world * inverse_bind))
                };
                joints.push(RenderJoint {
                    node,
                    inverse_bind: pose_rows(inverse_bind),
                    matrix,
                });
            }
            next.push((slot, base as u32));
        }
        next.sort_unstable_by_key(|&(slot, _)| slot);
        self.skin_joints = joints.len();
        self.joints_revision += 1;
        let sequence = self.sequence;
        let mut changed = false;
        for &slot in &self.skinned {
            if next.binary_search_by_key(&slot, |&(slot, _)| slot).is_err()
                && let Some(record) = self.snapshot.instances.get_mut(slot)
                && record.skin.take().is_some()
            {
                self.color_changes[slot] = sequence;
                changed = true;
            }
        }
        self.skinned.clear();
        for &(slot, base) in &next {
            let record = &mut self.snapshot.instances[slot];
            if record.skin != Some(base) {
                record.skin = Some(base);
                self.color_changes[slot] = sequence;
                changed = true;
            }
            self.skinned.push(slot);
        }
        if changed {
            self.color_revision = sequence;
        }
        self.skin_scratch = next;
        engine_profile::counter("extraction.skinned", self.skinned.len() as f64);
        engine_profile::counter("extraction.joints", self.snapshot.joints.len() as f64);
    }

    /// Fills this frame's decals (decision 039), their materials and
    /// textures, and their poses after the skins' joints.
    pub(crate) fn extract_decals(
        &mut self,
        world: &World,
        decals: &mut DecalExtraction,
        gpu: bool,
    ) {
        decals.extract(world, &mut self.snapshot, gpu);
        let listed = &self.snapshot.joints[self.skin_joints.min(self.snapshot.joints.len())..];
        if listed != self.decal_joints.as_slice() {
            self.decal_joints.clear();
            self.decal_joints.extend_from_slice(listed);
            self.joints_revision += 1;
        }
    }

    /// Lists the frame's reflection probes (decision 043).
    pub(crate) fn extract_probes(&mut self, world: &World, probes: &mut ProbeExtraction) {
        probes.extract(world, &mut self.snapshot);
    }

    /// Creates the source stamp window used by GPU-presented frames.
    pub(crate) fn ensure_source_scan(&mut self, world: &mut World) {
        if self.source_stamps.is_none() {
            self.source_stamps = Some(SystemState::new(world));
        }
    }

    /// Whether any Mesh3d, Cube or Tint changed, or a Tint was removed,
    /// since the previous call, consuming that window. The caller advances
    /// the world change tick afterwards.
    fn sources_changed(&mut self, world: &World) -> bool {
        let Some(stamps) = self.source_stamps.as_mut() else {
            return true;
        };
        let Ok((meshes, cubes, tints)) = stamps.get(world) else {
            // Conservatively run the full pass.
            return true;
        };
        // A removed tint changes no tick: the count reveals it, and an
        // insertion elsewhere in the same window changes a tick.
        let tint_count = tints.iter().len();
        let mut changed = self.tint_count != Some(tint_count);
        self.tint_count = Some(tint_count);
        for tints in tints
            .contiguous_iter()
            .expect("Tint uses dense table storage")
        {
            let (_, ticks) = ContiguousRef::split(tints);
            let window = ChangeWindow::new(ticks.last_run(), ticks.this_run());
            changed |= window.any(ticks.changed());
        }
        for meshes in meshes
            .contiguous_iter()
            .expect("Mesh3d uses dense table storage")
        {
            let (_, ticks) = ContiguousRef::split(meshes);
            let window = ChangeWindow::new(ticks.last_run(), ticks.this_run());
            changed |= window.any(ticks.changed());
        }
        for cubes in cubes
            .contiguous_iter()
            .expect("Cube uses dense table storage")
        {
            let (_, ticks) = ContiguousRef::split(cubes);
            let window = ChangeWindow::new(ticks.last_run(), ticks.this_run());
            changed |= window.any(ticks.changed());
        }
        stamps.meta_mut().set_last_run(world.read_change_tick());
        changed
    }

    /// Row caches must never name a slot their entity no longer owns; call
    /// after freeing slots, before any later lookup.
    fn forget_freed_rows(&mut self) {
        for cached in &mut self.query_slots {
            if let Some((entity, index)) = *cached
                && self.owners.get(index) != Some(&Some(entity))
            {
                *cached = None;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare(
        &mut self,
        world: &World,
        query: &mut RenderQuery,
        cube_mesh: MeshHandle,
        aspect: f32,
        culling: bool,
        budget: usize,
        gpu: bool,
    ) {
        let _profile = engine_profile::span("cpu", "extraction_retained_updates_culling");
        self.sequence = self
            .sequence
            .checked_add(1)
            .expect("render sequence exhausted");
        let sequence = self.sequence;
        if self.gpu && !gpu {
            // GPU frames neither culled nor prepared CPU poses. Recompute both
            // for every slot, and their rows, on the first CPU frame after them.
            self.view = None;
            for frame in &mut self.frames {
                frame.pose_revision = 0;
            }
        }
        self.culling = culling;
        self.budget = budget;
        let globals = world.resource::<GlobalTransforms>();
        let stamps_changed = gpu && self.sources_changed(world);
        let scan_tick = world.read_change_tick();
        let aged = self
            .last_scan
            .is_none_or(|last| scan_tick.get().wrapping_sub(last.get()) > MAX_CHANGE_AGE);
        self.last_scan = Some(scan_tick);
        let assets = world.resource::<AssetLibrary>();
        let revision = assets.revision();
        let assets_changed = self.asset_revision != Some(revision) || revision.1 == u64::MAX;
        if assets_changed {
            self.sources.clear();
            self.asset_revision = Some(revision);
        }
        self.sources.mesh_used.fill(false);
        self.sources.material_used.fill(false);
        let view = world.resource::<Camera>().view_projection(aspect);
        crate::effects::extract_lighting(world, &mut self.snapshot);
        let total = query.iter(world).len();
        if gpu
            && self.gpu
            && self.gpu_topology == Some(globals.topology_revision)
            && !aged
            && !assets_changed
            && !stamps_changed
            && total == self.snapshot.render_candidates
        {
            // Same renderables, sources, assets and node indices: candidates,
            // slots and colors are unchanged, and the GPU culls. Only the camera
            // moves on to this frame.
            self.snapshot.view_projection = view;
            engine_profile::counter("extraction.gpu_structure_reused", 1.0);
            return;
        }
        engine_profile::counter("extraction.gpu_structure_reused", 0.0);
        let view_key = (view.to_cols_array().map(f32::to_bits), culling);
        let view_changed = !gpu && self.view != Some(view_key);
        if view_changed {
            self.view = Some(view_key);
            self.frustum = culling.then(|| Frustum::new(view)).flatten();
        }
        let hint_count = total.min(MAX_TRANSFORM_NODES);
        if self.query_slots.capacity() < hint_count {
            self.query_slots
                .reserve_exact(hint_count.saturating_sub(self.query_slots.len()));
        }
        self.query_slots.resize(hint_count, None);
        // GPU frames list every candidate; the GPU applies the budget.
        let output_bound = if gpu { total } else { total.min(budget) }.min(MAX_TRANSFORM_NODES);
        if self.visible.capacity() < output_bound {
            self.visible
                .reserve_exact(output_bound.saturating_sub(self.visible.len()));
        }
        self.snapshot.view_projection = view;
        self.snapshot.render_candidates = total;
        self.snapshot.visible_instances = 0;
        self.snapshot.culled_instances = 0;
        self.snapshot.invalid_transforms = 0;
        self.snapshot.missing_assets = 0;
        self.snapshot.dropped_instances = 0;
        let presented = world.resource::<PresentedTransforms>();
        let mut structure_changed = assets_changed;
        let mut sources_changed = assets_changed;
        let mut submitted = 0;
        let mut casters = 0;
        let mut casters_changed = false;
        let mut seen = 0;
        let mut resolved_count = 0;
        let mut pose_count = 0;
        let mut color_count = 0;
        let mut culling_tests = 0;
        let mut culling_interior = 0;
        let mut culling_reused = 0;
        // Slots can only be admitted up to the query size during this pass.
        let visited_words = (self.owners.len() + total)
            .min(MAX_TRANSFORM_NODES)
            .div_ceil(64);
        self.visited.clear();
        self.visited.resize(visited_words, 0);
        for (row, (entity, has_transform, cube, mesh3d, tint)) in query.iter(world).enumerate() {
            if !has_transform {
                self.snapshot.invalid_transforms += 1;
                continue;
            }
            let Some(index) = self.slot(entity, row, world, globals, cube_mesh) else {
                self.snapshot.invalid_transforms += 1;
                continue;
            };
            self.visited[index / 64] |= 1 << (index % 64);
            seen += 1;
            let frame = &mut self.frames[index];
            let source = SourceKey::new(
                cube.as_ref().map(|value| value.last_changed()),
                mesh3d.as_ref().map(|value| value.last_changed()),
                tint.as_ref().map(|value| value.last_changed()),
            );
            // Every live slot is (re)resolved when assets change or the window
            // ages; new owners start from an unknown source key.
            let source_changed = aged || assets_changed || frame.source != source;
            if source_changed {
                let record = &mut self.snapshot.instances[index];
                resolved_count += 1;
                frame.source = source;
                let resolved = self.sources.resolve(
                    assets,
                    cube_mesh,
                    cube.as_deref(),
                    mesh3d.as_deref(),
                    tint.as_deref(),
                );
                frame.mesh = resolved
                    .as_ref()
                    .map_or(NO_SOURCE, |value| value.mesh_index as u32);
                frame.material = resolved
                    .as_ref()
                    .and_then(|value| value.material_index)
                    .map_or(NO_SOURCE, |material| material as u32);
                if let Some(ResolvedSource {
                    mesh,
                    material,
                    color,
                    surface,
                    ..
                }) = resolved
                {
                    // Opacity decides the pass of materials without maps
                    // (decision 026): switching between opaque and
                    // transparent regroups draws. Materials are immutable,
                    // so their alpha modes change only with their handles.
                    let keys_changed = record.mesh != mesh
                        || record.material != material
                        || (record.color[3] < 1.0) != (color[3] < 1.0);
                    structure_changed |= keys_changed;
                    sources_changed |= keys_changed;
                    record.mesh = mesh;
                    record.material = material;
                    let surface_bits = |surface: Surface| {
                        [
                            surface.roughness,
                            surface.metalness,
                            surface.emission,
                            surface.ripples,
                        ]
                        .map(f32::to_bits)
                    };
                    if self.color_changes[index] == 0
                        || record.color.map(f32::to_bits) != color.map(f32::to_bits)
                        || surface_bits(record.surface) != surface_bits(surface)
                    {
                        record.color = color;
                        record.surface = surface;
                        self.color_changes[index] = sequence;
                        color_count += 1;
                    }
                }
            }
            if gpu {
                // Node indices only; blending, validity and culling run on the
                // GPU. Invalid admitted nodes stay candidates and are hidden there.
                let mut hint = TransformIndex(self.pose_indices[index] as usize);
                let Some((global, _)) = globals.get_indexed_with_revision(entity, &mut hint) else {
                    self.snapshot.invalid_transforms += 1;
                    continue;
                };
                self.pose_indices[index] = hint.0 as u32;
                // Counters partition candidates as on the CPU path; here
                // `visible_instances` counts valid listed candidates before
                // GPU culling.
                if global.matrix().is_none() {
                    self.snapshot.invalid_transforms += 1;
                } else if frame.mesh == NO_SOURCE {
                    self.snapshot.missing_assets += 1;
                } else {
                    self.snapshot.visible_instances += 1;
                }
                if frame.mesh == NO_SOURCE {
                    continue;
                }
                if let Some(previous) = self.visible.get_mut(submitted) {
                    structure_changed |= *previous != index;
                    *previous = index;
                } else {
                    self.visible.push(index);
                    structure_changed = true;
                }
                submitted += 1;
                self.sources.mesh_used[frame.mesh as usize] = true;
                if frame.material != NO_SOURCE {
                    self.sources.material_used[frame.material as usize] = true;
                }
                continue;
            }
            // extract_frame prepared presentation from this same refreshed
            // hierarchy immediately before this pass. Every admitted display
            // is valid exactly when its authoritative pose is valid, so the
            // presented read alone keeps the authoritative validity gate.
            // The slot's pose index is the store hint; it is validated against
            // the full Entity before use.
            let mut hint = TransformIndex(self.pose_indices[index] as usize);
            let pose = presented.get_indexed(entity, &mut hint);
            debug_assert_eq!(
                pose.map(|pose| pose.matrix().is_some()),
                globals.get(entity).map(|global| global.matrix().is_some()),
                "presented validity must match the authoritative boundary"
            );
            let Some((model, pose_revision)) =
                pose.and_then(|pose| Some((pose.matrix()?, pose.revision())))
            else {
                self.snapshot.invalid_transforms += 1;
                continue;
            };
            // Frames read the prepared pose through this validated index
            // instead of copying the matrix here.
            self.pose_indices[index] = hint.0 as u32;
            let pose_changed = frame.pose_revision != pose_revision
                || pose_revision == 0
                || pose_revision == u64::MAX;
            if pose_changed {
                frame.pose_revision = pose_revision;
                self.pose_changes[index] = sequence;
                // Written with the stamp while the matrix is at hand, so no
                // later pass re-reads the presented store to repack it.
                self.pose_rows[index] = pose_rows(model);
                pose_count += 1;
            }
            if frame.mesh == NO_SOURCE {
                self.snapshot.missing_assets += 1;
                continue;
            }
            // Culled or visible, the slot casts shadows.
            if let Some(previous) = self.casters.get_mut(casters) {
                casters_changed |= *previous != index;
                *previous = index;
            } else {
                self.casters.push(index);
                casters_changed = true;
            }
            casters += 1;
            if view_changed || pose_changed || source_changed {
                let bounds = self.sources.bounds[frame.mesh as usize];
                let skinned = self.sources.skinned[frame.mesh as usize];
                frame.outside = !skinned
                    && self.frustum.as_ref().is_some_and(|frustum| {
                        culling_tests += 1;
                        let classification = frustum.classify(model, bounds);
                        culling_interior += usize::from(classification == Classification::Interior);
                        classification.outside()
                    });
            } else if self.frustum.is_some() {
                culling_reused += 1;
            }
            if frame.outside {
                self.snapshot.culled_instances += 1;
            } else {
                self.snapshot.visible_instances += 1;
                if submitted == budget {
                    self.snapshot.dropped_instances += 1;
                    continue;
                }
                if let Some(previous) = self.visible.get_mut(submitted) {
                    structure_changed |= *previous != index;
                    *previous = index;
                } else {
                    self.visible.push(index);
                    structure_changed = true;
                }
                submitted += 1;
            }
            self.sources.mesh_used[frame.mesh as usize] = true;
            if frame.material != NO_SOURCE {
                self.sources.material_used[frame.material as usize] = true;
            }
        }
        if submitted != self.visible.len() {
            self.visible.truncate(submitted);
            structure_changed = true;
        }
        if !gpu && casters != self.casters.len() {
            self.casters.truncate(casters);
            casters_changed = true;
        }
        if casters_changed {
            self.caster_revision = sequence;
        }
        // Each entity owns one slot and visits it at most once, so equal counts
        // prove every owned slot was visited.
        if seen != self.entities.len() {
            self.free
                .reserve_exact(self.owners.len().saturating_sub(self.free.len()));
            for (index, owner) in self.owners.iter_mut().enumerate() {
                if self.visited[index / 64] & (1 << (index % 64)) == 0
                    && let Some(entity) = owner.take()
                {
                    self.entities.remove(&entity);
                    self.frames[index].mesh = NO_SOURCE;
                    self.free.push(index);
                    structure_changed = true;
                    sources_changed = true;
                }
            }
            self.forget_freed_rows();
        }
        let ownership_changed = self.sources.publish(&mut self.snapshot, assets_changed);
        structure_changed |= ownership_changed;
        sources_changed |= ownership_changed;
        if structure_changed {
            self.structure_revision = sequence;
        }
        if sources_changed {
            self.source_revision = sequence;
        }
        if pose_count != 0 {
            self.pose_revision = sequence;
        }
        if color_count != 0 {
            self.color_revision = sequence;
        }
        self.gpu = gpu;
        self.gpu_topology = gpu.then_some(globals.topology_revision);
        engine_profile::counter("extraction.candidates", total as f64);
        engine_profile::counter("extraction.instances", submitted as f64);
        engine_profile::counter("extraction.culled", self.snapshot.culled_instances as f64);
        engine_profile::counter("extraction.retained_resolved", resolved_count as f64);
        engine_profile::counter("extraction.retained_pose_updates", pose_count as f64);
        engine_profile::counter("extraction.retained_color_updates", color_count as f64);
        engine_profile::counter("extraction.culling_tests", culling_tests as f64);
        engine_profile::counter("extraction.culling_interior", culling_interior as f64);
        engine_profile::counter("extraction.culling_reused", culling_reused as f64);
        engine_profile::counter(
            "extraction.instance_capacity_bytes",
            (self.snapshot.instances.capacity() * size_of::<RenderInstance>()) as f64,
        );
        engine_profile::counter(
            "memory.retained_render_capacity_bytes",
            (self.owners.capacity() * size_of::<Option<Entity>>()
                + self.frames.capacity() * size_of::<SlotFrame>()
                + self.visited.capacity() * size_of::<u64>()
                + self.query_slots.capacity() * size_of::<Option<(Entity, usize)>>()
                + (self.visible.capacity() + self.casters.capacity() + self.free.capacity())
                    * size_of::<usize>()
                + self.pose_indices.capacity() * size_of::<u32>()
                + (self.pose_changes.capacity() + self.color_changes.capacity()) * size_of::<u64>()
                + self.pose_rows.capacity() * size_of::<PoseRows>()) as f64,
        );
    }
}

#[cfg(test)]
mod tests;
