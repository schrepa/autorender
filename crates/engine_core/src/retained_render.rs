//! Engine-owned render records. Structural admission resolves immutable assets;
//! ordinary frames update poses and visibility without rebuilding those records.

use std::{
    collections::HashMap,
    hash::{BuildHasherDefault, Hasher},
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
    Aabb, AlphaMode, AssetLibrary, Camera, Cube, GlobalTransforms, GpuPresentation,
    MAX_TRANSFORM_NODES, Material, MaterialBinding, MaterialHandle, Mesh3d, MeshHandle,
    PresentedTransforms, RenderInstance, RenderJoint, RenderMaterial, RenderMesh, RenderQuery,
    RenderSnapshot, RenderTexture, Surface, TextureHandle, Tint, Transform, TransformIndex,
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

/// Whether any `T` changed within the window `components` last ran over.
fn changed_in_window<T: Component>(components: &Query<'_, '_, Ref<'static, T>>) -> bool {
    components
        .contiguous_iter()
        .expect("render sources use dense table storage")
        .any(|tables| {
            let (_, ticks) = ContiguousRef::split(tables);
            ChangeWindow::new(ticks.last_run(), ticks.this_run()).any(ticks.changed())
        })
}

static NEXT_STREAM: AtomicU64 = AtomicU64::new(1);

/// Hashes asset handles and bindings, a few integers each, with one
/// multiply per word instead of SipHash: structural passes look one up per
/// object.
#[derive(Default)]
struct HandleHasher(u64);

impl Hasher for HandleHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.write_u64(u64::from(byte));
        }
    }

    fn write_u32(&mut self, value: u32) {
        self.write_u64(u64::from(value));
    }

    fn write_u64(&mut self, value: u64) {
        let mixed = (self.0 ^ value).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        self.0 = mixed ^ (mixed >> 32);
    }

    fn write_usize(&mut self, value: usize) {
        self.write_u64(value as u64);
    }
}

type HandleMap<K, V> = HashMap<K, V, BuildHasherDefault<HandleHasher>>;

/// What places a material's objects among draws besides its binding's
/// contents: its pass, alpha mode, sides and transmission. A replacement
/// that keeps it keeps its objects' draw keys.
fn draw_class(material: &Material) -> (bool, u8, bool, bool) {
    (
        material.is_transparent(),
        match material.alpha_mode() {
            AlphaMode::Opaque => 0,
            AlphaMode::Mask { .. } => 1,
            AlphaMode::Blend => 2,
        },
        material.is_double_sided(),
        material.transmission().factor > 0.0,
    )
}

/// What a publish changed in the snapshot's source lists.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Published {
    Unchanged,
    /// Listed materials' contents, in place: slots keep their draw keys.
    Contents,
    /// The meshes or materials listed.
    Sources,
}

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
    /// Slot ownership, draw keys and the listed meshes and materials.
    /// Visibility changes alone preserve this stamp so renderers can retain
    /// slot buckets, and so do listed materials replaced in place,
    /// which advance only [`Self::structure_revision`].
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
    /// Advances whenever the snapshot's particle emitters change; frames
    /// that keep it keep every emitter record.
    pub fn emitters_revision(self) -> u64 {
        self.owner.emitters_revision
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
    /// culling to the GPU. Then `visible_indices` lists every
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
            presented_seconds,
            environment,
            lights,
            dropped_lights,
            invalid_lights,
            emitters,
            dropped_particles,
            invalid_emitters,
            decals,
            decal_materials,
            decal_textures,
            dropped_decals,
            invalid_decals,
            reflection_probes,
            dropped_reflection_probes,
            terrain,
            deformation,
            cells,
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
        snapshot.presented_seconds = *presented_seconds;
        snapshot.environment = *environment;
        snapshot.lights.clear();
        snapshot.dropped_lights = *dropped_lights;
        snapshot.invalid_lights = *invalid_lights;
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
        snapshot.invalid_emitters = *invalid_emitters;
        snapshot.decals.clone_from(decals);
        snapshot.decal_materials.clone_from(decal_materials);
        snapshot.decal_textures.clone_from(decal_textures);
        snapshot.dropped_decals = *dropped_decals;
        snapshot.invalid_decals = *invalid_decals;
        snapshot.reflection_probes.clone_from(reflection_probes);
        snapshot.dropped_reflection_probes = *dropped_reflection_probes;
        snapshot.terrain.clone_from(terrain);
        snapshot.deformation.clone_from(deformation);
        snapshot.cells = *cells;
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
    // Meshes with joints: their joints place them, so their
    // bounds never cull.
    skinned: Vec<bool>,
    // Materials with maps and the textures they read.
    materials: Vec<RenderMaterial>,
    textures: Vec<RenderTexture>,
    mesh_indices: HandleMap<MeshHandle, usize>,
    // Materials with maps to their binding's index in `materials`, which
    // holds the first material of each binding.
    material_indices: HandleMap<MaterialHandle, usize>,
    binding_indices: HandleMap<MaterialBinding, usize>,
    texture_indices: HandleMap<TextureHandle, usize>,
    // The version each object's material resolved at, with maps or not:
    // a replacement re-resolves only its objects.
    versions: HandleMap<MaterialHandle, u64>,
    mesh_used: Vec<bool>,
    material_used: Vec<bool>,
    previous_mesh_used: Vec<bool>,
    previous_material_used: Vec<bool>,
    // Listed materials changed in place since the last publish.
    contents_changed: bool,
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
        self.versions.clear();
        self.mesh_used.clear();
        self.material_used.clear();
        self.previous_mesh_used.clear();
        self.previous_material_used.clear();
        self.contents_changed = false;
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
            let version = assets.material_version(mesh3d.material).unwrap_or(0);
            self.versions.insert(mesh3d.material, version);
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
                            version: assets.material_version(material_handle).unwrap_or(0),
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

    /// Materials replaced since their objects resolved. A
    /// replaced material alone drawing as its binding is updated in place
    /// when its draw class stands; every other one forgets its binding, and
    /// a binding it held forgets the materials drawing as it. Returns the
    /// materials whose objects resolve again.
    fn replaced(&mut self, assets: &AssetLibrary) -> Vec<MaterialHandle> {
        let mut handles = Vec::new();
        for (&handle, version) in &mut self.versions {
            let current = assets.material_version(handle);
            if current != Some(*version) {
                *version = current.unwrap_or(u64::MAX);
                handles.push(handle);
            }
        }
        for position in 0..handles.len() {
            let handle = handles[position];
            // Objects without maps follow on their own when they resolve.
            let Some(&index) = self.material_indices.get(&handle) else {
                continue;
            };
            if self.update_in_place(assets, handle, index) {
                continue;
            }
            self.material_indices.remove(&handle);
            let source = self.materials[index];
            if source.handle != handle {
                continue;
            }
            // The binding drew as this material's old contents.
            if let Some(binding) = source.material.binding()
                && self.binding_indices.get(&binding) == Some(&index)
            {
                self.binding_indices.remove(&binding);
            }
            let sharing: Vec<MaterialHandle> = self
                .material_indices
                .iter()
                .filter(|&(_, &other)| other == index)
                .map(|(&other, _)| other)
                .collect();
            for other in sharing {
                self.material_indices.remove(&other);
                if !handles.contains(&other) {
                    handles.push(other);
                }
            }
        }
        handles
    }

    /// Updates listed material `index`, which replaced `handle` alone draws
    /// as, to `handle`'s contents when its draw class stands: its objects
    /// keep their draw keys, and only the listed contents change.
    fn update_in_place(
        &mut self,
        assets: &AssetLibrary,
        handle: MaterialHandle,
        index: usize,
    ) -> bool {
        let Some(material) = assets.material(handle) else {
            return false;
        };
        let source = self.materials[index];
        if source.handle != handle
            || !material.has_maps()
            || draw_class(&source.material) != draw_class(material)
            || material
                .textures()
                .any(|texture| assets.texture(texture).is_none())
            || self
                .material_indices
                .iter()
                .any(|(&other, &other_index)| other_index == index && other != handle)
        {
            return false;
        }
        if let Some(binding) = source.material.binding()
            && self.binding_indices.get(&binding) == Some(&index)
        {
            self.binding_indices.remove(&binding);
        }
        let binding = material.binding().expect("materials with maps bind");
        self.binding_indices.entry(binding).or_insert(index);
        for texture in material.textures() {
            self.texture_indices.entry(texture).or_insert_with(|| {
                self.textures.push(RenderTexture {
                    handle: texture,
                    texture: Arc::clone(assets.texture(texture).expect("checked above")),
                });
                self.textures.len() - 1
            });
        }
        self.materials[index] = RenderMaterial {
            handle,
            material: *material,
            version: assets.material_version(handle).unwrap_or(0),
        };
        self.contents_changed = true;
        true
    }

    /// Lists the used sources in `snapshot` when they changed: `Sources`
    /// when the used meshes or materials changed (or `force`), `Contents`
    /// when only listed materials changed in place.
    fn publish(&mut self, snapshot: &mut RenderSnapshot, force: bool) -> Published {
        let sets_changed = force
            || self.mesh_used != self.previous_mesh_used
            || self.material_used != self.previous_material_used;
        if !sets_changed && !self.contents_changed {
            return Published::Unchanged;
        }
        self.contents_changed = false;
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
        if sets_changed {
            Published::Sources
        } else {
            Published::Contents
        }
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
    emitters_revision: u64,
    snapshot: RenderSnapshot,
    // The engine's cube, which `Cube` components draw.
    cube_mesh: MeshHandle,
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
    material_revision: Option<u64>,
    sources: Sources,
    last_scan: Option<Tick>,
    view: Option<([u32; 16], bool)>,
    frustum: Option<Frustum>,
    culling: bool,
    budget: usize,
    // Instances whose assets were missing at the last instance pass: frames
    // that skip the pass keep its count, which decals' add to.
    missing_sources: usize,
    // Mode of the last preparation, and the hierarchy topology its candidate
    // node indices belong to when it was GPU-presented.
    gpu: bool,
    gpu_topology: Option<u64>,
    source_stamps: Option<SourceStamps>,
    // Tints at the last source scan (removals change no tick).
    tint_count: Option<usize>,
}

/// What one instance pass changed and counted.
#[derive(Default)]
struct Pass {
    structure_changed: bool,
    sources_changed: bool,
    casters_changed: bool,
    // Slots submitted and listed as casters so far, and owned slots visited.
    submitted: usize,
    casters: usize,
    seen: usize,
    // Profile counters.
    resolved: usize,
    poses: usize,
    colors: usize,
    culling_tests: usize,
    culling_interior: usize,
    culling_reused: usize,
}

impl RetainedRender {
    /// The slot `entity` draws in, while it has one.
    pub(crate) fn slot_of(&self, entity: Entity) -> Option<u32> {
        self.entities.get(&entity).map(|&slot| slot as u32)
    }

    /// Records that draw `Cube` components with `cube_mesh`, frustum cull
    /// when `culling`, and submit at most `budget` instances a frame.
    pub(crate) fn new(cube_mesh: MeshHandle, culling: bool, budget: usize) -> Self {
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
            emitters_revision: 0,
            snapshot: RenderSnapshot::default(),
            cube_mesh,
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
            material_revision: None,
            sources: Sources::default(),
            last_scan: None,
            view: None,
            frustum: None,
            culling,
            budget,
            missing_sources: 0,
            gpu: false,
            gpu_topology: None,
            source_stamps: None,
            tint_count: None,
        }
    }

    /// Frustum culling of the frames prepared from now on.
    pub(crate) fn set_culling(&mut self, enabled: bool) {
        self.culling = enabled;
    }

    pub(crate) fn culling(&self) -> bool {
        self.culling
    }

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
                    mesh: self.cube_mesh,
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

    /// Fills this frame's local lights and particle emitters.
    /// A light on an entity whose slot draws a mesh stays in that
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
        self.emitters_revision = particles.revision;
    }

    /// Gives skinned slots their joint palettes in
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
            let mut inputs = std::mem::take(&mut skins.inputs);
            let previous = inputs.len();
            let mut changed = skins.key != Some(key);
            let mut count = 0;
            for (entity, skin) in skins.query.iter(world) {
                changed |= skin.last_changed().is_newer_than(skins.tick, this_run);
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
            skins.inputs = inputs;
            skins.tick = this_run;
            if !changed {
                // Last frame's decals follow the skins' joints: they list
                // theirs again.
                self.snapshot.joints.truncate(skins.joints);
                engine_profile::counter("extraction.skinned", skins.skinned.len() as f64);
                engine_profile::counter("extraction.joints", self.snapshot.joints.len() as f64);
                return;
            }
            skins.key = Some(key);
        } else {
            // CPU-prepared joints carry this frame's poses.
            skins.key = None;
        }
        let joints = &mut self.snapshot.joints;
        joints.clear();
        let mut next = std::mem::take(&mut skins.scratch);
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
        skins.joints = joints.len();
        self.joints_revision += 1;
        let sequence = self.sequence;
        let mut changed = false;
        for &slot in &skins.skinned {
            if next.binary_search_by_key(&slot, |&(slot, _)| slot).is_err()
                && let Some(record) = self.snapshot.instances.get_mut(slot)
                && record.skin.take().is_some()
            {
                self.color_changes[slot] = sequence;
                changed = true;
            }
        }
        skins.skinned.clear();
        for &(slot, base) in &next {
            let record = &mut self.snapshot.instances[slot];
            if record.skin != Some(base) {
                record.skin = Some(base);
                self.color_changes[slot] = sequence;
                changed = true;
            }
            skins.skinned.push(slot);
        }
        if changed {
            self.color_revision = sequence;
        }
        skins.scratch = next;
        engine_profile::counter("extraction.skinned", skins.skinned.len() as f64);
        engine_profile::counter("extraction.joints", self.snapshot.joints.len() as f64);
    }

    /// Fills this frame's decals, their materials and
    /// textures, and their poses after the skins' joints.
    pub(crate) fn extract_decals(
        &mut self,
        world: &World,
        decals: &mut DecalExtraction,
        gpu: bool,
    ) {
        if decals.extract(world, &mut self.snapshot, gpu) {
            self.joints_revision += 1;
        }
    }

    /// Lists the frame's reflection probes.
    pub(crate) fn extract_probes(&mut self, world: &World, probes: &mut ProbeExtraction) {
        probes.extract(world, &mut self.snapshot);
    }

    /// Records world streaming's state at this frame.
    pub(crate) fn set_cells(&mut self, cells: crate::CellStats) {
        self.snapshot.cells = cells;
    }

    /// Lays the frame's deformation map and lists its stamps.
    pub(crate) fn extract_deformation(
        &mut self,
        world: &mut World,
        deformation: &mut crate::deformation::DeformationExtraction,
    ) {
        deformation.extract(world, &mut self.snapshot.deformation);
    }

    /// Lists the frame's terrains.
    pub(crate) fn extract_terrain(
        &mut self,
        world: &World,
        terrain: &mut crate::terrain::TerrainExtraction,
        aspect: f32,
    ) {
        terrain.extract(world, aspect, &mut self.snapshot);
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
        let changed = self.tint_count != Some(tint_count)
            || changed_in_window(&tints)
            || changed_in_window(&meshes)
            || changed_in_window(&cubes);
        self.tint_count = Some(tint_count);
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

    /// The instance pass: admits and retires slots, resolves their sources
    /// and, on CPU-prepared frames, reads their prepared poses and culls
    /// them. GPU-presented frames list candidates for the GPU instead, and
    /// keep the last pass's records while nothing structural changed.
    pub(crate) fn prepare(
        &mut self,
        world: &World,
        query: &mut RenderQuery,
        aspect: f32,
        gpu: bool,
    ) {
        let _profile = engine_profile::span("cpu", "extraction_retained_updates_culling");
        self.sequence = self
            .sequence
            .checked_add(1)
            .expect("render sequence exhausted");
        if self.gpu && !gpu {
            // GPU frames neither culled nor prepared CPU poses. Recompute both
            // for every slot, and their rows, on the first CPU frame after them.
            self.view = None;
            for frame in &mut self.frames {
                frame.pose_revision = 0;
            }
        }
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
        let material_revision = assets.material_revision();
        // Objects whose materials were replaced in place resolve again.
        let mut replaced = Vec::new();
        if assets_changed {
            self.sources.clear();
            self.asset_revision = Some(revision);
        } else if self.material_revision != Some(material_revision) || material_revision == u64::MAX
        {
            replaced = self.sources.replaced(assets);
        }
        self.material_revision = Some(material_revision);
        let view = world.resource::<Camera>().view_projection(aspect);
        self.snapshot.view_projection = view;
        crate::effects::extract_lighting(world, &mut self.snapshot);
        let total = query.iter(world).len();
        if gpu
            && self.gpu
            && self.gpu_topology == Some(globals.topology_revision)
            && !aged
            && !assets_changed
            && !stamps_changed
            && total == self.snapshot.render_candidates
            && (replaced.is_empty() || self.refresh_replaced(world, query, &replaced))
        {
            // Same renderables, sources, assets and node indices: candidates
            // and slots are unchanged, and the GPU culls. Only the camera and
            // replaced materials' objects move on to this frame.
            self.snapshot.missing_assets = self.missing_sources;
            engine_profile::counter("extraction.gpu_structure_reused", 1.0);
            return;
        }
        engine_profile::counter("extraction.gpu_structure_reused", 0.0);
        let view_changed = !gpu && self.set_view(view);
        self.begin_pass(total, gpu);
        let presented = world.resource::<PresentedTransforms>();
        let mut pass = Pass {
            structure_changed: assets_changed,
            sources_changed: assets_changed,
            ..Pass::default()
        };
        for (row, (entity, has_transform, cube, mesh3d, tint)) in query.iter(world).enumerate() {
            if !has_transform {
                self.snapshot.invalid_transforms += 1;
                continue;
            }
            let Some(index) = self.slot(entity, row, world, globals) else {
                self.snapshot.invalid_transforms += 1;
                continue;
            };
            self.visited[index / 64] |= 1 << (index % 64);
            pass.seen += 1;
            let source = SourceKey::new(
                cube.as_ref().map(|value| value.last_changed()),
                mesh3d.as_ref().map(|value| value.last_changed()),
                tint.as_ref().map(|value| value.last_changed()),
            );
            // Every live slot is (re)resolved when assets change or the window
            // ages, and objects of replaced materials; new owners start from
            // an unknown source key.
            let source_changed = aged
                || assets_changed
                || self.frames[index].source != source
                || mesh3d
                    .as_ref()
                    .is_some_and(|mesh3d| replaced.contains(&mesh3d.material));
            if source_changed {
                self.frames[index].source = source;
                let resolved = self.sources.resolve(
                    assets,
                    self.cube_mesh,
                    cube.as_deref(),
                    mesh3d.as_deref(),
                    tint.as_deref(),
                );
                self.apply_source(index, resolved, &mut pass);
            }
            if gpu {
                self.list_candidate(entity, index, globals, &mut pass);
            } else {
                self.cull(
                    entity,
                    index,
                    (presented, globals),
                    view_changed || source_changed,
                    &mut pass,
                );
            }
        }
        self.finish_pass(pass, assets_changed, gpu, globals.topology_revision);
    }

    /// Re-resolves the objects of `replaced` materials on a GPU-presented
    /// frame that keeps its candidates: their colors, surfaces and draw
    /// keys, and the sources listed. False, for the full pass to run, when
    /// one of them lost its sources.
    fn refresh_replaced(
        &mut self,
        world: &World,
        query: &mut RenderQuery,
        replaced: &[MaterialHandle],
    ) -> bool {
        let assets = world.resource::<AssetLibrary>();
        let mut pass = Pass::default();
        let mut moved = false;
        for (entity, _, cube, mesh3d, tint) in query.iter(world) {
            let Some(mesh3d) = mesh3d.as_deref() else {
                continue;
            };
            if !replaced.contains(&mesh3d.material) {
                continue;
            }
            let Some(&index) = self.entities.get(&entity) else {
                continue;
            };
            let before = self.frames[index];
            let resolved = self.sources.resolve(
                assets,
                self.cube_mesh,
                cube.as_deref(),
                Some(mesh3d),
                tint.as_deref(),
            );
            if resolved.is_none() {
                return false;
            }
            self.apply_source(index, resolved, &mut pass);
            moved |= self.frames[index].material != before.material;
        }
        if moved {
            // Listed candidates use the sources: mark them again.
            self.sources.mesh_used.fill(false);
            self.sources.material_used.fill(false);
            for position in 0..self.visible.len() {
                self.mark_used(self.frames[self.visible[position]]);
            }
        }
        let published = self.sources.publish(&mut self.snapshot, false);
        let sequence = self.sequence;
        if pass.structure_changed || published != Published::Unchanged {
            self.structure_revision = sequence;
        }
        if pass.sources_changed || published == Published::Sources {
            self.source_revision = sequence;
        }
        if pass.colors != 0 {
            self.color_revision = sequence;
        }
        engine_profile::counter("extraction.retained_resolved", pass.resolved as f64);
        true
    }

    /// Whether the camera's view or culling changed since the last
    /// CPU-prepared frame, whose frustum then follows them.
    fn set_view(&mut self, view: Mat4) -> bool {
        let key = (view.to_cols_array().map(f32::to_bits), self.culling);
        if self.view == Some(key) {
            return false;
        }
        self.view = Some(key);
        self.frustum = self.culling.then(|| Frustum::new(view)).flatten();
        true
    }

    /// Sizes the pass's scratch for `total` renderables and clears its
    /// counts.
    fn begin_pass(&mut self, total: usize, gpu: bool) {
        let hint_count = total.min(MAX_TRANSFORM_NODES);
        if self.query_slots.capacity() < hint_count {
            self.query_slots
                .reserve_exact(hint_count.saturating_sub(self.query_slots.len()));
        }
        self.query_slots.resize(hint_count, None);
        // GPU frames list every candidate; the GPU applies the budget.
        let output_bound =
            if gpu { total } else { total.min(self.budget) }.min(MAX_TRANSFORM_NODES);
        if self.visible.capacity() < output_bound {
            self.visible
                .reserve_exact(output_bound.saturating_sub(self.visible.len()));
        }
        self.snapshot.render_candidates = total;
        self.snapshot.visible_instances = 0;
        self.snapshot.culled_instances = 0;
        self.snapshot.invalid_transforms = 0;
        self.snapshot.missing_assets = 0;
        self.snapshot.dropped_instances = 0;
        self.sources.mesh_used.fill(false);
        self.sources.material_used.fill(false);
        // Slots can only be admitted up to the query size during this pass.
        let visited_words = (self.owners.len() + total)
            .min(MAX_TRANSFORM_NODES)
            .div_ceil(64);
        self.visited.clear();
        self.visited.resize(visited_words, 0);
    }

    /// Records slot `index`'s newly resolved source: its mesh, material,
    /// color and surface, or none while its assets are missing.
    #[inline]
    fn apply_source(&mut self, index: usize, resolved: Option<ResolvedSource>, pass: &mut Pass) {
        pass.resolved += 1;
        let frame = &mut self.frames[index];
        frame.mesh = resolved
            .as_ref()
            .map_or(NO_SOURCE, |value| value.mesh_index as u32);
        frame.material = resolved
            .as_ref()
            .and_then(|value| value.material_index)
            .map_or(NO_SOURCE, |material| material as u32);
        let Some(ResolvedSource {
            mesh,
            material,
            color,
            surface,
            ..
        }) = resolved
        else {
            return;
        };
        let record = &mut self.snapshot.instances[index];
        // Opacity decides the pass of materials without maps:
        // switching between opaque and transparent regroups draws. Materials
        // are immutable, so their alpha modes change only with their handles.
        let keys_changed = record.mesh != mesh
            || record.material != material
            || (record.color[3] < 1.0) != (color[3] < 1.0);
        pass.structure_changed |= keys_changed;
        pass.sources_changed |= keys_changed;
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
            self.color_changes[index] = self.sequence;
            pass.colors += 1;
        }
    }

    /// Lists slot `index` as a GPU-presented candidate by its node index;
    /// blending, validity and culling run on the GPU, which hides invalid
    /// admitted nodes.
    #[inline]
    fn list_candidate(
        &mut self,
        entity: Entity,
        index: usize,
        globals: &GlobalTransforms,
        pass: &mut Pass,
    ) {
        let mut hint = TransformIndex(self.pose_indices[index] as usize);
        let Some((global, _)) = globals.get_indexed_with_revision(entity, &mut hint) else {
            self.snapshot.invalid_transforms += 1;
            return;
        };
        self.pose_indices[index] = hint.0 as u32;
        let frame = self.frames[index];
        // Counters partition candidates as on the CPU path; here
        // `visible_instances` counts valid listed candidates before GPU
        // culling.
        if global.matrix().is_none() {
            self.snapshot.invalid_transforms += 1;
        } else if frame.mesh == NO_SOURCE {
            self.snapshot.missing_assets += 1;
        } else {
            self.snapshot.visible_instances += 1;
        }
        if frame.mesh == NO_SOURCE {
            return;
        }
        self.submit(index, pass);
        self.mark_used(frame);
    }

    /// Reads slot `index`'s prepared pose and culls it, classifying it again
    /// when its pose changed or `reclassify` (a new view or source).
    #[inline]
    fn cull(
        &mut self,
        entity: Entity,
        index: usize,
        (presented, globals): (&PresentedTransforms, &GlobalTransforms),
        reclassify: bool,
        pass: &mut Pass,
    ) {
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
            return;
        };
        // Frames read the prepared pose through this validated index
        // instead of copying the matrix here.
        self.pose_indices[index] = hint.0 as u32;
        let frame = &mut self.frames[index];
        let pose_changed =
            frame.pose_revision != pose_revision || pose_revision == 0 || pose_revision == u64::MAX;
        if pose_changed {
            frame.pose_revision = pose_revision;
            self.pose_changes[index] = self.sequence;
            // Written with the stamp while the matrix is at hand, so no
            // later pass re-reads the presented store to repack it.
            self.pose_rows[index] = pose_rows(model);
            pass.poses += 1;
        }
        if frame.mesh == NO_SOURCE {
            self.snapshot.missing_assets += 1;
            return;
        }
        // Culled or visible, the slot casts shadows.
        if let Some(previous) = self.casters.get_mut(pass.casters) {
            pass.casters_changed |= *previous != index;
            *previous = index;
        } else {
            self.casters.push(index);
            pass.casters_changed = true;
        }
        pass.casters += 1;
        if reclassify || pose_changed {
            let bounds = self.sources.bounds[frame.mesh as usize];
            let skinned = self.sources.skinned[frame.mesh as usize];
            frame.outside = !skinned
                && self.frustum.as_ref().is_some_and(|frustum| {
                    pass.culling_tests += 1;
                    let classification = frustum.classify(model, bounds);
                    pass.culling_interior +=
                        usize::from(classification == Classification::Interior);
                    classification.outside()
                });
        } else if self.frustum.is_some() {
            pass.culling_reused += 1;
        }
        let frame = *frame;
        if frame.outside {
            self.snapshot.culled_instances += 1;
        } else {
            self.snapshot.visible_instances += 1;
            if pass.submitted == self.budget {
                self.snapshot.dropped_instances += 1;
                return;
            }
            self.submit(index, pass);
        }
        self.mark_used(frame);
    }

    /// Appends slot `index` to the submission order.
    #[inline]
    fn submit(&mut self, index: usize, pass: &mut Pass) {
        if let Some(previous) = self.visible.get_mut(pass.submitted) {
            pass.structure_changed |= *previous != index;
            *previous = index;
        } else {
            self.visible.push(index);
            pass.structure_changed = true;
        }
        pass.submitted += 1;
    }

    /// Marks the mesh and material of a listed or culled slot as used.
    #[inline]
    fn mark_used(&mut self, frame: SlotFrame) {
        self.sources.mesh_used[frame.mesh as usize] = true;
        if frame.material != NO_SOURCE {
            self.sources.material_used[frame.material as usize] = true;
        }
    }

    /// Retires the slots the pass did not visit, publishes the sources
    /// used, and advances the revisions of what changed.
    fn finish_pass(&mut self, mut pass: Pass, assets_changed: bool, gpu: bool, topology: u64) {
        if pass.submitted != self.visible.len() {
            self.visible.truncate(pass.submitted);
            pass.structure_changed = true;
        }
        if !gpu && pass.casters != self.casters.len() {
            self.casters.truncate(pass.casters);
            pass.casters_changed = true;
        }
        let sequence = self.sequence;
        if pass.casters_changed {
            self.caster_revision = sequence;
        }
        // Each entity owns one slot and visits it at most once, so equal counts
        // prove every owned slot was visited.
        if pass.seen != self.entities.len() {
            self.free
                .reserve_exact(self.owners.len().saturating_sub(self.free.len()));
            for (index, owner) in self.owners.iter_mut().enumerate() {
                if self.visited[index / 64] & (1 << (index % 64)) == 0
                    && let Some(entity) = owner.take()
                {
                    self.entities.remove(&entity);
                    self.frames[index].mesh = NO_SOURCE;
                    self.free.push(index);
                    pass.structure_changed = true;
                    pass.sources_changed = true;
                }
            }
            self.forget_freed_rows();
        }
        let published = self.sources.publish(&mut self.snapshot, assets_changed);
        if pass.structure_changed || published != Published::Unchanged {
            self.structure_revision = sequence;
        }
        if pass.sources_changed || published == Published::Sources {
            self.source_revision = sequence;
        }
        if pass.poses != 0 {
            self.pose_revision = sequence;
        }
        if pass.colors != 0 {
            self.color_revision = sequence;
        }
        self.gpu = gpu;
        self.gpu_topology = gpu.then_some(topology);
        self.missing_sources = self.snapshot.missing_assets;
        engine_profile::counter(
            "extraction.candidates",
            self.snapshot.render_candidates as f64,
        );
        engine_profile::counter("extraction.instances", pass.submitted as f64);
        engine_profile::counter("extraction.culled", self.snapshot.culled_instances as f64);
        engine_profile::counter("extraction.retained_resolved", pass.resolved as f64);
        engine_profile::counter("extraction.retained_pose_updates", pass.poses as f64);
        engine_profile::counter("extraction.retained_color_updates", pass.colors as f64);
        engine_profile::counter("extraction.culling_tests", pass.culling_tests as f64);
        engine_profile::counter("extraction.culling_interior", pass.culling_interior as f64);
        engine_profile::counter("extraction.culling_reused", pass.culling_reused as f64);
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
