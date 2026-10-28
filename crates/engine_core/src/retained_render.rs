use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use bevy_ecs::{
    change_detection::{ContiguousRef, Tick},
    entity::{Entity, EntityHashMap},
    prelude::*,
    system::SystemState,
};
use glam::Mat4;

use crate::collections::HashMap;
use crate::hierarchy::HierarchyScratch;

use crate::{
    AlphaMode, AssetLibrary, Camera, Cube, GlobalTransforms, GpuPresentation, MAX_TRANSFORM_NODES,
    Material, MaterialBinding, MaterialHandle, Mesh3d, MeshHandle, PresentedTransforms,
    RenderInstance, RenderMaterial, RenderMesh, RenderQuery, RenderSnapshot, RenderTexture,
    Surface, TextureHandle, Tint, Transform, TransformIndex,
    probes::ProbeExtraction,
    ticks::{ChangeWindow, ScanClock, consume_through},
};

type SourceStamps = SystemState<(
    Query<'static, 'static, Ref<'static, Mesh3d>>,
    Query<'static, 'static, Ref<'static, Cube>>,
    Query<'static, 'static, Ref<'static, Tint>>,
)>;

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

fn draw_class(material: &Material) -> (bool, bool, bool) {
    (
        matches!(material.alpha_mode(), AlphaMode::Mask { .. }),
        material.is_double_sided(),
        material.is_foliage(),
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Published {
    Unchanged,
    Contents,
    Sources,
}

pub type PoseRows = [[f32; 4]; 3];

pub(crate) const IDENTITY_ROWS: PoseRows = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
];

#[inline]
pub(crate) fn pose_rows(model: Mat4) -> PoseRows {
    let rows = model.transpose();
    [
        rows.x_axis.to_array(),
        rows.y_axis.to_array(),
        rows.z_axis.to_array(),
    ]
}

#[derive(Clone, Copy)]
pub struct RenderFrame<'a> {
    owner: &'a RetainedRender,
    poses: &'a PresentedTransforms,
    hierarchy: &'a HierarchyScratch,
}

impl<'a> RenderFrame<'a> {
    pub fn snapshot(self) -> &'a RenderSnapshot {
        &self.owner.snapshot
    }
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
    pub fn source_revision(self) -> u64 {
        self.owner.source_revision
    }
    pub fn pose_revision(self) -> u64 {
        self.owner.pose_revision
    }
    pub fn color_revision(self) -> u64 {
        self.owner.color_revision
    }
    pub fn lights_revision(self) -> u64 {
        self.owner.lights_revision
    }
    pub fn emitters_revision(self) -> u64 {
        self.owner.emitters_revision
    }
    pub fn visible_indices(self) -> &'a [usize] {
        &self.owner.visible
    }
    pub fn casters(self) -> &'a [usize] {
        if self.owner.gpu {
            &self.owner.visible
        } else {
            &self.owner.casters
        }
    }
    pub fn caster_revision(self) -> u64 {
        if self.owner.gpu {
            self.owner.structure_revision
        } else {
            self.owner.caster_revision
        }
    }
    pub fn pose_changes(self) -> &'a [u64] {
        &self.owner.pose_changes
    }
    pub fn pose_rows(self) -> &'a [PoseRows] {
        &self.owner.pose_rows
    }
    pub fn gpu(self) -> Option<GpuPresentation<'a>> {
        self.owner
            .gpu
            .then(|| self.poses.gpu_presentation(self.hierarchy))
    }
    pub fn slot_nodes(self) -> &'a [u32] {
        &self.owner.pose_indices
    }
    pub fn culling(self) -> bool {
        self.owner.culling
    }
    pub fn instance_budget(self) -> usize {
        self.owner.budget
    }
    pub fn color_changes(self) -> &'a [u64] {
        &self.owner.color_changes
    }

    pub(crate) fn copy_to(self, snapshot: &mut RenderSnapshot) {
        let RenderSnapshot {
            view_projection,
            camera_position,
            detail_eye,
            surface_under_eye,
            elapsed_seconds,
            presented_seconds,
            environment,
            lights,
            particles,
            reflection_probes,
            terrain,
            tables,
            instances,
            meshes,
            materials,
            textures,
            render_candidates,
            visible_instances,
            dropped_instances,
            missing_assets,
            invalid_transforms,
        } = self.snapshot();
        snapshot.view_projection = *view_projection;
        snapshot.camera_position = *camera_position;
        snapshot.detail_eye = *detail_eye;
        snapshot.surface_under_eye = *surface_under_eye;
        snapshot.elapsed_seconds = *elapsed_seconds;
        snapshot.presented_seconds = *presented_seconds;
        snapshot.environment = *environment;
        snapshot.lights.list.clear();
        snapshot.lights.dropped = lights.dropped;
        snapshot.lights.invalid = lights.invalid;
        for light in &lights.list {
            let Some(slot) = light.slot else {
                snapshot.lights.list.push(*light);
                continue;
            };
            let model = self
                .owner
                .pose_indices
                .get(slot as usize)
                .and_then(|&index| self.poses.prepared_matrix(index as usize));
            let Some(model) = model else {
                snapshot.lights.dropped += 1;
                continue;
            };
            snapshot.lights.list.push(light.posed(model));
        }
        snapshot.particles.clone_from(particles);
        snapshot.reflection_probes.clone_from(reflection_probes);
        snapshot.terrain.clone_from(terrain);
        snapshot.tables.clone_from(tables);
        snapshot.instances.clear();
        snapshot
            .instances
            .extend(self.visible_indices().iter().map(|&slot| RenderInstance {
                model: self.model(slot),
                ..instances[slot]
            }));
        snapshot.meshes.clone_from(meshes);
        snapshot.materials.clone_from(materials);
        snapshot.textures.clone_from(textures);
        snapshot.render_candidates = *render_candidates;
        snapshot.visible_instances = *visible_instances;
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

#[derive(Clone, Copy)]
struct SlotFrame {
    pose_revision: u64,
    mesh: u32,
    material: u32,
    source: SourceKey,
}

impl SlotFrame {
    const NEW: Self = Self {
        pose_revision: 0,
        mesh: NO_SOURCE,
        material: NO_SOURCE,
        source: SourceKey::UNKNOWN,
    };
}

#[derive(Default)]
struct Sources {
    meshes: Vec<RenderMesh>,
    materials: Vec<RenderMaterial>,
    textures: Vec<RenderTexture>,
    mesh_indices: HashMap<MeshHandle, usize>,
    material_indices: HashMap<MaterialHandle, usize>,
    binding_indices: HashMap<MaterialBinding, usize>,
    texture_indices: HashMap<TextureHandle, usize>,
    versions: HashMap<MaterialHandle, u64>,
    mesh_used: Vec<bool>,
    material_used: Vec<bool>,
    previous_mesh_used: Vec<bool>,
    previous_material_used: Vec<bool>,
    contents_changed: bool,
    texture_listed: Vec<bool>,
}

impl Sources {
    fn clear(&mut self) {
        self.meshes.clear();
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
            (cube_mesh, [r, g, b, 1.0], None, Surface::MATTE)
        };
        let color = tint.map_or(color, |tint| tint.apply(color));
        let rendered = RenderMesh::of(assets, handle)?;
        if let Some((_, material)) = mapped {
            for texture in material.textures() {
                assets.texture(texture)?;
            }
        }
        let mesh_index = *self.mesh_indices.entry(handle).or_insert_with(|| {
            let index = self.meshes.len();
            self.meshes.push(rendered);
            self.mesh_used.push(false);
            index
        });
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
    emitters_revision: u64,
    snapshot: RenderSnapshot,
    cube_mesh: MeshHandle,
    owners: Vec<Option<Entity>>,
    frames: Vec<SlotFrame>,
    visited: Vec<u64>,
    pub(crate) entities: EntityHashMap<usize>,
    free: Vec<usize>,
    query_slots: Vec<Option<(Entity, usize)>>,
    visible: Vec<usize>,
    casters: Vec<usize>,
    pose_indices: Vec<u32>,
    pose_changes: Vec<u64>,
    pose_rows: Vec<PoseRows>,
    color_changes: Vec<u64>,
    asset_revision: Option<(u64, u64)>,
    material_revision: Option<u64>,
    sources: Sources,
    scans: ScanClock,
    culling: bool,
    budget: usize,
    missing_sources: usize,
    gpu: bool,
    gpu_topology: Option<u64>,
    source_stamps: Option<SourceStamps>,
    tint_count: Option<usize>,
}

#[derive(Default)]
struct Pass {
    structure_changed: bool,
    sources_changed: bool,
    casters_changed: bool,
    submitted: usize,
    casters: usize,
    seen: usize,
    resolved: usize,
    poses: usize,
    colors: usize,
}

impl RetainedRender {
    pub(crate) fn slot_of(&self, entity: Entity) -> Option<u32> {
        self.entities.get(&entity).map(|&slot| slot as u32)
    }

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
            scans: ScanClock::default(),
            culling,
            budget,
            missing_sources: 0,
            gpu: false,
            gpu_topology: None,
            source_stamps: None,
            tint_count: None,
        }
    }

    pub(crate) fn set_culling(&mut self, enabled: bool) {
        self.culling = enabled;
    }

    pub(crate) fn culling(&self) -> bool {
        self.culling
    }

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
            debug_assert_eq!(self.owners[*index], Some(entity), "an entity owns its slot");
            return Some(*index);
        }
        let index = if let Some(&index) = self.entities.get(&entity) {
            index
        } else {
            globals.get(entity)?;
            if self.free.is_empty() && self.owners.len() == MAX_TRANSFORM_NODES {
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
                self.snapshot.instances.push(RenderInstance {
                    mesh: self.cube_mesh,
                    material: None,
                    model: Mat4::IDENTITY,
                    color: [1.0; 4],
                    surface: Surface::MATTE,
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

    pub(crate) fn extract_probes(&mut self, world: &World, probes: &mut ProbeExtraction) {
        probes.extract(world, &mut self.snapshot);
    }

    pub(crate) fn set_tables(&mut self, tables: crate::RenderTables) {
        self.snapshot.tables = tables;
    }

    pub(crate) fn extract_terrain(&mut self, world: &World, aspect: f32) {
        crate::extract_terrains(world, aspect, &mut self.snapshot);
    }

    pub(crate) fn ensure_source_scan(&mut self, world: &mut World) {
        if self.source_stamps.is_none() {
            self.source_stamps = Some(SystemState::new(world));
        }
    }

    fn sources_changed(&mut self, world: &World) -> bool {
        let Some(stamps) = self.source_stamps.as_mut() else {
            return true;
        };
        let Ok((meshes, cubes, tints)) = stamps.get(world) else {
            return true;
        };
        let tint_count = tints.iter().len();
        let changed = self.tint_count != Some(tint_count)
            || changed_in_window(&tints)
            || changed_in_window(&meshes)
            || changed_in_window(&cubes);
        self.tint_count = Some(tint_count);
        consume_through(stamps, world.read_change_tick());
        changed
    }

    fn forget_freed_rows(&mut self) {
        for cached in &mut self.query_slots {
            if let Some((entity, index)) = *cached
                && self.owners.get(index) != Some(&Some(entity))
            {
                *cached = None;
            }
        }
    }

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
            for frame in &mut self.frames {
                frame.pose_revision = 0;
            }
        }
        let globals = world.resource::<GlobalTransforms>();
        let stamps_changed = gpu && self.sources_changed(world);
        let scan_tick = world.read_change_tick();
        let aged = self.scans.aged(scan_tick);
        self.scans.scanned(scan_tick);
        let assets = world.resource::<AssetLibrary>();
        let revision = assets.revision();
        let assets_changed = self.asset_revision != Some(revision) || revision.1 == u64::MAX;
        let material_revision = assets.material_revision();
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
            && self.gpu_topology == Some(globals.topology_revision())
            && !aged
            && !assets_changed
            && !stamps_changed
            && total == self.snapshot.render_candidates
            && (replaced.is_empty() || self.refresh_replaced(world, query, &replaced))
        {
            self.snapshot.missing_assets = self.missing_sources;
            engine_profile::counter("extraction.gpu_structure_reused", 1.0);
            return;
        }
        engine_profile::counter("extraction.gpu_structure_reused", 0.0);
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
                self.list_presented(entity, index, (presented, globals), &mut pass);
            }
        }
        self.finish_pass(pass, assets_changed, gpu, globals.topology_revision());
    }

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

    fn begin_pass(&mut self, total: usize, gpu: bool) {
        let hint_count = total.min(MAX_TRANSFORM_NODES);
        if self.query_slots.capacity() < hint_count {
            self.query_slots
                .reserve_exact(hint_count.saturating_sub(self.query_slots.len()));
        }
        self.query_slots.resize(hint_count, None);
        let output_bound =
            if gpu { total } else { total.min(self.budget) }.min(MAX_TRANSFORM_NODES);
        if self.visible.capacity() < output_bound {
            self.visible
                .reserve_exact(output_bound.saturating_sub(self.visible.len()));
        }
        self.snapshot.render_candidates = total;
        self.snapshot.visible_instances = 0;
        self.snapshot.invalid_transforms = 0;
        self.snapshot.missing_assets = 0;
        self.snapshot.dropped_instances = 0;
        self.sources.mesh_used.fill(false);
        self.sources.material_used.fill(false);
        let visited_words = (self.owners.len() + total)
            .min(MAX_TRANSFORM_NODES)
            .div_ceil(64);
        self.visited.clear();
        self.visited.resize(visited_words, 0);
    }

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

    #[inline]
    fn list_presented(
        &mut self,
        entity: Entity,
        index: usize,
        (presented, globals): (&PresentedTransforms, &GlobalTransforms),
        pass: &mut Pass,
    ) {
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
        self.pose_indices[index] = hint.0 as u32;
        let frame = &mut self.frames[index];
        let pose_changed =
            frame.pose_revision != pose_revision || pose_revision == 0 || pose_revision == u64::MAX;
        if pose_changed {
            frame.pose_revision = pose_revision;
            self.pose_changes[index] = self.sequence;
            self.pose_rows[index] = pose_rows(model);
            pass.poses += 1;
        }
        if frame.mesh == NO_SOURCE {
            self.snapshot.missing_assets += 1;
            return;
        }
        if let Some(previous) = self.casters.get_mut(pass.casters) {
            pass.casters_changed |= *previous != index;
            *previous = index;
        } else {
            self.casters.push(index);
            pass.casters_changed = true;
        }
        pass.casters += 1;
        let frame = *frame;
        self.snapshot.visible_instances += 1;
        if pass.submitted == self.budget {
            self.snapshot.dropped_instances += 1;
            return;
        }
        self.submit(index, pass);
        self.mark_used(frame);
    }

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

    #[inline]
    fn mark_used(&mut self, frame: SlotFrame) {
        self.sources.mesh_used[frame.mesh as usize] = true;
        if frame.material != NO_SOURCE {
            self.sources.material_used[frame.material as usize] = true;
        }
    }

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
        engine_profile::counter("extraction.retained_resolved", pass.resolved as f64);
        engine_profile::counter("extraction.retained_pose_updates", pass.poses as f64);
        engine_profile::counter("extraction.retained_color_updates", pass.colors as f64);
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
