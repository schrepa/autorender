use std::sync::Arc;

use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemChangeTick;
use engine_core::collections::HashMap;
use engine_core::{
    AlphaMode, AssetLibrary, GlobalTransforms, MAX_TRANSFORM_NODES, Material, MaterialBinding,
    MaterialHandle, Mesh, Mesh3d, MeshHandle, PresentationState, PresentedTransforms,
    ScanClock, Stamp, Surface, TextureHandle, Transform, TransformIndex,
};
use engine_places::meshes::{Cube, Instances, Tint};
use engine_places::splines::SplineMesh;
use glam::Mat4;

use crate::settings::Culling;

pub(crate) type PoseRows = [[f32; 4]; 3];

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

fn draw_class(material: &Material) -> (bool, bool, bool) {
    (
        matches!(material.alpha_mode(), AlphaMode::Mask { .. }),
        material.is_double_sided(),
        material.is_foliage(),
    )
}

type RenderQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        Has<Transform>,
        Option<Ref<'static, Cube>>,
        Option<Ref<'static, Mesh3d>>,
        Option<Ref<'static, Tint>>,
        Has<SplineMesh>,
    ),
    Or<(With<Cube>, With<Mesh3d>)>,
>;

type UnbuiltSplines<'w, 's> = Query<'w, 's, &'static Mesh3d, (With<SplineMesh>, Without<Cube>)>;

fn unbuilt(mesh3d: &Mesh3d) -> bool {
    mesh3d.mesh == MeshHandle::default()
}

fn skipped(cube: bool, mesh3d: Option<&Mesh3d>, spline: bool) -> bool {
    spline && !cube && mesh3d.is_some_and(unbuilt)
}

type SourceChanges<'w, 's> =
    Query<'w, 's, (), Or<(Changed<Mesh3d>, Changed<Cube>, Changed<Tint>)>>;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Published {
    Unchanged,
    Contents,
    Sources,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Instance {
    pub mesh: MeshHandle,
    pub material: Option<MaterialHandle>,
    pub color: [f32; 4],
    pub surface: Surface,
}

#[derive(Clone, Debug)]
pub(crate) struct SourceMesh {
    pub handle: MeshHandle,
    pub mesh: Arc<Mesh>,
}

impl SourceMesh {
    fn of(assets: &AssetLibrary, handle: MeshHandle) -> Option<Self> {
        let mesh = assets.mesh(handle)?;
        assets.mesh_lods(handle)?;
        Some(Self {
            handle,
            mesh: Arc::clone(mesh),
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SourceMaterial {
    pub handle: MaterialHandle,
    pub material: Material,
}

#[derive(Clone, Debug)]
pub(crate) struct SourceTexture {
    pub handle: TextureHandle,
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
    fn new(cube: Option<u32>, mesh: Option<u32>, tint: Option<u32>) -> Self {
        let source = mesh.or(cube);
        Self {
            ticks: [source.unwrap_or(0), tint.unwrap_or(0)],
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
pub(crate) struct Snapshot {
    pub view_projection: Mat4,
    pub instances: Vec<Instance>,
    pub entities: HashMap<Entity, usize>,
    pub sourced: Vec<bool>,
    pub source_revision: u64,
    pub meshes: Vec<SourceMesh>,
    pub materials: Vec<SourceMaterial>,
    pub textures: Vec<SourceTexture>,
    pub render_candidates: usize,
    pub visible_instances: usize,
    pub dropped_instances: usize,
    pub missing_assets: usize,
    pub invalid_transforms: usize,
}

#[derive(Default)]
struct Sources {
    meshes: Vec<SourceMesh>,
    materials: Vec<SourceMaterial>,
    textures: Vec<SourceTexture>,
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
        let rendered = SourceMesh::of(assets, handle)?;
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
        let material_index = match mapped {
            Some((material_handle, material)) => {
                Some(self.material_index(assets, material_handle, material)?)
            }
            None => None,
        };
        Some(ResolvedSource {
            mesh_index,
            material_index,
            mesh: handle,
            material: material_index.map(|index| self.materials[index].handle),
            color,
            surface,
        })
    }

    fn material_index(
        &mut self,
        assets: &AssetLibrary,
        handle: MaterialHandle,
        material: &Material,
    ) -> Option<usize> {
        if let Some(&index) = self.material_indices.get(&handle) {
            return Some(index);
        }
        let binding = material.binding()?;
        let index = match self.binding_indices.get(&binding) {
            Some(&index) => index,
            None => {
                for texture in material.textures() {
                    if self.texture_indices.contains_key(&texture) {
                        continue;
                    }
                    assets.texture(texture)?;
                    self.textures.push(SourceTexture { handle: texture });
                    self.texture_indices.insert(texture, self.textures.len() - 1);
                }
                self.materials.push(SourceMaterial {
                    handle,
                    material: *material,
                });
                self.material_used.push(false);
                let index = self.materials.len() - 1;
                self.binding_indices.insert(binding, index);
                index
            }
        };
        self.material_indices.insert(handle, index);
        Some(index)
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
        let Some(binding) = material.binding() else {
            return false;
        };
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
        self.binding_indices.entry(binding).or_insert(index);
        for texture in material.textures() {
            if self.texture_indices.contains_key(&texture) {
                continue;
            }
            if assets.texture(texture).is_none() {
                return false;
            }
            self.textures.push(SourceTexture { handle: texture });
            self.texture_indices.insert(texture, self.textures.len() - 1);
        }
        self.materials[index] = SourceMaterial {
            handle,
            material: *material,
        };
        self.contents_changed = true;
        true
    }

    fn publish(&mut self, snapshot: &mut Snapshot, force: bool) -> Published {
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
                let Some(&index) = self.texture_indices.get(&texture) else {
                    continue;
                };
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

#[derive(Resource)]
pub(crate) struct RetainedRender {
    identity: Stamp,
    sequence: u64,
    structure_revision: u64,
    pose_revision: u64,
    color_revision: u64,
    caster_revision: u64,
    snapshot: Snapshot,
    owners: Vec<Option<Entity>>,
    frames: Vec<SlotFrame>,
    visited: Vec<u64>,
    free: Vec<usize>,
    query_slots: Vec<Option<(Entity, usize)>>,
    visible: Vec<usize>,
    casters: Vec<usize>,
    hints: Vec<TransformIndex>,
    pose_indices: Vec<u32>,
    pose_changes: Vec<u64>,
    pose_rows: Vec<PoseRows>,
    color_changes: Vec<u64>,
    asset_revision: Option<(u64, u64)>,
    material_revision: Option<u64>,
    sources: Sources,
    scans: ScanClock,
    missing_sources: usize,
    gpu: bool,
    gpu_topology: Option<u64>,
    tint_count: Option<usize>,
}

impl Default for RetainedRender {
    fn default() -> Self {
        Self {
            identity: Stamp::next(),
            sequence: 0,
            structure_revision: 0,
            pose_revision: 0,
            color_revision: 0,
            caster_revision: 0,
            snapshot: Snapshot::default(),
            owners: Vec::new(),
            frames: Vec::new(),
            visited: Vec::new(),
            free: Vec::new(),
            query_slots: Vec::new(),
            visible: Vec::new(),
            casters: Vec::new(),
            hints: Vec::new(),
            pose_indices: Vec::new(),
            pose_changes: Vec::new(),
            pose_rows: Vec::new(),
            color_changes: Vec::new(),
            asset_revision: None,
            material_revision: None,
            sources: Sources::default(),
            scans: ScanClock::default(),
            missing_sources: 0,
            gpu: false,
            gpu_topology: None,
            tint_count: None,
        }
    }
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

struct Inputs<'a, 'w, 's> {
    query: &'a RenderQuery<'w, 's>,
    globals: &'a GlobalTransforms,
    presented: &'a PresentedTransforms,
    assets: &'a AssetLibrary,
    cube_mesh: MeshHandle,
    budget: usize,
    unbuilt: usize,
}

#[expect(
    clippy::too_many_arguments,
    reason = "the retained pass reads the scene's drawable entities and the presentation"
)]
pub(crate) fn retain_instances(
    (query, splines): (RenderQuery, UnbuiltSplines),
    changes: SourceChanges,
    (tints, ticks): (Query<(), With<Tint>>, SystemChangeTick),
    globals: Res<GlobalTransforms>,
    presented: Res<PresentedTransforms>,
    presentation: Res<PresentationState>,
    assets: Res<AssetLibrary>,
    view: Res<engine_core::View>,
    culling: Res<Culling>,
    mut retained: ResMut<RetainedRender>,
    mut instances: ResMut<Instances>,
    mut published: Local<Option<(Stamp, u64, u64)>>,
) {
    let gpu = presentation.gpu_pending();
    let tint_count = tints.iter().len();
    let stamps_changed = gpu && (retained.tint_count != Some(tint_count) || !changes.is_empty());
    if gpu {
        retained.tint_count = Some(tint_count);
    }
    let inputs = Inputs {
        query: &query,
        globals: &globals,
        presented: &presented,
        assets: &assets,
        cube_mesh: assets.cube_mesh(),
        budget: culling.budget(),
        unbuilt: splines.iter().filter(|mesh3d| unbuilt(mesh3d)).count(),
    };
    retained.prepare(
        &inputs,
        view.view_projection(),
        (gpu, stamps_changed),
        ticks,
    );
    let key = (
        retained.identity,
        retained.snapshot.source_revision,
        retained.structure_revision,
    );
    if *published == Some(key) {
        return;
    }
    *published = Some(key);
    let drawn = retained.drawn_slots();
    if instances.slots != drawn {
        instances.slots = drawn;
    }
}

impl RetainedRender {
    pub(crate) fn snapshot(&self) -> &Snapshot {
        &self.snapshot
    }

    pub(crate) fn identity(&self) -> Stamp {
        self.identity
    }

    pub(crate) fn structure_revision(&self) -> u64 {
        self.structure_revision
    }

    pub(crate) fn source_revision(&self) -> u64 {
        self.snapshot.source_revision
    }

    pub(crate) fn pose_revision(&self) -> u64 {
        self.pose_revision
    }

    pub(crate) fn color_revision(&self) -> u64 {
        self.color_revision
    }

    pub(crate) fn visible_indices(&self) -> &[usize] {
        &self.visible
    }

    pub(crate) fn casters(&self) -> &[usize] {
        if self.gpu {
            &self.visible
        } else {
            &self.casters
        }
    }

    pub(crate) fn caster_revision(&self) -> u64 {
        if self.gpu {
            self.structure_revision
        } else {
            self.caster_revision
        }
    }

    pub(crate) fn pose_changes(&self) -> &[u64] {
        &self.pose_changes
    }

    pub(crate) fn pose_rows(&self) -> &[PoseRows] {
        &self.pose_rows
    }

    pub(crate) fn slot_nodes(&self) -> &[u32] {
        &self.pose_indices
    }

    pub(crate) fn color_changes(&self) -> &[u64] {
        &self.color_changes
    }

    pub(crate) fn gpu(&self) -> bool {
        self.gpu
    }

    fn drawn_slots(&self) -> Vec<(Entity, u32)> {
        let mut drawn: Vec<(Entity, u32)> = self
            .snapshot
            .entities
            .iter()
            .filter(|&(_, &index)| self.snapshot.sourced[index])
            .map(|(&entity, &index)| (entity, index as u32))
            .collect();
        drawn.sort_unstable();
        drawn
    }

    fn recycle(&mut self, inputs: &Inputs<'_, '_, '_>) {
        self.free
            .reserve_exact(self.owners.len().saturating_sub(self.free.len()));
        for (index, owner) in self.owners.iter_mut().enumerate() {
            if let Some(old) = *owner
                && (!matches!(inputs.query.get(old), Ok((_, true, cube, mesh3d, _, spline))
                    if !skipped(cube.is_some(), mesh3d.as_deref(), spline))
                    || inputs.globals.get(old).is_none())
            {
                self.snapshot.entities.remove(&old);
                *owner = None;
                self.frames[index].mesh = NO_SOURCE;
                self.snapshot.sourced[index] = false;
                self.free.push(index);
            }
        }
        self.forget_freed_rows();
    }

    fn slot(&mut self, entity: Entity, row: usize, inputs: &Inputs<'_, '_, '_>) -> Option<usize> {
        if let Some(Some((cached_entity, index))) = self.query_slots.get(row)
            && *cached_entity == entity
        {
            return Some(*index);
        }
        let index = if let Some(&index) = self.snapshot.entities.get(&entity) {
            index
        } else {
            inputs.globals.get(entity)?;
            if self.free.is_empty() && self.owners.len() == MAX_TRANSFORM_NODES {
                self.recycle(inputs);
            }
            let index = if let Some(index) = self.free.pop() {
                self.owners[index] = Some(entity);
                self.frames[index] = SlotFrame::NEW;
                self.snapshot.sourced[index] = false;
                self.hints[index] = TransformIndex::default();
                self.pose_indices[index] = u32::MAX;
                self.pose_changes[index] = 0;
                self.color_changes[index] = 0;
                index
            } else {
                if self.owners.len() == MAX_TRANSFORM_NODES {
                    return None;
                }
                self.push_slot(entity, inputs.cube_mesh)
            };
            self.snapshot.entities.insert(entity, index);
            self.structure_revision = self.sequence;
            self.snapshot.source_revision = self.sequence;
            index
        };
        if let Some(cached) = self.query_slots.get_mut(row) {
            *cached = Some((entity, index));
        }
        Some(index)
    }

    fn push_slot(&mut self, entity: Entity, cube_mesh: MeshHandle) -> usize {
        let index = self.owners.len();
        if index == self.owners.capacity() {
            let capacity = (index.max(16) * 2).min(MAX_TRANSFORM_NODES);
            self.owners.reserve_exact(capacity - index);
            self.frames.reserve_exact(capacity - index);
            self.snapshot.sourced.reserve_exact(capacity - index);
            self.snapshot.instances.reserve_exact(capacity - index);
            self.hints.reserve_exact(capacity - index);
            self.pose_indices.reserve_exact(capacity - index);
            self.pose_changes.reserve_exact(capacity - index);
            self.pose_rows.reserve_exact(capacity - index);
            self.color_changes.reserve_exact(capacity - index);
        }
        self.owners.push(Some(entity));
        self.frames.push(SlotFrame::NEW);
        self.snapshot.sourced.push(false);
        self.snapshot.instances.push(Instance {
            mesh: cube_mesh,
            material: None,
            color: [1.0; 4],
            surface: Surface::MATTE,
        });
        self.hints.push(TransformIndex::default());
        self.pose_indices.push(u32::MAX);
        self.pose_changes.push(0);
        self.pose_rows.push(IDENTITY_ROWS);
        self.color_changes.push(0);
        index
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

    fn prepare(
        &mut self,
        inputs: &Inputs<'_, '_, '_>,
        view_projection: Mat4,
        (gpu, stamps_changed): (bool, bool),
        ticks: SystemChangeTick,
    ) {
        let _profile = engine_core::profile::span("cpu", "extraction_retained_updates_culling");
        self.sequence = self.sequence.saturating_add(1);
        if self.gpu && !gpu {
            for frame in &mut self.frames {
                frame.pose_revision = 0;
            }
        }
        let scan_tick = ticks.this_run();
        let aged = self.scans.aged(scan_tick);
        self.scans.scanned(scan_tick);
        let assets = inputs.assets;
        let revision = assets.revision();
        let assets_changed = self.asset_revision != Some(revision) || revision.1 == u64::MAX;
        let material_revision = assets.material_revision();
        let mut replaced = Vec::new();
        if assets_changed {
            self.sources.clear();
            self.asset_revision = Some(revision);
        } else if self.material_revision != Some(material_revision)
            || material_revision == u64::MAX
        {
            replaced = self.sources.replaced(assets);
        }
        self.material_revision = Some(material_revision);
        self.snapshot.view_projection = view_projection;
        let total = inputs.query.iter().len() - inputs.unbuilt;
        if gpu
            && self.gpu
            && self.gpu_topology == Some(inputs.globals.topology_revision())
            && !aged
            && !assets_changed
            && !stamps_changed
            && total == self.snapshot.render_candidates
            && (replaced.is_empty() || self.refresh_replaced(inputs, &replaced))
        {
            self.snapshot.missing_assets = self.missing_sources;
            engine_core::profile::counter("extraction.gpu_structure_reused", 1.0);
            return;
        }
        engine_core::profile::counter("extraction.gpu_structure_reused", 0.0);
        self.begin_pass(total, gpu, inputs.budget);
        let mut pass = Pass {
            structure_changed: assets_changed,
            sources_changed: assets_changed,
            ..Pass::default()
        };
        let rows = inputs
            .query
            .iter()
            .filter(|(_, _, cube, mesh3d, _, spline)| {
                !skipped(cube.is_some(), mesh3d.as_deref(), *spline)
            });
        for (row, (entity, has_transform, cube, mesh3d, tint, _)) in rows.enumerate() {
            if !has_transform {
                self.snapshot.invalid_transforms += 1;
                continue;
            }
            let Some(index) = self.slot(entity, row, inputs) else {
                self.snapshot.invalid_transforms += 1;
                continue;
            };
            self.visited[index / 64] |= 1 << (index % 64);
            pass.seen += 1;
            let source = SourceKey::new(
                cube.as_ref().map(|value| value.last_changed().get()),
                mesh3d.as_ref().map(|value| value.last_changed().get()),
                tint.as_ref().map(|value| value.last_changed().get()),
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
                    inputs.cube_mesh,
                    cube.as_deref(),
                    mesh3d.as_deref(),
                    tint.as_deref(),
                );
                self.apply_source(index, resolved, &mut pass);
            }
            if gpu {
                self.list_candidate(entity, index, inputs.globals, &mut pass);
            } else {
                self.list_presented(entity, index, inputs, &mut pass);
            }
        }
        self.finish_pass(pass, assets_changed, gpu, inputs.globals.topology_revision());
    }

    fn refresh_replaced(&mut self, inputs: &Inputs<'_, '_, '_>, replaced: &[MaterialHandle]) -> bool {
        let mut pass = Pass::default();
        let mut moved = false;
        for (entity, _, cube, mesh3d, tint, spline) in inputs.query.iter() {
            let Some(mesh3d) = mesh3d.as_deref() else {
                continue;
            };
            if skipped(cube.is_some(), Some(mesh3d), spline) {
                continue;
            }
            if !replaced.contains(&mesh3d.material) {
                continue;
            }
            let Some(&index) = self.snapshot.entities.get(&entity) else {
                continue;
            };
            let before = self.frames[index];
            let resolved = self.sources.resolve(
                inputs.assets,
                inputs.cube_mesh,
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
            self.snapshot.source_revision = sequence;
        }
        if pass.colors != 0 {
            self.color_revision = sequence;
        }
        engine_core::profile::counter("extraction.retained_resolved", pass.resolved as f64);
        true
    }

    fn begin_pass(&mut self, total: usize, gpu: bool, budget: usize) {
        let hint_count = total.min(MAX_TRANSFORM_NODES);
        if self.query_slots.capacity() < hint_count {
            self.query_slots
                .reserve_exact(hint_count.saturating_sub(self.query_slots.len()));
        }
        self.query_slots.resize(hint_count, None);
        let output_bound = if gpu { total } else { total.min(budget) }.min(MAX_TRANSFORM_NODES);
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
        self.snapshot.sourced[index] = resolved.is_some();
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
        let Some((global, _)) = globals.get_indexed_with_revision(entity, &mut self.hints[index])
        else {
            self.snapshot.invalid_transforms += 1;
            return;
        };
        self.pose_indices[index] = self.hints[index].index() as u32;
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
        inputs: &Inputs<'_, '_, '_>,
        pass: &mut Pass,
    ) {
        let pose = inputs.presented.get_indexed(entity, &mut self.hints[index]);
        let Some((model, pose_revision)) =
            pose.and_then(|pose| Some((pose.matrix()?, pose.revision())))
        else {
            self.snapshot.invalid_transforms += 1;
            return;
        };
        self.pose_indices[index] = self.hints[index].index() as u32;
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
        if pass.submitted == inputs.budget {
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
        if pass.seen != self.snapshot.entities.len() {
            self.free
                .reserve_exact(self.owners.len().saturating_sub(self.free.len()));
            for (index, owner) in self.owners.iter_mut().enumerate() {
                if self.visited[index / 64] & (1 << (index % 64)) == 0
                    && let Some(entity) = owner.take()
                {
                    self.snapshot.entities.remove(&entity);
                    self.frames[index].mesh = NO_SOURCE;
                    self.snapshot.sourced[index] = false;
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
            self.snapshot.source_revision = sequence;
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
    }
}
