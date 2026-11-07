use std::collections::BTreeSet;

use bevy_ecs::prelude::*;
use engine_core::collections::HashMap;
use engine_core::{
    AssetLibrary, Feed, FeedSettings, GlobalTransforms, Material, MaterialBinding, MaterialHandle,
    Mesh3d, MeshHandle, Revision, Surface, Transform,
};
use engine_places::meshes::{Cube, FirstPersonLayer, Instances, Tint};
use glam::Mat4;

type RenderQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        Has<Transform>,
        Option<&'static Cube>,
        Option<&'static Mesh3d>,
        Option<&'static Tint>,
        Option<&'static FirstPersonLayer>,
    ),
    Or<(With<Cube>, With<Mesh3d>)>,
>;

type SourceChanges<'w, 's> = Query<
    'w,
    's,
    Entity,
    Or<(Changed<Mesh3d>, Changed<Cube>, Changed<Tint>, Changed<FirstPersonLayer>)>,
>;

fn unbuilt(cube: bool, mesh3d: Option<&Mesh3d>) -> bool {
    !cube && mesh3d.is_some_and(|mesh3d| mesh3d.mesh == MeshHandle::default())
}

fn textures_live(assets: &AssetLibrary, material: &Material) -> bool {
    material
        .textures()
        .all(|texture| assets.texture(texture).is_some())
}

fn surface_bits(surface: Surface) -> [u32; 4] {
    [
        surface.roughness,
        surface.metalness,
        surface.emission,
        surface.ripples,
    ]
    .map(f32::to_bits)
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Instance {
    pub mesh: MeshHandle,
    pub source: Option<MaterialHandle>,
    pub material: Option<MaterialHandle>,
    pub color: [f32; 4],
    pub surface: Surface,
    pub flags: u32,
}

impl Instance {
    pub(crate) fn first_person(&self) -> bool {
        self.flags != 0
    }
}

#[derive(Clone)]
struct HeldMaterial {
    handle: MaterialHandle,
    material: Material,
    canonical: Option<MaterialHandle>,
    stale: bool,
}

#[derive(Default)]
struct Materials {
    held: Vec<Option<HeldMaterial>>,
    bindings: HashMap<MaterialBinding, MaterialHandle>,
}

impl Materials {
    fn resolve(&mut self, assets: &AssetLibrary, handle: MaterialHandle) -> Option<HeldMaterial> {
        let at = handle.slot();
        if let Some(Some(held)) = self.held.get(at)
            && held.handle == handle
        {
            return Some(held.clone());
        }
        let material = assets.material(handle)?.clone();
        if !textures_live(assets, &material) {
            return None;
        }
        let canonical = material
            .binding()
            .map(|binding| *self.bindings.entry(binding).or_insert(handle));
        if self.held.len() <= at {
            self.held.resize(at + 1, None);
        }
        let held = HeldMaterial {
            handle,
            material,
            canonical,
            stale: false,
        };
        self.held[at] = Some(held.clone());
        Some(held)
    }

    fn current(&self, handle: MaterialHandle) -> bool {
        matches!(
            self.held.get(handle.slot()),
            Some(Some(held)) if held.handle == handle && !held.stale
        )
    }

    fn settle(&mut self, assets: &AssetLibrary) -> bool {
        let mut mapped_changed = false;
        for at in 0..self.held.len() {
            let Some(held) = self.held[at].clone() else {
                continue;
            };
            let live = assets
                .material(held.handle)
                .filter(|material| textures_live(assets, material));
            match live {
                Some(material) if *material == held.material => {}
                Some(material) if material.binding() == held.material.binding() => {
                    self.held[at] = Some(HeldMaterial {
                        material: material.clone(),
                        stale: true,
                        ..held
                    });
                }
                _ => mapped_changed |= self.forget(at, held),
            }
        }
        for at in 0..self.held.len() {
            let Some(held) = self.held[at].clone() else {
                continue;
            };
            if let Some(canonical) = held.canonical
                && canonical != held.handle
                && !matches!(
                    self.held.get(canonical.slot()),
                    Some(Some(other)) if other.handle == canonical
                )
            {
                self.forget(at, held);
                mapped_changed = true;
            }
        }
        mapped_changed
    }

    fn forget(&mut self, at: usize, held: HeldMaterial) -> bool {
        self.held[at] = None;
        let Some(binding) = held.material.binding() else {
            return false;
        };
        if self.bindings.get(&binding) == Some(&held.handle) {
            self.bindings.remove(&binding);
        }
        true
    }

    fn settled(&mut self) {
        for held in self.held.iter_mut().flatten() {
            held.stale = false;
        }
    }
}

#[derive(Resource)]
pub(crate) struct RetainedRender {
    instances: Vec<Instance>,
    owners: Vec<Option<Entity>>,
    free: Vec<usize>,
    removed: Vec<Entity>,
    homeless: BTreeSet<Entity>,
    unresolved: BTreeSet<Entity>,
    unbuilt: BTreeSet<Entity>,
    hidden: BTreeSet<Entity>,
    materials: Materials,
    asset_revision: Option<(u64, u64)>,
    topology: Option<Revision>,
    words: Option<u64>,
    colors: Feed<u32>,
    changed: bool,
    render_candidates: usize,
    invalid_matrices: usize,
}

impl Default for RetainedRender {
    fn default() -> Self {
        Self {
            instances: Vec::new(),
            owners: Vec::new(),
            free: Vec::new(),
            removed: Vec::new(),
            homeless: BTreeSet::new(),
            unresolved: BTreeSet::new(),
            unbuilt: BTreeSet::new(),
            hidden: BTreeSet::new(),
            materials: Materials::default(),
            asset_revision: None,
            topology: None,
            words: None,
            colors: Feed::new(FeedSettings::default().instance_colors as usize),
            changed: false,
            render_candidates: 0,
            invalid_matrices: 0,
        }
    }
}

struct Inputs<'a, 'w, 's> {
    query: &'a RenderQuery<'w, 's>,
    globals: &'a GlobalTransforms,
    assets: &'a AssetLibrary,
}

pub(crate) fn retain_instances(
    query: RenderQuery,
    changes: SourceChanges,
    globals: Res<GlobalTransforms>,
    assets: Res<AssetLibrary>,
    feeds: Res<FeedSettings>,
    mut retained: ResMut<RetainedRender>,
    mut instances: ResMut<Instances>,
) {
    let _profile = engine_core::profile::span("cpu", "extraction_retained_updates_culling");
    let inputs = Inputs {
        query: &query,
        globals: &globals,
        assets: &assets,
    };
    let retained = &mut *retained;
    retained.changed = false;
    retained.colors.resize(feeds.instance_colors as usize);
    let assets_moved = retained.settle_assets(&inputs, &mut instances);
    for entity in &changes {
        retained.refresh(entity, &inputs, &mut instances);
    }
    for entity in std::mem::take(&mut retained.removed) {
        retained.refresh(entity, &inputs, &mut instances);
    }
    let topology_moved = retained.settle_topology(&inputs, &mut instances);
    retained.settle_words(&inputs, &mut instances);
    retained.render_candidates = query
        .iter()
        .len()
        .saturating_sub(retained.unbuilt.len() + retained.hidden.len());
    if retained.changed || assets_moved || topology_moved {
        retained.recount(&globals);
    }
}

pub(crate) fn source_removed(
    In(entity): In<Entity>,
    instances: Res<Instances>,
    mut retained: ResMut<RetainedRender>,
) {
    if instances.slots.contains_key(&entity)
        || retained.homeless.contains(&entity)
        || retained.unresolved.contains(&entity)
        || retained.unbuilt.contains(&entity)
    {
        retained.removed.push(entity);
    }
}

impl RetainedRender {
    pub(crate) fn changed(&self) -> bool {
        self.changed
    }

    pub(crate) fn instances(&self) -> &[Instance] {
        &self.instances
    }

    pub(crate) fn owner(&self, slot: usize) -> Option<Entity> {
        self.owners.get(slot).copied().flatten()
    }

    pub(crate) fn drawn(&self) -> impl Iterator<Item = (usize, &Instance)> + '_ {
        self.owners
            .iter()
            .zip(&self.instances)
            .enumerate()
            .filter(|(_, (owner, _))| owner.is_some())
            .map(|(slot, (_, instance))| (slot, instance))
    }

    pub(crate) fn colors(&self) -> &Feed<u32> {
        &self.colors
    }

    pub(crate) fn render_candidates(&self) -> usize {
        self.render_candidates
    }

    pub(crate) fn listed_instances(&self) -> usize {
        (self.owners.len() - self.free.len()).saturating_sub(self.invalid_matrices)
    }

    pub(crate) fn missing_assets(&self) -> usize {
        self.unresolved.len()
    }

    pub(crate) fn invalid_transforms(&self) -> usize {
        self.homeless.len() + self.invalid_matrices
    }

    pub(crate) fn model(&self, slot: usize, globals: &GlobalTransforms) -> Mat4 {
        self.owner(slot)
            .and_then(|entity| globals.get(entity))
            .and_then(|global| global.matrix())
            .unwrap_or(Mat4::IDENTITY)
    }

    fn recount(&mut self, globals: &GlobalTransforms) {
        self.invalid_matrices = self
            .owners
            .iter()
            .flatten()
            .filter(|&&entity| {
                globals
                    .get(entity)
                    .and_then(|global| global.matrix())
                    .is_none()
            })
            .count();
    }

    fn settle_assets(
        &mut self,
        inputs: &Inputs<'_, '_, '_>,
        instances: &mut ResMut<'_, Instances>,
    ) -> bool {
        let assets = inputs.assets;
        let revision = Some(assets.revision());
        if self.asset_revision == revision {
            return false;
        }
        self.asset_revision = revision;
        self.changed |= self.materials.settle(assets);
        let cube = assets.cube_mesh();
        for slot in 0..self.owners.len() {
            let Some(entity) = self.owners[slot] else {
                continue;
            };
            let instance = &self.instances[slot];
            let current = assets.mesh_lods(instance.mesh).is_some()
                && match instance.source {
                    Some(source) => self.materials.current(source),
                    None => instance.mesh == cube,
                };
            if !current {
                self.refresh(entity, inputs, instances);
            }
        }
        self.materials.settled();
        for entity in std::mem::take(&mut self.unresolved) {
            self.refresh(entity, inputs, instances);
        }
        true
    }

    fn settle_topology(
        &mut self,
        inputs: &Inputs<'_, '_, '_>,
        instances: &mut ResMut<'_, Instances>,
    ) -> bool {
        let topology = Some(inputs.globals.topology_revision());
        if self.topology == topology {
            return false;
        }
        self.topology = topology;
        for slot in 0..self.owners.len() {
            if let Some(entity) = self.owners[slot]
                && inputs.globals.get(entity).is_none()
            {
                self.refresh(entity, inputs, instances);
            }
        }
        for entity in std::mem::take(&mut self.homeless) {
            self.refresh(entity, inputs, instances);
        }
        true
    }

    fn settle_words(
        &mut self,
        inputs: &Inputs<'_, '_, '_>,
        instances: &mut ResMut<'_, Instances>,
    ) {
        let words = Some(inputs.globals.words_revision());
        if self.words == words {
            return;
        }
        self.words = words;
        let undrawn: Vec<Entity> = self
            .owners
            .iter()
            .flatten()
            .chain(&self.homeless)
            .chain(&self.unresolved)
            .chain(&self.unbuilt)
            .copied()
            .filter(|&entity| !inputs.globals.drawn(entity))
            .collect();
        let hidden = std::mem::take(&mut self.hidden);
        for entity in undrawn.into_iter().chain(hidden) {
            self.refresh(entity, inputs, instances);
        }
    }

    fn refresh(
        &mut self,
        entity: Entity,
        inputs: &Inputs<'_, '_, '_>,
        instances: &mut ResMut<'_, Instances>,
    ) {
        let Ok((_, has_transform, cube, mesh3d, tint, layer)) = inputs.query.get(entity) else {
            self.forget(entity, instances);
            return;
        };
        if !inputs.globals.drawn(entity) {
            self.forget(entity, instances);
            self.hidden.insert(entity);
            return;
        }
        self.hidden.remove(&entity);
        if unbuilt(cube.is_some(), mesh3d) {
            self.forget(entity, instances);
            self.unbuilt.insert(entity);
            return;
        }
        self.unbuilt.remove(&entity);
        if !has_transform || inputs.globals.get(entity).is_none() {
            self.release(entity, instances);
            self.unresolved.remove(&entity);
            self.homeless.insert(entity);
            return;
        }
        self.homeless.remove(&entity);
        let Some(next) = self.resolve(inputs.assets, cube, mesh3d, tint, layer) else {
            self.release(entity, instances);
            self.unresolved.insert(entity);
            return;
        };
        self.unresolved.remove(&entity);
        let (slot, fresh) = self.slot(entity, next, instances);
        self.apply(slot, fresh, next);
    }

    fn resolve(
        &mut self,
        assets: &AssetLibrary,
        cube: Option<&Cube>,
        mesh3d: Option<&Mesh3d>,
        tint: Option<&Tint>,
        layer: Option<&FirstPersonLayer>,
    ) -> Option<Instance> {
        let (mesh, source, material, color, surface) = if let Some(mesh3d) = mesh3d {
            let held = self.materials.resolve(assets, mesh3d.material)?;
            (
                mesh3d.mesh,
                Some(mesh3d.material),
                held.canonical,
                held.material.color(),
                held.material.surface(),
            )
        } else {
            let [r, g, b, _] = cube?.color;
            (
                assets.cube_mesh(),
                None,
                None,
                [r, g, b, 1.0],
                Surface::MATTE,
            )
        };
        assets.mesh_lods(mesh)?;
        Some(Instance {
            mesh,
            source,
            material,
            color: tint.map_or(color, |tint| tint.apply(color)),
            surface,
            flags: layer.map_or(0, FirstPersonLayer::slot_flags),
        })
    }

    fn slot(
        &mut self,
        entity: Entity,
        next: Instance,
        instances: &mut ResMut<'_, Instances>,
    ) -> (usize, bool) {
        if let Some(&slot) = instances.slots.get(&entity) {
            return (slot as usize, false);
        }
        let slot = if let Some(slot) = self.free.pop() {
            self.owners[slot] = Some(entity);
            slot
        } else {
            self.owners.push(Some(entity));
            self.instances.push(next);
            self.owners.len() - 1
        };
        instances.slots.insert(entity, slot as u32);
        (slot, true)
    }

    fn apply(&mut self, slot: usize, fresh: bool, next: Instance) {
        let record = &mut self.instances[slot];
        self.changed |= fresh
            || record.mesh != next.mesh
            || record.material != next.material
            || record.first_person() != next.first_person();
        let restyled = fresh
            || record.color.map(f32::to_bits) != next.color.map(f32::to_bits)
            || surface_bits(record.surface) != surface_bits(next.surface)
            || record.flags != next.flags;
        *record = next;
        if restyled {
            self.colors.push(slot as u32);
        }
    }

    fn release(&mut self, entity: Entity, instances: &mut ResMut<'_, Instances>) {
        let Some(&slot) = instances.slots.get(&entity) else {
            return;
        };
        instances.slots.remove(&entity);
        self.owners[slot as usize] = None;
        self.free.push(slot as usize);
        self.changed = true;
    }

    fn forget(&mut self, entity: Entity, instances: &mut ResMut<'_, Instances>) {
        self.release(entity, instances);
        self.homeless.remove(&entity);
        self.unresolved.remove(&entity);
        self.unbuilt.remove(&entity);
        self.hidden.remove(&entity);
    }
}
