use engine::prelude::*;
use engine_core::GlobalTransforms;
use engine_math::collections::{HashMap, HashSet};

use crate::world::{InstanceHandle, NaniteWorld};

#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct NaniteMesh {
    pub mesh_id: u32,
    pub custom_id: u32,
    pub visible: bool,
}

impl NaniteMesh {
    #[must_use]
    pub const fn new(mesh_id: u32) -> Self {
        Self {
            mesh_id,
            custom_id: 0,
            visible: true,
        }
    }

    #[must_use]
    pub const fn with_custom_id(mut self, custom_id: u32) -> Self {
        self.custom_id = custom_id;
        self
    }

    #[must_use]
    pub const fn with_visible(mut self, visible: bool) -> Self {
        self.visible = visible;
        self
    }
}

#[derive(Clone, Copy)]
struct SyncedEntry {
    handle: Option<InstanceHandle>,
    mesh_id: u32,
}

#[derive(Resource, Default)]
pub(crate) struct NaniteEcsSync {
    entries: HashMap<Entity, SyncedEntry>,
    last_clear_epoch: u64,
}

fn uniform_scale(scale: Vec3) -> f32 {
    engine::math::max(
        engine::math::max(scale.x.abs(), scale.y.abs()),
        scale.z.abs(),
    )
}

fn resolve_pose(
    entity: Entity,
    local: &Transform,
    globals: Option<&GlobalTransforms>,
) -> (Vec3, Quat, f32) {
    if let Some(mat) = globals
        .and_then(|g| g.get(entity))
        .and_then(|gt| gt.matrix())
    {
        let (scale, rot, pos) = mat.to_scale_rotation_translation();
        if pos.is_finite() && rot.is_finite() && scale.is_finite() {
            return (pos, rot.normalize(), uniform_scale(scale));
        }
    }
    (
        local.translation,
        local.rotation.normalize(),
        uniform_scale(local.scale),
    )
}

pub(crate) fn sync_ecs_nanite_instances(
    world: &mut NaniteWorld,
    sync: &mut NaniteEcsSync,
    globals: Option<&GlobalTransforms>,
    query: &Query<(Entity, &NaniteMesh, &Transform)>,
) {
    if sync.last_clear_epoch != world.clear_epoch {
        sync.entries.clear();
        sync.last_clear_epoch = world.clear_epoch;
    }
    if query.is_empty() {
        if !sync.entries.is_empty() {
            for (_, entry) in sync.entries.drain() {
                if let Some(handle) = entry.handle {
                    let _ = world.remove_instance(handle);
                }
            }
        }
        return;
    }
    let mut seen: HashSet<Entity> = HashSet::default();
    for (entity, mesh, transform) in query {
        seen.insert(entity);
        let (pos, rot, scale) = resolve_pose(entity, transform, globals);
        let prev = sync.entries.get(&entity).copied();
        if !mesh.visible {
            if let Some(entry) = prev {
                if let Some(handle) = entry.handle {
                    let _ = world.remove_instance(handle);
                }
                sync.entries.insert(
                    entity,
                    SyncedEntry {
                        handle: None,
                        mesh_id: mesh.mesh_id,
                    },
                );
            }
            continue;
        }
        match prev {
            Some(SyncedEntry {
                handle: Some(handle),
                mesh_id,
            }) if mesh_id == mesh.mesh_id => {
                if !world.set_instance_transform(handle, pos, rot, scale) {
                    let next =
                        world.add_instance_custom(mesh.mesh_id, pos, rot, scale, mesh.custom_id);
                    sync.entries.insert(
                        entity,
                        SyncedEntry {
                            handle: next,
                            mesh_id: mesh.mesh_id,
                        },
                    );
                } else {
                    let _ = world.set_instance_custom_id(handle, mesh.custom_id);
                }
            }
            other => {
                if let Some(SyncedEntry {
                    handle: Some(old_handle),
                    ..
                }) = other
                {
                    let _ = world.remove_instance(old_handle);
                }
                let next =
                    world.add_instance_custom(mesh.mesh_id, pos, rot, scale, mesh.custom_id);
                sync.entries.insert(
                    entity,
                    SyncedEntry {
                        handle: next,
                        mesh_id: mesh.mesh_id,
                    },
                );
            }
        }
    }
    if sync.entries.len() > seen.len() {
        let stale: Vec<Entity> = sync
            .entries
            .keys()
            .copied()
            .filter(|e| !seen.contains(e))
            .collect();
        for entity in stale {
            if let Some(entry) = sync.entries.remove(&entity)
                && let Some(handle) = entry.handle
            {
                let _ = world.remove_instance(handle);
            }
        }
    }
}
