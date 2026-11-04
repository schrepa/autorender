use bevy_ecs::prelude::*;
use engine_core::{
    AlphaMode, AssetLibrary, AssetResidency, DeviceLimits, MaterialHandle, MeshHandle,
    ResidentMesh, Spawner,
};
use engine_places::shadows::{CasterMap, Refresh, Region, ShadowChanges, ShadowMaps};

use crate::MeshesBrick;
use crate::batches::BatchPlan;
use crate::layouts::{MOVING_MAP, MOVING_PAIRS, SKY_MAP};
use crate::order::MeshPlan;
use crate::retained::RetainedRender;

pub(crate) const MAPS: usize = SKY_MAP as usize + 1;
pub(crate) const ID_CAPACITIES: usize = MAPS - 1 + MOVING_PAIRS as usize;
pub(crate) const STILLNESS_BYTES: u64 = 16;

pub(crate) fn map_start(map: usize, capacity: usize) -> usize {
    if map <= MOVING_MAP as usize {
        map * capacity
    } else {
        (map + MOVING_PAIRS as usize - 1) * capacity
    }
}

pub(crate) fn map_factor(map: usize) -> u32 {
    if map == MOVING_MAP as usize {
        MOVING_PAIRS
    } else {
        1
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ShadowGroup {
    pub mesh: MeshHandle,
    pub material: Option<MaterialHandle>,
    pub start: u32,
    pub capacity: u32,
}

fn caster_range(mesh: Option<&ResidentMesh>, with_levels: bool, error: f32) -> (u32, u32, i32) {
    let Some(mesh) = mesh else {
        return (0, 0, 0);
    };
    let mut range = mesh.caster_range(0, (0, mesh.index_count()));
    if with_levels {
        for (level, &(offset, count, lod_error)) in mesh.lods().iter().enumerate() {
            if lod_error > error {
                break;
            }
            range = mesh.caster_range(level + 1, (offset, count));
        }
    }
    (range.1, mesh.first_index() + range.0, mesh.base_vertex())
}

#[derive(Resource, Default)]
pub(crate) struct CasterPlan {
    pub groups: Vec<ShadowGroup>,
    pub total: u32,
    pub batch_groups: Vec<u32>,
    pub template: Vec<u32>,
    pub starts: Vec<u32>,
    pub id_capacity: usize,
    pub arg_capacity: usize,
    pub stillness_capacity: usize,
    pub regions_in_args: bool,
    pub maps: Vec<Option<usize>>,
    grouped: Option<u64>,
}

impl CasterPlan {
    pub(crate) fn batch_group(&self, index: usize) -> u32 {
        self.batch_groups.get(index).copied().unwrap_or(0)
    }

    fn clear(&mut self) {
        self.groups.clear();
        self.batch_groups.clear();
        self.starts.clear();
        self.total = 0;
        self.grouped = None;
    }

    fn plan(&mut self, batches: &BatchPlan, assets: &AssetLibrary) -> bool {
        if self.grouped == Some(batches.generation()) {
            return false;
        }
        self.grouped = Some(batches.generation());
        self.groups.clear();
        self.batch_groups.clear();
        let mut plain: Option<(MeshHandle, u32)> = None;
        for batch in &batches.batches {
            let masked = batch.material.filter(|&material| {
                assets
                    .material(material)
                    .is_some_and(|held| matches!(held.alpha_mode(), AlphaMode::Mask { .. }))
            });
            let shared = plain
                .filter(|&(mesh, _)| masked.is_none() && mesh == batch.mesh)
                .map(|(_, group)| group);
            let group = shared.unwrap_or_else(|| {
                self.groups.push(ShadowGroup {
                    mesh: batch.mesh,
                    material: masked,
                    start: 0,
                    capacity: 0,
                });
                self.groups.len() as u32 - 1
            });
            if masked.is_none() {
                plain = Some((batch.mesh, group));
            }
            self.groups[group as usize].capacity += batch.instances.end - batch.instances.start;
            self.batch_groups.push(group);
        }
        let mut start = 0;
        for group in &mut self.groups {
            group.start = start;
            start += group.capacity;
        }
        self.starts.clear();
        self.starts
            .extend(self.groups.iter().map(|group| group.start));
        self.total = start;
        true
    }

    fn outputs(&mut self, slots: usize, first_instance: bool) -> bool {
        let mut grown = false;
        if slots > self.stillness_capacity {
            self.stillness_capacity = slots.max(1_024).next_power_of_two();
            grown = true;
        }
        let ids = (self.total as usize).max(1);
        if ids > self.id_capacity {
            self.id_capacity = ids.next_power_of_two();
            grown = true;
        }
        let groups = self.groups.len();
        if groups > self.arg_capacity {
            self.arg_capacity = groups.next_power_of_two();
            grown = true;
        }
        self.regions_in_args =
            first_instance && u32::try_from(ID_CAPACITIES * self.id_capacity).is_ok();
        grown
    }

    fn template(&mut self, errors: &[f32; MAPS], residency: &AssetResidency, with_levels: bool) {
        self.template.clear();
        for (map, &error) in errors.iter().enumerate() {
            let start = map_start(map, self.id_capacity) as u32;
            for group in &self.groups {
                let (count, first_index, base_vertex) =
                    caster_range(residency.mesh(group.mesh), with_levels, error);
                let first_instance = if self.regions_in_args {
                    start + map_factor(map) * group.start
                } else {
                    0
                };
                self.template
                    .extend([count, 0, first_index, base_vertex as u32, first_instance]);
            }
        }
    }
}

fn map_of(regions: &[Region], at: usize) -> Option<usize> {
    let region = regions.get(at)?;
    Some(match region.map {
        CasterMap::Region => regions[..at]
            .iter()
            .filter(|region| region.map == CasterMap::Region)
            .count(),
        CasterMap::Moving => MOVING_MAP as usize,
        CasterMap::Sky => SKY_MAP as usize,
    })
}

fn errors(maps: &ShadowMaps) -> [f32; MAPS] {
    let mut errors = [0.0; MAPS];
    for (at, region) in maps.regions.iter().enumerate() {
        let (Some(map), Refresh::Kept(kept)) = (map_of(&maps.regions, at), region.refresh) else {
            continue;
        };
        if map != SKY_MAP as usize {
            errors[map] = 0.5 * kept.texel;
        }
    }
    errors
}

#[derive(Component)]
pub(crate) struct CasterChanges;

pub(crate) fn plan_casters(
    retained: Res<RetainedRender>,
    batches: Res<BatchPlan>,
    decided: Res<MeshPlan>,
    (maps, limits, residency, assets): (
        Res<ShadowMaps>,
        Res<DeviceLimits>,
        Res<AssetResidency>,
        Res<AssetLibrary>,
    ),
    own: Query<Entity, With<CasterChanges>>,
    mut plan: ResMut<CasterPlan>,
    mut spawner: Spawner<MeshesBrick>,
) {
    let plan = &mut *plan;
    if !maps.on {
        plan.clear();
        plan.maps.clear();
        engine_core::profile::counter("render.shadow_groups", 0.0);
        return;
    }
    let groups_changed = plan.plan(&batches, &assets);
    let grown = plan.outputs(retained.instances().len(), limits.indirect_first_instance);
    plan.template(&errors(&maps), &residency, decided.lod_levels);
    plan.maps.clear();
    plan.maps
        .extend((0..maps.regions.len()).map(|at| map_of(&maps.regions, at)));
    if groups_changed || grown {
        let change = ShadowChanges {
            boxes: Vec::new(),
            all: true,
            casters: groups_changed,
        };
        match own.single() {
            Ok(entity) => spawner.insert(entity, change),
            Err(_) => {
                spawner.spawn((CasterChanges, change));
            }
        }
    }
    engine_core::profile::counter("render.shadow_groups", plan.groups.len() as f64);
}
