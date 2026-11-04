use bevy_ecs::prelude::*;
use engine_core::collections::HashMap;
use engine_core::{
    AlphaMode, AssetResidency, DeviceLimits, MaterialHandle, MeshHandle, ResidentMesh, Spawner,
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
    pub source_index: usize,
    pub material: Option<MaterialHandle>,
    pub start: u32,
    pub capacity: u32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct CasterDraw {
    pub index_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub lods: Vec<(u32, u32, f32)>,
}

impl CasterDraw {
    fn of(mesh: Option<&ResidentMesh>, with_levels: bool) -> Self {
        let Some(mesh) = mesh else {
            return Self::default();
        };
        let first_index = mesh.first_index();
        let lods = if with_levels {
            mesh.lods()
                .iter()
                .enumerate()
                .map(|(level, &(offset, count, error))| {
                    let (offset, count) = mesh.caster_range(level + 1, (offset, count));
                    (first_index + offset, count, error)
                })
                .collect()
        } else {
            Vec::new()
        };
        let (offset, count) = mesh.caster_range(0, (0, mesh.index_count()));
        Self {
            index_count: count,
            first_index: first_index + offset,
            base_vertex: mesh.base_vertex(),
            lods,
        }
    }

    fn level_within(&self, error: f32) -> u8 {
        self.lods
            .iter()
            .take_while(|&&(_, _, lod_error)| lod_error <= error)
            .count() as u8
    }

    fn range(&self, level: u8) -> (u32, u32, i32) {
        match level.checked_sub(1) {
            Some(lod) => {
                let (first_index, index_count, _) = self.lods[lod as usize];
                (index_count, first_index, self.base_vertex)
            }
            None => (self.index_count, self.first_index, self.base_vertex),
        }
    }
}

type GroupIndices = HashMap<(MeshHandle, Option<MaterialHandle>), u32>;

#[derive(Resource, Default)]
pub(crate) struct CasterPlan {
    pub groups: Vec<ShadowGroup>,
    pub masked: Vec<bool>,
    pub total: u32,
    pub batch_groups: Vec<u32>,
    pub template: Vec<u32>,
    pub starts: Vec<u32>,
    pub id_capacity: usize,
    pub arg_capacity: usize,
    pub stillness_capacity: usize,
    pub regions_in_args: bool,
    pub maps: Vec<Option<usize>>,
    pub generation: u64,
    groups_generation: u64,
    key: Option<(engine_core::Stamp, u64)>,
    draws_key: Option<(u64, u64, bool)>,
    draws: Vec<CasterDraw>,
    levels: Vec<u8>,
}

impl CasterPlan {
    pub(crate) fn batch_group(&self, index: usize) -> u32 {
        self.batch_groups.get(index).copied().unwrap_or(0)
    }

    fn clear(&mut self) {
        self.groups.clear();
        self.masked.clear();
        self.batch_groups.clear();
        self.starts.clear();
        self.total = 0;
        self.key = None;
        self.draws.clear();
        self.levels.clear();
    }

    fn plan(&mut self, retained: &RetainedRender, batches: &BatchPlan) -> bool {
        let key = (retained.identity(), retained.source_revision());
        if self.key == Some(key) {
            return false;
        }
        self.key = Some(key);
        self.groups.clear();
        self.masked.clear();
        self.batch_groups.clear();
        let snapshot = retained.snapshot();
        let mut indices = GroupIndices::default();
        let mut counts = Vec::new();
        let masked = |material: MaterialHandle| {
            snapshot.materials.iter().any(|source| {
                source.handle == material
                    && matches!(source.material.alpha_mode(), AlphaMode::Mask { .. })
            })
        };
        let key = |mesh, material: Option<MaterialHandle>| {
            (mesh, material.filter(|&material| masked(material)))
        };
        let mut add = |groups: &mut Vec<ShadowGroup>, (mesh, material), source_index, count| {
            let index = match indices.get(&(mesh, material)) {
                Some(&index) => index,
                None => {
                    groups.push(ShadowGroup {
                        mesh,
                        source_index,
                        material,
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
        };
        for batch in &batches.batches {
            let count = batch.instances.end - batch.instances.start;
            let group = add(
                &mut self.groups,
                key(batch.mesh, batch.material),
                batch.source_index,
                count,
            );
            self.batch_groups.push(group);
        }
        let mut start = 0;
        for (group, count) in self.groups.iter_mut().zip(&counts) {
            group.start = start;
            group.capacity = *count;
            start += count;
        }
        self.masked
            .extend(self.groups.iter().map(|group| group.material.is_some()));
        self.starts.clear();
        self.starts.extend(self.groups.iter().map(|group| group.start));
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

    fn template(&mut self, errors: &[f32; MAPS], force: bool) -> bool {
        let groups = self.groups.len();
        let levels: Vec<u8> = errors
            .iter()
            .flat_map(|&error| self.draws.iter().map(move |draw| draw.level_within(error)))
            .collect();
        if !force && levels == self.levels {
            return false;
        }
        self.levels = levels;
        self.template.clear();
        for map in 0..MAPS {
            let start = map_start(map, self.id_capacity) as u32;
            for (index, group) in self.groups.iter().enumerate() {
                let (count, first_index, base_vertex) =
                    self.draws[index].range(self.levels[map * groups + index]);
                let first_instance = if self.regions_in_args {
                    start + map_factor(map) * group.start
                } else {
                    0
                };
                self.template
                    .extend([count, 0, first_index, base_vertex as u32, first_instance]);
            }
        }
        true
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
    (maps, limits, residency): (Res<ShadowMaps>, Res<DeviceLimits>, Res<AssetResidency>),
    own: Query<Entity, With<CasterChanges>>,
    mut plan: ResMut<CasterPlan>,
    mut spawner: Spawner<MeshesBrick>,
) {
    let plan = &mut *plan;
    if !maps.on {
        if !plan.groups.is_empty() {
            plan.groups_generation += 1;
            plan.generation += 1;
        }
        plan.clear();
        plan.maps.clear();
        engine_core::profile::counter("render.shadow_groups", 0.0);
        return;
    }
    let groups_changed = plan.plan(&retained, &batches);
    if groups_changed {
        plan.groups_generation += 1;
    }
    let mut changed = groups_changed;
    let draws_key = (plan.groups_generation, residency.layout(), decided.lod_levels);
    if plan.draws_key != Some(draws_key) {
        plan.draws_key = Some(draws_key);
        plan.draws = plan
            .groups
            .iter()
            .map(|group| CasterDraw::of(residency.mesh(group.mesh), decided.lod_levels))
            .collect();
        plan.levels.clear();
        changed = true;
    }
    let slots = retained.snapshot().instances.len();
    let grown = plan.outputs(slots, limits.indirect_first_instance);
    if plan.template(&errors(&maps), grown || changed) {
        changed = true;
    }
    plan.maps.clear();
    plan.maps
        .extend((0..maps.regions.len()).map(|at| map_of(&maps.regions, at)));
    if changed {
        plan.generation += 1;
    }
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
