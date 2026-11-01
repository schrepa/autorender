use bevy_ecs::prelude::*;
use engine_core::collections::HashMap;
use engine_core::{AlphaMode, DeviceLimits, MaterialHandle, MeshHandle, Spawner};
use engine_places::shadows::{CasterMap, Refresh, Region, ShadowChanges, ShadowMaps};

use crate::MeshesBrick;
use crate::batches::BatchPlan;
use crate::layouts::{BatchData, CandidateData, MOVING_MAP, MOVING_PAIRS, SKY_MAP};
use crate::order::{MeshLevels, MeshPlan};
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Source {
    Listed,
    Presented,
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
    pub lods: Vec<(u32, u32, f32)>,
}

impl CasterDraw {
    fn of(levels: &MeshLevels, with_levels: bool) -> Self {
        let lods = if with_levels {
            levels
                .lods
                .iter()
                .enumerate()
                .map(|(level, &(offset, count, error))| {
                    let (offset, count) = levels.caster_range(level + 1, (offset, count));
                    (offset, count, error)
                })
                .collect()
        } else {
            Vec::new()
        };
        let (offset, count) = levels.caster_range(0, (0, levels.index_count));
        Self {
            index_count: count,
            first_index: offset,
            lods,
        }
    }

    fn level_within(&self, error: f32) -> u8 {
        self.lods
            .iter()
            .take_while(|&&(_, _, lod_error)| lod_error <= error)
            .count() as u8
    }

    fn range(&self, level: u8) -> (u32, u32) {
        match level.checked_sub(1) {
            Some(lod) => {
                let (first_index, index_count, _) = self.lods[lod as usize];
                (index_count, first_index)
            }
            None => (self.index_count, self.first_index),
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
    pub source: Option<Source>,
    pub listed: Vec<CandidateData>,
    pub records: Vec<BatchData>,
    pub template: Vec<u32>,
    pub starts: Vec<u32>,
    pub id_capacity: usize,
    pub arg_capacity: usize,
    pub stillness_capacity: usize,
    pub regions_in_args: bool,
    pub maps: Vec<Option<usize>>,
    pub generation: u64,
    key: Option<(u8, engine_core::Stamp, u64)>,
    draws: Vec<CasterDraw>,
    levels: Vec<u8>,
}

impl CasterPlan {
    pub(crate) fn batch_group(&self, index: usize) -> u32 {
        self.batch_groups.get(index).copied().unwrap_or(0)
    }

    fn plan(&mut self, retained: &RetainedRender, batches: &BatchPlan) -> bool {
        let gpu = retained.gpu();
        let (source, key) = if gpu {
            (
                Source::Presented,
                (0, retained.identity(), retained.source_revision()),
            )
        } else {
            (
                Source::Listed,
                (
                    1,
                    retained.identity(),
                    retained.caster_revision().max(retained.source_revision()),
                ),
            )
        };
        self.source = Some(source);
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
        if gpu {
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
        } else {
            let sources: HashMap<MeshHandle, usize> = snapshot
                .meshes
                .iter()
                .enumerate()
                .map(|(index, source)| (source.handle, index))
                .collect();
            self.listed.clear();
            for &slot in retained.casters() {
                let instance = &snapshot.instances[slot];
                let Some(&source_index) = sources.get(&instance.mesh) else {
                    continue;
                };
                let group = add(
                    &mut self.groups,
                    key(instance.mesh, instance.material),
                    source_index,
                    1,
                );
                self.listed.push(CandidateData {
                    slot: slot as u32,
                    node: 0,
                    batch: group,
                    position: 0,
                });
            }
        }
        let mut start = 0;
        self.records.clear();
        for (group, count) in self.groups.iter_mut().zip(&counts) {
            group.start = start;
            group.capacity = *count;
            start += count;
            let mesh = &snapshot.meshes[group.source_index].mesh;
            let bounds = mesh.bounds();
            self.records.push(BatchData {
                start: group.start,
                count: *count,
                index_count: mesh.indices().len() as u32,
                first_index: 0,
                base_vertex: 0,
                first_instance: 0,
                first_plane: 0,
                lod_ranges: [0; 4],
                lod_errors: [0.0, 0.0, 1.0, 0.0],
                group: self.records.len() as u32,
                bounds_min: bounds.min().extend(0.0).to_array(),
                bounds_max: bounds.max().extend(0.0).to_array(),
            });
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
                let (count, first_index) = self.draws[index].range(self.levels[map * groups + index]);
                let first_instance = if self.regions_in_args {
                    start + map_factor(map) * group.start
                } else {
                    0
                };
                self.template.extend([count, 0, first_index, 0, first_instance]);
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

#[expect(
    clippy::too_many_arguments,
    reason = "the caster plan joins the draw plan, the shadow maps and the device"
)]
pub(crate) fn plan_casters(
    retained: Res<RetainedRender>,
    batches: Res<BatchPlan>,
    decided: Res<MeshPlan>,
    maps: Res<ShadowMaps>,
    limits: Res<DeviceLimits>,
    own: Query<Entity, With<CasterChanges>>,
    mut plan: ResMut<CasterPlan>,
    mut spawner: Spawner<MeshesBrick>,
) {
    let plan = &mut *plan;
    let mut changed = false;
    if plan.plan(&retained, &batches) {
        let snapshot = retained.snapshot();
        plan.draws = plan
            .groups
            .iter()
            .map(|group| {
                let levels = MeshLevels::of(&snapshot.meshes[group.source_index]);
                CasterDraw::of(&levels, decided.lod_levels)
            })
            .collect();
        plan.levels.clear();
        changed = true;
    }
    let slots = retained.snapshot().instances.len();
    let classify = plan.source == Some(Source::Presented);
    let grown = plan.outputs(if classify { slots } else { 0 }, limits.indirect_first_instance);
    if plan.template(&errors(&maps), grown || changed) {
        changed = true;
    }
    plan.maps.clear();
    plan.maps
        .extend((0..maps.regions.len()).map(|at| map_of(&maps.regions, at)));
    if changed {
        plan.generation += 1;
    }
    if grown {
        let change = ShadowChanges {
            boxes: Vec::new(),
            all: true,
        };
        match own.single() {
            Ok(entity) => spawner.insert(entity, change),
            Err(_) => {
                spawner.spawn((CasterChanges, change));
            }
        }
    }
}
