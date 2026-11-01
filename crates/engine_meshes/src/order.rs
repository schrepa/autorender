use bevy_ecs::prelude::*;
use engine_core::{
    AlphaMode, AssetLibrary, AssetResidency, Detail, DeviceLimits, MaterialHandle, Mesh,
    ResidentMesh, View,
};
use engine_places::meshes::MOVED_SLOT;
use glam::Vec3;

use crate::batches::{BatchPlan, DrawBatch};
use crate::layouts::LOD_LEVELS;
use crate::presentation::frustum_uniform;
use crate::retained::RetainedRender;
use crate::settings::{Culling, OcclusionCulling, SubpixelCulling};
use crate::uploads::PoseData;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct BatchLook {
    pub masked: bool,
    pub double_sided: bool,
    pub extended: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct MeshLevels {
    pub index_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub plan: Vec<(u32, u32, f32)>,
}

impl MeshLevels {
    pub(crate) fn of(mesh: Option<&ResidentMesh>) -> Self {
        let Some(mesh) = mesh else {
            return Self::default();
        };
        Self {
            index_count: mesh.index_count(),
            first_index: mesh.first_index(),
            base_vertex: mesh.base_vertex(),
            plan: mesh.plan().to_vec(),
        }
    }

    pub(crate) fn levels(&self) -> usize {
        1 + self.plan.len().min(LOD_LEVELS as usize - 1)
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct BatchMeta {
    pub index_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub bounds: (Vec3, Vec3),
    pub lods: Vec<(u32, u32, f32)>,
    pub levels: MeshLevels,
}

#[derive(Resource, Default)]
pub(crate) struct DrawOrder {
    pub order: Vec<u32>,
    pub masked: Vec<u32>,
    pub looks: Vec<BatchLook>,
    pub metas: Vec<BatchMeta>,
    pub generation: u64,
    keys: Vec<(u8, u32, Option<MaterialHandle>, u8, u32)>,
    planned: Option<(u64, u64, u64, u64)>,
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct MeshPlan {
    pub gpu: bool,
    pub culling: bool,
    pub occlusion_enabled: bool,
    pub occlusion: bool,
    pub subpixel_enabled: bool,
    pub subpixel: bool,
    pub budget: u32,
    pub lod_levels: bool,
}

pub(crate) fn extended(mesh: &Mesh) -> bool {
    mesh.colors().is_some() || mesh.uv1().is_some()
}

fn instance_limit(limits: &DeviceLimits) -> Option<usize> {
    let record = size_of::<PoseData>().max(6 * size_of::<u32>()) as u64;
    (limits.storage_room != 0)
        .then(|| (limits.storage_room / record).min(u64::from(MOVED_SLOT)) as usize)
}

fn look(retained: &RetainedRender, assets: &AssetLibrary, batch: &DrawBatch) -> BatchLook {
    let snapshot = retained.snapshot();
    let material = batch.material.and_then(|handle| assets.material(handle));
    BatchLook {
        masked: material
            .is_some_and(|material| matches!(material.alpha_mode(), AlphaMode::Mask { .. })),
        double_sided: material.is_some_and(|material| material.is_double_sided()),
        extended: extended(&snapshot.meshes[batch.source_index].mesh),
    }
}

fn meta(retained: &RetainedRender, residency: &AssetResidency, batch: &DrawBatch) -> BatchMeta {
    let source = &retained.snapshot().meshes[batch.source_index];
    let levels = MeshLevels::of(residency.mesh(batch.mesh));
    let bounds = source.mesh.bounds();
    let first_index = levels.first_index;
    BatchMeta {
        index_count: levels.index_count,
        first_index,
        base_vertex: levels.base_vertex,
        bounds: (bounds.min(), bounds.max()),
        lods: levels.plan[..levels.levels() - 1]
            .iter()
            .map(|&(offset, count, error)| (first_index + offset, count, error))
            .collect(),
        levels,
    }
}

fn flat_batch(retained: &RetainedRender, plan: &BatchPlan, batch: &DrawBatch) -> bool {
    let snapshot = retained.snapshot();
    if plan
        .ordered_instances
        .get(batch.instances.start as usize)
        .and_then(|&slot| snapshot.instances.get(slot))
        .is_none()
    {
        return false;
    }
    let bounds = snapshot.meshes[batch.source_index].mesh.bounds();
    let size = bounds.max() - bounds.min();
    let model = glam::Mat4::IDENTITY;
    let extent = |row: usize| {
        model.x_axis[row].abs() * size.x
            + model.y_axis[row].abs() * size.y
            + model.z_axis[row].abs() * size.z
    };
    extent(1) < 0.1 * engine_core::math::max(extent(0), extent(2))
}

#[expect(
    clippy::too_many_arguments,
    reason = "the plan joins the retained instances, the assets and the culling settings"
)]
pub(crate) fn plan_draws(
    retained: Res<RetainedRender>,
    (assets, residency): (Res<AssetLibrary>, Res<AssetResidency>),
    (view, detail, limits): (Res<View>, Res<Detail>, Res<DeviceLimits>),
    culling: Res<Culling>,
    occlusion: Res<OcclusionCulling>,
    subpixel: Res<SubpixelCulling>,
    mut plan: ResMut<BatchPlan>,
    mut order: ResMut<DrawOrder>,
    mut decided: ResMut<MeshPlan>,
) {
    let plan = &mut *plan;
    plan.instance_limit = instance_limit(&limits);
    {
        let _profile = engine_core::profile::span("render", "batch_plan");
        plan.refused = plan.prepare_frame(&retained).err();
    }
    let key = (
        plan.generation(),
        assets.material_revision(),
        residency.generation(),
        residency.layout(),
    );
    if order.planned != Some(key) {
        order.planned = Some(key);
        order_draws(&retained, (&assets, &residency), plan, &mut order);
    }
    let (planes, absolute) = frustum_uniform(view.view_projection());
    let bounded = absolute.iter().flatten().all(|value| {
        value.is_finite() && *value <= engine_core::presentation_bounds::MAX_COEFFICIENT
    }) && planes.iter().flatten().all(|value| value.is_finite());
    let gpu = retained.gpu();
    let frustum = culling.frustum && bounded;
    let candidates = plan.ordered_instances.len();
    let unbudgeted = frustum && culling.budget() >= candidates;
    decided.set_if_neq(MeshPlan {
        gpu,
        culling: culling.frustum,
        occlusion_enabled: occlusion.enabled,
        occlusion: gpu && occlusion.enabled && unbudgeted && candidates != 0,
        subpixel_enabled: subpixel.enabled,
        subpixel: subpixel.enabled && unbudgeted && candidates != 0,
        budget: culling.budget().min(u32::MAX as usize) as u32,
        lod_levels: detail.enabled,
    });
}

fn order_draws(
    retained: &RetainedRender,
    (assets, residency): (&AssetLibrary, &AssetResidency),
    plan: &BatchPlan,
    order: &mut DrawOrder,
) {
    let mut keys = std::mem::take(&mut order.keys);
    keys.clear();
    order.looks.clear();
    order.metas.clear();
    for (index, batch) in plan.batches.iter().enumerate() {
        let look = look(retained, assets, batch);
        order.looks.push(look);
        order.metas.push(meta(retained, residency, batch));
        let material = batch.material.and_then(|handle| assets.material(handle));
        let variant = engine_render::material_variant(material, look.extended);
        let buffers = residency
            .mesh(batch.mesh)
            .and_then(ResidentMesh::placement)
            .map_or(u8::MAX, |placement| placement.streams as u8);
        let flat = flat_batch(retained, plan, batch);
        keys.push((
            u8::from(flat),
            variant,
            batch.material,
            buffers,
            index as u32,
        ));
    }
    keys.sort_unstable();
    order.order.clear();
    order.order.extend(keys.iter().map(|&(.., index)| index));
    order.masked.clear();
    order.masked.extend(
        keys.iter()
            .filter(|key| order.looks[key.4 as usize].masked)
            .map(|&(.., index)| index),
    );
    order.keys = keys;
    order.generation += 1;
}
