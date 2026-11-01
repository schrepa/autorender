use bevy_ecs::prelude::*;
use engine_core::{AlphaMode, AssetLibrary, Detail, MaterialHandle, Mesh, View};
use engine_places::meshes::MOVED_SLOT;
use glam::Vec3;

use crate::batches::{BatchPlan, DrawBatch};
use crate::layouts::LOD_LEVELS;
use crate::presentation::frustum_uniform;
use crate::retained::{RetainedRender, SourceMesh};
use crate::settings::{Culling, OcclusionCulling, SubpixelCulling};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct BatchLook {
    pub masked: bool,
    pub double_sided: bool,
    pub extended: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct MeshLevels {
    pub index_count: u32,
    pub lods: Vec<(u32, u32, f32)>,
    pub plan: Vec<(u32, u32, f32)>,
    pub casters: Vec<(u32, u32)>,
}

impl MeshLevels {
    pub(crate) fn of(source: &SourceMesh) -> Self {
        let index_count = source.mesh.indices().len() as u32;
        let Some(lods) = source.lods.as_deref() else {
            return Self {
                index_count,
                ..Self::default()
            };
        };
        let levels: Vec<(u32, u32, f32)> = lods
            .levels
            .iter()
            .map(|level| (level.first_index, level.index_count, level.error))
            .collect();
        let first = levels.first().copied().filter(|&(_, count, _)| count > 0);
        let empty = levels.last().copied().filter(|&(_, count, _)| count == 0);
        Self {
            index_count,
            plan: first.into_iter().chain(empty).collect(),
            lods: levels,
            casters: lods.casters.clone(),
        }
    }

    pub(crate) fn levels(&self) -> usize {
        1 + self.plan.len().min(LOD_LEVELS as usize - 1)
    }

    pub(crate) fn caster_range(&self, level: usize, own: (u32, u32)) -> (u32, u32) {
        self.casters.get(level).copied().unwrap_or(own)
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct BatchMeta {
    pub index_count: u32,
    pub bounds: (Vec3, Vec3),
    pub lods: Vec<(u32, u32, f32)>,
    pub impostor: f32,
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
    planned: Option<(u64, u64)>,
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct MeshPlan {
    pub gpu: bool,
    pub culling: bool,
    pub occlusion_enabled: bool,
    pub occlusion: bool,
    pub subpixel: bool,
    pub budget: u32,
    pub lod_levels: bool,
}

pub(crate) fn extended(mesh: &Mesh) -> bool {
    mesh.colors().is_some() || mesh.uv1().is_some()
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

fn meta(retained: &RetainedRender, batch: &DrawBatch) -> BatchMeta {
    let source = &retained.snapshot().meshes[batch.source_index];
    let levels = MeshLevels::of(source);
    let bounds = source.mesh.bounds();
    BatchMeta {
        index_count: levels.index_count,
        bounds: (bounds.min(), bounds.max()),
        lods: levels.plan[..levels.levels() - 1].to_vec(),
        impostor: 0.0,
        levels,
    }
}

fn flat_batch(retained: &RetainedRender, plan: &BatchPlan, batch: &DrawBatch) -> bool {
    if plan
        .ordered_instances
        .get(batch.instances.start as usize)
        .is_none()
    {
        return false;
    }
    let bounds = retained.snapshot().meshes[batch.source_index].mesh.bounds();
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
    assets: Res<AssetLibrary>,
    view: Res<View>,
    detail: Res<Detail>,
    culling: Res<Culling>,
    occlusion: Res<OcclusionCulling>,
    subpixel: Res<SubpixelCulling>,
    mut plan: ResMut<BatchPlan>,
    mut order: ResMut<DrawOrder>,
    mut decided: ResMut<MeshPlan>,
) {
    let plan = &mut *plan;
    plan.instance_limit = Some(MOVED_SLOT as usize);
    if plan.prepare_frame(&retained).is_err() {
        plan.batches.clear();
        plan.ordered_instances.clear();
    }
    let key = (plan.generation(), assets.material_revision());
    if order.planned != Some(key) {
        order.planned = Some(key);
        order_draws(&retained, &assets, plan, &mut order);
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
        culling: frustum,
        occlusion_enabled: occlusion.enabled,
        occlusion: gpu && occlusion.enabled && unbudgeted && candidates != 0,
        subpixel: subpixel.enabled && unbudgeted && candidates != 0,
        budget: culling.budget().min(u32::MAX as usize) as u32,
        lod_levels: detail.enabled,
    });
}

fn order_draws(
    retained: &RetainedRender,
    assets: &AssetLibrary,
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
        order.metas.push(meta(retained, batch));
        let material = batch.material.and_then(|handle| assets.material(handle));
        let variant = engine_render::material_variant(material, look.extended);
        let flat = flat_batch(retained, plan, batch);
        keys.push((
            u8::from(flat),
            variant,
            batch.material,
            u8::from(look.extended),
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
