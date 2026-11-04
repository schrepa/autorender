use bevy_ecs::prelude::*;
use engine_core::{Detail, DeviceLimits, View};
use engine_places::meshes::{Instances, MOVED_SLOT};
use engine_render::Resolve;

use crate::batches::BatchPlan;
use crate::presentation::{frustum_uniform, rasterized};
use crate::retained::RetainedRender;
use crate::settings::{Culling, OcclusionCulling, SubpixelCulling};
use crate::uploads::PoseData;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct BatchLook {
    pub masked: bool,
    pub double_sided: bool,
    pub extended: bool,
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct MeshPlan {
    pub culling: bool,
    pub occlusion_enabled: bool,
    pub occlusion: bool,
    pub subpixel_enabled: bool,
    pub subpixel: bool,
    pub budget: u32,
    pub lod_levels: bool,
}

fn instance_limit(limits: &DeviceLimits) -> Option<usize> {
    let record = size_of::<PoseData>().max(6 * size_of::<u32>()) as u64;
    (limits.storage_room != 0)
        .then(|| (limits.storage_room / record).min(u64::from(MOVED_SLOT)) as usize)
}

#[expect(
    clippy::too_many_arguments,
    reason = "the plan joins the retained instances and the culling settings"
)]
pub(crate) fn plan_draws(
    retained: Res<RetainedRender>,
    (view, detail, limits, resolve): (Res<View>, Res<Detail>, Res<DeviceLimits>, Res<Resolve>),
    culling: Res<Culling>,
    occlusion: Res<OcclusionCulling>,
    subpixel: Res<SubpixelCulling>,
    mut plan: ResMut<BatchPlan>,
    mut decided: ResMut<MeshPlan>,
    mut instances: ResMut<Instances>,
) {
    let plan = &mut *plan;
    plan.instance_limit = instance_limit(&limits);
    {
        let _profile = engine_core::profile::span("render", "batch_plan");
        plan.refused = plan.prepare_frame(&retained, view.view_projection()).err();
    }
    if plan.refused.is_none() {
        plan.grow_capacity(retained.instances().len());
    }
    let (capacity, candidates) = (
        plan.instance_capacity as u32,
        plan.ordered.len().min(u32::MAX as usize) as u32,
    );
    if (instances.capacity, instances.candidates) != (capacity, candidates) {
        instances.capacity = capacity;
        instances.candidates = candidates;
    }
    let (planes, absolute) = frustum_uniform(rasterized(&view, resolve.offset, view.temporal()));
    let bounded = absolute.iter().flatten().all(|value| {
        value.is_finite() && *value <= engine_core::presentation_bounds::MAX_COEFFICIENT
    }) && planes.iter().flatten().all(|value| value.is_finite());
    let frustum = culling.frustum && bounded;
    let candidates = plan.ordered.len();
    let unbudgeted = frustum && culling.budget() >= candidates;
    decided.set_if_neq(MeshPlan {
        culling: culling.frustum,
        occlusion_enabled: occlusion.enabled,
        occlusion: occlusion.enabled && unbudgeted && candidates != 0,
        subpixel_enabled: subpixel.enabled,
        subpixel: subpixel.enabled && unbudgeted && candidates != 0,
        budget: culling.budget().min(u32::MAX as usize) as u32,
        lod_levels: detail.enabled,
    });
}
