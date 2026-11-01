use bevy_ecs::prelude::*;
use engine_core::{Cut, settings, stats};

use crate::order::MeshPlan;
use crate::presentation::{CasterTotals, PresentationTotals};
use crate::retained::RetainedRender;

settings! {
    #[section("culling")]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneCulling")]
    pub struct Culling {
        pub frustum: bool = true => any,
        pub instances: Option<u32> = None => at_least(1),
    }
}

impl Culling {
    pub(crate) fn budget(&self) -> usize {
        self.instances.map_or(usize::MAX, |instances| instances as usize)
    }
}

settings! {
    #[section("occlusion_culling")]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneOcclusionCulling")]
    pub struct OcclusionCulling {
        pub enabled: bool = true => any,
    }
}

settings! {
    #[section("subpixel_culling")]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneSubpixelCulling")]
    pub struct SubpixelCulling {
        pub enabled: bool = true => any,
    }
}

settings! {
    #[section("impostors")]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneImpostors")]
    pub struct Impostors {
        pub enabled: bool = true => any,
        pub pixels: f32 = 128.0 => range(8.0, 4096.0),
        pub max: Option<u32> = None => any,
    }
}

stats! {
    #[section("meshes_stats")]
    pub struct MeshStats {
        pub candidates: u32,
        pub instances: u32,
        pub missing_assets: u32,
        pub invalid_transforms: u32,
        pub dropped_instances: u32,
        pub gpu_visible: Option<u32>,
        pub triangles: Option<u32>,
        pub read_back_age: Option<u64>,
        pub occlusion_active: bool,
        pub occluded: Option<u32>,
        pub subpixel_active: bool,
        pub subpixel: Option<u32>,
        pub lod0_instances: Option<u32>,
        pub lod1_instances: Option<u32>,
        pub lod2_instances: Option<u32>,
        pub impostor_instances: Option<u32>,
        pub casters: Option<u32>,
        pub moving_casters: Option<u32>,
        pub impostors: u32,
        pub impostor_cut: Cut,
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "the stats join the retained counters, the plan and both readbacks"
)]
pub(crate) fn mesh_stats(
    retained: Res<RetainedRender>,
    plan: Res<MeshPlan>,
    occlusion: Res<OcclusionCulling>,
    subpixel: Res<SubpixelCulling>,
    totals: Res<PresentationTotals>,
    casters: Res<CasterTotals>,
    frame: Res<engine_core::FrameTime>,
    mut stats: ResMut<MeshStats>,
) {
    let snapshot = retained.snapshot();
    let gpu = plan.gpu;
    let last = totals.last().filter(|_| gpu);
    let counted = |value: usize| u32::try_from(value).unwrap_or(u32::MAX);
    let next = MeshStats {
        candidates: counted(snapshot.render_candidates),
        instances: counted(retained.visible_indices().len()),
        missing_assets: counted(snapshot.missing_assets),
        invalid_transforms: counted(snapshot.invalid_transforms),
        dropped_instances: counted(snapshot.dropped_instances),
        gpu_visible: last.map(|totals| totals.visible),
        triangles: last.map(|totals| totals.triangles),
        read_back_age: totals.age(frame.frame).filter(|_| gpu),
        occlusion_active: gpu && plan.occlusion,
        occluded: last
            .filter(|_| occlusion.enabled)
            .map(|totals| totals.occluded),
        subpixel_active: gpu && plan.subpixel,
        subpixel: last.filter(|_| subpixel.enabled).map(|totals| totals.subpixel),
        lod0_instances: last.map(|totals| totals.levels[0]),
        lod1_instances: last.map(|totals| totals.levels[1]),
        lod2_instances: last.map(|totals| totals.levels[2]),
        impostor_instances: last.map(|totals| totals.levels[3]),
        casters: casters.last().map(|counts| counts[0]),
        moving_casters: casters.last().map(|counts| counts[1]),
        impostors: 0,
        impostor_cut: Cut::default(),
    };
    stats.set_if_neq(next);
}
