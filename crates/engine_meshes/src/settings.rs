use bevy_ecs::prelude::*;
use engine_core::{AssetResidency, Cut, Detail, ImpostorRecords, settings, stats};

use crate::assets::MeshAssets;
use crate::batches::BatchPlan;
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
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max: Option<u32> = None => budget,
    }
}

#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    engine_core::__private::serde::Serialize,
    engine_core::__private::serde::Deserialize,
    engine_core::__private::schemars::JsonSchema,
)]
#[serde(crate = "engine_core::__private::serde", default)]
#[schemars(crate = "engine_core::__private::schemars")]
pub struct DetailStats {
    pub lod0_instances: Option<u32>,
    pub lod1_instances: Option<u32>,
    pub lod2_instances: Option<u32>,
    pub impostor_instances: Option<u32>,
    pub impostors: u32,
    pub meshes_without_levels: u32,
    pub impostor_cut: Cut,
}

#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    engine_core::__private::serde::Serialize,
    engine_core::__private::serde::Deserialize,
    engine_core::__private::schemars::JsonSchema,
)]
#[serde(crate = "engine_core::__private::serde", default)]
#[schemars(crate = "engine_core::__private::schemars")]
pub struct CullingStats {
    pub active: bool,
    pub instances: Option<u32>,
}

stats! {
    #[section("meshes_stats")]
    pub struct MeshStats {
        pub invalid_transforms: u32,
        pub render_candidates: u32,
        pub visible_instances: u32,
        pub culled_instances: u32,
        pub dropped_instances: u32,
        pub missing_assets: u32,
        pub culling_enabled: bool,
        pub instances: u32,
        pub gpu_visible: Option<u32>,
        pub triangles: Option<u32>,
        pub read_back_age: Option<u32>,
        pub level_of_detail: DetailStats,
        pub occlusion_culling: CullingStats,
        pub subpixel_culling: CullingStats,
        pub casters: Option<u32>,
        pub moving_casters: Option<u32>,
    }
}

fn counted(value: usize) -> u32 {
    value.min(u32::MAX as usize) as u32
}

fn plain_triangles(batches: &BatchPlan, residency: &AssetResidency) -> u32 {
    let triangles: u64 = batches
        .batches
        .iter()
        .map(|batch| {
            u64::from(batch.instances.end - batch.instances.start)
                * u64::from(
                    residency
                        .mesh(batch.mesh)
                        .map_or(0, |mesh| mesh.index_count())
                        / 3,
                )
        })
        .sum();
    triangles.min(u64::from(u32::MAX)) as u32
}

#[expect(
    clippy::too_many_arguments,
    reason = "the stats join the retained counters, the plan, the residency and both readbacks"
)]
pub(crate) fn mesh_stats(
    (retained, batches, plan): (Res<RetainedRender>, Res<BatchPlan>, Res<MeshPlan>),
    (culling, occlusion, subpixel, detail): (
        Res<Culling>,
        Res<OcclusionCulling>,
        Res<SubpixelCulling>,
        Res<Detail>,
    ),
    residency: Res<AssetResidency>,
    totals: Res<PresentationTotals>,
    casters: Res<CasterTotals>,
    impostors: Query<&ImpostorRecords, With<MeshAssets>>,
    frame: Res<engine_core::FrameTime>,
    mut stats: ResMut<MeshStats>,
) {
    let snapshot = retained.snapshot();
    let gpu = plan.gpu;
    let last = totals.last().filter(|_| gpu);
    let budget = culling.budget();
    let instance_count = batches.ordered_instances.len();
    let gpu_visible = last.map(|totals| totals.visible);
    let instances = match gpu_visible {
        Some(visible) if gpu => visible.min(counted(budget)),
        None if gpu => instance_count.min(budget) as u32,
        _ => instance_count as u32,
    };
    let listed = counted(snapshot.visible_instances);
    let (visible, culled, dropped) = if gpu {
        let visible = gpu_visible.unwrap_or(listed);
        (
            visible,
            listed.saturating_sub(visible),
            visible.saturating_sub(instances),
        )
    } else {
        (listed, 0, counted(snapshot.dropped_instances))
    };
    let levels = last
        .filter(|_| detail.enabled)
        .map(|totals| totals.levels);
    let records = impostors.single().ok();
    let impostor_cut = records.map_or_else(Cut::default, |records| {
        Cut::of(records.asked, records.asked.saturating_sub(records.refused))
    });
    let next = MeshStats {
        invalid_transforms: counted(snapshot.invalid_transforms),
        render_candidates: counted(snapshot.render_candidates),
        visible_instances: visible,
        culled_instances: culled,
        dropped_instances: dropped,
        missing_assets: counted(snapshot.missing_assets),
        culling_enabled: culling.frustum,
        instances,
        gpu_visible,
        triangles: if gpu {
            last.map(|totals| totals.triangles)
        } else {
            Some(plain_triangles(&batches, &residency))
        },
        read_back_age: totals
            .age(frame.frame)
            .filter(|_| gpu)
            .map(|age| age.min(u64::from(u32::MAX)) as u32),
        level_of_detail: DetailStats {
            lod0_instances: levels.map(|levels| levels[0]),
            lod1_instances: levels.map(|levels| levels[1]),
            lod2_instances: levels.map(|levels| levels[2]),
            impostor_instances: levels.map(|levels| levels[3]),
            impostors: records.map_or(0, |records| records.live),
            meshes_without_levels: residency
                .meshes()
                .filter(|(_, mesh)| mesh.without_levels())
                .count() as u32,
            impostor_cut,
        },
        occlusion_culling: CullingStats {
            active: gpu && plan.occlusion,
            instances: last
                .filter(|_| occlusion.enabled)
                .map(|totals| totals.occluded),
        },
        subpixel_culling: CullingStats {
            active: gpu && plan.subpixel,
            instances: last
                .filter(|_| subpixel.enabled)
                .map(|totals| totals.subpixel),
        },
        casters: casters.last().map(|counts| counts[0]),
        moving_casters: casters.last().map(|counts| counts[1]),
    };
    stats.set_if_neq(next);
}
