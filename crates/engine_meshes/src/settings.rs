use bevy_ecs::prelude::*;
use engine_core::{AssetResidency, Cut, Detail, DrawnTable, settings, stats};
use engine_places::impostors::ImpostorRecords;

use crate::assets::MeshAssets;
use crate::MESH;
use crate::order::MeshPlan;
use crate::presentation::PresentationTotals;

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
        self.instances
            .map_or(usize::MAX, |instances| instances as usize)
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
        pub views: Option<u32>,
        pub refused_views: Option<u32>,
    }
}

fn counted(value: usize) -> u32 {
    value.min(u32::MAX as usize) as u32
}

pub(crate) fn mesh_stats(
    (table, plan): (Res<DrawnTable>, Res<MeshPlan>),
    (culling, occlusion, subpixel, detail): (
        Res<Culling>,
        Res<OcclusionCulling>,
        Res<SubpixelCulling>,
        Res<Detail>,
    ),
    residency: Res<AssetResidency>,
    totals: Res<PresentationTotals>,
    impostors: Query<&ImpostorRecords, With<MeshAssets>>,
    frame: Res<engine_core::FrameTime>,
    mut stats: ResMut<MeshStats>,
) {
    let last = totals.last();
    let budget = culling.budget();
    let census = table.census(MESH);
    let instance_count = census.drawn as usize;
    let gpu_visible = last.map(|totals| totals.visible);
    let instances = gpu_visible.map_or(instance_count.min(budget) as u32, |visible| {
        visible.min(counted(budget))
    });
    let listed = census.drawn;
    let visible = gpu_visible.unwrap_or(listed);
    let levels = last.filter(|_| detail.enabled).map(|totals| totals.levels);
    let records = impostors.single().ok();
    let impostor_cut = records.map_or_else(Cut::default, |records| {
        Cut::of(records.asked, records.asked.saturating_sub(records.refused))
    });
    let next = MeshStats {
        invalid_transforms: census.unplaced,
        render_candidates: census.drawn + census.unplaced + census.waiting,
        visible_instances: visible,
        culled_instances: listed.saturating_sub(visible),
        dropped_instances: visible.saturating_sub(instances),
        missing_assets: census.waiting + census.refused,
        culling_enabled: culling.frustum,
        instances,
        gpu_visible,
        triangles: last.map(|totals| totals.triangles),
        read_back_age: totals
            .age(frame.frame)
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
            active: plan.occlusion,
            instances: last
                .filter(|_| occlusion.enabled)
                .map(|totals| totals.occluded),
        },
        subpixel_culling: CullingStats {
            active: plan.subpixel,
            instances: last
                .filter(|_| subpixel.enabled)
                .map(|totals| totals.subpixel),
        },
        views: totals.views().map(|views| views.drawn),
        refused_views: totals.views().map(|views| views.refused),
    };
    stats.set_if_neq(next);
}
