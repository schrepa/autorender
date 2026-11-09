use bevy_ecs::change_detection::DetectChangesMut;
use bevy_ecs::resource::Resource;
use bevy_ecs::world::World;
use glam::{Mat4, Vec3, Vec4};

use crate::presentation_bounds::MAX_COEFFICIENT;
use crate::{DrawnState, DrawnTable, settings};

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
    pub fn budget(&self) -> u32 {
        self.instances.unwrap_or(u32::MAX)
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

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CullPlan {
    pub frustum: bool,
    pub occlusion: bool,
    pub subpixel: bool,
    pub budget: u32,
    pub over_budget: bool,
}

pub fn frustum_planes(view_projection: Mat4) -> [Vec4; 6] {
    let rows = view_projection.transpose();
    [
        rows.w_axis + rows.x_axis,
        rows.w_axis - rows.x_axis,
        rows.w_axis + rows.y_axis,
        rows.w_axis - rows.y_axis,
        rows.w_axis - rows.z_axis,
        rows.z_axis,
    ]
}

pub fn frustum_bounded(view_projection: Mat4) -> bool {
    view_projection
        .to_cols_array()
        .iter()
        .all(|value| value.is_finite() && value.abs() <= MAX_COEFFICIENT)
        && frustum_planes(view_projection)
            .iter()
            .all(|plane| plane.is_finite())
}

pub fn frustum_sees(planes: &[Vec4], low: Vec3, high: Vec3) -> bool {
    planes.iter().all(|plane| {
        let normal = plane.truncate();
        let support = Vec3::select(normal.cmpge(Vec3::ZERO), high, low);
        normal.dot(support) + plane.w >= 0.0
    })
}

pub(crate) fn settle(world: &mut World) {
    let culling = world.get_resource::<Culling>().cloned().unwrap_or_default();
    let occlusion = world
        .get_resource::<OcclusionCulling>()
        .is_none_or(|occlusion| occlusion.enabled);
    let subpixel = world
        .get_resource::<SubpixelCulling>()
        .is_none_or(|subpixel| subpixel.enabled);
    let drawn = world.get_resource::<DrawnTable>().map_or(0, |table| {
        table
            .states()
            .filter(|(_, state)| matches!(state, DrawnState::Drawn { .. }))
            .count()
    });
    let budget = culling.budget();
    let plan = CullPlan {
        frustum: culling.frustum,
        occlusion,
        subpixel,
        budget,
        over_budget: drawn > budget as usize,
    };
    if let Some(mut held) = world.get_resource_mut::<CullPlan>() {
        held.set_if_neq(plan);
    }
}
