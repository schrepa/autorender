use bevy_ecs::prelude::*;
use engine_core::{GlobalTransforms, Nearest, View};
use engine_places::probes::{Interior, ReflectionProbe};
use glam::Vec3;

use crate::ReflectionProbeSettings;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RenderReflectionProbe {
    pub(crate) key: u64,
    pub(crate) center: Vec3,
    pub(crate) extent: Vec3,
    pub(crate) blend: f32,
    pub(crate) interior: Option<Interior>,
}

#[derive(Resource, Default)]
pub(crate) struct ProbeList {
    pub(crate) list: Vec<RenderReflectionProbe>,
    pub(crate) cut: usize,
    candidates: Vec<(f32, RenderReflectionProbe)>,
    nearest: Nearest,
}

pub(crate) fn gather_probes(
    settings: Res<ReflectionProbeSettings>,
    view: Res<View>,
    globals: Res<GlobalTransforms>,
    probes: Query<(Entity, &ReflectionProbe)>,
    mut gathered: ResMut<ProbeList>,
) {
    let gathered = &mut *gathered;
    gathered.list.clear();
    gathered.cut = 0;
    let eye = view.detail_eye();
    gathered.candidates.clear();
    for (entity, probe) in &probes {
        if !crate::component::valid(probe) {
            continue;
        }
        let Some(center) = globals
            .get(entity)
            .and_then(|pose| pose.matrix())
            .map(|matrix| matrix.w_axis.truncate())
            .filter(|center| center.is_finite())
        else {
            continue;
        };
        let outside = ((eye - center).abs() - probe.extent).max(Vec3::ZERO);
        gathered.candidates.push((
            outside.length_squared(),
            RenderReflectionProbe {
                key: entity.to_bits(),
                center,
                extent: probe.extent,
                blend: probe.blend,
                interior: probe.interior,
            },
        ));
    }
    let kept = settings
        .max_probes
        .map_or(usize::MAX, |budget| budget as usize);
    let candidates = &gathered.candidates;
    let ranked = gathered
        .nearest
        .nearest_first(candidates.iter().map(|(distance, _)| *distance), kept);
    gathered.cut = candidates.len() - ranked.len();
    gathered
        .list
        .extend(ranked.map(|position| candidates[position].1));
}
