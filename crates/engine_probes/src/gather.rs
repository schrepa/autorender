use bevy_ecs::prelude::*;
use engine_core::{GlobalTransforms, TransformIndex, View};
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
    hints: Vec<TransformIndex>,
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
    for (position, (entity, probe)) in probes.iter().enumerate() {
        if !crate::component::valid(probe) {
            continue;
        }
        if gathered.hints.len() <= position {
            gathered
                .hints
                .resize(position + 1, TransformIndex::default());
        }
        let Some(center) = globals
            .get_indexed(entity, &mut gathered.hints[position])
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

#[derive(Default)]
struct Nearest {
    keys: Vec<u64>,
}

const POSITION: u64 = u32::MAX as u64;

impl Nearest {
    fn nearest_first(
        &mut self,
        distances: impl IntoIterator<Item = f32>,
        kept: usize,
    ) -> impl ExactSizeIterator<Item = usize> + '_ {
        self.keys.clear();
        self.keys.extend(
            distances
                .into_iter()
                .enumerate()
                .map(|(position, distance)| {
                    u64::from(total_order(distance)) << 32 | position as u64
                }),
        );
        if kept < self.keys.len() {
            self.keys.select_nth_unstable(kept);
            self.keys.truncate(kept);
        }
        self.keys.sort_unstable();
        self.keys.iter().map(|key| (key & POSITION) as usize)
    }
}

fn total_order(value: f32) -> u32 {
    let bits = value.to_bits();
    if bits >> 31 == 1 {
        !bits
    } else {
        bits | 1 << 31
    }
}
