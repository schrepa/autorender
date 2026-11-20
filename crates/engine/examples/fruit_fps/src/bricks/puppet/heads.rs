use std::collections::BTreeMap;

use engine::prelude::*;
use engine_core::math::max;
use engine_core::{Asset, Assets, Drawn, MeshFile};
use engine_meshpack::POSITION;

use crate::names::HEAD;
use crate::play::puppet::HeadCheck;
use crate::play::skinned::{Figure, JointAt};

#[derive(Default)]
pub(super) struct Crowns {
    known: BTreeMap<Asset<MeshFile>, f32>,
}

fn sliced(mesh: &MeshFile, bound: Vec4, check: HeadCheck) -> Option<f32> {
    let above: Vec<[f32; 3]> = mesh
        .floats3(POSITION, "$")
        .ok()?
        .into_iter()
        .filter(|at| at[1] > bound.y)
        .collect();
    let top = above.iter().map(|at| at[1]).reduce(max)?;
    let slice = max(check.slice_m, 1.0e-3);
    let count = ((top - bound.y) / slice) as usize + 1;
    let mut widths = vec![0.0_f32; count];
    for at in &above {
        let band = (((top - at[1]) / slice) as usize).min(count - 1);
        let reach = Vec3::new(at[0] - bound.x, 0.0, at[2] - bound.z).length();
        widths[band] = max(widths[band], reach);
    }
    let widest = widths.iter().copied().reduce(max)?;
    let band = widths
        .iter()
        .position(|width| *width >= widest * check.solid_share)?;
    Some(top - band as f32 * slice)
}

impl Crowns {
    pub(super) fn keep(&mut self, assets: &Assets) {
        self.known
            .retain(|mesh, _| assets.revision_of(*mesh).is_some());
    }

    pub(super) fn crown(
        &mut self,
        assets: &Assets,
        (figure, drawn): (&Figure, &Drawn),
        (check, head, fit): (HeadCheck, &JointAt, f32),
    ) -> Option<Vec3> {
        let skeleton = assets.get(figure.skeleton).ok()?;
        let inverse_bind = skeleton.joints().get(skeleton.find(HEAD)?)?.inverse_bind;
        let bound = inverse_bind.inverse().w_axis;
        let mesh = drawn.geometry?;
        let crown = match self.known.get(&mesh) {
            Some(known) => *known,
            None => {
                let found = sliced(assets.get(mesh).ok()?, bound, check)?;
                self.known.insert(mesh, found);
                found
            }
        };
        let local = inverse_bind.transform_point3(Vec3::new(bound.x, crown, bound.z));
        Some(head.position + head.rotation * (local * fit))
    }
}
