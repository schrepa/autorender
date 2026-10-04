//! What the per-feature extractions (lights, particles, decals and
//! reflection probes) share: where a frame reads world poses, and their
//! nearest-wins budgets (decision 029).

use bevy_ecs::prelude::{Entity, World};
use glam::Mat4;

use crate::{GlobalTransforms, PresentedTransforms, TransformIndex};

/// Where an extraction reads entities' world matrices: the poses prepared
/// for the frame or, on GPU-presented frames, which prepare none, the
/// authoritative ones.
#[derive(Clone, Copy)]
pub(crate) enum Poses<'w> {
    Prepared(&'w PresentedTransforms),
    Authoritative(&'w GlobalTransforms),
}

impl<'w> Poses<'w> {
    pub(crate) fn of(world: &'w World, gpu: bool) -> Self {
        if gpu {
            Self::Authoritative(world.resource())
        } else {
            Self::Prepared(world.resource())
        }
    }

    /// `entity`'s valid world matrix.
    pub(crate) fn model(self, entity: Entity) -> Option<Mat4> {
        match self {
            Self::Prepared(poses) => poses.get(entity)?.matrix(),
            Self::Authoritative(poses) => poses.get(entity)?.matrix(),
        }
    }

    /// [`Self::model`], found through the store hint `hint`.
    pub(crate) fn model_indexed(self, entity: Entity, hint: &mut TransformIndex) -> Option<Mat4> {
        match self {
            Self::Prepared(poses) => poses.get_indexed(entity, hint)?.matrix(),
            Self::Authoritative(poses) => poses.get_indexed(entity, hint)?.matrix(),
        }
    }
}

/// Ranks candidates for a nearest-wins budget: by distance, equally near
/// ones in their order.
#[derive(Default)]
pub(crate) struct Nearest {
    // Each candidate's distance, ordered as `f32::total_cmp` orders it, in
    // the high bits and its position in the low ones: sorting eight-byte
    // keys, not the candidates.
    keys: Vec<u64>,
}

const POSITION: u64 = u32::MAX as u64;

impl Nearest {
    /// The positions of the `kept` candidates nearest by `distances` (every
    /// one when fewer), nearest first.
    pub(crate) fn nearest_first(
        &mut self,
        distances: impl IntoIterator<Item = f32>,
        kept: usize,
    ) -> impl ExactSizeIterator<Item = usize> + '_ {
        self.select(distances, kept);
        self.keys.sort_unstable();
        self.positions()
    }

    /// [`Self::nearest_first`]'s positions in the candidates' order; every
    /// candidate, without ranking, when the budget keeps them all.
    pub(crate) fn kept_in_order(
        &mut self,
        distances: impl ExactSizeIterator<Item = f32>,
        kept: usize,
    ) -> impl ExactSizeIterator<Item = usize> + '_ {
        if distances.len() <= kept {
            self.keys.clear();
            self.keys.extend(0..distances.len() as u64);
        } else {
            self.select(distances, kept);
            self.keys.sort_unstable_by_key(|key| key & POSITION);
        }
        self.positions()
    }

    fn select(&mut self, distances: impl IntoIterator<Item = f32>, kept: usize) {
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
    }

    fn positions(&self) -> impl ExactSizeIterator<Item = usize> + '_ {
        self.keys.iter().map(|key| (key & POSITION) as usize)
    }
}

/// `value`'s bits ordered as [`f32::total_cmp`] orders values.
fn total_order(value: f32) -> u32 {
    let bits = value.to_bits();
    if bits >> 31 == 1 {
        !bits
    } else {
        bits | 1 << 31
    }
}
