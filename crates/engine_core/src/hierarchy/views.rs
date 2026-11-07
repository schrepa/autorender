use bevy_ecs::entity::Entity;

use super::{GlobalTransforms, Link, Revision};
use crate::{NodePose, PoseRows};

impl GlobalTransforms {
    #[inline]
    pub(crate) fn locals(&self) -> &[NodePose] {
        &self.locals
    }

    #[inline]
    pub(crate) fn local_revisions(&self) -> &[u64] {
        &self.local_revisions
    }

    #[inline]
    pub(crate) fn global_rows(&self) -> &[PoseRows] {
        &self.globals
    }

    #[inline]
    pub(crate) fn links(&self) -> &[Link] {
        &self.links
    }

    #[inline]
    pub(crate) fn order(&self) -> &[usize] {
        &self.order
    }

    #[inline]
    pub fn node_index(&self, entity: Entity) -> Option<usize> {
        self.indices.get(&entity).copied()
    }

    #[inline]
    pub fn topology_revision(&self) -> Revision {
        self.topology_revision
    }

    #[inline]
    pub(crate) fn revision(&self) -> u64 {
        self.revision
    }
}
