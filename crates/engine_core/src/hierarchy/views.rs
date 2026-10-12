//! Views of a hierarchy's nodes, by admitted index: what presentation,
//! interpolation and rendering read of [`GlobalTransforms`], and nothing
//! they could write.

use bevy_ecs::entity::Entity;

use super::{GlobalTransforms, Link, Source};
use crate::NodePose;

impl GlobalTransforms {
    /// Each node's source: its entity, parent and admission stamps.
    #[inline]
    pub(crate) fn sources(&self) -> &[Source] {
        &self.sources
    }

    /// Each node's cached local pose, in upload layout.
    #[inline]
    pub(crate) fn locals(&self) -> &[NodePose] {
        &self.locals
    }

    /// The boundary revision at which each node's local last changed.
    #[inline]
    pub(crate) fn local_revisions(&self) -> &[u64] {
        &self.local_revisions
    }

    /// Each node's interpolation record (`u32::MAX`: none).
    #[inline]
    pub(crate) fn history_hints(&self) -> &[u32] {
        &self.history_hints
    }

    /// Each node's topology: its parent, depth, children and global's kind.
    #[inline]
    pub(crate) fn links(&self) -> &[Link] {
        &self.links
    }

    /// The nodes, parents first.
    #[inline]
    pub(crate) fn order(&self) -> &[usize] {
        &self.order
    }

    /// The admitted index of `entity`.
    #[inline]
    pub(crate) fn node_index(&self, entity: Entity) -> Option<usize> {
        self.indices.get(&entity).copied()
    }

    /// The most transformed entities admitted ([`TransformNodeLimit`](super::TransformNodeLimit)).
    #[inline]
    pub(crate) fn node_limit(&self) -> usize {
        self.limit
    }

    /// Advances whenever topology rebuilds: node indices may have moved.
    #[inline]
    pub(crate) fn topology_revision(&self) -> u64 {
        self.topology_revision
    }

    /// Advances whenever a boundary observed a changed source, rebuilt
    /// topology or evaluated nodes: consumers of local and world state may
    /// skip their work otherwise.
    #[inline]
    pub(crate) fn revision(&self) -> u64 {
        self.revision
    }

    /// Both revisions, as consumers' caches key what they derived.
    #[inline]
    pub(crate) fn stamp(&self) -> (u64, u64) {
        (self.topology_revision, self.revision)
    }

    /// Takes the nodes changed since GPU presentation last gathered (`None`:
    /// too many to list), for [`Self::restore_changes`] after the gather.
    pub(crate) fn take_changes(&mut self) -> Option<Vec<u32>> {
        self.changes.take()
    }

    /// Gives back the change log [`Self::take_changes`] took, emptied for
    /// the next boundary.
    pub(crate) fn restore_changes(&mut self, nodes: Option<Vec<u32>>) {
        self.changes.restore(nodes);
    }
}
