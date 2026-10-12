//! What a propagated hierarchy holds, checked after every propagation in
//! debug builds: its node arrays have one entry per admitted node; its
//! entity index and its sources name each other; its order is every node
//! once, parents first; each node's cached parent, depth and children are
//! what its source and its parent's say, and a deferred global is a leaf's
//! over a stored one; and its report counts the invalid nodes it holds.

use super::{Global, GlobalTransforms, Link, MAX_HIERARCHY_DEPTH};

impl GlobalTransforms {
    /// Panics naming the first invariant a propagation left broken.
    pub(super) fn check(&self) {
        let nodes = self.sources.len();
        for (name, len) in [
            ("entities", self.entities.len()),
            ("locals", self.locals.len()),
            ("local revisions", self.local_revisions.len()),
            ("history hints", self.history_hints.len()),
            ("links", self.links.len()),
            ("globals", self.globals.len()),
            ("certificates", self.certificates.len()),
            ("revisions", self.revisions.len()),
            ("dirty flags", self.dirty.flags.len()),
            ("order", self.order.len()),
        ] {
            assert!(len == nodes, "hierarchy: {nodes} sources but {len} {name}");
        }
        assert!(
            self.indices.len() == nodes,
            "hierarchy: {nodes} sources but {} indexed entities",
            self.indices.len()
        );
        for (index, source) in self.sources.iter().enumerate() {
            assert!(
                self.indices.get(&source.entity) == Some(&index)
                    && self.entities[index] == source.entity,
                "hierarchy: node {index}'s entity {:?} is not indexed as it",
                source.entity
            );
        }
        // Every node once, each valid one after its parent.
        let mut position = vec![usize::MAX; nodes];
        for (at, &index) in self.order.iter().enumerate() {
            assert!(
                index < nodes && position[index] == usize::MAX,
                "hierarchy: node {index} is out of range or twice in the order"
            );
            position[index] = at;
        }
        for (index, link) in self.links.iter().enumerate() {
            let parent = self.sources[index]
                .parent
                .and_then(|parent| self.indices.get(&parent).copied());
            assert!(
                link.parent() == parent,
                "hierarchy: node {index} links parent {:?}, its source names {parent:?}",
                link.parent()
            );
            if !link.topology_valid() {
                assert!(
                    link.global == Global::Invalid && !link.has_children(),
                    "hierarchy: node {index}, outside valid topology, has a global or children"
                );
                continue;
            }
            let depth = match link.parent() {
                None => 0,
                Some(parent) => {
                    let above = &self.links[parent];
                    assert!(
                        above.topology_valid() && position[parent] < position[index],
                        "hierarchy: node {index}'s parent {parent} is invalid or after it in the order"
                    );
                    usize::from(above.depth()) + 1
                }
            };
            assert!(
                usize::from(link.depth()) == depth && depth <= MAX_HIERARCHY_DEPTH,
                "hierarchy: node {index} caches depth {}, its ancestry gives {depth}",
                link.depth()
            );
            // A deferred leaf composes its global from its parent's stored one
            // (a root's from identity) on every read.
            if link.global == Global::Deferred {
                assert!(
                    !link.has_children()
                        && link
                            .parent()
                            .is_none_or(|parent| self.links[parent].global == Global::Stored),
                    "hierarchy: node {index} defers its global with children, or its parent's is                      not stored"
                );
            }
            let children = self.children(index).collect::<Vec<_>>();
            assert!(
                children
                    .iter()
                    .all(|&child| self.links[child].parent() == Some(index)),
                "hierarchy: node {index} lists a child that is not its own"
            );
        }
        let linked = (0..nodes)
            .map(|index| self.children(index).count())
            .sum::<usize>();
        let parented = self
            .links
            .iter()
            .filter(|link| link.topology_valid() && link.parent().is_some())
            .count();
        assert!(
            linked == parented,
            "hierarchy: {linked} children are linked, {parented} valid nodes have a parent"
        );
        let invalid = self
            .links
            .iter()
            .filter(|link: &&Link| link.global == Global::Invalid)
            .count();
        assert!(
            self.report.invalid_transforms == invalid + self.report.overflow_transforms,
            "hierarchy: the report counts {} invalid transforms, the nodes {invalid} and {} \
             beyond the bound",
            self.report.invalid_transforms,
            self.report.overflow_transforms
        );
    }
}
