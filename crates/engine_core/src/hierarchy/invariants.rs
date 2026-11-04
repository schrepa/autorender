use super::{GlobalTransforms, MAX_HIERARCHY_DEPTH};

impl GlobalTransforms {
    pub(super) fn check(&self) {
        let nodes = self.sources.len();
        for (name, len) in [
            ("entities", self.entities.len()),
            ("locals", self.locals.len()),
            ("local revisions", self.local_revisions.len()),
            ("links", self.links.len()),
            ("globals", self.globals.len()),
            ("revisions", self.revisions.len()),
            ("dirty flags", self.dirty.flags.len()),
            ("order", self.order.len()),
        ] {
            assert!(len == nodes, "hierarchy_lengths_match {nodes} {len} {name}");
        }
        assert!(
            self.indices.len() == nodes,
            "hierarchy_indexed_match {nodes} {}",
            self.indices.len()
        );
        for (index, source) in self.sources.iter().enumerate() {
            assert!(
                self.indices.get(&source.entity) == Some(&index)
                    && self.entities[index] == source.entity,
                "hierarchy_entity_indexed {index} {:?}",
                source.entity
            );
        }
        let mut position = vec![usize::MAX; nodes];
        for (at, &index) in self.order.iter().enumerate() {
            assert!(
                index < nodes && position[index] == usize::MAX,
                "hierarchy_order_once {index}"
            );
            position[index] = at;
        }
        for (index, link) in self.links.iter().enumerate() {
            let parent = self.sources[index]
                .parent
                .and_then(|parent| self.indices.get(&parent).copied());
            assert!(
                link.parent() == parent,
                "hierarchy_parent_linked {index} {:?} {parent:?}",
                link.parent()
            );
            if !link.topology_valid() {
                assert!(
                    !link.has_global() && !link.has_children(),
                    "hierarchy_invalid_node_bare {index}"
                );
                continue;
            }
            let depth = match link.parent() {
                None => 0,
                Some(parent) => {
                    let above = &self.links[parent];
                    assert!(
                        above.topology_valid() && position[parent] < position[index],
                        "hierarchy_parent_first {index} {parent}"
                    );
                    usize::from(above.depth()) + 1
                }
            };
            assert!(
                usize::from(link.depth()) == depth && depth <= MAX_HIERARCHY_DEPTH,
                "hierarchy_depth_cached {index} {} {depth}",
                link.depth()
            );
            let children = self.children(index).collect::<Vec<_>>();
            assert!(
                children
                    .iter()
                    .all(|&child| self.links[child].parent() == Some(index)),
                "hierarchy_children_own {index}"
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
            "hierarchy_links_match_parents {linked} {parented}"
        );
        let invalid = self.links.iter().filter(|link| !link.has_global()).count();
        assert!(
            self.report.invalid_transforms == invalid + self.report.overflow_transforms,
            "hierarchy_invalid_counted {} {invalid} {}",
            self.report.invalid_transforms,
            self.report.overflow_transforms
        );
    }
}
