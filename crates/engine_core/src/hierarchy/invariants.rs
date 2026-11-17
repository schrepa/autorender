use super::{GlobalTransforms, MAX_HIERARCHY_DEPTH, SEGMENTS, UNPLACED};

impl GlobalTransforms {
    pub(super) fn check(&self) {
        let slots = self.links.len();
        for (name, len) in [
            ("entities", self.entities.len()),
            ("sources", self.sources.len()),
            ("locals", self.locals.len()),
            ("local revisions", self.local_revisions.len()),
            ("globals", self.globals.len()),
            ("revisions", self.revisions.len()),
            ("dirty flags", self.dirty.flags.len()),
            ("positions", self.slots.positions.len()),
            ("visits", self.walk.visits.len()),
            ("declared", self.declared.len()),
            ("words", self.words.len()),
        ] {
            assert!(len == slots, "hierarchy_lengths_match {slots} {len} {name}");
        }
        let live = self.len();
        assert!(
            self.slots.order.len() == live && self.indices.len() == live,
            "hierarchy_live_counted {live} {} {}",
            self.slots.order.len(),
            self.indices.len()
        );
        assert!(
            self.slots.starts[0] == 0 && self.slots.starts[SEGMENTS] == live,
            "hierarchy_segments_cover {live}"
        );
        for (at, &index) in self.slots.order.iter().enumerate() {
            assert!(
                self.slots.positions[index] == at,
                "hierarchy_position_kept {index} {at}"
            );
            let segment = self.links[index].segment();
            assert!(
                self.slots.starts[segment] <= at && at < self.slots.starts[segment + 1],
                "hierarchy_segment_kept {index} {segment}"
            );
        }
        for &index in &self.slots.free {
            assert!(
                self.slots.positions[index] == UNPLACED && !self.links[index].has_children(),
                "hierarchy_free_bare {index}"
            );
        }
        for &index in &self.slots.order {
            let source = &self.sources[index];
            assert!(
                self.indices.get(&source.entity) == Some(&index)
                    && self.entities[index] == source.entity,
                "hierarchy_entity_indexed {index} {:?}",
                source.entity
            );
            let link = &self.links[index];
            let parent = source
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
                assert!(
                    source.parent.is_none() || self.orphans.contains(&index),
                    "hierarchy_orphan_listed {index}"
                );
                continue;
            }
            let depth = match link.parent() {
                None => 0,
                Some(parent) => {
                    let above = &self.links[parent];
                    assert!(
                        above.topology_valid()
                            && self.slots.positions[parent] < self.slots.positions[index],
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
            assert!(
                self.children(index)
                    .all(|child| self.links[child].parent() == Some(index)),
                "hierarchy_children_own {index}"
            );
        }
        let linked = self
            .order
            .iter()
            .map(|&index| self.children(index).count())
            .sum::<usize>();
        let parented = self
            .order
            .iter()
            .filter(|&&index| {
                let link = &self.links[index];
                link.topology_valid() && link.parent().is_some()
            })
            .count();
        assert!(
            linked == parented,
            "hierarchy_links_match_parents {linked} {parented}"
        );
        let invalid = self
            .order
            .iter()
            .filter(|&&index| !self.links[index].has_global())
            .count();
        assert!(
            self.report.invalid_transforms == invalid + self.report.overflow_transforms,
            "hierarchy_invalid_counted {} {invalid} {}",
            self.report.invalid_transforms,
            self.report.overflow_transforms
        );
    }
}
