use bevy_ecs::prelude::*;
use glam::Mat4;

use super::{
    GlobalTransforms, Link, MAX_HIERARCHY_DEPTH, MAX_TRANSFORM_NODES, NO_NODE, Parent, SEGMENTS,
    Source, UNPLACED, pose_rows, words,
};
use crate::{NodeWord, Transform};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Visit {
    Idle,
    Pending,
    Visiting,
    Done,
}

impl GlobalTransforms {
    pub(super) fn lay_order(&mut self) {
        let mut starts = [0; SEGMENTS + 1];
        for link in &self.links {
            starts[link.segment() + 1] += 1;
        }
        for segment in 0..SEGMENTS {
            starts[segment + 1] += starts[segment];
        }
        self.slots.starts = starts;
        self.slots.order.clear();
        self.slots.order.resize(self.links.len(), 0);
        self.slots.positions.clear();
        self.slots.positions.resize(self.links.len(), UNPLACED);
        for (index, link) in self.links.iter().enumerate() {
            let segment = link.segment();
            let at = starts[segment];
            starts[segment] += 1;
            self.slots.order[at] = index;
            self.slots.positions[index] = at;
        }
    }

    fn order_insert(&mut self, index: usize, segment: usize) {
        let mut hole = self.slots.order.len();
        self.slots.order.push(index);
        self.slots.starts[SEGMENTS] += 1;
        for later in (segment + 1..SEGMENTS).rev() {
            let first = self.slots.starts[later];
            if first < hole {
                let moved = self.slots.order[first];
                self.slots.order[hole] = moved;
                self.slots.positions[moved] = hole;
            }
            hole = first;
            self.slots.starts[later] += 1;
        }
        self.slots.order[hole] = index;
        self.slots.positions[index] = hole;
    }

    fn order_remove(&mut self, index: usize) {
        let at = self.slots.positions[index];
        let segment = (0..SEGMENTS)
            .find(|&segment| self.slots.starts[segment] <= at && at < self.slots.starts[segment + 1])
            .expect("ordered_node_in_segment");
        let mut hole = at;
        for later in segment..SEGMENTS {
            let last = self.slots.starts[later + 1] - 1;
            if last != hole {
                let moved = self.slots.order[last];
                self.slots.order[hole] = moved;
                self.slots.positions[moved] = hole;
            }
            hole = last;
            self.slots.starts[later + 1] -= 1;
        }
        self.slots.order.pop();
        self.slots.positions[index] = UNPLACED;
    }

    pub(super) fn link_child(&mut self, parent: usize, child: usize) {
        let first = self.links[parent].first_child;
        self.links[child].next_sibling = first;
        self.links[child].prev_sibling = NO_NODE;
        if let Some(first) = Link::index(first) {
            self.links[first].prev_sibling = child as u32;
        }
        self.links[parent].first_child = child as u32;
    }

    fn unlink(&mut self, index: usize) {
        let link = self.links[index];
        let Some(parent) = link.parent().filter(|_| link.topology_valid) else {
            return;
        };
        match Link::index(link.prev_sibling) {
            Some(prev) => self.links[prev].next_sibling = link.next_sibling,
            None if self.links[parent].first_child == index as u32 => {
                self.links[parent].first_child = link.next_sibling;
            }
            None => {}
        }
        if let Some(next) = link.next_sibling() {
            self.links[next].prev_sibling = link.prev_sibling;
        }
        let link = &mut self.links[index];
        link.next_sibling = NO_NODE;
        link.prev_sibling = NO_NODE;
    }

    fn unsettle(&mut self, index: usize) {
        if self.walk.visits[index] == Visit::Idle {
            self.walk.visits[index] = Visit::Pending;
            self.walk.pending.push(index);
        }
    }

    fn parent_slot(&self, index: usize) -> Option<usize> {
        self.sources[index]
            .parent
            .and_then(|parent| self.indices.get(&parent).copied())
    }

    pub(super) fn restructure(&mut self, world: &World, placed: &mut Vec<Entity>) -> bool {
        if placed.is_empty() {
            return true;
        }
        let _profile = engine_profile::span("hierarchy", "restructure");
        placed.sort_unstable();
        placed.dedup();
        for &entity in placed.iter() {
            let held = world.get_entity(entity).ok();
            let transform = held.and_then(|held| held.get_ref::<Transform>());
            let parent = held
                .and_then(|held| held.get::<Parent>())
                .map(Parent::parent);
            match (self.indices.get(&entity).copied(), transform) {
                (Some(index), Some(transform))
                    if self.sources[index].transform_added == transform.added().get() =>
                {
                    if self.sources[index].parent != parent {
                        self.sources[index].parent = parent;
                        self.unsettle(index);
                    }
                }
                (Some(index), transform) => {
                    self.detach(index);
                    if let Some(transform) = transform
                        && !self.attach(world, entity, &transform, parent)
                    {
                        return false;
                    }
                }
                (None, Some(transform)) => {
                    if !self.attach(world, entity, &transform, parent) {
                        return false;
                    }
                }
                (None, None) => {}
            }
        }
        self.settle_links();
        true
    }

    fn attach(
        &mut self,
        world: &World,
        entity: Entity,
        transform: &Ref<'_, Transform>,
        parent: Option<Entity>,
    ) -> bool {
        if self.len() >= MAX_TRANSFORM_NODES {
            return false;
        }
        let index = match self.slots.free.pop() {
            Some(index) => index,
            None => {
                self.entities.push(Entity::PLACEHOLDER);
                self.sources.push(Source::EMPTY);
                self.locals.push([0.0; 10]);
                self.links.push(Link::DETACHED);
                self.globals.push(pose_rows(Mat4::IDENTITY));
                self.published.push(false);
                self.declared.push(words::Declared::default());
                self.words.push(NodeWord::ROOT);
                self.walk.visits.push(Visit::Idle);
                self.slots.positions.push(UNPLACED);
                self.dirty.flags.push(false);
                self.links.len() - 1
            }
        };
        self.entities[index] = entity;
        self.sources[index] = Source {
            entity,
            parent,
            transform_added: transform.added().get(),
        };
        self.locals[index] = transform.to_pose();
        self.links[index] = Link::DETACHED;
        self.globals[index] = pose_rows(Mat4::IDENTITY);
        self.published[index] = false;
        self.declared[index] = words::Declared::read(world, entity);
        self.words[index] = NodeWord::ROOT;
        self.dirty.flags[index] = false;
        self.indices.insert(entity, index);
        self.order_insert(index, 0);
        self.report.transforms += 1;
        self.report.invalid_transforms += 1;
        self.placed.mark(index as u32);
        self.unsettle(index);
        true
    }

    fn detach(&mut self, index: usize) {
        let mut child = self.links[index].first_child();
        while let Some(at) = child {
            child = self.links[at].next_sibling();
            self.unsettle(at);
        }
        self.unlink(index);
        if !self.links[index].valid {
            self.report.invalid_transforms -= 1;
        }
        self.report.transforms -= 1;
        self.order_remove(index);
        self.orphans.retain(|&orphan| orphan != index);
        self.indices.remove(&self.entities[index]);
        self.entities[index] = Entity::PLACEHOLDER;
        self.sources[index] = Source::EMPTY;
        self.links[index] = Link::DETACHED;
        self.published[index] = false;
        self.declared[index] = words::Declared::default();
        self.words[index] = NodeWord::ROOT;
        self.dirty.flags[index] = false;
        self.slots.free.push(index);
        self.placed.mark(index as u32);
    }

    fn settle_links(&mut self) {
        for orphan in std::mem::take(&mut self.orphans) {
            self.unsettle(orphan);
        }
        let mut at = 0;
        while let Some(&index) = self.walk.pending.get(at) {
            at += 1;
            if !self.live(index) {
                continue;
            }
            let mut child = self.links[index].first_child();
            while let Some(below) = child {
                child = self.links[below].next_sibling();
                self.unsettle(below);
            }
        }
        for at in 0..self.walk.pending.len() {
            let index = self.walk.pending[at];
            if !self.live(index) {
                continue;
            }
            let link = self.links[index];
            if link
                .parent()
                .filter(|_| link.topology_valid)
                .is_some_and(|parent| self.walk.visits[parent] == Visit::Idle)
            {
                self.unlink(index);
            }
        }
        for at in 0..self.walk.pending.len() {
            let index = self.walk.pending[at];
            if self.live(index) {
                let link = &mut self.links[index];
                link.first_child = NO_NODE;
                link.next_sibling = NO_NODE;
                link.prev_sibling = NO_NODE;
            }
        }
        let threshold = self.dense_threshold();
        let mut path = std::mem::take(&mut self.walk.path);
        for at in 0..self.walk.pending.len() {
            let start = self.walk.pending[at];
            if !self.live(start) || self.walk.visits[start] != Visit::Pending {
                continue;
            }
            path.clear();
            let mut cursor = Some(start);
            while let Some(index) = cursor {
                if self.walk.visits[index] != Visit::Pending {
                    break;
                }
                self.walk.visits[index] = Visit::Visiting;
                path.push(index);
                cursor = self.parent_slot(index);
            }
            while let Some(index) = path.pop() {
                self.place(index, threshold);
                self.walk.visits[index] = Visit::Done;
            }
        }
        self.walk.path = path;
        for at in 0..self.walk.pending.len() {
            let index = self.walk.pending[at];
            self.walk.visits[index] = Visit::Idle;
        }
        self.walk.pending.clear();
    }

    fn place(&mut self, index: usize, threshold: usize) {
        let before = self.links[index];
        let parent = self.parent_slot(index);
        let depth = match (self.sources[index].parent, parent) {
            (None, _) => Some(0),
            (Some(_), None) => None,
            (Some(_), Some(parent)) => {
                let above = self.links[parent];
                (self.walk.visits[parent] != Visit::Visiting
                    && above.topology_valid
                    && usize::from(above.depth) < MAX_HIERARCHY_DEPTH)
                    .then_some(above.depth + 1)
            }
        };
        let link = &mut self.links[index];
        link.parent = parent.map_or(NO_NODE, |parent| parent as u32);
        link.topology_valid = depth.is_some();
        link.depth = depth.unwrap_or(0);
        let after = *link;
        match parent {
            Some(parent) if after.topology_valid => self.link_child(parent, index),
            Some(_) | None if self.sources[index].parent.is_some() => self.orphans.push(index),
            Some(_) | None => {}
        }
        if before.segment() != after.segment() {
            self.order_remove(index);
            self.order_insert(index, after.segment());
        }
        let changed = (before.parent, before.depth, before.topology_valid)
            != (after.parent, after.depth, after.topology_valid);
        if changed {
            self.placed.mark(index as u32);
        }
        if changed || !self.published[index] {
            self.dirty.mark(index, threshold);
            self.worded.push(index);
        }
    }
}
