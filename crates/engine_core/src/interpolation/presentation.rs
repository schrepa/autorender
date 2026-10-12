//! Presentation reuses cached hierarchy indices. Broad motion blends directly
//! in parent-first order; sparse motion walks affected child links. Both paths
//! share composition into the canonical presented store; no ECS output mirrors.

use super::*;

const NO_LOCAL: u32 = u32::MAX;

/// Per-node traversal flags beside the canonical presented matrix/revision.
#[derive(Clone, Copy)]
pub(super) struct Display {
    // Sparse preparation indexes blended_locals; dense work reads source hints.
    local: u32,
    // Validity of the published matrix; kept until the next commit.
    valid: bool,
    changed: bool,
    snap_branch: bool,
    fallback: bool,
    processed: bool,
    touched: bool,
}

impl Default for Display {
    fn default() -> Self {
        Self {
            local: NO_LOCAL,
            valid: false,
            changed: false,
            snap_branch: false,
            fallback: false,
            processed: false,
            touched: false,
        }
    }
}

#[derive(Clone, Copy)]
struct Inherited {
    global: Mat4,
    global_valid: bool,
    changed: bool,
    snap_branch: bool,
    fallback: bool,
}

impl Display {
    #[inline]
    pub(super) fn valid(&self) -> bool {
        self.valid
    }

    #[inline]
    pub(super) fn set_valid(&mut self, valid: bool) {
        self.valid = valid;
    }

    #[inline]
    pub(super) fn reset(&mut self) {
        // Processed gates dependency traversal, not public pose reads. Keep the
        // canonical matrix/revision until this preparation commits its next
        // value; resetting traversal flags must not overwrite the last pose.
        self.local = NO_LOCAL;
        self.changed = false;
        self.snap_branch = false;
        self.fallback = false;
        self.processed = false;
        self.touched = false;
    }
}

impl Inherited {
    fn global(&self) -> Option<Mat4> {
        self.global_valid.then_some(self.global)
    }
}

fn touch<'a>(
    display: &'a mut [Display],
    touched: &mut Vec<usize>,
    index: usize,
) -> &'a mut Display {
    if !display[index].touched {
        display[index].touched = true;
        touched.push(index);
    }
    &mut display[index]
}

impl InterpolationState {
    /// The displayed value a child inherits once this node was processed.
    #[inline]
    fn inherited(&self, index: usize) -> Option<Inherited> {
        let display = &self.display[index];
        display.processed.then(|| Inherited {
            global: self.presented[index],
            global_valid: display.valid,
            changed: display.changed,
            snap_branch: display.snap_branch,
            fallback: display.fallback,
        })
    }

    pub(super) fn prepare_interpolated(
        &mut self,
        hierarchy: &HierarchyScratch,
        alpha: f32,
    ) -> usize {
        let preparation = engine_profile::span("interpolation", "prepare_blended_branches");
        let nodes = hierarchy.links().len();
        let histories = self.history.len();
        // This policy limits full-order visits to twice the opt-in count.
        let dense = histories >= nodes.div_ceil(2);
        if dense {
            // The ordered pass initializes each slot immediately before use.
            // No separate reset/scatter pass or dense touched-index list.
            self.touched_display.clear();
        } else if self.dense_display {
            // Sparse preparation will not revisit every previously dense slot.
            for display in &mut self.display {
                display.reset();
            }
            self.touched_display.clear();
        } else {
            for index in self.touched_display.drain(..) {
                if let Some(display) = self.display.get_mut(index) {
                    display.reset();
                }
            }
        }
        self.dense_display = dense;
        self.resize_display(nodes);
        self.blended_roots.clear();
        self.blended_locals.clear();
        self.display_work.clear();
        if !dense {
            // Touches remain bounded even when one root affects every descendant.
            if self.touched_display.capacity() < nodes {
                self.touched_display.reserve_exact(nodes);
            }
            if self.display_work.capacity() < nodes {
                self.display_work.reserve_exact(nodes);
            }
            if self.blended_roots.capacity() < histories {
                self.blended_roots.reserve_exact(histories);
            }
            if self.blended_locals.capacity() < histories {
                self.blended_locals.reserve_exact(histories);
            }
        }
        let (evaluated, index_reused) = if dense {
            self.prepare_dense(hierarchy, alpha)
        } else {
            let reused = self.prepare_sparse_inputs(hierarchy, alpha);
            (self.prepare_sparse(hierarchy), reused)
        };
        drop(preparation);
        engine_profile::counter("interpolation.dense_preparation", f64::from(dense));
        engine_profile::counter("interpolation.history_index_reused", index_reused as f64);
        engine_profile::counter("interpolation.evaluated_nodes", evaluated as f64);
        if !dense {
            // Sparse branches already committed directly. Remaining rows use the
            // authoritative current value, preserving the last prepared snapshot
            // independently from future authoritative propagation.
            for index in 0..self.display.len() {
                if !self.display[index].processed {
                    self.written_presented +=
                        usize::from(self.commit(index, hierarchy.global(index)));
                }
            }
        }
        engine_profile::counter(
            "interpolation.written_presented",
            self.written_presented as f64,
        );
        hierarchy.valid_count()
    }

    fn prepare_sparse_inputs(&mut self, hierarchy: &HierarchyScratch, alpha: f32) -> usize {
        let mut index_reused = 0;
        for record in 0..self.history.len() {
            let entity = self.history[record].entity;
            let cached = self.history_nodes[record] as usize;
            let index = if hierarchy
                .sources()
                .get(cached)
                .is_some_and(|source| source.entity == entity)
            {
                index_reused += 1;
                Some(cached)
            } else {
                let next = hierarchy.node_index(entity);
                self.history_nodes[record] = next.map_or(u32::MAX, |index| index as u32);
                next
            };
            let Some(index) = index else {
                continue;
            };
            let pose = &hierarchy.locals()[index];
            let current = &Transform::from_pose(pose);
            let interval = &self.intervals[record];
            debug_assert_eq!(
                interval.scales_compatible,
                compatible_scales(&self.previous[record], pose),
                "an interval's cached scale compatibility is current"
            );
            let blends = interval.blends(self.epoch, self.active);
            let snap_branch = !blends;
            let local = if blends && hierarchy.links()[index].has_global() {
                let previous = Transform::from_pose(previous_pose(
                    &self.previous[record],
                    interval,
                    self.epoch,
                    pose,
                ));
                let local = blend(&previous, current, alpha);
                (!same_local_bits(&local, current)).then_some(local)
            } else {
                None
            };
            let local = local.map(|local| {
                let index = self.blended_locals.len();
                self.blended_locals.push(local);
                index as u32
            });
            if snap_branch || local.is_some() {
                let display = touch(&mut self.display, &mut self.touched_display, index);
                display.snap_branch = snap_branch;
                display.local = local.unwrap_or(NO_LOCAL);
                if local.is_some() {
                    self.blended_roots.push(index);
                }
            }
        }

        index_reused
    }

    /// Stream parent-first; only active branches compose matrices. Snap flags
    /// also flow through inactive ancestors so a descendant cannot start a blend
    /// below a reset. Invalid topology never enters either traversal path.
    fn prepare_dense(&mut self, hierarchy: &HierarchyScratch, alpha: f32) -> (usize, usize) {
        let mut evaluated = 0;
        let mut index_reused = 0;
        let mut blocks = 0;
        // Gather only one fixed-size numeric block at a time. There is no second
        // full-pose array or per-frame AoS/SoA repacking pass. Lanes that are not
        // eligible keep stale finite poses whose results are never consumed.
        let mut previous = [Transform::default(); 4];
        let mut current = [Transform::default(); 4];
        for indices in hierarchy.order().chunks(4) {
            let mut eligible = [false; 4];
            let mut own_snap = [false; 4];
            for (lane, &index) in indices.iter().enumerate() {
                self.display[index].reset();
                let hint = hierarchy.history_hints()[index];
                if hint == u32::MAX {
                    continue;
                }
                let record = if self.history_nodes.get(hint as usize) == Some(&(index as u32)) {
                    index_reused += 1;
                    hint as usize
                } else if let Some(record) =
                    self.resolve_record(index, hierarchy.sources()[index].entity)
                {
                    record
                } else {
                    continue;
                };
                let link = &hierarchy.links()[index];
                let pose = &hierarchy.locals()[index];
                let interval = &self.intervals[record];
                debug_assert_eq!(
                    interval.scales_compatible,
                    compatible_scales(&self.previous[record], pose),
                    "an interval's cached scale compatibility is current"
                );
                let blends = interval.blends(self.epoch, self.active);
                own_snap[lane] = !blends;
                if blends && link.topology_valid() && link.has_global() {
                    current[lane] = Transform::from_pose(pose);
                    previous[lane] = Transform::from_pose(previous_pose(
                        &self.previous[record],
                        interval,
                        self.epoch,
                        pose,
                    ));
                    eligible[lane] = true;
                }
            }
            let blended = eligible.into_iter().any(|lane| lane).then(|| {
                blocks += 1;
                super::block::blend_block_matrices(&previous, &current, alpha)
            });
            // A parent's display value/reset/fallback must be committed before
            // any child consumes it, including dependencies inside this block.
            for (lane, &index) in indices.iter().enumerate() {
                let link = &hierarchy.links()[index];
                if !link.topology_valid() {
                    self.written_presented += usize::from(self.commit(index, None));
                    continue;
                }
                let parent = link.parent();
                let parent_display = parent.and_then(|parent| self.inherited(parent));
                let snap =
                    own_snap[lane] || parent.is_some_and(|parent| self.display[parent].snap_branch);
                self.display[index].snap_branch = snap;
                let local = blended
                    .as_ref()
                    .filter(|blended| !snap && eligible[lane] && blended.changed[lane])
                    .map(|blended| blended.matrices[lane]);
                if local.is_some() || parent_display.is_some() {
                    self.compose_display(hierarchy, index, parent_display, local);
                    evaluated += 1;
                } else {
                    self.written_presented +=
                        usize::from(self.commit(index, hierarchy.global(index)));
                }
            }
        }
        engine_profile::counter("interpolation.blend_blocks", blocks as f64);
        (evaluated, index_reused)
    }

    fn prepare_sparse(&mut self, hierarchy: &HierarchyScratch) -> usize {
        let mut evaluated = 0;
        for candidate in 0..self.blended_roots.len() {
            let root = self.blended_roots[candidate];
            if self.display[root].processed {
                continue;
            }
            // Parent candidates own their descendant traversal. A reset above
            // this candidate suppresses interpolation for its entire branch.
            let mut ancestor = Some(root);
            let mut covered = false;
            for _ in 0..=MAX_HIERARCHY_DEPTH {
                let Some(index) = ancestor else {
                    break;
                };
                if self.display[index].snap_branch
                    || (index != root && self.display[index].local != NO_LOCAL)
                {
                    covered = true;
                    break;
                }
                ancestor = hierarchy.links()[index].parent();
            }
            if covered {
                continue;
            }
            self.display_work.push(root);
            while let Some(index) = self.display_work.pop() {
                let parent_display = hierarchy.links()[index]
                    .parent()
                    .and_then(|parent| self.inherited(parent));
                let snap = self.display[index].snap_branch
                    || parent_display.is_some_and(|parent| parent.snap_branch);
                let blended = self.display[index].local;
                let local = if snap || blended == NO_LOCAL {
                    None
                } else {
                    Some(self.blended_locals[blended as usize].matrix())
                };
                touch(&mut self.display, &mut self.touched_display, index);
                self.compose_display(hierarchy, index, parent_display, local);
                evaluated += 1;
                self.display_work.extend(hierarchy.children(index));
            }
        }
        evaluated
    }

    /// `local` is the blended local matrix when this node interpolates.
    #[inline]
    fn compose_display(
        &mut self,
        hierarchy: &HierarchyScratch,
        index: usize,
        parent_display: Option<Inherited>,
        local: Option<Mat4>,
    ) {
        let link = &hierarchy.links()[index];
        let parent_changed = parent_display.is_some_and(|parent| parent.changed);
        let parent_global = if let Some(display) = parent_display {
            display.global()
        } else if let Some(parent) = link.parent() {
            hierarchy.global(parent)
        } else {
            Some(Mat4::IDENTITY)
        };
        let mut fallback = parent_display.is_some_and(|parent| parent.fallback);
        let mut changed = false;
        let global = if link.has_global() && !fallback && (local.is_some() || parent_changed) {
            let local = match local {
                Some(local) => local,
                None => Transform::from_pose(&hierarchy.locals()[index]).matrix(),
            };
            let matrix = parent_global.unwrap_or(Mat4::IDENTITY) * local;
            if valid_global_matrix(matrix) {
                // Only linked children read `changed`, so leaves skip comparing
                // with the authoritative matrix.
                changed = link.has_children() && Some(matrix) != hierarchy.global(index);
                Some(matrix)
            } else {
                fallback = true;
                hierarchy.global(index)
            }
        } else {
            hierarchy.global(index)
        };
        // Commit once. The common blended case no longer copies the current
        // matrix into scratch only to overwrite it with the interpolated one.
        let display = &mut self.display[index];
        display.snap_branch |= parent_display.is_some_and(|parent| parent.snap_branch);
        display.fallback = fallback;
        display.changed = changed;
        display.processed = true;
        self.written_presented += usize::from(self.commit(index, global));
    }
}
