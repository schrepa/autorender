use engine_render::{Attach, Count, Encode, PYRAMID_GROUP, Pass};

use crate::gpu::{Compaction, MeshesGpu};
use crate::layouts::{ARGS_GROUP, CULL_GROUP, GROUP, PARENT_GROUP};
use crate::presentation::PHASE_TWO;

impl MeshesGpu {
    pub(crate) fn encode_pass(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        if pass == self.passes.presentation {
            self.encode_presentation(cx);
        } else if pass == self.passes.occlusion {
            self.encode_occlusion(cx);
        } else if pass == self.passes.views {
            self.encode_views(cx);
        } else if pass == self.passes.occluded {
            return self.encode_occluded(cx);
        }
        0
    }

    fn compact(&self, compaction: &Compaction, offsets: &[u32], cx: &mut Encode<'_, Self>) {
        let candidates = self.state.candidates;
        let scan_groups = candidates.div_ceil(GROUP);
        cx.dispatch_at(&compaction.sum, Count::Groups([scan_groups, 1, 1]), offsets);
        cx.dispatch_at(&compaction.scan, Count::Groups([1, 1, 1]), offsets);
        cx.dispatch_at(
            &compaction.args,
            Count::Groups([self.state.groups.batch_count.div_ceil(ARGS_GROUP), 1, 1]),
            offsets,
        );
        cx.dispatch_at(&compaction.scatter, Count::Groups([scan_groups, 1, 1]), offsets);
    }

    fn encode_presentation(&self, cx: &mut Encode<'_, Self>) {
        if self.state.candidates != 0 {
            cx.clear(self.presented.counts);
        }
        let computes = &self.computes;
        let nodes = &self.state.nodes;
        for (level, &(start, end)) in nodes.level_ranges.iter().enumerate() {
            if end > start
                && let Some(pipeline) = computes.compose.get(level)
            {
                cx.dispatch(
                    pipeline,
                    Count::Groups([(end - start).div_ceil(PARENT_GROUP), 1, 1]),
                );
            }
        }
        let candidates = self.state.candidates;
        if candidates == 0 {
            return;
        }
        let workgroups = candidates.div_ceil(CULL_GROUP);
        cx.dispatch(&computes.cull, Count::Groups([workgroups, 1, 1]));
        if self.state.lod {
            cx.dispatch(&computes.fades, Count::Groups([workgroups, 1, 1]));
        }
        self.compact(&computes.one, &[], cx);
        cx.dispatch(&computes.gather_one, Count::Groups([1, 1, 1]));
    }

    fn encode_occlusion(&self, cx: &mut Encode<'_, Self>) {
        if !self.state.occlusion {
            return;
        }
        let computes = &self.computes;
        for (index, &(width, height)) in self.state.pyramid.written.iter().enumerate() {
            let pipeline = match index.checked_sub(1) {
                None => &computes.pyramid_first,
                Some(down) => match computes.pyramid_down.get(down) {
                    Some(pipeline) => pipeline,
                    None => break,
                },
            };
            cx.dispatch(
                pipeline,
                Count::Groups([
                    width.div_ceil(PYRAMID_GROUP),
                    height.div_ceil(PYRAMID_GROUP),
                    1,
                ]),
            );
        }
        let candidates = self.state.candidates;
        cx.dispatch(
            &computes.occlusion,
            Count::Groups([candidates.div_ceil(CULL_GROUP), 1, 1]),
        );
        self.compact(&computes.two, &[], cx);
        cx.dispatch(&computes.gather_two, Count::Groups([1, 1, 1]));
    }

    fn encode_views(&self, cx: &mut Encode<'_, Self>) {
        let groups = self.state.candidates.div_ceil(CULL_GROUP);
        for slot in &self.state.views {
            let offsets = [slot.frame_at];
            cx.dispatch_at(
                &self.computes.view_cull,
                Count::Groups([groups, 1, 1]),
                &offsets,
            );
            self.compact(&self.computes.views, &offsets, cx);
        }
    }

    fn encode_occluded(&self, cx: &mut Encode<'_, Self>) -> u32 {
        if !self.state.occlusion || self.state.instance_count == 0 {
            return 0;
        }
        let occluded = &self.occluded;
        let mut draw = cx.render(Attach::Declared, None);
        let mut drawn = 0;
        if self.state.draws.masked {
            drawn += self.draw_meshes(
                &mut draw,
                &occluded.streams,
                &occluded.depth,
                (PHASE_TWO, true),
            );
        }
        drawn += self.draw_meshes(
            &mut draw,
            &occluded.streams,
            &occluded.lit,
            (PHASE_TWO, false),
        );
        drawn + self.draw_cards(&mut draw, &occluded.streams, &occluded.card, PHASE_TWO)
    }
}
