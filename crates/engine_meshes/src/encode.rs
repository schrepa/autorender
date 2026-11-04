use engine_render::{Count, Encode, PYRAMID_GROUP, Pass};

use crate::gpu::{Compaction, MeshesGpu};
use crate::layouts::{ARGS_GROUP, CULL_GROUP, GROUP, PARENT_GROUP};

impl MeshesGpu {
    pub(crate) fn encode_pass(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        if pass == self.passes.presentation {
            self.encode_presentation(cx);
        } else if pass == self.passes.occlusion {
            self.encode_occlusion(cx);
        } else if pass == self.passes.casters {
            self.encode_casters(cx);
        } else if pass == self.passes.probes {
            self.encode_probes(cx);
        }
        0
    }

    fn compact(&self, compaction: &Compaction, cx: &mut Encode<'_, Self>) {
        let candidates = self.state.candidates;
        let scan_groups = candidates.div_ceil(GROUP);
        cx.dispatch(&compaction.sum, Count::Groups([scan_groups, 1, 1]));
        cx.dispatch(&compaction.scan, Count::Groups([1, 1, 1]));
        cx.dispatch(
            &compaction.args,
            Count::Groups([self.state.groups.batch_count.div_ceil(ARGS_GROUP), 1, 1]),
        );
        cx.dispatch(&compaction.scatter, Count::Groups([scan_groups, 1, 1]));
    }

    fn encode_presentation(&self, cx: &mut Encode<'_, Self>) {
        if self.state.candidates != 0 {
            cx.clear(self.presented.counts);
        }
        let computes = &self.computes;
        let nodes = &self.state.nodes;
        if nodes.shift {
            cx.dispatch(
                &computes.shift,
                Count::Groups([(nodes.capacity as u32).div_ceil(CULL_GROUP), 1, 1]),
            );
            if !nodes.rows.is_empty() {
                cx.dispatch(&computes.apply, Count::Groups([nodes.rows.len() as u32, 1, 1]));
            }
        }
        for (level, &(start, end)) in nodes.level_ranges.iter().enumerate() {
            if end > start
                && let Some(pipeline) = computes.compose.get(level)
            {
                cx.dispatch(pipeline, Count::Groups([(end - start).div_ceil(PARENT_GROUP), 1, 1]));
            }
        }
        let candidates = self.state.candidates;
        if candidates == 0 {
            return;
        }
        let workgroups = candidates.div_ceil(CULL_GROUP);
        cx.dispatch(&computes.cull, Count::Groups([workgroups, 1, 1]));
        if self.state.stillness.kept() && self.state.stillness.counting {
            cx.dispatch(&computes.count_still, Count::Groups([workgroups, 1, 1]));
        }
        if self.state.lod {
            cx.dispatch(&computes.fades, Count::Groups([workgroups, 1, 1]));
        }
        self.compact(&computes.one, cx);
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
                Count::Groups([width.div_ceil(PYRAMID_GROUP), height.div_ceil(PYRAMID_GROUP), 1]),
            );
        }
        let candidates = self.state.candidates;
        cx.dispatch(
            &computes.occlusion,
            Count::Groups([candidates.div_ceil(CULL_GROUP), 1, 1]),
        );
        self.compact(&computes.two, cx);
        cx.dispatch(&computes.gather_two, Count::Groups([1, 1, 1]));
    }

    fn encode_casters(&self, cx: &mut Encode<'_, Self>) {
        let casters = &self.state.casters;
        if !casters.runs || casters.candidates == 0 {
            return;
        }
        cx.dispatch(&self.computes.select, Count::Groups([casters.candidates.div_ceil(CULL_GROUP), 1, 1]));
        cx.dispatch(&self.computes.count_casters, Count::Groups([1, 1, 1]));
    }

    fn encode_probes(&self, cx: &mut Encode<'_, Self>) {
        let candidates = self.state.candidates;
        for &face in &self.state.faces {
            let face = face as usize;
            let (Some(cull), Some(compaction)) =
                (self.computes.probe.get(face), self.computes.faces.get(face))
            else {
                continue;
            };
            cx.dispatch(cull, Count::Groups([candidates.div_ceil(CULL_GROUP), 1, 1]));
            self.compact(compaction, cx);
        }
    }
}
