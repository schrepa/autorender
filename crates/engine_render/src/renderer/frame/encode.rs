use super::*;
use crate::graph::Plan;

impl Renderer {
    pub(super) fn encode_passes(
        &mut self,
        plan: &Plan,
        mut encoder: wgpu::CommandEncoder,
    ) -> (wgpu::CommandEncoder, u32, u8) {
        let mut draws = 0;
        let mut batch = 0;
        let mut timed = None;
        for planned in &plan.passes {
            let entry = self.graph.entry(planned.pass);
            if planned.batch != batch {
                self.mark_stages(&mut encoder, timed.take(), None);
                encoder = self.hand_batch(encoder, (batch, planned.batch));
                batch = planned.batch;
            }
            self.mark_stages(&mut encoder, timed.replace(entry.label), Some(entry.label));
            #[cfg(any(not(target_arch = "wasm32"), feature = "profiling"))]
            encoder.push_debug_group(entry.label);
            if let Some(held) = self.parts.get_mut(entry.part as usize) {
                let _owner = crate::memory::owned(held.brick);
                let mut core = engine_registry::EncodeCore { pass: entry.local };
                draws += held.held.encode(entry.local, &mut core);
            }
            #[cfg(any(not(target_arch = "wasm32"), feature = "profiling"))]
            encoder.pop_debug_group();
        }
        self.mark_stages(&mut encoder, timed, None);
        (encoder, draws, batch)
    }

    fn mark_stages(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        end: Option<&'static str>,
        begin: Option<&'static str>,
    ) {
        let span = |name: Option<&'static str>| self.core.profiler.span(name?);
        submit::mark_between(encoder, span(end), span(begin));
    }
}
