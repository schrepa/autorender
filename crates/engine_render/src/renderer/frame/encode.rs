use super::*;
use crate::parts::Encode;

impl Renderer {
    pub(super) fn plan_graph(
        &self,
        graph: &mut FrameGraph,
        info: &FrameInfo<'_>,
    ) -> Result<(), Error> {
        graph.clear();
        let view = self.view(info);
        for (decl, part) in self.parts.iter() {
            let _owner = crate::memory::owned(decl.brick);
            part.declare(&view, graph);
        }
        graph
            .compile(&self.table)
            .map_err(|invalid| invalid.error(&self.table))
    }

    pub(super) fn encode_passes(
        &mut self,
        graph: &FrameGraph,
        views: &[NodeViews],
        info: &FrameInfo<'_>,
        mut encoder: wgpu::CommandEncoder,
    ) -> (wgpu::CommandEncoder, u32, u8) {
        let mut draws = 0;
        let mut batch = 0;
        let mut timed = None;
        for (node, views) in graph.kept_passes().zip(views) {
            let entry = self.table.entry(node.pass());
            let next = entry.stage.batch();
            if next != batch {
                self.mark_stages(&mut encoder, timed.take(), None);
                encoder = self.hand_batch(encoder, (batch, next));
                batch = next;
            }
            self.mark_stages(&mut encoder, timed.replace(entry.span), Some(entry.span));
            #[cfg(any(not(target_arch = "wasm32"), feature = "profiling"))]
            encoder.push_debug_group(entry.name);
            let _owner = crate::memory::owned(self.parts.decl(entry.part).brick);
            let (core, table) = (&mut self.core, &self.table);
            draws += self
                .parts
                .with_part(entry.part, |part, parts| {
                    #[cfg(feature = "dev")]
                    part.encode_fault(&mut encoder);
                    part.encode(
                        entry.index,
                        &mut Encode {
                            core,
                            parts,
                            table,
                            frame: info,
                            node,
                            views,
                        },
                        &mut encoder,
                    )
                })
                .unwrap_or(0);
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
