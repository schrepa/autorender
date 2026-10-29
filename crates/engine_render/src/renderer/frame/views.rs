use super::*;
use crate::graph::{MAX_COLORS, Node, ResourceId};

const DEBUG_TARGETS: [&str; 3] = ["debug_ids", "debug_count", "debug_terrain"];

impl Renderer {
    pub(super) fn shows_debug_view(&self) -> bool {
        #[cfg(feature = "dev")]
        let forced = self.core.display.probe.forces();
        #[cfg(not(feature = "dev"))]
        let forced = None;
        forced.unwrap_or_else(|| {
            DEBUG_TARGETS
                .into_iter()
                .any(|name| self.parts.wants(self.table.resource(name)))
        })
    }

    pub(super) fn node_views(
        &self,
        graph: &FrameGraph,
        canvas: &wgpu::TextureView,
    ) -> Result<Vec<NodeViews>, Error> {
        graph
            .kept_passes()
            .map(|node| {
                let mut colors: [Option<wgpu::TextureView>; MAX_COLORS] = Default::default();
                for (slot, (resource, _)) in colors.iter_mut().zip(node.colors()) {
                    *slot = Some(self.attachment(node, resource, canvas)?);
                }
                let depth = node
                    .depth()
                    .map(|(resource, _)| self.attachment(node, resource, canvas))
                    .transpose()?;
                let whole = node.colors().any(|(resource, _)| {
                    matches!(self.table.resource_name(resource), "canvas" | "history")
                });
                Ok(NodeViews {
                    colors,
                    depth,
                    region: (!whole).then(|| self.core.display.region.render.pixels()),
                })
            })
            .collect()
    }

    fn attachment(
        &self,
        node: &Node,
        resource: ResourceId,
        canvas: &wgpu::TextureView,
    ) -> Result<wgpu::TextureView, Error> {
        let name = self.table.resource_name(resource);
        let view = match name {
            "canvas" => Some(canvas.clone()),
            "depth" => Some((*self.core.depth.view).clone()),
            _ => self
                .core
                .previous
                .attachment(name)
                .or_else(|| self.core.debug.attachment(name))
                .or_else(|| {
                    self.parts
                        .part(self.table.owner(resource))
                        .and_then(|part| part.attachment(name))
                }),
        };
        view.ok_or_else(|| {
            failed(&format!("$.passes.{}", self.table.entry(node.pass()).name))
                .expected(Expected::Present)
                .got(name)
        })
    }
}
