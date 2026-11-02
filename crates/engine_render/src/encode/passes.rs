use super::*;

impl Run {
    pub(crate) fn enter(&mut self, pass: u32) -> bool {
        let shares = self
            .machine
            .declared
            .passes
            .get(pass as usize)
            .is_some_and(shared);
        if self.merge && shares && self.open.as_ref().is_some_and(|open| open.shared) {
            return false;
        }
        if self.labelled.is_some_and(|known| self.same_group(known, pass)) {
            self.end_render();
        } else {
            self.leave();
        }
        true
    }

    pub(crate) fn same_group(&self, first: u32, second: u32) -> bool {
        if first == second {
            return true;
        }
        let passes = &self.machine.declared.passes;
        match (passes.get(first as usize), passes.get(second as usize)) {
            (Some(first), Some(second)) => {
                first.part == second.part && first.group.is_some() && first.group == second.group
            }
            (Some(_) | None, _) => false,
        }
    }

    pub(super) fn resolving(&self, pass: u32) -> Option<PassNames> {
        let declared = self.machine.declared.passes.get(pass as usize)?;
        let (names, _) = self.resolve?;
        (self.temporal && declared.name == crate::shared::COMPOSITE).then_some(names)
    }

    pub(crate) fn label_of(&self, pass: u32) -> Option<&'static str> {
        if let Some(names) = self.resolving(pass) {
            return Some(names.label);
        }
        let declared = self.machine.declared.passes.get(pass as usize)?;
        Some(declared.label.unwrap_or(declared.name))
    }

    pub(crate) fn group_of(&self, pass: u32) -> &'static str {
        if let Some(names) = self.resolving(pass) {
            return names.pass;
        }
        self.machine
            .declared
            .passes
            .get(pass as usize)
            .map_or("", |declared| declared.group.unwrap_or(declared.name))
    }

    pub(crate) fn stage_of(&self, pass: u32) -> Option<&'static str> {
        self.resolving(pass).and(self.resolve).map(|(_, stage)| stage)
    }

    pub(crate) fn span_of(&self, pass: u32) -> Option<&'static str> {
        if let Some(names) = self.resolving(pass) {
            return Some(names.span);
        }
        self.machine.declared.passes.get(pass as usize)?.span
    }

    pub(crate) fn open_encode_span(&mut self, timer: &GpuProfiler) {
        let Some(name) = self.timed.asked.take() else {
            return;
        };
        self.end_render();
        if let (Some(span), Some(encoder)) = (self.encode_span(timer, name), self.encoder.as_mut()) {
            span.mark(encoder, false);
        }
        self.timed.open = Some(name);
    }

    fn encode_span<'t>(&self, timer: &'t GpuProfiler, name: &'static str) -> Option<crate::profiling::Span<'t>> {
        let group = self.group_of(self.pass);
        if group == name {
            timer.span(name)
        } else {
            timer.span_in(group, name)
        }
    }

    pub(crate) fn close_encode_span(&mut self, timer: &GpuProfiler) {
        self.timed.asked = None;
        let Some(name) = self.timed.open.take() else {
            return;
        };
        self.end_render();
        if let (Some(span), Some(encoder)) = (self.encode_span(timer, name), self.encoder.as_mut()) {
            span.mark(encoder, true);
        }
    }

    pub(crate) fn flush_clears(&mut self) {
        if self.queued.cleared.is_empty() {
            return;
        }
        self.end_render();
        for buffer in std::mem::take(&mut self.queued.cleared) {
            if let (Some(Some((made, _))), Some(encoder)) = (self.buffers.get(buffer as usize), self.encoder.as_mut()) {
                encoder.clear_buffer(made, 0, None);
            }
        }
    }

    pub(crate) fn group(&mut self, pass: u32) {
        if !GROUPED || self.labelled.is_some_and(|known| self.same_group(known, pass)) {
            return;
        }
        let name = self.group_of(pass);
        if let Some(encoder) = self.encoder.as_mut() {
            if self.labelled.is_some() {
                encoder.pop_debug_group();
            }
            encoder.push_debug_group(name);
            self.labelled = Some(pass);
        }
    }

    pub(crate) fn leave(&mut self) {
        self.end_render();
        if GROUPED
            && self.labelled.take().is_some()
            && let Some(encoder) = self.encoder.as_mut()
        {
            encoder.pop_debug_group();
        }
    }

    fn layer_set(&self, layer: Option<crate::Layer>) -> Arc<[SetFormat]> {
        self.layer_sets
            .iter()
            .find(|(known, _)| *known == layer)
            .map_or_else(
                || Arc::from(set_formats(&self.machine.outputs, layer)),
                |(_, set)| Arc::clone(set),
            )
    }

    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    pub(crate) fn begin_render(&mut self, attach: AttachRaw, view: Option<u32>) -> bool {
        let Some(offset) = self.view_offset(self.part, view) else {
            let error = fault(self.path("render.view"), ErrorCode::OutOfRange);
            self.fault(error);
            return false;
        };
        let Some(pass) = self.pass_decls.get(self.pass as usize).cloned() else {
            return false;
        };
        let Some(plan) = self.plan.clone() else {
            return false;
        };
        let Some(planned) = plan.passes.get(self.planned_at) else {
            return false;
        };
        let layer = self.pass_layer();
        let is_shared = shared(&pass);
        let mut colors = std::mem::take(&mut self.spare.colors);
        colors.clear();
        let mut depth: Option<(
            u64,
            wgpu::TextureView,
            wgpu::TextureFormat,
            Option<Ops>,
            bool,
        )> = None;
        let mut main = false;
        let ops_of = |id: Option<crate::graph::ResourceId>| {
            planned
                .colors
                .iter()
                .find(|(known, _)| Some(*known) == id)
                .map(|(_, ops)| *ops)
        };
        let depth_of = |id: Option<crate::graph::ResourceId>| -> Option<Option<Ops>> {
            planned
                .depth
                .and_then(|(known, ops)| (Some(known) == id).then_some(ops))
        };
        let declared_ops =
            |access: crate::Access, clear: Option<Clear>, fallback: Clear| match access {
                crate::Access::Load => Ops {
                    load: Load::Load,
                    store: true,
                },
                crate::Access::Clear => Ops {
                    load: Load::Clear(clear.unwrap_or(fallback)),
                    store: true,
                },
            };
        let key = target_key;
        match attach {
            AttachRaw::Declared => {
                if let Some(layer) = layer {
                    if !own_colors_only(&pass) {
                        main = !self.debug;
                        if self.debug {
                            for index in 0..self.debug_targets.len() {
                                let target = Target::Debug(index as u32);
                                let ops = ops_of(self.machine.ids.target(target)).unwrap_or(Ops {
                                    load: Load::Load,
                                    store: true,
                                });
                                let made = &self.debug_targets[index];
                                colors.push((
                                    key(target),
                                    Some(made.view.clone()),
                                    made.texture.format(),
                                    ops,
                                ));
                            }
                        } else {
                            let set = self.layer_set(Some(layer));
                            for &(target, format, _, written) in set.iter() {
                                let id = self.machine.ids.target(target);
                                match ops_of(id).filter(|_| written) {
                                    Some(ops) => colors.push((
                                        key(target),
                                        self.view_of(target, true).cloned(),
                                        format,
                                        ops,
                                    )),
                                    None => colors.push((
                                        key(target),
                                        None,
                                        format,
                                        Ops {
                                            load: Load::Load,
                                            store: false,
                                        },
                                    )),
                                }
                            }
                        }
                    }
                    if let (Some(id), Some(made)) = (
                        self.machine.ids.target(Target::Depth),
                        self.view_of(Target::Depth, true),
                    ) {
                        let ops = depth_of(Some(id)).unwrap_or(Some(Ops {
                            load: Load::Load,
                            store: true,
                        }));
                        depth = Some((key(Target::Depth), made.clone(), DEPTH_FORMAT, ops, true));
                    }
                }
                if pass.name == crate::shared::COMPOSITE || pass.name == crate::shared::DEBUG_VIEW {
                    let id = self.machine.ids.target(Target::Canvas);
                    if let (Some(ops), Some(canvas)) = (ops_of(id), self.canvas.clone()) {
                        let format = self.canvas_format();
                        colors.insert(0, (key(Target::Canvas), Some(canvas), format, ops));
                    }
                }
                if pass.name == crate::shared::COMPOSITE {
                    let set = self.layer_set(None);
                    for &(target, format, _, _) in
                        set.iter().skip(1).filter(|(_, _, _, written)| *written)
                    {
                        let id = self.machine.ids.target(target);
                        let ops = ops_of(id).unwrap_or(Ops {
                            load: Load::Load,
                            store: true,
                        });
                        colors.push((
                            key(target),
                            self.view_of(target, true).cloned(),
                            format,
                            ops,
                        ));
                    }
                }
                for &(texture, access) in &pass.colors {
                    let id = self.machine.ids.textures.get(texture as usize).copied();
                    let ops = ops_of(id).unwrap_or_else(|| declared_ops(access, pass.clear_of(texture), Clear::Color(crate::graph::TRANSPARENT)));
                    let format = self.machine.declared.textures[texture as usize].format;
                    let view = self.attachment_view(texture, 0, 0);
                    colors.push((u64::from(texture) + 1_000, view, format, ops));
                }
                if depth.is_none()
                    && let Some(&(texture, access)) = pass.depths.first()
                {
                    let id = self.machine.ids.textures.get(texture as usize).copied();
                    let ops = depth_of(id).unwrap_or_else(|| Some(declared_ops(access, pass.clear_of(texture), Clear::Depth(FAR))));
                    let format = self.machine.declared.textures[texture as usize].format;
                    if let Some(view) = self.attachment_view(texture, 0, 0) {
                        depth = Some((u64::from(texture) + 2_000, view, format, ops, false));
                    }
                }
            }
            AttachRaw::Color(texture, mip, ref layers) => {
                let Some(&(_, access)) = pass.colors.iter().find(|(known, _)| *known == texture) else {
                    return self.mismatch(pass.name);
                };
                let id = self.machine.ids.textures.get(texture as usize).copied();
                let planned = ops_of(id).unwrap_or_else(|| declared_ops(access, pass.clear_of(texture), Clear::Color(crate::graph::TRANSPARENT)));
                let ops = Ops { store: true, ..planned };
                let format = self.machine.declared.textures[texture as usize].format;
                for at in layers.clone() {
                    let view = self.attachment_view(texture, mip, at);
                    let attachment = (u64::from(texture) * 4_096 + u64::from(at)) * 64 + u64::from(mip) + 10_000;
                    colors.push((attachment, view, format, ops));
                }
            }
            AttachRaw::Depth(texture, layer) => {
                if !self.depth_attachment(&pass, texture, layer.unwrap_or(0), &mut depth) {
                    return self.mismatch(pass.name);
                }
            }
            AttachRaw::ColorDepth(color, layer, texture) => {
                let Some(&(_, access)) = pass.colors.iter().find(|(known, _)| *known == color) else {
                    return self.mismatch(pass.name);
                };
                let id = self.machine.ids.textures.get(color as usize).copied();
                let planned = ops_of(id).unwrap_or_else(|| declared_ops(access, pass.clear_of(color), Clear::Color(crate::graph::TRANSPARENT)));
                let ops = Ops { store: true, ..planned };
                let format = self.machine.declared.textures[color as usize].format;
                let at = layer.unwrap_or(0);
                let view = self.attachment_view(color, 0, at);
                colors.push(((u64::from(color) * 4_096 + u64::from(at)) * 64 + 10_000, view, format, ops));
                if !self.depth_attachment(&pass, texture, 0, &mut depth) {
                    return self.mismatch(pass.name);
                }
            }
            AttachRaw::Colors(ref attached) => {
                for &(texture, at) in attached {
                    let Some(&(_, access)) = pass.colors.iter().find(|(known, _)| *known == texture) else {
                        return self.mismatch(pass.name);
                    };
                    let id = self.machine.ids.textures.get(texture as usize).copied();
                    let planned = ops_of(id).unwrap_or_else(|| {
                        declared_ops(
                            access,
                            pass.clear_of(texture),
                            Clear::Color(crate::graph::TRANSPARENT),
                        )
                    });
                    let ops = Ops {
                        store: true,
                        ..planned
                    };
                    let format = self.machine.declared.textures[texture as usize].format;
                    let view = self.attachment_view(texture, 0, at);
                    colors.push((
                        (u64::from(texture) * 4_096 + u64::from(at)) * 64 + 10_000,
                        view,
                        format,
                        ops,
                    ));
                }
            }
        }
        let mut attachments = std::mem::take(&mut self.spare.attachments);
        attachments.clear();
        attachments.extend(
            colors
                .iter()
                .map(|(key, ..)| *key)
                .chain(depth.as_ref().map(|(key, ..)| *key)),
        );
        if let Some(open) = &mut self.open
            && self.merge
            && open.shared
            && is_shared
            && open.attachments == attachments
        {
            if open.offset != offset {
                open.offset = offset;
                open.set.frame = None;
            }
            if GROUPED && open.label != Some(self.pass) && self.labelled != Some(self.pass) {
                if open.label.is_some() {
                    open.pass.pop_debug_group();
                }
                open.pass.push_debug_group(pass.name);
                open.label = Some(self.pass);
            }
            colors.clear();
            self.spare.colors = colors;
            self.spare.attachments = attachments;
            return true;
        }
        self.end_render();
        let current = self.pass;
        self.group(current);
        let label = self.label_of(current).unwrap_or(pass.name);
        let layout = self.pass_layout(
            &colors,
            depth
                .as_ref()
                .map(|(_, _, format, _, view)| (*format, *view)),
            (self.debug && layer.is_some(), main),
        );
        let Some(encoder) = self.encoder.as_mut() else {
            return false;
        };
        let color_attachments: [Option<wgpu::RenderPassColorAttachment<'_>>; MOST_COLORS] =
            std::array::from_fn(|at| {
                colors.get(at).and_then(|(_, view, _, ops)| {
                    view.as_ref().map(|view| wgpu::RenderPassColorAttachment {
                        view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: color_ops(*ops),
                    })
                })
            });
        let depth_attachment =
            depth.as_ref().map(
                |(_, view, _, ops, _)| wgpu::RenderPassDepthStencilAttachment {
                    view,
                    depth_ops: ops.map(depth_ops),
                    stencil_ops: None,
                },
            );
        let mut render = encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(label),
                color_attachments: &color_attachments[..colors.len().min(MOST_COLORS)],
                depth_stencil_attachment: depth_attachment,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            })
            .forget_lifetime();
        if layer.is_some()
            && let Some(region) = self.region
        {
            crate::targets::draw_within(&mut render, region);
        }
        self.open = Some(Open {
            pass: render,
            attachments,
            layout,
            offset,
            shared: is_shared,
            layered: layer.is_some(),
            label: None,
            set: Set::default(),
        });
        colors.clear();
        self.spare.colors = colors;
        true
    }

    fn pass_layout(
        &mut self,
        colors: &[(u64, Option<wgpu::TextureView>, wgpu::TextureFormat, Ops)],
        depth: Option<(wgpu::TextureFormat, bool)>,
        (debug, main): (bool, bool),
    ) -> u32 {
        let formats = || {
            colors
                .iter()
                .map(|(_, view, format, _)| view.as_ref().map(|_| *format))
        };
        let found = self.layouts.iter().position(|known| {
            known.depth == depth
                && known.debug == debug
                && known.main == main
                && known.formats.iter().copied().eq(formats())
        });
        let at = found.unwrap_or_else(|| {
            self.layouts.push(PassLayout {
                formats: formats().collect(),
                depth,
                debug,
                main,
            });
            self.layouts.len() - 1
        });
        at as u32
    }

    pub(super) fn mismatch(&mut self, name: &str) -> bool {
        let error = fault(
            self.path(&format!("passes.{name}.attach")),
            ErrorCode::PassMismatch,
        );
        self.fault(error);
        false
    }

    pub(super) fn depth_attachment(
        &mut self,
        pass: &crate::install::PassDecl,
        texture: u32,
        layer: u32,
        depth: &mut Option<(
            u64,
            wgpu::TextureView,
            wgpu::TextureFormat,
            Option<Ops>,
            bool,
        )>,
    ) -> bool {
        let Some(&(_, access)) = pass.depths.iter().find(|(known, _)| *known == texture) else {
            return false;
        };
        let id = self.machine.ids.textures.get(texture as usize).copied();
        let planned = self
            .planned()
            .and_then(|planned| planned.depth)
            .and_then(|(known, ops)| (Some(known) == id).then_some(ops));
        let ops = planned
            .flatten()
            .map(|ops| Ops { store: true, ..ops })
            .or_else(|| {
                Some(match access {
                    crate::Access::Load => Ops {
                        load: Load::Load,
                        store: true,
                    },
                    crate::Access::Clear => Ops {
                        load: Load::Clear(pass.clear_of(texture).unwrap_or(Clear::Depth(FAR))),
                        store: true,
                    },
                })
            });
        let format = self.machine.declared.textures[texture as usize].format;
        if let Some(view) = self.attachment_view(texture, 0, layer) {
            *depth = Some((
                (u64::from(texture) * 4_096 + u64::from(layer)) * 64 + 20_000,
                view,
                format,
                ops,
                false,
            ));
        }
        true
    }

    pub(crate) fn canvas_format(&self) -> wgpu::TextureFormat {
        self.surface
    }

    pub(super) fn attachment_view(&mut self, texture: u32, mip: u32, layer: u32) -> Option<wgpu::TextureView> {
        let parity = self.parity;
        let made = self.textures.get_mut(texture as usize)?;
        let made = match parity {
            1 if made[1].is_some() => made[1].as_mut()?,
            _ => made[0].as_mut()?,
        };
        made.slice(
            &(mip..mip + 1),
            &(layer..layer + 1),
            Some(wgpu::TextureViewDimension::D2),
        )
        .cloned()
    }

    pub(crate) fn end_render(&mut self) {
        self.computing = None;
        self.end_open();
    }

    pub(super) fn end_open(&mut self) {
        if let Some(mut open) = self.open.take() {
            if GROUPED && open.label.is_some() {
                open.pass.pop_debug_group();
            }
            self.spare.attachments = std::mem::take(&mut open.attachments);
        }
    }
}
