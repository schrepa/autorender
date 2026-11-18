use super::*;

impl Run {
    pub(crate) fn enter(&mut self, pass: u32) -> bool {
        let shares = self
            .machine
            .declared
            .passes
            .get(pass as usize)
            .is_some_and(shared);
        if self.planning.merge && shares && self.recording.open().is_some_and(|open| open.shared) {
            return false;
        }
        if self
            .planning
            .labelled
            .is_some_and(|known| self.same_group(known, pass))
        {
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

    pub(crate) fn start_runs(&mut self) {
        self.planning.runs.clear();
        self.planning.idle.clear();
    }

    pub(crate) fn forget_ran(&mut self, part: u32) {
        let sets = &mut self.sets;
        for &pass in sets.fixed.part_passes.get(part as usize).into_iter().flatten() {
            for set in &mut sets.list {
                set.ran.remove(pass as usize);
            }
        }
    }

    pub(crate) fn forget_global(&mut self, part: u32) {
        let sets = &mut self.sets;
        for &pass in sets.fixed.part_globals.get(part as usize).into_iter().flatten() {
            for set in &mut sets.list {
                set.ran.remove(pass as usize);
            }
        }
    }

    pub(crate) fn forget_viewed(&mut self) {
        let sets = &mut self.sets;
        sets.list[sets.at].ran.remove_all(&sets.passes);
    }

    pub(crate) fn ran_now(&mut self, pass: usize) {
        let (at, global) = (self.sets.at, !self.sets.passes.contains(pass));
        for (index, set) in self.sets.list.iter_mut().enumerate() {
            if global || index == at {
                set.ran.insert(pass);
            }
        }
    }

    pub(crate) fn skip_kind(&mut self, kind: engine_core::DrawKind, reason: &'static str) {
        let part = self.part;
        if !self
            .planning
            .idle
            .iter()
            .any(|&(known, held, _)| known == part && held == kind)
        {
            self.planning.idle.push((part, kind, reason));
        }
    }

    pub(crate) fn drop_idle_scenes(&mut self) {
        if self.planning.idle.is_empty() {
            return;
        }
        for (index, pass) in self.machine.declared.passes.iter().enumerate() {
            let Some(bit) = pass.scene.filter(|scene| scene.core).and_then(|scene| scene.bit)
            else {
                continue;
            };
            if !self.planning.runs.contains(index) {
                continue;
            }
            let (installs, idle) = (&self.installs, &self.planning.idle);
            let scene = bit.index();
            let listed = installs.members.get(scene as usize).map_or(&[][..], Vec::as_slice);
            let skipped: Option<Vec<(u32, &'static str)>> = listed
                .iter()
                .filter_map(|&at| installs.draw_entries.get(at as usize))
                .map(|member| {
                    idle.iter()
                        .find(|&&(part, kind, _)| part == member.part && kind == member.decl.kind)
                        .map(|&(.., reason)| (member.kind, reason))
                })
                .collect();
            let Some(skipped) = skipped.filter(|skipped| !skipped.is_empty()) else {
                continue;
            };
            self.planning.runs.remove(index);
            let kinds = &mut self.tallies.kinds;
            kinds.scened(scene);
            for (kind, reason) in skipped {
                kinds.begin(scene, kind);
                kinds.skip(reason);
                kinds.end();
            }
        }
    }

    pub(crate) fn label_of(&self, pass: u32) -> Option<&'static str> {
        let declared = self.machine.declared.passes.get(pass as usize)?;
        Some(declared.label.unwrap_or(declared.name))
    }

    pub(crate) fn group_of(&self, pass: u32) -> &'static str {
        self.machine
            .declared
            .passes
            .get(pass as usize)
            .map_or("", |declared| declared.group.unwrap_or(declared.name))
    }

    pub(crate) fn span_of(&self, pass: u32) -> Option<&'static str> {
        self.machine.declared.passes.get(pass as usize)?.span
    }

    pub(crate) fn open_encode_span(&mut self, timer: &GpuProfiler) {
        let Some(name) = self.timed.asked.take() else {
            return;
        };
        self.end_render();
        if let (Some(span), Some(encoder)) =
            (self.encode_span(timer, name), self.recording.encoder())
        {
            span.mark(encoder, false);
        }
        self.timed.open = Some(name);
    }

    fn encode_span<'t>(
        &self,
        timer: &'t GpuProfiler,
        name: &'static str,
    ) -> Option<crate::profiling::Span<'t>> {
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
        if let (Some(span), Some(encoder)) =
            (self.encode_span(timer, name), self.recording.encoder())
        {
            span.mark(encoder, true);
        }
    }

    pub(crate) fn flush_clears(&mut self) {
        if self.queued.cleared.is_empty() {
            return;
        }
        self.end_render();
        for buffer in std::mem::take(&mut self.queued.cleared) {
            self.owe(buffer);
            let made = self.buffer_held(buffer).map(|(made, _)| made.clone());
            if let (Some(made), Some(encoder)) = (made, self.recording.encoder()) {
                encoder.clear_buffer(&made, 0, None);
            }
        }
    }

    pub(crate) fn group(&mut self, pass: u32) {
        if !GROUPED
            || self
                .planning
                .labelled
                .is_some_and(|known| self.same_group(known, pass))
        {
            return;
        }
        let name = self.group_of(pass);
        if let Some(encoder) = self.recording.encoder() {
            if self.planning.labelled.is_some() {
                encoder.pop_debug_group();
            }
            encoder.push_debug_group(name);
            self.planning.labelled = Some(pass);
        }
    }

    pub(crate) fn leave(&mut self) {
        self.end_render();
        if GROUPED
            && self.planning.labelled.take().is_some()
            && let Some(encoder) = self.recording.encoder()
        {
            encoder.pop_debug_group();
        }
    }

    fn pass_set(&self, pass: u32) -> Arc<[SetFormat]> {
        self.planning
            .pass_sets
            .get(pass as usize)
            .map_or_else(|| Arc::from(Vec::new()), Arc::clone)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    pub(crate) fn begin_render(&mut self, attach: AttachRaw) -> bool {
        self.pay_arena();
        let offset = self.view_offset();
        let Some(pass) = self.planning.pass_decls.get(self.pass as usize).cloned() else {
            return false;
        };
        let Some(plan) = self.planning.plan.clone() else {
            return false;
        };
        let Some(planned) = plan.passes.get(self.planning.planned_at) else {
            return false;
        };
        let is_shared = shared(&pass);
        let mut colors = std::mem::take(&mut self.spare.colors);
        colors.clear();
        let mut depth: Option<(u64, wgpu::TextureView, wgpu::TextureFormat, Option<Ops>)> = None;
        let mut main = false;
        let mut plain = false;
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
        let kept = self.canvas.carried;
        let clear = self.canvas.clear;
        let loaded = self.canvas.load;
        let carried = move |target: Target, ops: Ops| match target {
            Target::Shared(_) if kept => Ops {
                load: Load::Load,
                ..ops
            },
            Target::Shared(_) if matches!(ops.load, Load::Clear(Clear::Color(_))) => Ops {
                load: Load::Clear(Clear::Color(clear)),
                ..ops
            },
            Target::Canvas if loaded => Ops {
                load: Load::Load,
                ..ops
            },
            Target::Shared(_) | Target::Canvas => ops,
        };
        match attach {
            AttachRaw::Declared => {
                let at = 0;
                let declared = &self.machine.declared;
                let key = |target: Target| declared.target_key(target);
                if is_shared {
                    let set = self.pass_set(self.pass);
                    main = lit_set(&set);
                    plain = !set.is_empty() && !main;
                    for &(target, format, _) in set.iter() {
                        let id = self.machine.ids.target(target);
                        match ops_of(id) {
                            Some(ops) => colors.push((
                                key(target),
                                self.view_of(target).cloned(),
                                format,
                                carried(target, ops),
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
                    if let Some(target) = declared.depth_target()
                        && let (Some(id), Some(made), Some(format)) = (
                            self.machine.ids.target(target),
                            self.view_of(target),
                            declared.target_format(target),
                        )
                    {
                        let ops = depth_of(Some(id))
                            .unwrap_or(Some(Ops {
                                load: Load::Load,
                                store: true,
                            }))
                            .map(|ops| carried(target, ops));
                        let attached = key(target) | u64::from(ops.is_none());
                        depth = Some((attached, made.clone(), format, ops));
                    }
                }
                for &(name, _) in &pass.targets {
                    let Some(target) = declared.target_of(name) else {
                        continue;
                    };
                    let ops = ops_of(self.machine.ids.target(target)).unwrap_or(Ops {
                        load: Load::Load,
                        store: true,
                    });
                    let (view, format) = match target {
                        Target::Canvas => (self.view_canvas(), Some(self.canvas_format())),
                        Target::Shared(_) => (
                            self.view_of(target).cloned(),
                            declared.target_format(target),
                        ),
                    };
                    let Some(format) = format else {
                        return self.mismatch(pass.name);
                    };
                    if view.is_some() || self.recording.collecting() {
                        colors.push((key(target), view, format, carried(target, ops)));
                    }
                }
                for &(texture, access) in &pass.colors {
                    let id = self.machine.ids.textures.get(texture as usize).copied();
                    let ops = ops_of(id).unwrap_or_else(|| {
                        declared_ops(
                            access,
                            pass.clear_of(texture),
                            Clear::Color(crate::graph::TRANSPARENT),
                        )
                    });
                    let format = self.machine.declared.textures[texture as usize].format;
                    let view = self.attachment_view(texture, 0, at);
                    let attached = (u64::from(texture) * 4_096 + u64::from(at)) * 64 + 1_000;
                    colors.push((attached, view, format, ops));
                }
                if depth.is_none()
                    && let Some(&(texture, access)) = pass.depths.first()
                {
                    let id = self.machine.ids.textures.get(texture as usize).copied();
                    let ops = depth_of(id).unwrap_or_else(|| {
                        Some(declared_ops(
                            access,
                            pass.clear_of(texture),
                            Clear::Depth(FAR),
                        ))
                    });
                    let format = self.machine.declared.textures[texture as usize].format;
                    if let Some(view) = self.attachment_view(texture, 0, at) {
                        let attached = (u64::from(texture) * 4_096 + u64::from(at)) * 64 + 2_000;
                        depth = Some((attached, view, format, ops));
                    }
                }
            }
            AttachRaw::Color(texture, mip, ref layers) => {
                let Some(&(_, access)) = pass.colors.iter().find(|(known, _)| *known == texture)
                else {
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
                for at in layers.clone() {
                    let view = self.attachment_view(texture, mip, at);
                    let attachment =
                        (u64::from(texture) * 4_096 + u64::from(at)) * 64 + u64::from(mip) + 10_000;
                    colors.push((attachment, view, format, ops));
                }
            }
            AttachRaw::Depth(texture, layer) => {
                let layer = layer.unwrap_or(0);
                if !self.depth_attachment(&pass, texture, layer, &mut depth) {
                    return self.mismatch(pass.name);
                }
            }
            AttachRaw::ColorDepth(color, layer, texture) => {
                let Some(&(_, access)) = pass.colors.iter().find(|(known, _)| *known == color)
                else {
                    return self.mismatch(pass.name);
                };
                let id = self.machine.ids.textures.get(color as usize).copied();
                let planned = ops_of(id).unwrap_or_else(|| {
                    declared_ops(
                        access,
                        pass.clear_of(color),
                        Clear::Color(crate::graph::TRANSPARENT),
                    )
                });
                let ops = Ops {
                    store: true,
                    ..planned
                };
                let format = self.machine.declared.textures[color as usize].format;
                let at = layer.unwrap_or(0);
                let view = self.attachment_view(color, 0, at);
                colors.push((
                    (u64::from(color) * 4_096 + u64::from(at)) * 64 + 10_000,
                    view,
                    format,
                    ops,
                ));
                if !self.depth_attachment(&pass, texture, at, &mut depth) {
                    return self.mismatch(pass.name);
                }
            }
            AttachRaw::Colors(ref attached) => {
                for &(texture, at) in attached {
                    let Some(&(_, access)) =
                        pass.colors.iter().find(|(known, _)| *known == texture)
                    else {
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
        let merge = self.planning.merge;
        let differ = self.recording.open_mut().is_some_and(|open| {
            merge && open.shared && is_shared && open.attachments != attachments
        });
        if let Some(open) = self.recording.open_mut()
            && self.planning.merge
            && open.shared
            && is_shared
            && open.attachments == attachments
        {
            if let Some(merged) = self.sets.counted.merged.get_mut(self.pass as usize) {
                *merged += 1;
            }
            open.offset = offset;
            if GROUPED && open.label != Some(self.pass) && self.planning.labelled != Some(self.pass)
            {
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
        let collecting = self.recording.collecting();
        let layout = self.machine.layout_id(pass_layout(
            &colors,
            depth.as_ref().map(|(_, _, format, _)| *format),
            (plain, main),
            collecting.then(|| self.machine.declared.target_key(Target::Canvas)),
        ));
        if collecting {
            self.recording.collected(super::recording::Opened {
                layout,
                shared: is_shared,
            });
            colors.clear();
            self.spare.colors = colors;
            self.spare.attachments = attachments;
            return true;
        }
        let Some(encoder) = self.recording.encoder() else {
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
            depth
                .as_ref()
                .map(|(_, view, format, ops)| wgpu::RenderPassDepthStencilAttachment {
                    view,
                    depth_ops: ops.map(depth_ops),
                    stencil_ops: format.has_stencil_aspect().then(|| stencil_ops(*ops)),
                });
        let render = encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(label),
                color_attachments: &color_attachments[..colors.len().min(MOST_COLORS)],
                depth_stencil_attachment: depth_attachment,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            })
            .forget_lifetime();
        self.opened(is_shared, differ);
        self.bound.forget();
        self.recording.transition(Step::Render(Box::new(Open {
            pass: render,
            attachments,
            layout,
            offset,
            shared: is_shared,
            layered: is_shared,
            label: None,
        })));
        colors.clear();
        self.spare.colors = colors;
        true
    }

    fn opened(&mut self, shared: bool, differ: bool) {
        let pass = self.pass as usize;
        let counted = &mut self.sets.counted;
        if let Some(opened) = counted.opened.get_mut(pass) {
            *opened += 1;
        }
        let why = match (shared, self.planning.merge, differ, counted.shared_open) {
            (false, ..) => None,
            (true, false, ..) => Some("merging_off"),
            (true, true, true, _) => Some("attachments_differ"),
            (true, true, false, true) => Some("group_boundary"),
            (true, true, false, false) => None,
        };
        counted.shared_open = shared;
        let Some(last) = counted.why.get_mut(pass) else {
            return;
        };
        if why.is_none() || *last == why {
            return;
        }
        *last = why;
        if let (Some(why), Some(declared)) = (why, self.machine.declared.passes.get(pass))
            && engine_profile::recording()
        {
            engine_profile::event(
                "pass_merge",
                declared.name,
                &[("why", engine_profile::Fact::Text(why))],
            );
        }
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
        depth: &mut Option<(u64, wgpu::TextureView, wgpu::TextureFormat, Option<Ops>)>,
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
            ));
        }
        true
    }

    pub(crate) fn canvas_format(&self) -> wgpu::TextureFormat {
        self.canvas.format
    }

    pub(crate) fn clear_canvas(&mut self) {
        let Some(canvas) = self.canvas.view.clone() else {
            return;
        };
        self.leave();
        let Some(encoder) = self.recording.encoder() else {
            return;
        };
        drop(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("engine/canvas-clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &canvas,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        }));
    }

    pub(super) fn attachment_view(
        &mut self,
        texture: u32,
        mip: u32,
        layer: u32,
    ) -> Option<wgpu::TextureView> {
        let made = self
            .texture_tables_mut(texture)
            .textures
            .get_mut(texture as usize)?
            .as_mut()?;
        made.slice(
            &(mip..mip + 1),
            &(layer..layer + 1),
            Some(wgpu::TextureViewDimension::D2),
        )
        .cloned()
    }

    pub(crate) fn end_render(&mut self) {
        let ended = self.recording.transition(Step::EndPass);
        self.closed(ended);
    }

    pub(crate) fn collect(&mut self) {
        self.leave();
        self.recording.collect();
    }

    pub(crate) fn record(&mut self) {
        self.recording.record();
    }

    pub(super) fn end_open(&mut self) {
        let ended = self.recording.transition(Step::EndRender);
        self.closed(ended);
    }

    pub(crate) fn take_encoder(&mut self) -> Option<wgpu::CommandEncoder> {
        self.end_render();
        match self.recording.transition(Step::Take) {
            Ended::Encoder(encoder) => Some(encoder),
            Ended::Nothing | Ended::Render(_) => None,
        }
    }

    fn closed(&mut self, ended: Ended) {
        if let Ended::Render(mut open) = ended {
            if GROUPED && open.label.is_some() {
                open.pass.pop_debug_group();
            }
            self.spare.attachments = std::mem::take(&mut open.attachments);
        }
    }
}

fn pass_layout(
    colors: &[(u64, Option<wgpu::TextureView>, wgpu::TextureFormat, Ops)],
    depth: Option<wgpu::TextureFormat>,
    (plain, main): (bool, bool),
    canvas: Option<u64>,
) -> PassLayout {
    let mut formats = [None; MOST_COLORS];
    for (format, (key, view, known, _)) in formats.iter_mut().zip(colors) {
        *format = (view.is_some() || canvas == Some(*key)).then_some(*known);
    }
    PassLayout {
        formats,
        count: colors.len().min(MOST_COLORS),
        depth,
        plain,
        main,
    }
}

fn color_ops(ops: Ops) -> wgpu::Operations<wgpu::Color> {
    wgpu::Operations {
        load: match ops.load {
            Load::Clear(Clear::Color([r, g, b, a])) => {
                wgpu::LoadOp::Clear(wgpu::Color { r, g, b, a })
            }
            Load::Clear(Clear::Depth(_)) | Load::Load => wgpu::LoadOp::Load,
        },
        store: if ops.store {
            wgpu::StoreOp::Store
        } else {
            wgpu::StoreOp::Discard
        },
    }
}

fn stencil_ops(ops: Option<Ops>) -> wgpu::Operations<u32> {
    wgpu::Operations {
        load: match ops.map(|ops| ops.load) {
            Some(Load::Clear(_)) => wgpu::LoadOp::Clear(0),
            Some(Load::Load) | None => wgpu::LoadOp::Load,
        },
        store: wgpu::StoreOp::Store,
    }
}

fn depth_ops(ops: Ops) -> wgpu::Operations<f32> {
    wgpu::Operations {
        load: match ops.load {
            Load::Clear(Clear::Depth(depth)) => wgpu::LoadOp::Clear(depth),
            Load::Clear(Clear::Color(_)) | Load::Load => wgpu::LoadOp::Load,
        },
        store: if ops.store {
            wgpu::StoreOp::Store
        } else {
            wgpu::StoreOp::Discard
        },
    }
}
