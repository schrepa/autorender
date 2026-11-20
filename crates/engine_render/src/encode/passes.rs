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
        if self.planning.labelled == Some(pass) {
            self.end_render();
        } else {
            self.leave();
        }
        true
    }

    pub(crate) fn start_runs(&mut self) {
        self.planning.runs.clear();
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

    pub(crate) fn label_of(&self, pass: u32) -> Option<&'static str> {
        let declared = self.machine.declared.passes.get(pass as usize)?;
        Some(declared.label.unwrap_or(declared.name))
    }

    pub(crate) fn span_of(&self, pass: u32) -> Option<&'static str> {
        self.machine.declared.passes.get(pass as usize)?.span
    }

    pub(crate) fn group(&mut self, pass: u32) {
        if !GROUPED || self.planning.labelled == Some(pass) {
            return;
        }
        let name = self
            .machine
            .declared
            .passes
            .get(pass as usize)
            .map_or("", |declared| declared.name);
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

    pub(crate) fn record_readbacks(&mut self) {
        self.leave();
        #[cfg(feature = "dev")]
        self.read_work();
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
    pub(crate) fn begin_render(&mut self) -> bool {
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
        let kept = self.canvas.carried;
        let clear = self.canvas.clear;
        let loaded = self.canvas.load;
        let carried = move |target: Target, ops: Ops| match target {
            Target::Shared(_) if kept => Ops {
                load: Load::Load,
                ..ops
            },
            Target::Canvas if loaded => Ops {
                load: Load::Load,
                ..ops
            },
            Target::Canvas if matches!(ops.load, Load::Clear(Clear::Color(_))) => Ops {
                load: Load::Clear(Clear::Color(clear)),
                ..ops
            },
            Target::Shared(_) | Target::Canvas => ops,
        };
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
        let mut attachments = std::mem::take(&mut self.spare.attachments);
        attachments.clear();
        attachments.extend(
            colors
                .iter()
                .map(|(key, ..)| *key)
                .chain(depth.as_ref().map(|(key, ..)| *key)),
        );
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
        if let Some(opened) = self.sets.counted.opened.get_mut(self.pass as usize) {
            *opened += 1;
        }
        self.bound.forget();
        self.recording.transition(Step::Render(Box::new(Open {
            pass: render,
            attachments,
            layout,
            offset,
            shared: is_shared,
            label: None,
        })));
        colors.clear();
        self.spare.colors = colors;
        true
    }

    fn mismatch(&mut self, name: &str) -> bool {
        let error = fault(
            self.path(&format!("passes.{name}.attach")),
            ErrorCode::PassMismatch,
        );
        self.fault(error);
        false
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
