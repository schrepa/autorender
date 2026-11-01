use super::*;

impl Run {
    pub(super) fn copied_texture(&self, copied: Copied) -> Option<wgpu::Texture> {
        match copied {
            Copied::Texture(texture) => self.written_texture(texture).map(|made| made.texture.clone()),
            Copied::Target(view) => {
                let target = final_target(view);
                match target {
                    Target::Debug(index) => self.debug_targets.get(index as usize).map(|made| made.texture.clone()),
                    Target::Scene | Target::Depth | Target::Output(_) | Target::Canvas | Target::Copy(..) => {
                        self.targets.iter().find(|(known, _)| *known == target).map(|(_, made)| made.texture.clone())
                    }
                }
            }
            Copied::Canvas => self.canvas_texture.clone(),
            Copied::Buffer(_) => None,
        }
    }

    pub(crate) fn copy(&mut self, pass: u32) {
        self.end_render();
        let Some(at) = self.copies.iter().position(|(known, _)| *known == pass) else {
            return;
        };
        let (_, regions) = self.copies.swap_remove(at);
        let Some(decl) = self.machine.declared.passes.get(pass as usize).cloned() else {
            return;
        };
        if decl.part != self.part {
            let error = fault(self.path("copy"), ErrorCode::PassMismatch);
            self.fault(error);
            return;
        }
        let Work::Copy { from, to } = decl.work else {
            return;
        };
        let (source, target) = (self.copied_texture(from), self.copied_texture(to));
        let buffer = |copied: Copied| match copied {
            Copied::Buffer(buffer) => self.buffers.get(buffer as usize).and_then(Option::as_ref).map(|(made, _)| made.clone()),
            Copied::Texture(_) | Copied::Target(_) | Copied::Canvas => None,
        };
        let (from_buffer, to_buffer) = (buffer(from), buffer(to));
        self.group(pass);
        let Some(encoder) = self.encoder.as_mut() else {
            return;
        };
        for region in &regions {
            match region {
                crate::CopyRegion::Bytes { from: at, to: into, size } => {
                    if let (Some(source), Some(target)) = (&from_buffer, &to_buffer) {
                        encoder.copy_buffer_to_buffer(source, *at, target, *into, *size);
                    }
                }
                crate::CopyRegion::Texels { from: at, to: into, size } => {
                    if let (Some(source), Some(target)) = (&source, &target) {
                        encoder.copy_texture_to_texture(
                            wgpu::TexelCopyTextureInfo { texture: source, mip_level: 0, origin: *at, aspect: wgpu::TextureAspect::All },
                            wgpu::TexelCopyTextureInfo { texture: target, mip_level: 0, origin: *into, aspect: wgpu::TextureAspect::All },
                            *size,
                        );
                    }
                }
                crate::CopyRegion::Levels { from: (from_mip, at), to: (to_mip, into), size } => {
                    if let (Some(source), Some(target)) = (&source, &target) {
                        encoder.copy_texture_to_texture(
                            wgpu::TexelCopyTextureInfo { texture: source, mip_level: *from_mip, origin: *at, aspect: wgpu::TextureAspect::All },
                            wgpu::TexelCopyTextureInfo { texture: target, mip_level: *to_mip, origin: *into, aspect: wgpu::TextureAspect::All },
                            *size,
                        );
                    }
                }
                crate::CopyRegion::ToBuffer { from: at, to: layout, size } => {
                    if let (Some(source), Some(target)) = (&source, &to_buffer) {
                        let aspect = if source.format().has_depth_aspect() {
                            wgpu::TextureAspect::DepthOnly
                        } else {
                            wgpu::TextureAspect::All
                        };
                        encoder.copy_texture_to_buffer(
                            wgpu::TexelCopyTextureInfo { texture: source, mip_level: 0, origin: *at, aspect },
                            wgpu::TexelCopyBufferInfo { buffer: target, layout: *layout },
                            *size,
                        );
                    }
                }
            }
        }
    }

    pub(crate) fn clear_buffer(&mut self, buffer: u32) {
        self.end_render();
        let Some(Some((made, _))) = self.buffers.get(buffer as usize) else {
            return;
        };
        let made = made.clone();
        let current = self.pass;
        self.group(current);
        if let Some(encoder) = self.encoder.as_mut() {
            encoder.clear_buffer(&made, 0, None);
        }
    }

    pub(crate) fn copy_view(&mut self, layer: crate::Layer, view: ViewTarget) {
        self.end_render();
        let from = final_target(view);
        let source = self.targets.iter().find(|(known, _)| *known == from).map(|(_, made)| made.texture.clone());
        let target = self.targets.iter().find(|(known, _)| *known == Target::Copy(layer, view)).map(|(_, made)| made.texture.clone());
        if let (Some(source), Some(target), Some(encoder)) = (source, target, self.encoder.as_mut()) {
            encoder.copy_texture_to_texture(source.as_image_copy(), target.as_image_copy(), source.size());
        }
    }

    pub(crate) fn record_readbacks(&mut self) {
        self.leave();
        let asked = std::mem::take(&mut self.asked);
        let (slots, most) = (self.readback_slots, self.readback_slot_bytes);
        for request in asked {
            let Some(Some(buffer)) = self.buffers.get(request.buffer as usize).map(|made| made.as_ref().map(|(buffer, _)| buffer.clone())) else {
                continue;
            };
            let bytes = request.range.end.saturating_sub(request.range.start).next_multiple_of(4);
            let ring = match self.rings.iter().position(|ring| ring.part == request.part && ring.tag == request.tag) {
                Some(at) => at,
                None => {
                    self.rings.push(Ring { part: request.part, tag: request.tag, slots: Vec::new() });
                    self.rings.len() - 1
                }
            };
            let once = most.is_some_and(|most| bytes > most);
            let idle = |slot: &Slot| !slot.once && slot.state.load(Ordering::Acquire) == 0 && slot.bytes >= bytes;
            if once || (!self.rings[ring].slots.iter().any(idle) && self.rings[ring].kept() < slots) {
                let staging = crate::memory::buffer(
                    &self.device,
                    &wgpu::BufferDescriptor {
                        label: Some("engine/readback"),
                        size: bytes.max(4),
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    },
                );
                self.rings[ring].slots.push(Slot {
                    buffer: staging,
                    bytes,
                    state: Arc::new(AtomicU8::new(0)),
                    asked: None,
                    once,
                });
                if once {
                    self.overflows += 1;
                }
            }
            let found = if once {
                self.rings[ring].slots.last_mut()
            } else {
                self.rings[ring].slots.iter_mut().find(|slot| idle(slot))
            };
            let Some(slot) = found else {
                continue;
            };
            let Some(encoder) = self.encoder.as_mut() else {
                continue;
            };
            encoder.copy_buffer_to_buffer(&buffer, request.range.start, &slot.buffer, 0, bytes);
            if request.then == AfterRead::Clear {
                encoder.clear_buffer(&buffer, request.range.start, Some(bytes));
            }
            slot.state.store(1, Ordering::Release);
            slot.asked = Some((self.asked_frame, request.deliver));
        }
    }

    pub(crate) fn map_readbacks(&mut self) {
        for ring in &self.rings {
            for slot in &ring.slots {
                if slot.state.load(Ordering::Acquire) != 1 {
                    continue;
                }
                slot.state.store(2, Ordering::Release);
                let mapped = Arc::clone(&slot.state);
                slot.buffer.slice(..slot.bytes).map_async(
                    wgpu::MapMode::Read,
                    crate::platform::map_callback(move |result| {
                        mapped.store(if result.is_ok() { 3 } else { 0 }, Ordering::Release);
                    }),
                );
            }
        }
    }

    pub(crate) fn deliver(&mut self, world: &mut World) {
        for delivered in std::mem::take(&mut self.queued.delivered) {
            (delivered.deliver)(world, delivered.frame, delivered.tag, &delivered.bytes);
        }
        #[cfg(not(target_arch = "wasm32"))]
        crate::platform::run_arrived_maps();
        for ring in &mut self.rings {
            for slot in &mut ring.slots {
                if slot.state.load(Ordering::Acquire) != 3 {
                    continue;
                }
                if let Some((frame, deliver)) = slot.asked.take()
                    && let Ok(view) = slot.buffer.slice(..slot.bytes).get_mapped_range()
                {
                    deliver(world, frame, ring.tag, &view);
                }
                slot.buffer.unmap();
                slot.state.store(0, Ordering::Release);
            }
            ring.slots.retain(|slot| !(slot.once && slot.state.load(Ordering::Acquire) == 0 && slot.asked.is_none()));
        }
    }
}
