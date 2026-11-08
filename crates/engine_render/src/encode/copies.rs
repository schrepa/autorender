use super::*;

fn to_buffer_copy(
    encoder: &mut wgpu::CommandEncoder,
    (source, mip, at): (&wgpu::Texture, u32, wgpu::Origin3d),
    (target, layout): (&wgpu::Buffer, wgpu::TexelCopyBufferLayout),
    size: wgpu::Extent3d,
) {
    let aspect = if source.format().has_depth_aspect() {
        wgpu::TextureAspect::DepthOnly
    } else {
        wgpu::TextureAspect::All
    };
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture: source, mip_level: mip, origin: at, aspect },
        wgpu::TexelCopyBufferInfo { buffer: target, layout },
        size,
    );
}

impl Run {
    pub(super) fn copied_texture(&self, copied: Copied) -> Option<wgpu::Texture> {
        match copied {
            Copied::Texture(texture) => self.written_texture(texture).map(|made| made.texture.clone()),
            Copied::Target(view) => {
                let target = final_target(view);
                self.set()
                    .targets
                    .iter()
                    .find(|(known, _)| *known == target)
                    .map(|(_, made)| made.texture.clone())
            }
            Copied::Canvas => self.canvas.texture.clone(),
            Copied::Buffer(_) => None,
        }
    }

    pub(crate) fn copy(&mut self, pass: u32) -> bool {
        self.end_render();
        let Some(at) = self.queued.copies.iter().position(|(known, _)| *known == pass) else {
            return false;
        };
        let (_, regions) = self.queued.copies.swap_remove(at);
        let Some(decl) = self.planning.pass_decls.get(pass as usize).cloned() else {
            return false;
        };
        if decl.part != self.part {
            let error = fault(self.path("copy"), ErrorCode::PassMismatch);
            self.fault(error);
            return false;
        }
        let Work::Copy { from, to } = decl.work else {
            return false;
        };
        self.group(pass);
        self.record_copy(from, to, &regions);
        true
    }

    pub(crate) fn copy_at_end(&mut self) {
        self.leave();
        for (from, to, regions) in std::mem::take(&mut self.queued.ending) {
            self.record_copy(from, to, &regions);
        }
    }

    fn record_copy(&mut self, from: Copied, to: Copied, regions: &[crate::CopyRegion]) {
        let (source, target) = (self.copied_texture(from), self.copied_texture(to));
        let buffer = |copied: Copied| match copied {
            Copied::Buffer(buffer) => self.buffer_held(buffer).map(|(made, _)| made.clone()),
            Copied::Texture(_) | Copied::Target(_) | Copied::Canvas => None,
        };
        let (from_buffer, to_buffer) = (buffer(from), buffer(to));
        let Some(encoder) = self.recording.encoder() else {
            return;
        };
        for region in regions {
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
                        to_buffer_copy(encoder, (source, 0, *at), (target, *layout), *size);
                    }
                }
                crate::CopyRegion::LevelToBuffer { from: (mip, at), to: layout, size } => {
                    if let (Some(source), Some(target)) = (&source, &to_buffer) {
                        to_buffer_copy(encoder, (source, *mip, *at), (target, *layout), *size);
                    }
                }
            }
        }
    }

    pub(crate) fn clear_buffer(&mut self, buffer: u32) {
        self.end_render();
        let Some((made, _)) = self.buffer_held(buffer) else {
            return;
        };
        let made = made.clone();
        let current = self.pass;
        self.group(current);
        if let Some(encoder) = self.recording.encoder() {
            encoder.clear_buffer(&made, 0, None);
        }
    }

    pub(crate) fn copy_view(&mut self, layer: engine_core::AttachPoint, view: ViewTarget) {
        self.end_render();
        let from = final_target(view);
        let targets = &self.set().targets;
        let source = targets.iter().find(|(known, _)| *known == from).map(|(_, made)| made.texture.clone());
        let target = targets.iter().find(|(known, _)| *known == Target::Copy(layer, view)).map(|(_, made)| made.texture.clone());
        if let (Some(source), Some(target), Some(encoder)) = (source, target, self.recording.encoder()) {
            encoder.copy_texture_to_texture(source.as_image_copy(), target.as_image_copy(), source.size());
        }
    }

    pub(crate) fn record_readbacks(&mut self) {
        self.leave();
        let asked = std::mem::take(&mut self.queued.asked);
        let (slots, most) = (self.reading.slots, self.reading.slot_bytes);
        let written = self.written();
        let device = self.device.clone();
        for request in asked {
            let Some(buffer) = self.buffer_held(request.buffer).map(|(buffer, _)| buffer.clone()) else {
                continue;
            };
            let bytes = request.range.end.saturating_sub(request.range.start).next_multiple_of(4);
            let over = most.is_some_and(|most| bytes > most);
            if over {
                self.reading.list.overflowed();
            }
            let keep = if request.once || over { Keep::Once } else { Keep::Ring(slots) };
            let reader = Reader::Part { part: request.part, tag: request.tag };
            let Some(encoder) = self.recording.encoder() else {
                continue;
            };
            let asked = (reader, written, Some(request.deliver));
            if !self.reading.list.record((&device, encoder), (&buffer, request.range.clone()), asked, keep) {
                continue;
            }
            if request.then == AfterRead::Clear {
                encoder.clear_buffer(&buffer, request.range.start, Some(bytes));
            }
        }
    }
}
