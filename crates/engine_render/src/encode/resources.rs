use engine_core::Expected;

use super::*;

impl Run {
    pub(crate) fn mipmaps(&mut self, texture: u32, layers: Range<u32>) -> bool {
        let Some(made) = self
            .written_texture(texture)
            .map(|made| made.texture.clone())
        else {
            let error = fault(self.path("mipmaps"), ErrorCode::InvalidValue);
            self.fault(error);
            return false;
        };
        if !crate::mipmaps::drawn(made.format(), made.dimension()) {
            let error = fault(self.path("mipmaps"), ErrorCode::UnsupportedFeature)
                .expected(Expected::Supported)
                .got(made.format());
            self.fault(error);
            return false;
        }
        self.owed_mips.retain(|(owed, _)| *owed != texture);
        self.owed_mips.push((texture, layers));
        self.record_mips();
        !self.owed_mips.iter().any(|(owed, _)| *owed == texture)
    }

    pub(crate) fn record_mips(&mut self) {
        let device = self.device.clone();
        let mut encoder = self.queued.before.take().unwrap_or_else(|| {
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("engine/part-mips"),
            })
        });
        for (texture, layers) in std::mem::take(&mut self.owed_mips) {
            let Some(made) = self
                .written_texture(texture)
                .map(|made| made.texture.clone())
            else {
                continue;
            };
            let (mut compiler, assets) = self.split();
            let levels = made.mip_level_count();
            let recorded = layers.clone().all(|layer| {
                assets
                    .materials
                    .mips()
                    .record((&device, &mut encoder), &mut compiler, &made, Some(layer))
                    .is_some_and(|written| written + 1 >= levels)
            });
            if !recorded {
                self.owed_mips.push((texture, layers));
            }
        }
        self.queued.before = Some(encoder);
    }

    pub(crate) fn grow_buffer(&mut self, buffer: u32, bytes: u64) -> Allocation {
        let Some(held) = self
            .buffer_tables_mut(buffer)
            .buffers
            .get_mut(buffer as usize)
        else {
            return Allocation::Fresh;
        };
        if held.as_ref().is_some_and(|(_, known)| *known >= bytes) {
            return Allocation::Kept;
        }
        let previous = held.take();
        let allocation = self.allocate_buffer(buffer, bytes.max(16).next_power_of_two());
        let Some((old, kept)) = previous else {
            return allocation;
        };
        let copied = match self.buffer_held(buffer) {
            Some((new, _)) => {
                let mut encoder =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("engine/part-grow"),
                        });
                encoder.copy_buffer_to_buffer(&old, 0, new, 0, kept);
                self.queue.submit_now([encoder.finish()]);
                Allocation::Kept
            }
            None => Allocation::Fresh,
        };
        crate::memory::release_buffer(&old);
        copied
    }
}
