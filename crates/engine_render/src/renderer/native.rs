use super::*;

impl Renderer {
    pub fn gpu_allocations(&self) -> serde_json::Value {
        let Some(report) = self.core.device.generate_allocator_report() else {
            return serde_json::Value::Null;
        };
        let mut named = std::collections::BTreeMap::<&str, (u64, u64)>::new();
        for allocation in &report.allocations {
            let (count, bytes) = named.entry(allocation.name.as_str()).or_default();
            *count += 1;
            *bytes += allocation.size;
        }
        let mut rows: Vec<(&str, (u64, u64))> = named.into_iter().collect();
        rows.sort_by(|a, b| b.1.1.cmp(&a.1.1).then_with(|| a.0.cmp(b.0)));
        serde_json::json!({
            "reserved_bytes": report.total_reserved_bytes,
            "allocated_bytes": report.total_allocated_bytes,
            "blocks": report.blocks.len(),
            "allocations": rows
                .iter()
                .map(|(name, (count, bytes))| {
                    serde_json::json!({ "name": name, "count": count, "bytes": bytes })
                })
                .collect::<Vec<_>>(),
        })
    }

    pub fn wait_for_gpu(&self) -> Result<(), Error> {
        self.core
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map(drop)
            .map_err(|error| failed_with("$.gpu", error))
    }

    pub fn read_offscreen(&self) -> Result<Option<(u32, u32, Vec<u8>)>, Error> {
        let Some(texture) = self.core.display.offscreen() else {
            return Ok(None);
        };
        let (width, height) = (texture.width(), texture.height());
        let row = (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buffer = self.core.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine/offscreen-readback"),
            size: u64::from(row) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            self.core
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("engine/offscreen-readback"),
                });
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(height),
                },
            },
            texture.size(),
        );
        self.core.queue.submit([encoder.finish()]);
        let mapped = std::sync::Arc::new(std::sync::OnceLock::new());
        let result = std::sync::Arc::clone(&mapped);
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |outcome| {
                let _ = result.set(outcome);
            });
        self.core
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| failed_with("$.offscreen", error))?;
        match mapped.get() {
            Some(Ok(())) => {}
            Some(Err(error)) => {
                return Err(failed_with("$.offscreen", error));
            }
            None => return Err(failed("$.offscreen").expected(Expected::Present)),
        }
        let view = buffer
            .slice(..)
            .get_mapped_range()
            .map_err(|error| failed_with("$.offscreen", error))?;
        let mut pixels = Vec::with_capacity((width * height * 4) as usize);
        for rows in view.chunks_exact(row as usize) {
            for texel in rows[..(width * 4) as usize].as_chunks::<4>().0 {
                let [blue, green, red, alpha] = *texel;
                pixels.extend_from_slice(&[red, green, blue, alpha]);
            }
        }
        drop(view);
        buffer.unmap();
        Ok(Some((width, height, pixels)))
    }
}
