#![expect(unsafe_code, reason = "starts and stops a graphics debugger's capture")]

use super::*;

#[derive(Clone, Debug)]
pub struct GpuMark(wgpu::SubmissionIndex);

impl Renderer {
    pub fn start_debugger_capture(&mut self) -> bool {
        if self.debugger_capture {
            return false;
        }
        self.debugger_capture = true;
        // SAFETY: no other capture is active (the flag pairs every start
        // with one stop); the debugger's own rules are its documentation's.
        unsafe {
            self.device.start_graphics_debugger_capture();
        }
        true
    }

    pub fn stop_debugger_capture(&mut self) -> Result<(), RenderError> {
        if !self.debugger_capture {
            return Ok(());
        }
        self.wait_for_gpu()?;
        self.debugger_capture = false;
        // SAFETY: the capture started above is the one active.
        unsafe {
            self.device.stop_graphics_debugger_capture();
        }
        Ok(())
    }

    pub fn wait_for_gpu(&self) -> Result<(), RenderError> {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map(drop)
            .map_err(|error| render_error("The GPU failed", error))
    }

    pub fn gpu_mark(&self) -> GpuMark {
        GpuMark(self.queue.submit([]))
    }

    pub fn wait_for_mark(&self, mark: GpuMark) -> Result<(), RenderError> {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(mark.0),
                timeout: None,
            })
            .map(drop)
            .map_err(|error| render_error("The GPU failed", error))
    }

    pub fn read_offscreen(&self) -> Result<Option<(u32, u32, Vec<u8>)>, RenderError> {
        let Some(texture) = self.display.offscreen() else {
            return Ok(None);
        };
        let (width, height) = (texture.width(), texture.height());
        let row = (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buffer = crate::memory::buffer(&self.device, &wgpu::BufferDescriptor {
            label: Some("engine/offscreen-readback"),
            size: u64::from(row) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
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
        self.queue.submit([encoder.finish()]);
        let mapped = std::sync::Arc::new(std::sync::OnceLock::new());
        let result = std::sync::Arc::clone(&mapped);
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |outcome| {
                let _ = result.set(outcome);
            });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| render_error("Could not read the offscreen frame", error))?;
        match mapped.get() {
            Some(Ok(())) => {}
            Some(Err(error)) => {
                return Err(render_error("Could not read the offscreen frame", error));
            }
            None => return Err(RenderError("The offscreen frame never mapped.".into())),
        }
        let view = buffer
            .slice(..)
            .get_mapped_range()
            .map_err(|error| render_error("Could not read the offscreen frame", error))?;
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
