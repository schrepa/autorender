#![expect(unsafe_code, reason = "starts and stops a graphics debugger's capture")]

use super::*;

#[derive(Clone, Debug)]
pub struct GpuMark(wgpu::SubmissionIndex);

impl Renderer {
    pub fn start_debugger_capture(&mut self) -> bool {
        if self.core.debugger_capture {
            return false;
        }
        self.core.debugger_capture = true;
        // SAFETY: no other capture is active (the flag pairs every start
        // with one stop); the debugger's own rules are its documentation's.
        unsafe {
            self.core.device.start_graphics_debugger_capture();
        }
        true
    }

    pub fn stop_debugger_capture(&mut self) -> Result<(), Error> {
        if !self.core.debugger_capture {
            return Ok(());
        }
        self.wait_for_gpu()?;
        self.core.debugger_capture = false;
        // SAFETY: the capture started above is the one active.
        unsafe {
            self.core.device.stop_graphics_debugger_capture();
        }
        Ok(())
    }

    pub fn wait_for_gpu(&self) -> Result<(), Error> {
        self.core
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map(drop)
            .map_err(|error| failed_with("$.gpu", error))
    }

    pub fn gpu_mark(&self) -> GpuMark {
        GpuMark(self.core.queue.submit([]))
    }

    pub fn wait_for_mark(&self, mark: GpuMark) -> Result<(), Error> {
        self.core
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(mark.0),
                timeout: None,
            })
            .map(drop)
            .map_err(|error| failed_with("$.gpu", error))
    }

    pub fn read_offscreen(&self) -> Result<Option<(u32, u32, Vec<u8>)>, Error> {
        let Some(texture) = self.core.display.offscreen() else {
            return Ok(None);
        };
        let (width, height) = (texture.width(), texture.height());
        let row = (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buffer = crate::memory::buffer(
            &self.core.device,
            &wgpu::BufferDescriptor {
                label: Some("engine/offscreen-readback"),
                size: u64::from(row) * u64::from(height),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            },
        );
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
