use super::*;

impl Renderer {
    pub fn wait_for_gpu(&self) -> Result<(), Error> {
        self.core
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map(drop)
            .map_err(|error| failed_with("$.gpu", error))
    }
}
