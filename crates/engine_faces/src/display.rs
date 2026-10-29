pub trait MeteredExposure {
    fn metered(&self) -> Option<(&wgpu::Buffer, u64)>;
}
