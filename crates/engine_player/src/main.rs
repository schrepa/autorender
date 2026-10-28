#[cfg(not(target_arch = "wasm32"))]
fn main() -> std::process::ExitCode {
    engine_player::native::main(std::env::args().skip(1))
}

#[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
#[global_allocator]
static ALLOCATOR: engine::profiling::TrackingAllocator<std::alloc::System> =
    engine::profiling::TrackingAllocator::new(std::alloc::System);

#[cfg(target_arch = "wasm32")]
fn main() {}
