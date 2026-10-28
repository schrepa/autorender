#[cfg(not(target_arch = "wasm32"))]
#[path = "../../engine/examples/app_template.rs"]
mod app;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> std::process::ExitCode {
    engine_native::main("Template", std::env::args().skip(1), app::build)
}

#[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
#[global_allocator]
static ALLOCATOR: engine::profiling::TrackingAllocator<std::alloc::System> =
    engine::profiling::TrackingAllocator::new(std::alloc::System);

#[cfg(target_arch = "wasm32")]
fn main() {}
