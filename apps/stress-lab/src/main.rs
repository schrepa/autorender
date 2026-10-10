//! The Stress Lab natively, in a window or offscreen:
//! `cargo run -p stress-lab --release -- --preset metro`; `--help` lists the
//! options.

#[cfg(not(target_arch = "wasm32"))]
fn main() -> std::process::ExitCode {
    stress_lab::native::main(std::env::args().skip(1))
}

// Profiling captures count the program's allocations, as the page's do.
#[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
#[global_allocator]
static ALLOCATOR: engine::profiling::TrackingAllocator<std::alloc::System> =
    engine::profiling::TrackingAllocator::new(std::alloc::System);

// The page runs the Stress Lab in the browser.
#[cfg(target_arch = "wasm32")]
fn main() {}
