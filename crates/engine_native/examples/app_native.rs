//! A new game's native entry: the game in a window on DX12, Vulkan or Metal,
//! or offscreen, profiled, captured or as a development session as the
//! host's options ask (`--help` lists them). `cargo xtask new app NAME`
//! writes this file as the app's `src/main.rs`, with the app's library as
//! `app`; lint compiles it with the app template.

// The game: the app's library.
#[cfg(not(target_arch = "wasm32"))]
#[path = "../../engine/examples/app_template.rs"]
mod app;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> std::process::ExitCode {
    engine_native::main("Template", std::env::args().skip(1), app::build)
}

// Profiling builds count the program's allocations, as the page's do.
#[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
#[global_allocator]
static ALLOCATOR: engine::profiling::TrackingAllocator<std::alloc::System> =
    engine::profiling::TrackingAllocator::new(std::alloc::System);

// The page runs the game in the browser.
#[cfg(target_arch = "wasm32")]
fn main() {}
