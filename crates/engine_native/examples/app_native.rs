//! A new game's native entry: the game in a window on DX12, Vulkan or Metal.
//! `cargo xtask new app NAME` writes this file as the app's `src/main.rs`,
//! with the app's library as `app`; lint compiles it with the app template.

// The game: the app's library.
#[cfg(not(target_arch = "wasm32"))]
#[path = "../../engine/examples/app_template.rs"]
mod app;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> std::process::ExitCode {
    engine_native::main(engine_native::Window::new("Template"), app::build)
}

// The page runs the game in the browser.
#[cfg(target_arch = "wasm32")]
fn main() {}
