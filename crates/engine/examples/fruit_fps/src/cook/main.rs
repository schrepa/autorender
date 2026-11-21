#[cfg(not(target_arch = "wasm32"))]
mod chunks;
#[cfg(not(target_arch = "wasm32"))]
mod frame;
#[cfg(not(target_arch = "wasm32"))]
mod level;
#[cfg(not(target_arch = "wasm32"))]
mod list;
#[cfg(not(target_arch = "wasm32"))]
mod meshes;
#[cfg(not(target_arch = "wasm32"))]
mod sets;
#[cfg(not(target_arch = "wasm32"))]
mod sound;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> std::process::ExitCode {
    list::cook()
}

#[cfg(target_arch = "wasm32")]
fn main() {}
