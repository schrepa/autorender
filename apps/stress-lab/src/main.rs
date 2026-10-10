//! The Stress Lab natively, in a window or offscreen:
//! `cargo run -p stress-lab --release -- --preset metro`; `--help` lists the
//! options.

#[cfg(not(target_arch = "wasm32"))]
fn main() -> std::process::ExitCode {
    stress_lab::native::main(std::env::args().skip(1))
}

// The page runs the Stress Lab in the browser.
#[cfg(target_arch = "wasm32")]
fn main() {}
