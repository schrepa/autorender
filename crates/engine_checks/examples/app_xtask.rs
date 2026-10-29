#[path = "../../engine/examples/app_template.rs"]
mod app;

fn main() -> std::process::ExitCode {
    engine_checks::app::main(env!("CARGO_MANIFEST_DIR"), app::build)
}
