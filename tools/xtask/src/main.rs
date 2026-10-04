mod authoring;
mod benchmarking;
mod development;
mod importing;
mod iteration_build;
mod pipeline;
mod process;
mod profiling;

use std::{env, process::ExitCode};

fn main() -> ExitCode {
    match execute() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("xtask: {error}");
            ExitCode::FAILURE
        }
    }
}

fn execute() -> Result<(), String> {
    let mut arguments = env::args().skip(1);
    let task = arguments.next().unwrap_or_else(|| "help".to_owned());
    let options: Vec<_> = arguments.collect();
    let root = process::repository_root()?;
    if task == "dev" {
        return pipeline::dev(&root, dev_port(&options)?);
    }
    if task == "agent" {
        return development::agent(&root, &options);
    }
    if task == "scene" {
        return development::scene(&options);
    }
    if task == "asset" {
        return importing::asset(&options);
    }
    if task == "benchmark" {
        return benchmarking::benchmark(&root, &options);
    }
    if task == "profile" {
        return match options.as_slice() {
            [] => profiling::build(&root, false),
            [option] if option == "--attribution" => profiling::build(&root, true),
            _ => Err("usage: profile [--attribution]".into()),
        };
    }
    if task == "build-dev" {
        let [option, output] = options.as_slice() else {
            return Err("usage: build-dev --output web/pkg-iterations/GENERATION".into());
        };
        if option != "--output" {
            return Err("build-dev requires --output".into());
        }
        let result = iteration_build::build(&root, std::path::Path::new(output));
        println!("{result}");
        return if result["ok"] == true {
            Ok(())
        } else {
            Err("development build failed; see JSON diagnostics".into())
        };
    }
    if let Some(argument) = options.first() {
        return Err(format!(
            "unexpected argument {argument:?}; run `cargo xtask help` for usage"
        ));
    }
    match task.as_str() {
        "build" => pipeline::build(&root),
        "lint" => pipeline::lint(&root),
        "test" => pipeline::test(&root),
        "docs" => pipeline::docs(&root),
        "assets" => authoring::validate_sources(&root),
        "schema" => authoring::write_schema(&root),
        "help" | "--help" | "-h" => {
            println!(
                "Usage: cargo xtask <command>\n\n\
                 Commands:\n\
                   build  Build all release Wasm clients and bundle the web host\n\
                   profile [--attribution]  Build an isolated instrumented Stress Lab in web/dist-profile\n\
                          --attribution: hot helpers out of line for sampling profilers, in web/dist-attribution\n\
                   dev [--port N]  Build and serve on loopback (default port 5173)\n\
                   agent watch [--web-port N] [--port N] [--game playground|orbit|stress]\n\
                          Watch Rust, build, and replace identified browser sessions\n\
                   agent  Browser session commands and development build status\n\
                   scene  Validate or save revision-checked scene sources as JSON\n\
                   asset import --file PATH --id ID (--output PATH | --into PATH --expected-revision N) [--replace] [--as-prefab]\n\
                          [--virtual-triangles N | --no-virtual-geometry]\n\
                          Import a static textured GLB with structured results; dense opaque primitives\n\
                          (512 triangles and up by default) become virtual meshes\n\
                   assets Validate authored scene sources without changing them\n\
                   benchmark [--url URL] [--samples N] [--warmup N] [--repeats N] [--mode step|render|present] [--setup-ticks N]\n\
                          Measure fixed browser workloads with culling off/on; --output PATH, --baseline PATH, --max-regression-percent N\n\
                   schema Write the generated versioned scene JSON schema\n\
                   lint   Check source/schema, formatting, native/Wasm Clippy, TypeScript\n\
                   test   Run native workspace, scene schema, and host input tests\n\
                   docs   Generate native workspace and browser API documentation\n\
                   help   Show this message"
            );
            Ok(())
        }
        _ => Err(format!(
            "unknown command {task:?}; run `cargo xtask help` for usage"
        )),
    }
}

fn dev_port(arguments: &[String]) -> Result<u16, String> {
    match arguments {
        [] => Ok(5173),
        [option, value] if option == "--port" => value
            .parse::<u16>()
            .ok()
            .filter(|port| *port != 0)
            .ok_or_else(|| "dev --port requires an integer from 1 to 65535".to_owned()),
        _ => Err("usage: cargo xtask dev [--port N]".to_owned()),
    }
}
