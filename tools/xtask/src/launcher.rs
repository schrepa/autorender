//! Fast discovery and storage operations; engine code lives in a helper.

mod commands;

use std::{
    env,
    path::PathBuf,
    process::{Command, ExitCode},
};
use xtask::{builds, cache::Store};

fn main() -> ExitCode {
    let result = execute();
    match result {
        Ok(code) => ExitCode::from(code),
        Err(error) => {
            eprintln!("xtask: {error}");
            ExitCode::FAILURE
        }
    }
}

struct Cleanup<'a>(&'a std::path::Path);
impl Drop for Cleanup<'_> {
    fn drop(&mut self) {
        xtask::cache::finish(self.0);
    }
}

fn execute() -> Result<u8, String> {
    let mut args: Vec<String> = env::args().skip(1).collect();
    let from = if let Some(index) = args
        .iter()
        .take_while(|a| a.as_str() != "--")
        .position(|a| a == "--workspace")
    {
        if index + 1 >= args.len() {
            return Err("--workspace requires a path".into());
        }
        let path = PathBuf::from(args.remove(index + 1));
        args.remove(index);
        path
    } else {
        env::current_dir().map_err(|e| e.to_string())?
    };
    let root = xtask::workspace(&from)?;
    let command = args.first().map(String::as_str).unwrap_or("help");
    if command == "help" {
        commands::help(args.get(1..).unwrap_or_default())?;
        return Ok(0);
    }
    if ["measure", "checks"].contains(&command) && args.get(1).is_some_and(|s| s == "--schema") {
        println!("{}", xtask::execution::schema(command));
        return Ok(0);
    }
    if command == "cache" {
        let mut store = Store::open(&root)?;
        let result = match args
            .iter()
            .skip(1)
            .map(String::as_str)
            .collect::<Vec<_>>()
            .as_slice()
        {
            [] | ["status"] => store.status()?,
            ["prune"] => store.collect(0)?,
            ["configure", "--max-bytes", bytes] => {
                store.configure(bytes.parse().map_err(|_| "invalid byte budget")?)?
            }
            _ => return Err("usage: cache status | prune | configure --max-bytes N".into()),
        };
        println!(
            "{}",
            serde_json::to_string_pretty(&result).map_err(|e| e.to_string())?
        );
        return Ok(if result["within_budget"] == false {
            1
        } else {
            0
        });
    }
    // Discovery and explicit cache inspection do no preparation and need no
    // second collection. Other commands release all leases before one sweep.
    let _cleanup = Cleanup(&root);
    if command == "prepare-binaries" {
        xtask::builds::prepare_binaries(&root, &args[1..])?;
        return Ok(0);
    }
    if ["prepare-native", "run-native"].contains(&command) {
        return xtask::native::command(&root, &args[1..], command == "run-native");
    }
    if command == "selfcheck" && args.len() == 2 && args[1] == "tooling" {
        xtask::tooling_check::run(&root)?;
        return Ok(0);
    }
    if command == "tooling-child" {
        return xtask::tooling_check::child(&args[1..]);
    }
    let helper = builds::helper(&root)?;
    if command == "prepare" && args.get(1).is_some_and(|s| s == "helper") {
        println!(
            "{}",
            serde_json::json!({"helper": helper.data(), "workspace": root})
        );
        return Ok(0);
    }
    let status = Command::new(
        helper
            .data()
            .join(format!("engine-xtask{}", env::consts::EXE_SUFFIX)),
    )
    .args(&args)
    .current_dir(&root)
    .env("ENGINE_TOOL_WORKSPACE", &root)
    .status()
    .map_err(|e| e.to_string())?;
    drop(helper);
    Ok(status
        .code()
        .and_then(|c| u8::try_from(c).ok())
        .unwrap_or(1))
}
