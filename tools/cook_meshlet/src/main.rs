use std::path::Path;
use std::process::ExitCode;

use cook::Refusal;
use cook_meshlet::{cook_gltf, run_self_check, verify_manifest_file};
use serde::Serialize;
use serde_json::json;

fn emit_refusal(no: &Refusal) {
    println!(
        "{}",
        json!({
            "status": "fail",
            "error": no.code,
            "file": no.file,
            "detail": no.detail,
        })
    );
}

fn emit_result<T: Serialize>(result: Result<T, Refusal>) -> ExitCode {
    match result {
        Ok(value) => match serde_json::to_string_pretty(&value) {
            Ok(text) => {
                println!("{text}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                emit_refusal(&Refusal::new("Serialize", "stdout", error));
                ExitCode::FAILURE
            }
        },
        Err(no) => {
            emit_refusal(&no);
            ExitCode::FAILURE
        }
    }
}

fn usage() -> ExitCode {
    emit_refusal(&Refusal::new(
        "Usage",
        "cook_meshlet",
        "cook_meshlet cook <input.glb|.gltf> <output_dir> [--name <stem>] [--material <id>] | cook_meshlet verify <manifest.mlm> | cook_meshlet self-check",
    ));
    ExitCode::from(2)
}

fn run_cook(args: &[&str]) -> Result<cook_meshlet::CookOutput, Refusal> {
    let Some((&input, rest)) = args.split_first() else {
        return Err(Refusal::new("Usage", "cook", "missing <input.glb|.gltf>"));
    };
    let Some((&output_dir, mut flags)) = rest.split_first() else {
        return Err(Refusal::new("Usage", "cook", "missing <output_dir>"));
    };
    let mut name_override: Option<&str> = None;
    let mut material_override: Option<u8> = None;
    while let Some((&flag, tail)) = flags.split_first() {
        match flag {
            "--name" => {
                let Some((&val, next)) = tail.split_first() else {
                    return Err(Refusal::new("Usage", "cook", "--name requires a value"));
                };
                name_override = Some(val);
                flags = next;
            }
            "--material" => {
                let Some((&val, next)) = tail.split_first() else {
                    return Err(Refusal::new("Usage", "cook", "--material requires a u8 id"));
                };
                let parsed: u8 = val.parse().map_err(|error| {
                    Refusal::new("InvalidMaterial", "cook", format!("`{val}`: {error}"))
                })?;
                material_override = Some(parsed);
                flags = next;
            }
            other => {
                return Err(Refusal::new(
                    "UnknownFlag",
                    "cook",
                    format!("unknown option `{other}`"),
                ));
            }
        }
    }
    cook_gltf(
        Path::new(input),
        Path::new(output_dir),
        name_override,
        material_override,
    )
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let words: Vec<&str> = args.iter().map(String::as_str).collect();
    match words.as_slice() {
        ["cook", rest @ ..] => emit_result(run_cook(rest)),
        ["verify", manifest] => emit_result(verify_manifest_file(Path::new(manifest))),
        ["self-check"] => emit_result(run_self_check()),
        _ => usage(),
    }
}
