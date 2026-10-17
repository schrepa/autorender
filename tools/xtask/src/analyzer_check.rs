//! Analyzer integration checks using an explicitly supplied compiler fixture.
//! Real SPIR-V reports are validated separately by gpu --report.

use crate::{analyzer::Analyzer, builds, files};
use serde_json::{Value, json};
use std::{fs, path::Path, process::Command};

const COMPILER: &str = r#"
use std::{env, fs};
fn main() {
    let args: Vec<_> = env::args().collect();
    let arg = |name| args.windows(2).find(|pair| pair[0] == name).map(|pair| pair[1].as_str()).expect("compiler option");
    let target = arg("-c");
    let bytes = fs::read("input.spv").expect("input");
    fs::write(arg("--isa"), format!("v_mov_b32 v0, {}\n", bytes[0])).expect("isa");
    fs::write(arg("-a"), format!("DEVICE,USED_SGPRS,USED_VGPRS\n{target},1,2\n")).expect("csv");
    if bytes[0] == 255 { std::process::exit(9); }
}
"#;

pub fn run(root: &Path, output: &Path) -> Result<(), String> {
    let project = output.join("cargo-fixture");
    fs::create_dir_all(project.join("src/bin")).map_err(|e| e.to_string())?;
    fs::write(project.join("src/bin/rga.rs"), COMPILER).map_err(|e| e.to_string())?;
    let built = builds::cargo(&project, &["--bin", "rga"], "fixture")?;
    let bundle = output.join("analyzer-fixture");
    fs::create_dir_all(&bundle).map_err(|e| e.to_string())?;
    fs::copy(
        built.executable("rga")?,
        bundle.join(format!("rga{}", std::env::consts::EXE_SUFFIX)),
    )
    .map_err(|e| e.to_string())?;
    drop(built);
    let input = output.join("input.spv");
    fs::write(&input, [1]).map_err(|e| e.to_string())?;
    let cold = invocation(root, output, &bundle, &input, "cold", false)?;
    let warm = invocation(root, output, &bundle, &input, "warm", false)?;
    let direct = invocation(root, output, &bundle, &input, "disabled", true)?;
    require(
        cold["cache"]["compiled"] == 3
            && warm["cache"]["hits"] == 3
            && direct["cache"]["compiled"] == 3,
        "analyzer hit/miss admission differs",
    )?;
    for target in crate::analyzer::TARGETS {
        for report in [&warm, &direct] {
            require(
                cold["amd"][target]["resources"] == report["amd"][target]["resources"],
                "cache changed analyzer observations",
            )?;
            require(
                files::hash(Path::new(
                    cold["amd"][target]["isa"].as_str().ok_or("missing ISA")?,
                ))? == files::hash(Path::new(
                    report["amd"][target]["isa"].as_str().ok_or("missing ISA")?,
                ))?,
                "cache changed analyzer bytes",
            )?;
        }
    }
    fs::write(&input, [2]).map_err(|e| e.to_string())?;
    require(
        invocation(root, output, &bundle, &input, "changed-input", false)?["cache"]["compiled"]
            == 3,
        "changed compiler input hit cache",
    )?;
    fs::write(bundle.join("compiler-resource"), "changed bundle").map_err(|e| e.to_string())?;
    require(
        invocation(root, output, &bundle, &input, "changed-tool", false)?["cache"]["compiled"] == 3,
        "changed compiler bundle hit cache",
    )?;
    fs::write(&input, [255]).map_err(|e| e.to_string())?;
    for name in ["failure", "failure-again"] {
        let failed = invocation(root, output, &bundle, &input, name, false)?;
        require(
            failed["cache"]["failed"] == 3 && failed["cache"]["hits"] == 0,
            "failed compiler outputs were admitted as success",
        )?;
    }
    Ok(())
}

fn require(condition: bool, message: &str) -> Result<(), String> {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}

fn invocation(
    root: &Path,
    output: &Path,
    bundle: &Path,
    input: &Path,
    label: &str,
    disabled: bool,
) -> Result<Value, String> {
    let folder = output.join(format!("analyzer-{label}"));
    let result = Command::new(std::env::current_exe().map_err(|e| e.to_string())?)
        .current_dir(root)
        .args(["tooling-child", "analyzer"])
        .arg(input)
        .arg(&folder)
        .arg(if disabled { "disabled" } else { "enabled" })
        .env("ENGINE_RGA_ROOT", bundle)
        .env(
            "LD_LIBRARY_PATH",
            if label == "warm" {
                "irrelevant-cargo-output"
            } else {
                ""
            },
        )
        .env(
            "LANG",
            if label == "warm" {
                "unrelated-parent-locale"
            } else {
                "C"
            },
        )
        .output()
        .map_err(|e| e.to_string())?;
    if !result.status.success() {
        return Err(String::from_utf8_lossy(&result.stderr).into_owned());
    }
    files::json_file(&folder.join("report.json"))
}

pub fn child(args: &[String]) -> Result<u8, String> {
    let input = Path::new(args.get(1).ok_or("missing shader input")?);
    let folder = Path::new(args.get(2).ok_or("missing output")?);
    let analyzer = Analyzer::open(Path::new("."), args.get(3).is_some_and(|s| s == "enabled"))?
        .ok_or("fixture analyzer unavailable")?;
    let amd = analyzer.report(input, "main", "compute", folder, "fixture");
    files::atomic_json(
        &folder.join("report.json"),
        &json!({"amd": amd, "cache": analyzer.telemetry()}),
    )?;
    Ok(0)
}
