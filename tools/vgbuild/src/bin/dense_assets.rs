//! Writes the Stress Lab's dense virtual geometry (decision 044): procedural
//! high-poly meshes as `EVG1` files and the manifest the web host reads.
//!
//! `cargo run -p engine_vgbuild --release --bin dense_assets -- [DIRECTORY]`
//! (default `web/public/stress/dense`).

use std::path::PathBuf;
use std::time::Instant;

use engine_vgbuild::{build_virtual_mesh, procedural};

fn main() -> Result<(), String> {
    let directory = std::env::args()
        .nth(1)
        .map_or_else(|| PathBuf::from("web/public/stress/dense"), PathBuf::from);
    std::fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
    let meshes = [
        ("boulder-a.evg", procedural::boulder(7, 3)),
        ("boulder-b.evg", procedural::boulder(7, 29)),
        ("column.evg", procedural::column(768, 384)),
        ("terrain.evg", procedural::terrain(512, 64.0, 5)),
    ];
    let mut names = Vec::new();
    for (name, geometry) in meshes {
        let started = Instant::now();
        let bytes = build_virtual_mesh(&geometry.source())?;
        std::fs::write(directory.join(name), &bytes).map_err(|error| error.to_string())?;
        println!(
            "{name}: {} triangles, {:.1} MB, {:.1} s",
            geometry.indices.len() / 3,
            bytes.len() as f64 / 1.0e6,
            started.elapsed().as_secs_f64()
        );
        names.push(format!("\"{name}\""));
    }
    let manifest = format!("{{\"version\":1,\"meshes\":[{}]}}\n", names.join(","));
    std::fs::write(directory.join("manifest.json"), manifest).map_err(|error| error.to_string())?;
    Ok(())
}
