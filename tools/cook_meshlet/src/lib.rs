mod gi_bake;
mod gltf_load;
mod pbr_pack;
mod synth;

use std::path::{Path, PathBuf};
use std::time::Instant;

use cook::Refusal;
use engine_meshlet::{
    BuildStats, MeshletManifest, VerifyReport, build, decode_compressed_pages, verify,
};
use serde::Serialize;

pub use gi_bake::{BakeGiOutput, bake_gi};
pub use gltf_load::{compute_curvature_ao, load_gltf_mesh};
pub use pbr_pack::{PackPbrOutput, pack_pbr};
pub use synth::{build_multimaterial_foliage_mesh, build_terrain_column_mesh};

#[derive(Clone, Debug, Serialize)]
pub struct CookOutput {
    pub stem: String,
    pub mlm_path: String,
    pub mlp_path: String,
    pub build_ms: f64,
    pub stats: BuildStats,
}

#[derive(Clone, Debug, Serialize)]
pub struct VerifyOutput {
    pub status: &'static str,
    pub mlm_path: String,
    pub mlp_path: String,
    pub verify_ms: f64,
    pub report: VerifyReport,
}

#[derive(Clone, Debug, Serialize)]
pub struct SelfCheckCase {
    pub name: &'static str,
    pub build_ms: f64,
    pub verify_ms: f64,
    pub stats: BuildStats,
    pub report: VerifyReport,
}

#[derive(Clone, Debug, Serialize)]
pub struct SelfCheckOutput {
    pub status: &'static str,
    pub cases: Vec<SelfCheckCase>,
}

pub fn cook_gltf(
    input_path: &Path,
    output_dir: &Path,
    name_override: Option<&str>,
    material_override: Option<u8>,
) -> Result<CookOutput, Refusal> {
    let stem = match name_override {
        Some(name) if !name.is_empty() => name.to_owned(),
        _ => input_path
            .file_stem()
            .and_then(|s| s.to_str())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                Refusal::new(
                    "InvalidName",
                    &input_path.display().to_string(),
                    "cannot derive output stem from input filename",
                )
            })?
            .to_owned(),
    };
    let source = load_gltf_mesh(input_path, material_override)?;
    let start = Instant::now();
    let built = build(&source).map_err(|error| {
        Refusal::new(
            "BuildFailed",
            &input_path.display().to_string(),
            format!("{error}"),
        )
    })?;
    let build_ms = start.elapsed().as_secs_f64() * 1000.0;
    std::fs::create_dir_all(output_dir).map_err(|error| {
        Refusal::new(
            "OutputDir",
            &output_dir.display().to_string(),
            format!("{error}"),
        )
    })?;
    let mlm_path = output_dir.join(format!("{stem}.mlm"));
    let mlp_path = output_dir.join(format!("{stem}.mlp"));
    std::fs::write(&mlm_path, built.manifest.to_bytes()).map_err(|error| {
        Refusal::new(
            "WriteFailed",
            &mlm_path.display().to_string(),
            format!("{error}"),
        )
    })?;
    std::fs::write(&mlp_path, built.pages_to_bytes()).map_err(|error| {
        Refusal::new(
            "WriteFailed",
            &mlp_path.display().to_string(),
            format!("{error}"),
        )
    })?;
    Ok(CookOutput {
        stem,
        mlm_path: mlm_path.display().to_string(),
        mlp_path: mlp_path.display().to_string(),
        build_ms,
        stats: built.stats,
    })
}

pub fn verify_manifest_file(manifest_path: &Path) -> Result<VerifyOutput, Refusal> {
    let start = Instant::now();
    let mlm_str = manifest_path.display().to_string();
    let mlm_bytes = std::fs::read(manifest_path)
        .map_err(|error| Refusal::new("FileUnreadable", &mlm_str, format!("{error}")))?;
    let manifest = MeshletManifest::from_bytes(&mlm_bytes)
        .map_err(|error| Refusal::new("ManifestInvalid", &mlm_str, format!("{error}")))?;
    let reencoded = manifest.to_bytes();
    if reencoded != mlm_bytes {
        return Err(Refusal::new(
            "ManifestRoundtrip",
            &mlm_str,
            "manifest binary roundtrip mismatch",
        ));
    }
    let mlp_path: PathBuf = manifest_path.with_extension("mlp");
    let mlp_str = mlp_path.display().to_string();
    let mlp_bytes = std::fs::read(&mlp_path)
        .map_err(|error| Refusal::new("FileUnreadable", &mlp_str, format!("{error}")))?;
    let page_words = decode_compressed_pages(&mlp_bytes, &manifest.page_spans)
        .map_err(|error| Refusal::new("PagesInvalid", &mlp_str, format!("{error}")))?;
    let report = verify(&manifest, &page_words)
        .map_err(|error| Refusal::new("VerifyFailed", &mlm_str, format!("{error}")))?;
    let verify_ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok(VerifyOutput {
        status: "pass",
        mlm_path: mlm_str,
        mlp_path: mlp_str,
        verify_ms,
        report,
    })
}

fn check_synthetic(
    name: &'static str,
    source: &engine_meshlet::SourceMesh,
) -> Result<SelfCheckCase, Refusal> {
    let t0 = Instant::now();
    let built = build(source)
        .map_err(|error| Refusal::new("BuildFailed", name, format!("{error}")))?;
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let t1 = Instant::now();
    let mlm_bytes = built.manifest.to_bytes();
    let decoded_manifest = MeshletManifest::from_bytes(&mlm_bytes)
        .map_err(|error| Refusal::new("ManifestRoundtrip", name, format!("{error}")))?;
    if decoded_manifest != built.manifest {
        return Err(Refusal::new(
            "ManifestRoundtrip",
            name,
            "decoded manifest does not match original",
        ));
    }
    let mlp_bytes = built.pages_to_bytes();
    let decoded_words = decode_compressed_pages(&mlp_bytes, &decoded_manifest.page_spans)
        .map_err(|error| Refusal::new("PagesRoundtrip", name, format!("{error}")))?;
    let report = verify(&decoded_manifest, &decoded_words)
        .map_err(|error| Refusal::new("VerifyFailed", name, format!("{error}")))?;
    let verify_ms = t1.elapsed().as_secs_f64() * 1000.0;
    Ok(SelfCheckCase {
        name,
        build_ms,
        verify_ms,
        stats: built.stats,
        report,
    })
}

pub fn run_self_check() -> Result<SelfCheckOutput, Refusal> {
    let terrain = build_terrain_column_mesh();
    let case_terrain = check_synthetic("terrain_column_60k", &terrain)?;
    let foliage = build_multimaterial_foliage_mesh();
    let case_foliage = check_synthetic("multimaterial_foliage_5k", &foliage)?;
    Ok(SelfCheckOutput {
        status: "pass",
        cases: vec![case_terrain, case_foliage],
    })
}
