use std::path::PathBuf;

fn main() {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("set by cargo"))
        .join("../../vendor/meshoptimizer");
    let sources: Vec<PathBuf> = std::fs::read_dir(root.join("src"))
        .expect("vendored meshoptimizer sources")
        .map(|entry| entry.expect("readable source entry").path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "cpp"))
        .collect();
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .files(&sources)
        .file("src/clusterlod.cpp")
        .include(root.join("src"))
        .include(root.join("demo"))
        .opt_level(2)
        .flag_if_supported("-ffp-contract=off")
        .warnings(false);
    build.compile("meshoptimizer");
    for source in &sources {
        println!("cargo:rerun-if-changed={}", source.display());
    }
    println!(
        "cargo:rerun-if-changed={}",
        root.join("demo/clusterlod.h").display()
    );
    println!("cargo:rerun-if-changed=src/clusterlod.cpp");
}
