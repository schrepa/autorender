//! Bakes a scene's generated terrain into an `ETR1` file of
//! every tile of every level, which scenes then reference with `file` and
//! stream by byte range instead of generating.
//!
//! ```text
//! cargo run -p engine_assetc --release --bin terrain_bake -- SCENE.json ENTITY OUT.etr [STEP]
//! ```
//!
//! `ENTITY` names the scene entity whose `terrain` has layers; heightmap
//! files load from beside the scene, and hydrology layers bake first. Heights are quantized to `STEP` meters
//! (default 1/256).

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use engine_core::{TerrainSource, terrain_data::Generator};
use engine_scene::{SceneDocument, SceneFileKind, SceneFiles};

fn main() {
    if let Err(error) = run() {
        eprintln!("terrain_bake: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let [scene, entity, output, rest @ ..] = arguments.as_slice() else {
        return Err("usage: terrain_bake SCENE.json ENTITY OUT.etr [STEP]".into());
    };
    let step = match rest {
        [] => 1.0 / 256.0,
        [step] => step
            .parse::<f32>()
            .map_err(|error| format!("step {step}: {error}"))?,
        _ => return Err("usage: terrain_bake SCENE.json ENTITY OUT.etr [STEP]".into()),
    };
    let source = std::fs::read_to_string(scene).map_err(|error| format!("{scene}: {error}"))?;
    let document = SceneDocument::from_json(&source).map_err(|error| error.to_string())?;
    let folder = Path::new(scene).parent().unwrap_or(Path::new("."));
    let mut files = SceneFiles::new();
    for file in document.files() {
        if file.kind == SceneFileKind::Buffer {
            let bytes = std::fs::read(folder.join(&file.uri))
                .map_err(|error| format!("{}: {error}", file.uri))?;
            files.add_buffer(&file.uri, bytes);
        }
    }
    let (index, object) = document
        .entities
        .iter()
        .enumerate()
        .find(|(_, object)| object.id == *entity)
        .ok_or_else(|| format!("{scene} has no entity {entity}"))?;
    let terrain = object
        .terrain
        .as_ref()
        .ok_or_else(|| format!("entity {entity} has no terrain"))?
        .prepare_component(
            &format!("$.entities[{index}].terrain"),
            Some(&files),
            (&document.materials, &document.meshes),
        )
        .map_err(|error| error.to_string())?;
    let TerrainSource::Generated(layers) = &terrain.source else {
        return Err(format!("entity {entity}'s terrain already reads a file"));
    };
    let pyramid = terrain.pyramid;
    let total = pyramid.tile_count();
    let start = Instant::now();
    // Hydrology bakes before tiles generate from it.
    let layers = &Arc::new(layers.bake_all(&pyramid));
    let mut done = 0;
    let bytes = engine_core::terrain_data::encode_file(&pyramid, step, |key| {
        let mut generator = Generator::new(Arc::clone(layers), &pyramid, key);
        let mut budget = u64::MAX;
        let tile = generator
            .advance(&mut budget)
            .expect("an unlimited budget finishes the tile");
        done += 1;
        if done % 64 == 0 || done == total {
            eprintln!(
                "{done}/{total} tiles, {:.1} s",
                start.elapsed().as_secs_f64()
            );
        }
        tile
    })
    .map_err(|error| error.to_string())?;
    std::fs::write(output, &bytes).map_err(|error| format!("{output}: {error}"))?;
    println!(
        "{output}: {} tiles in {} levels, {:.1} MiB, {:.1} s",
        total,
        pyramid.levels(),
        bytes.len() as f64 / 1_048_576.0,
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
