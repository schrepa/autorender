use std::path::Path;

use engine_assetc::{Imported, import_glb, import_glb_prefab};
use engine_scene::{SceneDocument, SceneFileKind, SceneFiles};
use serde_json::json;
use sha2::{Digest, Sha256};

use super::Observations;
use super::simplify::PROP;

const IMPORTS: [(&str, bool); 2] = [("scene", false), ("prefab", true)];

fn import(bytes: &[u8], prefab: bool) -> Result<Imported, String> {
    let imported = if prefab {
        import_glb_prefab(bytes, "prop")
    } else {
        import_glb(bytes, "prop")
    };
    imported.map_err(|error| format!("{PROP}: {error}"))
}

pub fn run(root: &Path, observed: &mut Observations) -> Result<(), String> {
    let bytes =
        std::fs::read(root.join(PROP)).map_err(|error| format!("cannot read {PROP}: {error}"))?;
    for (name, prefab) in IMPORTS {
        let imported = import(&bytes, prefab)?;
        let again = import(&bytes, prefab)?;
        let text = imported
            .document
            .to_json()
            .map_err(|error| error.to_string())?;
        observed.check(
            again.document.to_json().ok().as_ref() == Some(&text)
                && again.files.len() == imported.files.len()
                && again
                    .files
                    .iter()
                    .zip(&imported.files)
                    .all(|(a, b)| a.uri == b.uri && a.bytes == b.bytes),
            || format!("gltf {name}: importing twice gives different bytes"),
        );
        let reparsed = SceneDocument::from_json(&text).map_err(|error| error.to_string())?;
        observed.check(reparsed.to_json().ok().as_ref() == Some(&text), || {
            format!("gltf {name}: the document does not reparse to the same text")
        });
        observed.hash(format!("gltf {name} document"), text.as_bytes());
        let mut files = SceneFiles::default();
        let kinds: Vec<_> = reparsed.files();
        for file in &imported.files {
            let hash = format!("{:x}", Sha256::digest(&file.bytes));
            observed.check(file.uri.contains(&format!(".{}.", &hash[..12])), || {
                format!("gltf {name}: {} is not named by its content", file.uri)
            });
            observed.hash(format!("gltf {name} file {}", file.uri), &file.bytes);
            match kinds
                .iter()
                .find(|kind| kind.uri == file.uri)
                .map(|kind| kind.kind)
            {
                Some(SceneFileKind::Image) => {
                    let image = engine_native::files::decode_image(&file.uri, &file.bytes)?;
                    files.add_image(&file.uri, image.width, image.height, image.pixels);
                }
                Some(SceneFileKind::Buffer) => files.add_buffer(&file.uri, file.bytes.clone()),
                None => observed.check(false, || {
                    format!("gltf {name}: the document names no file {}", file.uri)
                }),
            }
        }
        let mut engine = crate::bricks::standard_engine();
        let scene = reparsed
            .instantiate_with(engine.app.world_mut(), &files)
            .map_err(|error| format!("gltf {name}: the import does not instantiate: {error}"))?;
        observed.record(
            format!("gltf {name}"),
            json!({
                "entities": scene.entity_count(),
                "meshes": scene.meshes().len(),
                "materials": scene.materials().len(),
                "textures": scene.textures().len(),
                "files": imported.files.len(),
                "ignored extensions": imported.ignored_extensions,
            }),
        );
    }
    Ok(())
}
