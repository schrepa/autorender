//! The gltf corpus: the importer (engine_assetc's `import_glb` and its
//! prefab importer) over the garden prop (tools/xtask/pure/garden-prop.glb):
//! as a scene of ordinary meshes, as one of virtual meshes (built by
//! vgbuild), and as a prefab. Each import must load back as the native host
//! loads scenes (its document reparses to the same text, its files are
//! named by their content, and the scene instantiates in an engine from its
//! files), and importing twice must give the same bytes. Each document and
//! file is held to goldens.

use std::path::Path;

use engine::EnginePlugins;
use engine_assetc::{ImportOptions, Imported, import_glb_prefab_with, import_glb_with};
use engine_core::{Engine, EngineConfig};
use engine_scene::{SceneDocument, SceneFileKind, SceneFiles};
use serde_json::json;
use sha2::{Digest, Sha256};

use super::Observations;
use super::simplify::GARDEN_PROP;

/// The imports: their name, whether a prefab, and the triangles from which
/// a primitive imports as a virtual mesh (the prop's two primitives hold 52
/// and 24).
const IMPORTS: [(&str, bool, Option<u32>); 3] = [
    ("scene", false, None),
    ("virtual", false, Some(16)),
    ("prefab", true, None),
];

fn import(bytes: &[u8], prefab: bool, options: &ImportOptions) -> Result<Imported, String> {
    let imported = if prefab {
        import_glb_prefab_with(bytes, "garden", options)
    } else {
        import_glb_with(bytes, "garden", options)
    };
    imported.map_err(|error| format!("{GARDEN_PROP}: {error}"))
}

pub fn run(root: &Path, observed: &mut Observations) -> Result<(), String> {
    let bytes = std::fs::read(root.join(GARDEN_PROP))
        .map_err(|error| format!("cannot read {GARDEN_PROP}: {error}"))?;
    for (name, prefab, virtual_triangles) in IMPORTS {
        let options = ImportOptions { virtual_triangles };
        let imported = import(&bytes, prefab, &options)?;
        let again = import(&bytes, prefab, &options)?;
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
        // The document reparses to the same text.
        let reparsed = SceneDocument::from_json(&text).map_err(|error| error.to_string())?;
        observed.check(reparsed.to_json().ok().as_ref() == Some(&text), || {
            format!("gltf {name}: the document does not reparse to the same text")
        });
        observed.hash(format!("gltf {name} document"), text.as_bytes());
        let mut files = SceneFiles::default();
        let kinds: Vec<_> = reparsed.files();
        for file in &imported.files {
            // Files are named by their content: `ID/NAME.HASH.EXTENSION`.
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
                Some(SceneFileKind::VirtualMesh) => {
                    files.add_virtual_mesh(&file.uri, file.bytes.clone(), None);
                }
                Some(SceneFileKind::Terrain) => {
                    files.add_terrain(&file.uri, file.bytes.clone(), None);
                }
                Some(SceneFileKind::Buffer) => files.add_buffer(&file.uri, file.bytes.clone()),
                None => observed.check(false, || {
                    format!("gltf {name}: the document names no file {}", file.uri)
                }),
            }
        }
        let mut engine = Engine::new(EngineConfig::default());
        engine.app.add_plugins(EnginePlugins);
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
