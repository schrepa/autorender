//! Native source import; cooked output is the runtime's validated scene format.

use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
};

use engine_assetc::{
    DEFAULT_VIRTUAL_TRIANGLES, IMPORTER_VERSION, ImportError, ImportOptions, ImportedFile,
    MAX_INPUT_BYTES, import_glb_prefab_with, import_glb_with, source_sha256,
};
use engine_dev::authoring;
use engine_scene::{MeshSource, SceneDocument};
use serde_json::{Value, json};

pub fn asset(arguments: &[String]) -> Result<(), String> {
    let result = command(arguments);
    let succeeded = result.is_ok();
    println!(
        "{}",
        match result {
            Ok(result) => json!({"version":1,"ok":true,"result":result}),
            Err(error) => json!({"version":1,"ok":false,"error":error}),
        }
    );
    if succeeded {
        Ok(())
    } else {
        Err("asset import failed; see its JSON result".into())
    }
}

fn command(arguments: &[String]) -> Result<Value, ImportError> {
    let options = Options::parse(arguments)?;
    let bytes = read_source(&options.file)?;
    let import_options = ImportOptions {
        virtual_triangles: options.virtual_triangles,
    };
    let result = if options.as_prefab {
        import_glb_prefab_with(&bytes, &options.id, &import_options)?
    } else {
        import_glb_with(&bytes, &options.id, &import_options)?
    };
    let imported = result.document;
    let file_bytes: usize = result.files.iter().map(|file| file.bytes.len()).sum();
    let virtual_meshes = imported
        .meshes
        .iter()
        .filter(|mesh| matches!(mesh.source, MeshSource::Virtual { .. }))
        .count();
    let counts = json!({"textures":imported.textures.len(),"materials":imported.materials.len(),"meshes":imported.meshes.len(),"virtual_meshes":virtual_meshes,"entities":imported.entities.len(),"prefabs":imported.prefabs.len(),"prefab_entities":imported.prefabs.iter().map(|prefab| prefab.entities.len()).sum::<usize>(),"animations":imported.animations.len(),"skins":imported.skins.len(),"files":result.files.len(),"file_bytes":file_bytes});
    // Animated and skinned models keep their nodes (decisions 035, 036).
    let node_transforms = if imported.animations.is_empty() && imported.skins.is_empty() {
        "baked"
    } else {
        "hierarchy"
    };
    let (report, document) = if let Some(output) = &options.output {
        let source = imported.to_json()?;
        // Files first: a scene is never written without them.
        write_files(output, &result.files)?;
        write_new(output, &source)?;
        (
            json!({"path":output,"revision":imported.revision.to_string(),"persisted":true,"source_bytes":source.len(),"mode":"create"}),
            imported,
        )
    } else {
        let target = options.into.as_ref().expect("validated target");
        let original = authoring::read_scene_source(target).map_err(dev_error)?;
        let current = SceneDocument::from_json(&original)?;
        let revision = options.revision.expect("validated revision");
        if current.revision != revision {
            return Err(ImportError::new(
                "revision_conflict",
                "$.revision",
                format!(
                    "expected revision {revision}, saved source is {}",
                    current.revision
                ),
            ));
        }
        let candidate = merge(current, imported, &options.id, options.replace)?;
        let source = candidate.to_json()?;
        write_files(target, &result.files)?;
        let report = authoring::save_scene(target, &source, revision).map_err(dev_error)?;
        (
            json!({"path":report.path,"revision":report.revision,"persisted":report.persisted,"source_bytes":report.source_bytes,"mode":if options.replace {"replace_namespace"} else {"append"}}),
            candidate,
        )
    };
    Ok(json!({
        "save":report,"scene":document.id,"imported":counts,
        "ignored_extensions":result.ignored_extensions,
        "provenance":{"source":options.file,"source_sha256":source_sha256(&bytes),"importer_version":IMPORTER_VERSION,"namespace":options.id,"settings":{"format":"glb2","authoring":if options.as_prefab {"prefab"} else {"flat"},"coordinates":"right-handed-y-up","node_transforms":node_transforms,"materials":"gltf-metallic-roughness","images":"files-unchanged","meshes":"emb-files","virtual_meshes":{"format":"evg1-files","min_triangles":options.virtual_triangles},"animations":"ean1-files"}}
    }))
}

struct Options {
    file: PathBuf,
    id: String,
    output: Option<PathBuf>,
    into: Option<PathBuf>,
    revision: Option<u64>,
    replace: bool,
    as_prefab: bool,
    // Triangles from which primitives import as virtual meshes (decision
    // 044); `None` imports none.
    virtual_triangles: Option<u32>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            file: PathBuf::new(),
            id: String::new(),
            output: None,
            into: None,
            revision: None,
            replace: false,
            as_prefab: false,
            virtual_triangles: Some(DEFAULT_VIRTUAL_TRIANGLES),
        }
    }
}

impl Options {
    fn parse(arguments: &[String]) -> Result<Self, ImportError> {
        let invalid = || {
            ImportError::new(
                "invalid_arguments",
                "$",
                "usage: cargo xtask asset import --file MODEL.glb --id NAMESPACE [--as-prefab] [--virtual-triangles N | --no-virtual-geometry] (--output NEW.scene.json | --into EXISTING.scene.json --expected-revision N [--replace])",
            )
        };
        if arguments.first().is_none_or(|command| command != "import") {
            return Err(invalid());
        }
        let mut values = BTreeMap::new();
        let mut arguments = arguments[1..].iter();
        let mut replace = false;
        let mut as_prefab = false;
        let mut no_virtual = false;
        while let Some(flag) = arguments.next() {
            if flag == "--replace" || flag == "--as-prefab" || flag == "--no-virtual-geometry" {
                let present = match flag.as_str() {
                    "--replace" => &mut replace,
                    "--as-prefab" => &mut as_prefab,
                    _ => &mut no_virtual,
                };
                if *present {
                    return Err(invalid());
                }
                *present = true;
                continue;
            }
            if !matches!(
                flag.as_str(),
                "--file"
                    | "--id"
                    | "--output"
                    | "--into"
                    | "--expected-revision"
                    | "--virtual-triangles"
            ) {
                return Err(invalid());
            }
            let value = arguments
                .next()
                .filter(|value| !value.starts_with("--"))
                .ok_or_else(invalid)?;
            if values.insert(flag.as_str(), value.as_str()).is_some() {
                return Err(invalid());
            }
        }
        let file = PathBuf::from(values.get("--file").ok_or_else(invalid)?);
        let id = values.get("--id").ok_or_else(invalid)?.to_string();
        let output = values.get("--output").map(PathBuf::from);
        let into = values.get("--into").map(PathBuf::from);
        let revision = values
            .get("--expected-revision")
            .map(|revision| {
                revision.parse::<u64>().map_err(|_| {
                    ImportError::new(
                        "invalid_revision",
                        "--expected-revision",
                        "expected an unsigned 64-bit decimal integer",
                    )
                })
            })
            .transpose()?;
        let virtual_triangles = values
            .get("--virtual-triangles")
            .map(|triangles| {
                triangles
                    .parse::<u32>()
                    .ok()
                    .filter(|triangles| *triangles >= 1)
                    .ok_or_else(|| {
                        ImportError::new(
                            "invalid_arguments",
                            "--virtual-triangles",
                            "expected a triangle count of at least 1",
                        )
                    })
            })
            .transpose()?;
        if output.is_some() == into.is_some()
            || (into.is_some() != revision.is_some())
            || (replace && into.is_none())
            || (no_virtual && virtual_triangles.is_some())
        {
            return Err(invalid());
        }
        let virtual_triangles = if no_virtual {
            None
        } else {
            Some(virtual_triangles.unwrap_or(DEFAULT_VIRTUAL_TRIANGLES))
        };
        Ok(Self {
            file,
            id,
            output,
            into,
            revision,
            replace,
            as_prefab,
            virtual_triangles,
        })
    }
}

fn merge(
    mut current: SceneDocument,
    mut imported: SceneDocument,
    id: &str,
    replace: bool,
) -> Result<SceneDocument, ImportError> {
    let as_prefab = !imported.prefabs.is_empty();
    let existing_prefab = current.prefabs.iter().any(|prefab| prefab.id == id);
    let existing_flat = current
        .entities
        .iter()
        .any(|entity| generated(&entity.id, id, Kind::Entity));
    let existing_assets = current
        .meshes
        .iter()
        .any(|mesh| generated(&mesh.id, id, Kind::Mesh))
        || current
            .materials
            .iter()
            .any(|material| generated(&material.id, id, Kind::Material))
        || current
            .textures
            .iter()
            .any(|texture| generated(&texture.id, id, Kind::Texture))
        || current
            .animations
            .iter()
            .any(|animation| generated(&animation.id, id, Kind::Animation))
        || current
            .skins
            .iter()
            .any(|skin| generated(&skin.id, id, Kind::Skin));
    if (as_prefab && existing_flat)
        || (!as_prefab && existing_prefab)
        || (existing_assets && !existing_flat && !existing_prefab)
    {
        return Err(ImportError::new(
            "namespace_mode_conflict",
            "--id",
            "namespace belongs to another or ambiguous import mode; use a new namespace or explicitly migrate its authoring data before import",
        ));
    }
    // Upgrading keeps the meaning of the scene's own content.
    current.upgrade();
    if replace {
        let retained = if as_prefab {
            current
                .prefabs
                .iter()
                .filter(|prefab| prefab.id == id)
                .flat_map(|prefab| prefab.entities.iter())
                .collect::<Vec<_>>()
        } else {
            current
                .entities
                .iter()
                .filter(|entity| generated(&entity.id, id, Kind::Entity))
                .collect::<Vec<_>>()
        };
        let replacements = if as_prefab {
            imported
                .prefabs
                .iter_mut()
                .flat_map(|prefab| prefab.entities.iter_mut())
                .collect::<Vec<_>>()
        } else {
            imported.entities.iter_mut().collect::<Vec<_>>()
        };
        preserve_colliders(&retained, replacements)?;
        let previous: BTreeMap<_, _> = current
            .entities
            .iter()
            .filter(|entity| generated(&entity.id, id, Kind::Entity))
            .map(|entity| (entity.id.clone(), entity.clone()))
            .collect();
        if as_prefab {
            current.prefabs.retain(|prefab| prefab.id != id);
        } else {
            current
                .entities
                .retain(|entity| !generated(&entity.id, id, Kind::Entity));
        }
        current
            .meshes
            .retain(|mesh| !generated(&mesh.id, id, Kind::Mesh));
        current
            .materials
            .retain(|material| !generated(&material.id, id, Kind::Material));
        current
            .textures
            .retain(|texture| !generated(&texture.id, id, Kind::Texture));
        current
            .animations
            .retain(|animation| !generated(&animation.id, id, Kind::Animation));
        current
            .skins
            .retain(|skin| !generated(&skin.id, id, Kind::Skin));
        for mut entity in imported.entities {
            if let Some(old) = previous.get(&entity.id) {
                // Hierarchy nodes' transforms and parents are source data;
                // other entities keep their placement.
                if !node_entity(&entity.id, id) {
                    entity.transform = old.transform;
                    entity.parent.clone_from(&old.parent);
                }
                entity.name.clone_from(&old.name);
                entity.tags.clone_from(&old.tags);
                entity.collider = old.collider;
                entity.tint = old.tint;
                // A chosen clip stays while the import still has it.
                if let Some(player) = &old.animation
                    && imported
                        .animations
                        .iter()
                        .any(|animation| animation.id == player.clip)
                {
                    entity.animation = Some(player.clone());
                }
            }
            current.entities.push(entity);
        }
    } else {
        current.entities.extend(imported.entities);
    }
    current.meshes.extend(imported.meshes);
    current.materials.extend(imported.materials);
    current.textures.extend(imported.textures);
    current.animations.extend(imported.animations);
    current.skins.extend(imported.skins);
    current.prefabs.extend(imported.prefabs);
    current.revision = current.revision.checked_add(1).ok_or_else(|| {
        ImportError::new(
            "revision_exhausted",
            "$.revision",
            "scene revision cannot advance past u64::MAX",
        )
    })?;
    current.validate()?;
    Ok(current)
}

fn preserve_colliders(
    previous: &[&engine_scene::SceneObject],
    replacements: Vec<&mut engine_scene::SceneObject>,
) -> Result<(), ImportError> {
    let mut replacements: BTreeMap<_, _> = replacements
        .into_iter()
        .map(|entity| (entity.id.clone(), entity))
        .collect();
    for old in previous {
        let Some(collider) = old.collider else {
            continue;
        };
        let Some(replacement) = replacements.get_mut(&old.id) else {
            return Err(ImportError::new(
                "collider_reimport_conflict",
                "--replace",
                format!(
                    "reimport removes '{}' with an authored collider; explicitly remove or relocate that collider in source first",
                    old.id
                ),
            ));
        };
        replacement.collider = Some(collider);
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum Kind {
    Entity,
    Mesh,
    Material,
    Texture,
    Animation,
    Skin,
}

fn decimal(value: &str) -> bool {
    !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit())
}

/// A hierarchy import's node entity, `NAMESPACE.nodeN`.
fn node_entity(value: &str, namespace: &str) -> bool {
    value
        .strip_prefix(namespace)
        .and_then(|suffix| suffix.strip_prefix(".node"))
        .is_some_and(decimal)
}

fn generated(value: &str, namespace: &str, kind: Kind) -> bool {
    let Some(suffix) = value
        .strip_prefix(namespace)
        .and_then(|suffix| suffix.strip_prefix('.'))
    else {
        return false;
    };
    match kind {
        Kind::Texture => suffix.strip_prefix("image").is_some_and(decimal),
        Kind::Material => suffix.strip_prefix("material").is_some_and(decimal),
        Kind::Animation => suffix.strip_prefix("anim").is_some_and(decimal),
        Kind::Skin => suffix.strip_prefix("skin").is_some_and(decimal),
        // Hierarchy imports' root and node entities.
        Kind::Entity if suffix == "root" || node_entity(value, namespace) => true,
        Kind::Entity | Kind::Mesh => {
            let suffix = if matches!(kind, Kind::Mesh) {
                let Some(suffix) = suffix.strip_suffix(".mesh") else {
                    return false;
                };
                suffix
            } else {
                suffix
            };
            suffix
                .strip_prefix("node")
                .and_then(|suffix| suffix.split_once(".p"))
                .is_some_and(|(node, primitive)| decimal(node) && decimal(primitive))
        }
    }
}

fn read_source(path: &Path) -> Result<Vec<u8>, ImportError> {
    let file = File::open(path).map_err(|error| io_error(path, error))?;
    if file
        .metadata()
        .map_err(|error| io_error(path, error))?
        .len()
        > MAX_INPUT_BYTES as u64
    {
        return Err(ImportError::new(
            "input_too_large",
            path.display().to_string(),
            "GLB source exceeds the format's 4 GiB length field",
        ));
    }
    let mut bytes = Vec::new();
    file.take((MAX_INPUT_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|error| io_error(path, error))?;
    if bytes.len() > MAX_INPUT_BYTES {
        return Err(ImportError::new(
            "input_too_large",
            path.display().to_string(),
            "GLB source exceeds the format's 4 GiB length field",
        ));
    }
    Ok(bytes)
}

/// Writes imported files beside `scene`, replacing earlier imports of the
/// same namespace.
fn write_files(scene: &Path, files: &[ImportedFile]) -> Result<(), ImportError> {
    let folder = scene.parent().unwrap_or(Path::new("."));
    for file in files {
        let path = folder.join(&file.uri);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|error| io_error(parent, error))?;
        }
        std::fs::write(&path, &file.bytes).map_err(|error| io_error(&path, error))?;
    }
    Ok(())
}

fn write_new(path: &Path, source: &str) -> Result<(), ImportError> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::AlreadyExists { ImportError::new("output_exists", path.display().to_string(), "output exists; use a new path or revision-checked --into with explicit --replace") } else { io_error(path, error) }
    })?;
    let result = file
        .write_all(source.as_bytes())
        .and_then(|()| file.sync_all());
    drop(file);
    if let Err(error) = result {
        let cleanup = std::fs::remove_file(path);
        return Err(ImportError::new(
            "source_io",
            path.display().to_string(),
            match cleanup {
                Ok(()) => error.to_string(),
                Err(cleanup) => format!("{error}; cannot remove incomplete output: {cleanup}"),
            },
        ));
    }
    Ok(())
}

fn io_error(path: &Path, error: std::io::Error) -> ImportError {
    ImportError::new("source_io", path.display().to_string(), error.to_string())
}
fn dev_error(error: engine_dev::DevError) -> ImportError {
    ImportError::new(error.code, error.path, error.message)
}
