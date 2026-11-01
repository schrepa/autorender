use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
};

use engine_assetc::{
    IMPORTER_VERSION, ImportedFile, MAX_INPUT_BYTES, import_glb, import_glb_prefab, source_sha256,
};
use engine_core::{Error, ErrorCode, Expected};
use engine_dev::authoring;
use engine_scene::SceneDocument;
use serde_json::{Value, json};

const COLLIDER: &str = "collider";

const TINT: &str = "tint";

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

fn command(arguments: &[String]) -> Result<Value, Error> {
    let options = Options::parse(arguments)?;
    let bytes = read_source(&options.file)?;
    let result = if options.as_prefab {
        import_glb_prefab(&bytes, &options.id)?
    } else {
        import_glb(&bytes, &options.id)?
    };
    let imported = result.document;
    let file_bytes: usize = result.files.iter().map(|file| file.bytes.len()).sum();
    let counts = json!({"textures":imported.textures.len(),"materials":imported.materials.len(),"meshes":imported.meshes.len(),"entities":imported.entities.len(),"prefabs":imported.prefabs.len(),"prefab_entities":imported.prefabs.iter().map(|prefab| prefab.entities.len()).sum::<usize>(),"files":result.files.len(),"file_bytes":file_bytes});
    let (report, document) = if let Some(output) = &options.output {
        let source = imported.to_json()?;
        write_files(output, &result.files)?;
        write_new(output, &source)?;
        (
            json!({"path":output,"revision":imported.revision.to_string(),"persisted":true,"source_bytes":source.len(),"mode":"create"}),
            imported,
        )
    } else {
        let target = options.into.as_ref().expect("target_validated");
        let original = authoring::read_scene_source(target)?;
        let current = SceneDocument::from_json(&original)?;
        let revision = options.revision.expect("revision_validated");
        if current.revision != revision {
            return Err(Error::new(ErrorCode::RevisionConflict, "$.revision")
                .got(revision)
                .nearest(current.revision));
        }
        let candidate = merge(current, imported, &options.id, options.replace)?;
        let source = candidate.to_json()?;
        write_files(target, &result.files)?;
        let report = authoring::save_scene(target, &source, revision)?;
        (
            json!({"path":report.path,"revision":report.revision,"persisted":report.persisted,"source_bytes":report.source_bytes,"mode":if options.replace {"replace_namespace"} else {"append"}}),
            candidate,
        )
    };
    Ok(json!({
        "save":report,"scene":document.id,"imported":counts,
        "ignored_extensions":result.ignored_extensions,
        "provenance":{"source":options.file,"source_sha256":source_sha256(&bytes),"importer_version":IMPORTER_VERSION,"namespace":options.id,"settings":{"format":"glb2","authoring":if options.as_prefab {"prefab"} else {"flat"},"coordinates":"right-handed-y-up","node_transforms":"baked","materials":"gltf-metallic-roughness","images":"files-unchanged","meshes":"emb-files"}}
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
}

const VALUED: [&str; 5] = [
    "--file",
    "--id",
    "--output",
    "--into",
    "--expected-revision",
];

fn argument(flag: &str) -> Error {
    Error::new(
        ErrorCode::InvalidArguments,
        format!("$.{}", flag.trim_start_matches("--")),
    )
}

impl Options {
    fn parse(arguments: &[String]) -> Result<Self, Error> {
        if let Some(command) = arguments.first().filter(|command| *command != "import") {
            return Err(argument("command").got(command).variants(["import"]));
        }
        if arguments.is_empty() {
            return Err(argument("command").expected(Expected::Present));
        }
        let mut values = BTreeMap::new();
        let mut arguments = arguments[1..].iter();
        let mut replace = false;
        let mut as_prefab = false;
        while let Some(flag) = arguments.next() {
            if flag == "--replace" || flag == "--as-prefab" {
                let present = if flag == "--replace" {
                    &mut replace
                } else {
                    &mut as_prefab
                };
                if *present {
                    return Err(argument(flag).expected(Expected::Unique));
                }
                *present = true;
                continue;
            }
            if !VALUED.contains(&flag.as_str()) {
                return Err(argument("arguments")
                    .got(flag)
                    .variants(VALUED.into_iter().chain(["--replace", "--as-prefab"])));
            }
            let value = arguments
                .next()
                .filter(|value| !value.starts_with("--"))
                .ok_or_else(|| argument(flag).expected(Expected::Present))?;
            if values.insert(flag.as_str(), value.as_str()).is_some() {
                return Err(argument(flag).expected(Expected::Unique));
            }
        }
        let required = |flag: &str| {
            values
                .get(flag)
                .copied()
                .ok_or_else(|| argument(flag).expected(Expected::Present))
        };
        let file = PathBuf::from(required("--file")?);
        let id = required("--id")?.to_owned();
        let output = values.get("--output").map(PathBuf::from);
        let into = values.get("--into").map(PathBuf::from);
        let revision = values
            .get("--expected-revision")
            .map(|revision| {
                revision.parse::<u64>().map_err(|_| {
                    Error::new(ErrorCode::InvalidRevision, "$.expected-revision")
                        .expected(Expected::Integer)
                        .got(revision)
                })
            })
            .transpose()?;
        if output.is_some() == into.is_some() {
            let wanted = if output.is_some() {
                Expected::Absent
            } else {
                Expected::Present
            };
            return Err(argument("--into").expected(wanted));
        }
        if into.is_some() != revision.is_some() {
            let wanted = if into.is_some() {
                Expected::Present
            } else {
                Expected::Absent
            };
            return Err(argument("--expected-revision").expected(wanted));
        }
        if replace && into.is_none() {
            return Err(argument("--replace").expected(Expected::Absent));
        }
        Ok(Self {
            file,
            id,
            output,
            into,
            revision,
            replace,
            as_prefab,
        })
    }
}

fn merge(
    mut current: SceneDocument,
    mut imported: SceneDocument,
    id: &str,
    replace: bool,
) -> Result<SceneDocument, Error> {
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
            .any(|texture| generated(&texture.id, id, Kind::Texture));
    if (as_prefab && existing_flat)
        || (!as_prefab && existing_prefab)
        || (existing_assets && !existing_flat && !existing_prefab)
    {
        return Err(Error::new(ErrorCode::DuplicateId, "$.id").got(id));
    }
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
        for mut entity in imported.entities {
            if let Some(old) = previous.get(&entity.id) {
                if !node_entity(&entity.id, id) {
                    entity.transform = old.transform;
                    entity.parent.clone_from(&old.parent);
                }
                entity.name.clone_from(&old.name);
                entity.tags.clone_from(&old.tags);
                for kept in [COLLIDER, TINT] {
                    if let Some(value) = old.component(kept) {
                        entity.components.insert(kept.to_owned(), value.clone());
                    } else {
                        entity.components.remove(kept);
                    }
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
    current.prefabs.extend(imported.prefabs);
    current.revision = current
        .revision
        .checked_add(1)
        .ok_or_else(|| Error::new(ErrorCode::RevisionExhausted, "$.revision"))?;
    current.validate()?;
    Ok(current)
}

fn preserve_colliders(
    previous: &[&engine_scene::SceneObject],
    replacements: Vec<&mut engine_scene::SceneObject>,
) -> Result<(), Error> {
    let mut replacements: BTreeMap<_, _> = replacements
        .into_iter()
        .map(|entity| (entity.id.clone(), entity))
        .collect();
    for old in previous {
        let Some(collider) = old.component(COLLIDER) else {
            continue;
        };
        let Some(replacement) = replacements.get_mut(&old.id) else {
            return Err(Error::new(ErrorCode::InvalidValue, "$.replace")
                .expected(Expected::Present)
                .got(&old.id));
        };
        replacement
            .components
            .insert(COLLIDER.to_owned(), collider.clone());
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum Kind {
    Entity,
    Mesh,
    Material,
    Texture,
}

fn decimal(value: &str) -> bool {
    !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit())
}

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

fn read_source(path: &Path) -> Result<Vec<u8>, Error> {
    let too_large = |bytes: u64| {
        Error::new(ErrorCode::SourceTooLarge, path.display().to_string())
            .at_most(MAX_INPUT_BYTES as u64)
            .got(bytes)
    };
    let file = File::open(path).map_err(|error| io_error(path, &error))?;
    let length = file
        .metadata()
        .map_err(|error| io_error(path, &error))?
        .len();
    if length > MAX_INPUT_BYTES as u64 {
        return Err(too_large(length));
    }
    let mut bytes = Vec::new();
    file.take((MAX_INPUT_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|error| io_error(path, &error))?;
    if bytes.len() > MAX_INPUT_BYTES {
        return Err(too_large(bytes.len() as u64));
    }
    Ok(bytes)
}

fn write_files(scene: &Path, files: &[ImportedFile]) -> Result<(), Error> {
    let folder = scene.parent().unwrap_or(Path::new("."));
    for file in files {
        let path = folder.join(&file.uri);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|error| io_error(parent, &error))?;
        }
        std::fs::write(&path, &file.bytes).map_err(|error| io_error(&path, &error))?;
    }
    Ok(())
}

fn write_new(path: &Path, source: &str) -> Result<(), Error> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| {
            let failed = io_error(path, &error);
            if error.kind() == std::io::ErrorKind::AlreadyExists {
                failed.expected(Expected::Absent)
            } else {
                failed
            }
        })?;
    let result = file
        .write_all(source.as_bytes())
        .and_then(|()| file.sync_all());
    drop(file);
    if let Err(error) = result {
        return Err(match std::fs::remove_file(path) {
            Ok(()) => io_error(path, &error),
            Err(cleanup) => io_error(path, &error).got([kind(&error), kind(&cleanup)]),
        });
    }
    Ok(())
}

fn kind(error: &std::io::Error) -> String {
    format!("{:?}", error.kind())
}

fn io_error(path: &Path, error: &std::io::Error) -> Error {
    Error::new(ErrorCode::SourceIo, path.display().to_string()).got(kind(error))
}
