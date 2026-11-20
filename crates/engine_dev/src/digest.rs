use std::collections::BTreeMap;

use engine_core::{Assets, Material, MeshFile, Mover, Pace, Reach, Texture, content_hash};
use serde_json::{Map, Value, json};

pub fn digest(reach: &Reach<'_>) -> Value {
    let world = reach.world;
    let mut views: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let ticked = world.iter_entities().filter(|entity| {
        entity
            .get::<Mover>()
            .is_none_or(|mover| mover.pace == Pace::Tick)
    });
    for entity in ticked {
        for (name, value) in crate::entities::values(reach, entity.id(), &[]) {
            if value.get("unavailable").is_none() {
                views.entry(name).or_default().push(value.to_string());
            }
        }
    }
    let mut components = Map::new();
    for (name, mut views) in views {
        views.sort_unstable();
        components.insert(
            name,
            json!({"count": views.len(), "hash": hash(views.join("
").as_bytes())}),
        );
    }
    let assets = world
        .get_resource::<Assets>()
        .map_or(Value::Null, assets);
    let hash = hash(
        json!({"components": components, "assets": assets})
            .to_string()
            .as_bytes(),
    );
    json!({
        "entities": world.iter_entities().count(),
        "components": components,
        "assets": assets,
        "hash": hash,
    })
}

fn hash(bytes: &[u8]) -> String {
    format!("{:016x}", content_hash(bytes) as u64)
}

fn assets(assets: &Assets) -> Value {
    let kind = |entries: Vec<(usize, Vec<u8>)>| {
        let mut bytes = Vec::new();
        for (slot, content) in &entries {
            bytes.extend_from_slice(&(*slot as u64).to_le_bytes());
            bytes.extend_from_slice(&(content.len() as u64).to_le_bytes());
            bytes.extend_from_slice(content);
        }
        json!({"count": entries.len(), "hash": hash(&bytes)})
    };
    json!({
        "meshes": kind(
            assets
                .each::<MeshFile>()
                .map(|(mesh, held)| (mesh.slot(), mesh_bytes(held)))
                .collect()
        ),
        "materials": kind(
            assets
                .each::<Material>()
                .map(|(material, held)| (material.slot(), without_loads(&format!("{held:?}"))))
                .collect()
        ),
        "textures": kind(
            assets
                .each::<Texture>()
                .map(|(texture, held)| (texture.slot(), texture_bytes(held)))
                .collect()
        ),
    })
}

fn mesh_bytes(mesh: &MeshFile) -> Vec<u8> {
    let description = mesh.description();
    let mut bytes = format!("{description:?}").into_bytes();
    for at in 0..description.streams.len() {
        bytes.extend_from_slice(mesh.stream(at));
    }
    bytes.extend_from_slice(mesh.indices());
    bytes
}

fn texture_bytes(texture: &Texture) -> Vec<u8> {
    let mut bytes = format!(
        "{}x{} {:?}",
        texture.width(),
        texture.height(),
        texture.format()
    )
    .into_bytes();
    for level in texture.levels() {
        bytes.extend_from_slice(level);
    }
    bytes
}

fn without_loads(debug: &str) -> Vec<u8> {
    const LIBRARY: &str = "generation: ";
    let mut kept = String::with_capacity(debug.len());
    let mut rest = debug;
    while let Some(at) = rest.find(LIBRARY) {
        kept.push_str(&rest[..at]);
        let after = &rest[at + LIBRARY.len()..];
        let digits = after.bytes().take_while(u8::is_ascii_digit).count();
        rest = after[digits..]
            .strip_prefix(", ")
            .unwrap_or(&after[digits..]);
    }
    kept.push_str(rest);
    kept.into_bytes()
}
