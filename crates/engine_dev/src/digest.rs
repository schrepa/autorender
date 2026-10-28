use engine_core::bevy_ecs::world::World;
use engine_core::{AssetLibrary, Mesh, Texture, content_hash};
use serde_json::{Map, Value, json};

use crate::results::DigestResult;
use crate::{Context, DevTools};

impl DevTools {
    pub fn digest(&self, world: &World, context: Context<'_>) -> DigestResult {
        let mut components = Map::new();
        for (name, mut views) in self.component_views(world, context) {
            if views.is_empty() {
                continue;
            }
            views.sort_unstable();
            components.insert(
                name,
                json!({"count": views.len(), "hash": hash(views.join("\n").as_bytes())}),
            );
        }
        let assets = world
            .get_resource::<AssetLibrary>()
            .map_or(Value::Null, assets);
        let hash = hash(
            json!({"components": components, "assets": assets})
                .to_string()
                .as_bytes(),
        );
        DigestResult {
            entities: world.iter_entities().count(),
            components: Value::Object(components),
            assets,
            hash,
        }
    }
}

fn hash(bytes: &[u8]) -> String {
    format!("{:016x}", content_hash(bytes) as u64)
}

fn assets(library: &AssetLibrary) -> Value {
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
        "meshes": kind(library.meshes().map(|(slot, mesh)| (slot, mesh_bytes(mesh))).collect()),
        "materials": kind(
            library
                .materials()
                .map(|(slot, material)| (slot, without_libraries(&format!("{material:?}"))))
                .collect()
        ),
        "textures": kind(
            library
                .textures()
                .map(|(slot, texture)| (slot, texture_bytes(texture)))
                .collect()
        ),
        "clips": kind(
            library
                .clips()
                .map(|(slot, clip)| (slot, without_libraries(&format!("{clip:?}"))))
                .collect()
        ),
    })
}

fn mesh_bytes(mesh: &Mesh) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(mesh.byte_len());
    let mut floats = |values: &[f32]| {
        for value in values {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
    };
    for vertex in mesh.vertices() {
        floats(&vertex.position);
        floats(&vertex.normal);
        floats(&vertex.uv);
    }
    for color in mesh.colors().into_iter().flatten() {
        floats(color);
    }
    for uv in mesh.uv1().into_iter().flatten() {
        floats(uv);
    }
    for weights in mesh.weights().into_iter().flatten() {
        floats(weights);
    }
    let bounds = mesh.bounds();
    floats(&bounds.min().to_array());
    floats(&bounds.max().to_array());
    for index in mesh.indices() {
        bytes.extend_from_slice(&index.to_le_bytes());
    }
    for joints in mesh.joints().into_iter().flatten() {
        for joint in joints {
            bytes.extend_from_slice(&joint.to_le_bytes());
        }
    }
    bytes.extend_from_slice(&mesh.joint_count().to_le_bytes());
    if let Some(geometry) = mesh.virtual_geometry() {
        for word in geometry.words() {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
    }
    bytes
}

fn texture_bytes(texture: &Texture) -> Vec<u8> {
    let mut bytes = format!(
        "{}x{} {:?} {}",
        texture.width(),
        texture.height(),
        texture.format(),
        texture.has_emission_mask()
    )
    .into_bytes();
    for level in texture.levels() {
        bytes.extend_from_slice(level);
    }
    bytes
}

fn without_libraries(debug: &str) -> Vec<u8> {
    const LIBRARY: &str = "library: ";
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
