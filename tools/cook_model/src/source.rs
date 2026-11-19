use std::fmt::Display;

use cook::Refusal;
use glam::{Mat4, Quat, Vec3};
use gltf::{Gltf, buffer, image};

const REFUSED_EXTENSIONS: [&str; 3] = [
    "KHR_draco_mesh_compression",
    "EXT_meshopt_compression",
    "KHR_texture_transform",
];

pub struct Source {
    pub file: String,
    pub gltf: Gltf,
    json: serde_json::Value,
    parents: Vec<Option<usize>>,
}

fn json_chunk(bytes: &[u8]) -> Option<serde_json::Value> {
    let word = |at: usize| {
        bytes
            .get(at..at + 4)
            .map(|held| u32::from_le_bytes([held[0], held[1], held[2], held[3]]) as usize)
    };
    if !bytes.starts_with(b"glTF") {
        return serde_json::from_slice(bytes).ok();
    }
    let length = word(12)?;
    serde_json::from_slice(bytes.get(20..20 + length)?).ok()
}

pub fn local(node: &gltf::Node<'_>) -> Mat4 {
    let (translation, rotation, scale) = node.transform().decomposed();
    Mat4::from_scale_rotation_translation(
        Vec3::from_array(scale),
        Quat::from_array(rotation),
        Vec3::from_array(translation),
    )
}

pub fn node_label(node: &gltf::Node<'_>) -> String {
    match node.name() {
        Some(name) => format!("node `{name}`"),
        None => format!("nodes[{}]", node.index()),
    }
}

impl Source {
    pub fn read(file: &str, bytes: &[u8]) -> Result<Self, Refusal> {
        let refuse = |code, item: &str, detail: &dyn Display| {
            Refusal::new(code, file, format!("{item}: {detail}"))
        };
        let raw = Gltf::from_slice_without_validation(bytes)
            .map_err(|error| refuse("GltfParse", "file", &error))?;
        if let Some(extension) = raw.extensions_required().next() {
            return Err(refuse(
                "ExtensionRequired",
                "extensionsRequired",
                &extension,
            ));
        }
        if let Some(extension) = raw
            .extensions_used()
            .find(|extension| REFUSED_EXTENSIONS.contains(extension))
        {
            return Err(refuse("ExtensionRefused", "extensionsUsed", &extension));
        }
        let gltf =
            Gltf::from_slice(bytes).map_err(|error| refuse("GltfInvalid", "file", &error))?;
        for held in gltf.buffers() {
            if let buffer::Source::Uri(uri) = held.source() {
                return Err(refuse(
                    "ExternalUri",
                    &format!("buffers[{}]", held.index()),
                    &uri,
                ));
            }
        }
        for held in gltf.images() {
            if let image::Source::Uri { uri, .. } = held.source() {
                return Err(refuse(
                    "ExternalUri",
                    &format!("images[{}]", held.index()),
                    &uri,
                ));
            }
        }
        if gltf.buffers().count() > 0 && gltf.blob.is_none() {
            return Err(refuse(
                "NoBinaryChunk",
                "file",
                &"buffers need the GLB binary chunk",
            ));
        }
        let json = json_chunk(bytes)
            .ok_or_else(|| refuse("GltfParse", "file", &"the JSON chunk does not parse"))?;
        let mut parents = vec![None; gltf.nodes().count()];
        for node in gltf.nodes() {
            for child in node.children() {
                parents[child.index()] = Some(node.index());
            }
        }
        Ok(Self {
            file: file.to_owned(),
            gltf,
            json,
            parents,
        })
    }

    pub fn data<'s>(&'s self, held: &gltf::Buffer<'_>) -> Option<&'s [u8]> {
        match held.source() {
            buffer::Source::Bin => self.gltf.blob.as_deref(),
            buffer::Source::Uri(_) => None,
        }
    }

    pub fn view(&self, view: &buffer::View<'_>) -> Option<&[u8]> {
        self.data(&view.buffer())?
            .get(view.offset()..view.offset() + view.length())
    }

    pub fn extras(&self, table: &str, index: usize) -> serde_json::Value {
        self.json[table][index]["extras"].clone()
    }

    pub fn parent(&self, node: usize) -> Option<usize> {
        self.parents[node]
    }

    pub fn node(&self, index: usize) -> gltf::Node<'_> {
        self.gltf
            .nodes()
            .nth(index)
            .expect("node_index_from_document")
    }

    pub fn world(&self, node: usize) -> Mat4 {
        let own = local(&self.node(node));
        match self.parent(node) {
            Some(parent) => self.world(parent) * own,
            None => own,
        }
    }

    pub fn refuse(&self, code: &'static str, item: impl Display, detail: impl Display) -> Refusal {
        Refusal::new(code, &self.file, format!("{item}: {detail}"))
    }
}
