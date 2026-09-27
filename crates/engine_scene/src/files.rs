//! Files a scene references by relative path (decision 030): images and
//! binary mesh buffers beside the scene. The host reads them and decodes
//! images with the platform's own decoder before a scene loads, so the
//! runtime carries no image decoder and scene sources stay small.

use std::collections::HashMap;

use engine_core::{MeshVertex, Texture};

use crate::SceneError;

/// What a referenced file holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SceneFileKind {
    /// An image (PNG, JPEG or WebP), supplied decoded to RGBA8.
    Image,
    /// An engine mesh buffer (`EMB1`), supplied as bytes.
    Buffer,
}

/// A file a scene references, by its path relative to the scene.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SceneFile {
    pub uri: String,
    pub kind: SceneFileKind,
}

/// Contents of a scene's referenced files, supplied by the host.
#[derive(Clone, Debug, Default)]
pub struct SceneFiles {
    // Width, height and RGBA8 pixels of each decoded image.
    images: HashMap<String, (u32, u32, Vec<u8>)>,
    buffers: HashMap<String, Vec<u8>>,
}

impl SceneFiles {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds image `uri`, decoded: `width` × `height` RGBA8 sRGB pixels, rows
    /// top to bottom. Validated when a scene uses it.
    pub fn add_image(&mut self, uri: &str, width: u32, height: u32, pixels: Vec<u8>) {
        self.images.insert(uri.to_owned(), (width, height, pixels));
    }

    /// Adds buffer `uri`'s bytes.
    pub fn add_buffer(&mut self, uri: &str, bytes: Vec<u8>) {
        self.buffers.insert(uri.to_owned(), bytes);
    }

    /// Image `uri` as a texture, validated.
    pub(crate) fn texture(&self, uri: &str, path: &str) -> Result<Texture, SceneError> {
        let (width, height, pixels) = self.images.get(uri).ok_or_else(|| missing(uri, path))?;
        Texture::new(*width, *height, pixels.clone())
            .map_err(|error| SceneError::new("invalid_texture", path, error.to_string()))
    }

    /// Buffer `uri` decoded as a mesh buffer.
    pub(crate) fn mesh(
        &self,
        uri: &str,
        path: &str,
    ) -> Result<(Vec<MeshVertex>, Vec<u32>), SceneError> {
        let bytes = self.buffers.get(uri).ok_or_else(|| missing(uri, path))?;
        mesh_buffer(bytes, path)
    }
}

fn missing(uri: &str, path: &str) -> SceneError {
    SceneError::new(
        "missing_file",
        path,
        format!("the scene references {uri}, which was not supplied"),
    )
}

/// Checks that `uri` is a relative path inside the scene's folder.
pub(crate) fn valid_uri(uri: &str, path: &str) -> Result<(), SceneError> {
    let valid = !uri.is_empty()
        && uri.len() <= 256
        && uri
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"_./-".contains(&byte))
        && !uri.starts_with(['/', '.'])
        && uri
            .split('/')
            .all(|part| !part.is_empty() && part != "." && part != "..");
    if valid {
        Ok(())
    } else {
        Err(SceneError::new(
            "invalid_uri",
            path,
            "must be a relative path of letters, digits, '_', '.', '-' and '/' inside the scene's folder",
        ))
    }
}

/// Magic of an engine mesh buffer.
pub const MESH_BUFFER_MAGIC: [u8; 4] = *b"EMB1";

/// An engine mesh buffer: `EMB1`, the vertex and index counts (little-endian
/// u32), each vertex's position, normal and UV (eight little-endian f32),
/// then the u32 indices.
pub fn encode_mesh_buffer(vertices: &[MeshVertex], indices: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(12 + vertices.len() * 32 + indices.len() * 4);
    bytes.extend_from_slice(&MESH_BUFFER_MAGIC);
    bytes.extend_from_slice(&(vertices.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&(indices.len() as u32).to_le_bytes());
    for vertex in vertices {
        for value in vertex
            .position
            .iter()
            .chain(&vertex.normal)
            .chain(&vertex.uv)
        {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    for index in indices {
        bytes.extend_from_slice(&index.to_le_bytes());
    }
    bytes
}

fn mesh_buffer(bytes: &[u8], path: &str) -> Result<(Vec<MeshVertex>, Vec<u32>), SceneError> {
    let invalid = |rule: &str| SceneError::new("invalid_mesh", path, rule.to_owned());
    let word = |offset: usize| {
        bytes
            .get(offset..offset + 4)
            .map(|word| u32::from_le_bytes(word.try_into().expect("four bytes")))
    };
    if bytes.get(..4) != Some(&MESH_BUFFER_MAGIC[..]) {
        return Err(invalid("a mesh buffer starts with EMB1"));
    }
    let (Some(vertex_count), Some(index_count)) = (word(4), word(8)) else {
        return Err(invalid("a mesh buffer has vertex and index counts"));
    };
    let (vertex_count, index_count) = (vertex_count as usize, index_count as usize);
    let expected = vertex_count
        .checked_mul(32)
        .and_then(|vertices| vertices.checked_add(index_count.checked_mul(4)?))
        .and_then(|payload| payload.checked_add(12));
    if expected != Some(bytes.len()) {
        return Err(invalid(
            "the buffer's length must match its vertex and index counts",
        ));
    }
    let float = |offset: usize| {
        f32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("four bytes"))
    };
    let mut vertices = Vec::with_capacity(vertex_count);
    for vertex in 0..vertex_count {
        let base = 12 + vertex * 32;
        let values: [f32; 8] = std::array::from_fn(|index| float(base + index * 4));
        if !values[..3]
            .iter()
            .all(|value| value.is_finite() && value.abs() <= 1_000_000.0)
        {
            return Err(invalid(
                "vertex positions must be finite and within ±1000000",
            ));
        }
        vertices.push(MeshVertex {
            position: [values[0], values[1], values[2]],
            normal: [values[3], values[4], values[5]],
            uv: [values[6], values[7]],
        });
    }
    let first_index = 12 + vertex_count * 32;
    let indices = (0..index_count)
        .map(|index| {
            u32::from_le_bytes(
                bytes[first_index + index * 4..first_index + index * 4 + 4]
                    .try_into()
                    .expect("four bytes"),
            )
        })
        .collect();
    Ok((vertices, indices))
}
