//! Files a scene references by relative path: images and
//! binary mesh buffers beside the scene. The host reads them and decodes
//! images with the platform's own decoder before a scene loads, so the
//! runtime carries no image decoder and scene sources stay small. KTX2
//! images the host transcodes to a block-compressed format its GPU samples.

use engine_settings::ErrorCode;
use std::collections::BTreeMap;

use engine_core::{MeshVertex, Texture, TextureFormat, VirtualMesh};

use crate::SceneError;

/// What a referenced file holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SceneFileKind {
    /// An image (PNG, JPEG or WebP), supplied decoded to RGBA8.
    Image,
    /// An engine buffer, supplied as bytes: a mesh (`EMB1`, `EMB2`) or an
    /// animation (`EAN1`).
    Buffer,
    /// A virtual mesh (`EVG1`): hosts supply its prefix
    /// (`VirtualMesh::prefix_bytes` long; the whole file works too) and its
    /// URL, from which renderers stream the other pages.
    VirtualMesh,
    /// A terrain file (`ETR1`): hosts supply its prefix (its
    /// header and directory, `FileHeader::prefix_bytes` long; the whole
    /// file works too) and its URL, from which hosts fetch its tiles.
    Terrain,
}

/// A file a scene references, by its path relative to the scene.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SceneFile {
    pub uri: String,
    pub kind: SceneFileKind,
}

/// A terrain file's header, the URL its tiles stream from, and the whole
/// file when supplied whole.
pub(crate) type TerrainFile = (engine_core::FileHeader, Option<String>, Option<Vec<u8>>);

/// Format, width, height and levels of a transcoded image.
type CompressedImage = (TextureFormat, u32, u32, Vec<Vec<u8>>);

/// Contents of a scene's referenced files, supplied by the host.
#[derive(Clone, Debug, Default)]
pub struct SceneFiles {
    // Width, height and RGBA8 pixels of each decoded image.
    images: BTreeMap<String, (u32, u32, Vec<u8>)>,
    compressed: BTreeMap<String, CompressedImage>,
    buffers: BTreeMap<String, Vec<u8>>,
    // Each virtual mesh's prefix (or whole file) and its URL.
    virtual_meshes: BTreeMap<String, (Vec<u8>, Option<String>)>,
    // Each terrain file's prefix (or whole file) and its URL.
    terrains: BTreeMap<String, (Vec<u8>, Option<String>)>,
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

    /// Adds image `uri` transcoded to block-compressed `format`:
    /// `width` × `height`, its levels from the base down. Validated
    /// when a scene uses it.
    pub fn add_compressed_image(
        &mut self,
        uri: &str,
        format: TextureFormat,
        width: u32,
        height: u32,
        levels: Vec<Vec<u8>>,
    ) {
        self.compressed
            .insert(uri.to_owned(), (format, width, height, levels));
    }

    /// Adds buffer `uri`'s bytes.
    pub fn add_buffer(&mut self, uri: &str, bytes: Vec<u8>) {
        self.buffers.insert(uri.to_owned(), bytes);
    }

    /// Adds virtual mesh `uri`: its prefix or whole file, and
    /// the URL renderers fetch the pages a prefix lacks from.
    pub fn add_virtual_mesh(&mut self, uri: &str, bytes: Vec<u8>, url: Option<&str>) {
        self.virtual_meshes
            .insert(uri.to_owned(), (bytes, url.map(str::to_owned)));
    }

    /// Adds terrain file `uri`: its prefix or whole file, and
    /// the URL hosts fetch the tiles a prefix lacks from.
    pub fn add_terrain(&mut self, uri: &str, bytes: Vec<u8>, url: Option<&str>) {
        self.terrains
            .insert(uri.to_owned(), (bytes, url.map(str::to_owned)));
    }

    /// Buffer `uri`'s bytes.
    pub(crate) fn buffer(&self, uri: &str, path: &str) -> Result<&[u8], SceneError> {
        self.buffers
            .get(uri)
            .map(Vec::as_slice)
            .ok_or_else(|| missing(uri, path))
    }

    /// Terrain file `uri`, validated: its header, the URL
    /// its tiles stream from, and the whole file when supplied whole (as a
    /// terrain file or a buffer).
    pub(crate) fn terrain(&self, uri: &str, path: &str) -> Result<TerrainFile, SceneError> {
        let (bytes, url) = match self.terrains.get(uri) {
            Some((bytes, url)) => (bytes, url.clone()),
            None => (
                self.buffers.get(uri).ok_or_else(|| missing(uri, path))?,
                None,
            ),
        };
        let header = engine_core::FileHeader::parse(bytes)
            .map_err(|error| SceneError::new(ErrorCode::InvalidTerrain, path, error.to_string()))?;
        // The file's end: after its last tile.
        let end = header
            .pyramid()
            .level_tiles(0)
            .chain(
                (1..header.pyramid().levels())
                    .flat_map(|level| header.pyramid().level_tiles(level)),
            )
            .filter_map(|key| header.tile(key))
            .map(|tile| tile.offset + u64::from(tile.length))
            .max()
            .unwrap_or(0);
        let whole = bytes.len() as u64 >= end;
        if url.is_none() && !whole {
            return Err(SceneError::new(
                ErrorCode::InvalidTerrain,
                path,
                "a terrain file's prefix needs the URL its tiles stream from",
            ));
        }
        Ok((header, url, whole.then(|| bytes.clone())))
    }

    /// Image `uri` as a texture, validated, its alpha channel kept.
    pub(crate) fn texture(&self, uri: &str, path: &str) -> Result<Texture, SceneError> {
        if let Some((format, width, height, levels)) = self.compressed.get(uri) {
            return Texture::compressed(*format, *width, *height, levels.clone()).map_err(
                |error| SceneError::new(ErrorCode::InvalidTexture, path, error.to_string()),
            );
        }
        let (width, height, pixels) = self.images.get(uri).ok_or_else(|| missing(uri, path))?;
        Texture::from_rgba(*width, *height, pixels.clone())
            .map_err(|error| SceneError::new(ErrorCode::InvalidTexture, path, error.to_string()))
    }

    /// Buffer `uri` decoded as a mesh buffer.
    pub(crate) fn mesh(&self, uri: &str, path: &str) -> Result<MeshBuffer, SceneError> {
        let bytes = self.buffers.get(uri).ok_or_else(|| missing(uri, path))?;
        mesh_buffer(bytes, path)
    }

    /// Virtual mesh `uri`, validated: supplied as one, or a
    /// whole file supplied as a buffer.
    pub(crate) fn virtual_mesh(&self, uri: &str, path: &str) -> Result<VirtualMesh, SceneError> {
        let (bytes, url) = match self.virtual_meshes.get(uri) {
            Some((bytes, url)) => (bytes, url.as_deref()),
            None => (
                self.buffers.get(uri).ok_or_else(|| missing(uri, path))?,
                None,
            ),
        };
        let mesh = VirtualMesh::decode(bytes)
            .map_err(|error| SceneError::new(ErrorCode::InvalidMesh, path, error.to_string()))?;
        Ok(match url {
            Some(url) => mesh.with_source(url),
            None => mesh,
        })
    }

    /// Buffer `uri` decoded as an animation buffer's channels.
    pub(crate) fn animation(
        &self,
        uri: &str,
        path: &str,
    ) -> Result<Vec<engine_core::AnimationChannel>, SceneError> {
        let bytes = self.buffers.get(uri).ok_or_else(|| missing(uri, path))?;
        animation_buffer(bytes, path)
    }
}

fn missing(uri: &str, path: &str) -> SceneError {
    SceneError::new(
        ErrorCode::MissingFile,
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
            ErrorCode::InvalidUri,
            path,
            "must be a relative path of letters, digits, '_', '.', '-' and '/' inside the scene's folder",
        ))
    }
}

/// Magic of an engine mesh buffer.
pub const MESH_BUFFER_MAGIC: [u8; 4] = *b"EMB1";
/// Magic of an engine mesh buffer with optional streams.
pub const MESH_BUFFER_EXTRAS_MAGIC: [u8; 4] = *b"EMB2";

/// An engine mesh buffer: `EMB1`, the vertex and index counts (little-endian
/// u32), each vertex's position, normal and UV (eight little-endian f32),
/// then the u32 indices. Takes runtime or authored ([`crate::SceneVertex`])
/// vertices.
pub fn encode_mesh_buffer<V: Copy + Into<MeshVertex>>(vertices: &[V], indices: &[u32]) -> Vec<u8> {
    encode_mesh_buffer_with(vertices, indices, MeshStreams::default())
}

/// Optional per-vertex streams of a mesh buffer, one entry per vertex each:
/// vertex colors and second texture coordinates, skinning
/// joints and weights (both or neither).
#[derive(Clone, Copy, Debug, Default)]
pub struct MeshStreams<'a> {
    pub colors: Option<&'a [[f32; 4]]>,
    pub uv1: Option<&'a [[f32; 2]]>,
    pub joints: Option<&'a [[u16; 4]]>,
    pub weights: Option<&'a [[f32; 4]]>,
}

/// [`encode_mesh_buffer`] with optional streams. With any, the buffer is
/// `EMB2`: after the counts a u32 of flags (1 colors, 2 second
/// coordinates, 4 joints and weights), then the vertices, the colors (four
/// f32 each), the second coordinates (two f32 each), the joints (four u16
/// each), the weights (four f32 each) and the indices.
///
/// # Panics
/// Panics when only one of joints and weights is given.
pub fn encode_mesh_buffer_with<V: Copy + Into<MeshVertex>>(
    vertices: &[V],
    indices: &[u32],
    streams: MeshStreams<'_>,
) -> Vec<u8> {
    let MeshStreams {
        colors,
        uv1,
        joints,
        weights,
    } = streams;
    assert_eq!(
        joints.is_some(),
        weights.is_some(),
        "joints and weights come together"
    );
    let skin = joints.zip(weights);
    let extras = colors.is_some() || uv1.is_some() || skin.is_some();
    let mut bytes = Vec::with_capacity(
        16 + vertices.len() * (32 + if extras { 48 } else { 0 }) + indices.len() * 4,
    );
    bytes.extend_from_slice(if extras {
        &MESH_BUFFER_EXTRAS_MAGIC
    } else {
        &MESH_BUFFER_MAGIC
    });
    bytes.extend_from_slice(&(vertices.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&(indices.len() as u32).to_le_bytes());
    if extras {
        let flags = u32::from(colors.is_some())
            | u32::from(uv1.is_some()) << 1
            | u32::from(skin.is_some()) << 2;
        bytes.extend_from_slice(&flags.to_le_bytes());
    }
    for &vertex in vertices {
        let vertex: MeshVertex = vertex.into();
        for value in vertex
            .position
            .iter()
            .chain(&vertex.normal)
            .chain(&vertex.uv)
        {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    for value in colors
        .into_iter()
        .flatten()
        .flatten()
        .chain(uv1.into_iter().flatten().flatten())
    {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    if let Some((joints, weights)) = skin {
        for joint in joints.iter().flatten() {
            bytes.extend_from_slice(&joint.to_le_bytes());
        }
        for weight in weights.iter().flatten() {
            bytes.extend_from_slice(&weight.to_le_bytes());
        }
    }
    for index in indices {
        bytes.extend_from_slice(&index.to_le_bytes());
    }
    bytes
}

/// Joints and weights of skinned vertices.
pub(crate) type SkinStreams = (Vec<[u16; 4]>, Vec<[f32; 4]>);

/// A decoded mesh buffer.
pub(crate) struct MeshBuffer {
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
    pub colors: Option<Vec<[f32; 4]>>,
    pub uv1: Option<Vec<[f32; 2]>>,
    pub skin: Option<SkinStreams>,
}

fn mesh_buffer(bytes: &[u8], path: &str) -> Result<MeshBuffer, SceneError> {
    let invalid = |rule: &str| SceneError::new(ErrorCode::InvalidMesh, path, rule.to_owned());
    let word = |offset: usize| {
        bytes
            .get(offset..offset + 4)
            .map(|word| u32::from_le_bytes(word.try_into().expect("four bytes")))
    };
    let extras = match bytes.get(..4) {
        Some(magic) if magic == MESH_BUFFER_MAGIC => false,
        Some(magic) if magic == MESH_BUFFER_EXTRAS_MAGIC => true,
        _ => return Err(invalid("a mesh buffer starts with EMB1 or EMB2")),
    };
    let (Some(vertex_count), Some(index_count)) = (word(4), word(8)) else {
        return Err(invalid("a mesh buffer has vertex and index counts"));
    };
    let flags = if extras {
        match word(12) {
            Some(flags) if flags != 0 && flags & !7 == 0 => flags,
            _ => {
                return Err(invalid(
                    "EMB2 flags name colors (1), second coordinates (2) and joints and weights (4)",
                ));
            }
        }
    } else {
        0
    };
    let header = if extras { 16 } else { 12 };
    let (vertex_count, index_count) = (vertex_count as usize, index_count as usize);
    // Words per vertex: eight, four more with colors, two more with second
    // coordinates, six more with joints (two) and weights (four).
    let floats = 8
        + if flags & 1 != 0 { 4 } else { 0 }
        + if flags & 2 != 0 { 2 } else { 0 }
        + if flags & 4 != 0 { 6 } else { 0 };
    let expected = vertex_count
        .checked_mul(floats * 4)
        .and_then(|vertices| vertices.checked_add(index_count.checked_mul(4)?))
        .and_then(|payload| payload.checked_add(header));
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
        let base = header + vertex * 32;
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
    let mut offset = header + vertex_count * 32;
    let colors = (flags & 1 != 0).then(|| {
        let colors: Vec<[f32; 4]> = (0..vertex_count)
            .map(|vertex| std::array::from_fn(|lane| float(offset + (vertex * 4 + lane) * 4)))
            .collect();
        offset += vertex_count * 16;
        colors
    });
    let uv1 = (flags & 2 != 0).then(|| {
        let uv1: Vec<[f32; 2]> = (0..vertex_count)
            .map(|vertex| std::array::from_fn(|lane| float(offset + (vertex * 2 + lane) * 4)))
            .collect();
        offset += vertex_count * 8;
        uv1
    });
    let skin = (flags & 4 != 0).then(|| {
        let joints: Vec<[u16; 4]> = (0..vertex_count)
            .map(|vertex| {
                std::array::from_fn(|lane| {
                    let at = offset + (vertex * 4 + lane) * 2;
                    u16::from_le_bytes([bytes[at], bytes[at + 1]])
                })
            })
            .collect();
        offset += vertex_count * 8;
        let weights: Vec<[f32; 4]> = (0..vertex_count)
            .map(|vertex| std::array::from_fn(|lane| float(offset + (vertex * 4 + lane) * 4)))
            .collect();
        offset += vertex_count * 16;
        (joints, weights)
    });
    let indices = (0..index_count)
        .map(|index| {
            u32::from_le_bytes(
                bytes[offset + index * 4..offset + index * 4 + 4]
                    .try_into()
                    .expect("four bytes"),
            )
        })
        .collect();
    Ok(MeshBuffer {
        vertices,
        indices,
        colors,
        uv1,
        skin,
    })
}

/// Magic of an engine animation buffer.
pub const ANIMATION_BUFFER_MAGIC: [u8; 4] = *b"EAN1";

/// An engine animation buffer: `EAN1` and the channel count (little-endian
/// u32), then per channel its target (u32), property (u8: 0 translation, 1
/// rotation, 2 scale), interpolation (u8: 0 step, 1 linear, 2 cubic spline),
/// two zero bytes, the key count and value count (u32), the key times (f32)
/// and the values (three f32 each, four for rotations).
pub fn encode_animation_buffer(channels: &[engine_core::AnimationChannel]) -> Vec<u8> {
    use engine_core::{AnimatedProperty, KeyInterpolation, Keyframes};
    let mut bytes = ANIMATION_BUFFER_MAGIC.to_vec();
    bytes.extend_from_slice(&(channels.len() as u32).to_le_bytes());
    for channel in channels {
        bytes.extend_from_slice(&channel.target().to_le_bytes());
        bytes.push(match channel.property() {
            AnimatedProperty::Translation => 0,
            AnimatedProperty::Rotation => 1,
            AnimatedProperty::Scale => 2,
        });
        bytes.push(match channel.interpolation() {
            KeyInterpolation::Step => 0,
            KeyInterpolation::Linear => 1,
            KeyInterpolation::CubicSpline => 2,
        });
        bytes.extend_from_slice(&[0, 0]);
        let values: Vec<f32> = match channel.keyframes() {
            Keyframes::Vectors(values) => {
                values.iter().flat_map(|value| value.to_array()).collect()
            }
            Keyframes::Rotations(values) => {
                values.iter().flat_map(|value| value.to_array()).collect()
            }
        };
        let width = if channel.property() == AnimatedProperty::Rotation {
            4
        } else {
            3
        };
        bytes.extend_from_slice(&(channel.times().len() as u32).to_le_bytes());
        bytes.extend_from_slice(&((values.len() / width) as u32).to_le_bytes());
        for value in channel.times().iter().chain(&values) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

/// The channels of an `EAN1` buffer, validated as clips require.
pub(crate) fn animation_buffer(
    bytes: &[u8],
    path: &str,
) -> Result<Vec<engine_core::AnimationChannel>, SceneError> {
    use engine_core::{AnimatedProperty, AnimationChannel, KeyInterpolation, Keyframes, glam};
    let invalid = |rule: String| SceneError::new(ErrorCode::InvalidAnimation, path, rule);
    let word = |offset: usize| {
        bytes
            .get(offset..offset + 4)
            .map(|word| u32::from_le_bytes(word.try_into().expect("four bytes")))
    };
    if bytes.get(..4) != Some(&ANIMATION_BUFFER_MAGIC[..]) {
        return Err(invalid("an animation buffer starts with EAN1".into()));
    }
    let count = word(4).ok_or_else(|| invalid("an animation buffer has a channel count".into()))?;
    let mut offset = 8;
    let mut channels = Vec::new();
    for _ in 0..count {
        let header = bytes
            .get(offset..offset + 16)
            .ok_or_else(|| invalid("a channel header is cut short".into()))?;
        let target = u32::from_le_bytes(header[..4].try_into().expect("four bytes"));
        let property = match header[4] {
            0 => AnimatedProperty::Translation,
            1 => AnimatedProperty::Rotation,
            2 => AnimatedProperty::Scale,
            _ => return Err(invalid("a channel property is 0, 1 or 2".into())),
        };
        let interpolation = match header[5] {
            0 => KeyInterpolation::Step,
            1 => KeyInterpolation::Linear,
            2 => KeyInterpolation::CubicSpline,
            _ => return Err(invalid("a channel interpolation is 0, 1 or 2".into())),
        };
        let keys = u32::from_le_bytes(header[8..12].try_into().expect("four bytes")) as usize;
        let values = u32::from_le_bytes(header[12..16].try_into().expect("four bytes")) as usize;
        let width = if property == AnimatedProperty::Rotation {
            4
        } else {
            3
        };
        let floats = values
            .checked_mul(width)
            .and_then(|floats| floats.checked_add(keys))
            .ok_or_else(|| invalid("channel counts overflow".into()))?;
        offset += 16;
        let end = floats
            .checked_mul(4)
            .and_then(|length| length.checked_add(offset))
            .filter(|end| *end <= bytes.len())
            .ok_or_else(|| invalid("channel keys run past the buffer".into()))?;
        let floats: Vec<f32> = bytes[offset..end]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect();
        offset = end;
        let (times, values) = floats.split_at(keys);
        let keyframes = if property == AnimatedProperty::Rotation {
            Keyframes::Rotations(
                values
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|value| glam::Quat::from_array(*value))
                    .collect(),
            )
        } else {
            Keyframes::Vectors(
                values
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .map(|value| glam::Vec3::from_array(*value))
                    .collect(),
            )
        };
        channels.push(
            AnimationChannel::new(target, property, interpolation, times.to_vec(), keyframes)
                .map_err(|error| invalid(error.to_string()))?,
        );
    }
    if offset != bytes.len() {
        return Err(invalid(
            "an animation buffer ends after its last channel".into(),
        ));
    }
    Ok(channels)
}
