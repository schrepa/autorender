//! The files an authored scene references, loaded as every host loads them
//! ([`SceneLoad`]): the host reads what the load asks (files whole, and
//! streamed files' prefixes, whose headers say how long they are) and has
//! each image decoded as the load says ([`IMAGE_FORMATS`]: the engine's
//! decoder, the page's on its job workers; the page's Basis transcoder),
//! and the load holds them as the scene's files. Hosts
//! keep their transports: fetches relative to the scene's URL, or reads
//! beside its file.

use engine_core::{ErrorCode, FileHeader, TextureFormat, VirtualMesh};
use engine_scene::{SceneDocument, SceneFileKind, SceneFiles};
pub use engine_worker::{
    IMAGE_FORMATS, Image, ImageDecoder, ImageFormat, decode_image, image_format,
};

use crate::{Error, Result};

/// Bytes of a virtual mesh's header, which says how long its prefix is.
const VIRTUAL_HEADER: usize = 128;
/// Bytes of a terrain file's header, which says how long its prefix is.
const TERRAIN_HEADER: usize = 64;

/// Which host loads a scene's files, as [`IMAGE_FORMATS`] tells their
/// decoders apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadingHost {
    Page,
    Native,
}

/// A read the load asks of its host: file `uri` (relative to the scene),
/// whole or its first `prefix` bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileRead {
    pub uri: String,
    /// The bytes from the file's start to read; the whole file without.
    pub prefix: Option<usize>,
}

/// An image the host read, to decode as [`IMAGE_FORMATS`] says: its
/// decoder on this host and its bytes.
#[derive(Debug)]
pub struct ImageRead {
    pub decoder: ImageDecoder,
    pub bytes: Vec<u8>,
}

/// Where a file stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    /// Asked whole, or its header: what it is is known once read.
    Asked,
    /// A streamed file whose prefix is asked past its header.
    Prefix,
    /// An image its host decodes.
    Decoding,
    Held,
}

/// The files of one authored scene, loading.
#[derive(Debug)]
pub struct SceneLoad {
    host: LoadingHost,
    files: SceneFiles,
    // Each file, its kind and where it stands, in the scene's order.
    entries: Vec<(String, SceneFileKind, Stage)>,
    reads: Vec<FileRead>,
}

impl SceneLoad {
    /// The load on `host` of the files scene `source` references.
    pub fn new(source: &str, host: LoadingHost) -> Result<Self> {
        let document = SceneDocument::from_json(source)?;
        let entries: Vec<(String, SceneFileKind, Stage)> = document
            .files()
            .into_iter()
            .map(|file| (file.uri, file.kind, Stage::Asked))
            .collect();
        let reads = entries
            .iter()
            .map(|(uri, kind, _)| FileRead {
                uri: uri.clone(),
                prefix: match kind {
                    SceneFileKind::VirtualMesh => Some(VIRTUAL_HEADER),
                    SceneFileKind::Terrain => Some(TERRAIN_HEADER),
                    SceneFileKind::Image | SceneFileKind::Buffer => None,
                },
            })
            .collect();
        Ok(Self {
            host,
            files: SceneFiles::default(),
            entries,
            reads,
        })
    }

    /// The reads the load asks for since the last call; none once every
    /// file was read.
    pub fn take_reads(&mut self) -> Vec<FileRead> {
        std::mem::take(&mut self.reads)
    }

    /// The host read `bytes` of file `uri` as [`Self::take_reads`] asked; a
    /// streamed file streams from `url` (its URL or path). An image comes
    /// back, to decode as [`IMAGE_FORMATS`] says on this host.
    pub fn read(&mut self, uri: &str, bytes: Vec<u8>, url: &str) -> Result<Option<ImageRead>> {
        let at = self
            .entries
            .iter()
            .position(|(file, _, _)| file == uri)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::MissingFile,
                    "$",
                    format!("the scene names no file {uri}"),
                )
            })?;
        let (kind, stage) = (self.entries[at].1, self.entries[at].2);
        match kind {
            SceneFileKind::Buffer => {
                self.files.add_buffer(uri, bytes);
                self.entries[at].2 = Stage::Held;
                Ok(None)
            }
            SceneFileKind::Image => {
                let format = engine_worker::image_format(&bytes).ok_or_else(|| {
                    invalid_texture(format!("{uri} is not a PNG, JPEG or KTX2 image"))
                })?;
                let decoder = match self.host {
                    LoadingHost::Page => format.page,
                    LoadingHost::Native => format.native,
                }
                .ok_or_else(|| {
                    invalid_texture(format!(
                        "{uri} is a {} image, which only the page reads (IMAGE_FORMATS)",
                        format.what
                    ))
                })?;
                self.entries[at].2 = Stage::Decoding;
                Ok(Some(ImageRead { decoder, bytes }))
            }
            SceneFileKind::VirtualMesh | SceneFileKind::Terrain => {
                let virtual_mesh = kind == SceneFileKind::VirtualMesh;
                let header = if virtual_mesh {
                    VIRTUAL_HEADER
                } else {
                    TERRAIN_HEADER
                };
                let length = bytes.get(..header).and_then(|head| {
                    if virtual_mesh {
                        VirtualMesh::prefix_bytes(head)
                    } else {
                        FileHeader::prefix_bytes(head).ok()
                    }
                });
                let Some(length) = length else {
                    let what = if virtual_mesh {
                        "virtual mesh"
                    } else {
                        "terrain"
                    };
                    return Err(Error::new(
                        ErrorCode::InvalidMesh,
                        "$",
                        format!("{url} is not a {what} file of this engine version"),
                    ));
                };
                // A host that read past the prefix (a server ignoring
                // ranges sends the whole file) keeps just the prefix.
                if bytes.len() < length {
                    if stage == Stage::Prefix {
                        return Err(Error::new(
                            ErrorCode::MissingFile,
                            "$",
                            format!("{url} ends within its prefix"),
                        ));
                    }
                    self.entries[at].2 = Stage::Prefix;
                    self.reads.push(FileRead {
                        uri: uri.to_owned(),
                        prefix: Some(length),
                    });
                    return Ok(None);
                }
                let mut prefix = bytes;
                prefix.truncate(length);
                if virtual_mesh {
                    self.files.add_virtual_mesh(uri, prefix, Some(url));
                } else {
                    self.files.add_terrain(uri, prefix, Some(url));
                }
                self.entries[at].2 = Stage::Held;
                Ok(None)
            }
        }
    }

    /// Image `uri` decoded, as [`decode_image`] decodes it.
    pub fn image(&mut self, uri: &str, image: Image) {
        self.files
            .add_image(uri, image.width, image.height, image.pixels);
        self.held(uri);
    }

    /// Image `uri`'s `bytes` decoded in place with the engine's decoder, as
    /// hosts without workers decode them.
    pub fn decode(&mut self, uri: &str, bytes: &[u8]) -> Result<()> {
        let image = decode_image(uri, bytes).map_err(invalid_texture)?;
        self.image(uri, image);
        Ok(())
    }

    /// Image `uri` transcoded to block-compressed `format`, `width` ×
    /// `height`, with its `levels` from the base down.
    pub fn compressed_image(
        &mut self,
        uri: &str,
        format: TextureFormat,
        width: u32,
        height: u32,
        levels: Vec<Vec<u8>>,
    ) {
        self.files
            .add_compressed_image(uri, format, width, height, levels);
        self.held(uri);
    }

    /// The scene's files, once every one is held.
    pub fn finish(self) -> Result<SceneFiles> {
        if let Some((uri, _, _)) = self
            .entries
            .iter()
            .find(|(_, _, stage)| *stage != Stage::Held)
        {
            return Err(Error::new(
                ErrorCode::MissingFile,
                "$",
                format!("the scene's file {uri} did not load"),
            ));
        }
        Ok(self.files)
    }

    fn held(&mut self, uri: &str) {
        if let Some(entry) = self.entries.iter_mut().find(|(file, _, _)| file == uri) {
            entry.2 = Stage::Held;
        }
    }
}

fn invalid_texture(message: String) -> Error {
    Error::new(ErrorCode::InvalidTexture, "$", message)
}
