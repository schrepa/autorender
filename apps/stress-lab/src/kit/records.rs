//! The worlds' generated content as records ([`engine::Records`]): a mesh
//! or texture a world's code builds, kept by the hash of the Stress Lab's
//! code ([`crate::SOURCE`]), its kind and the inputs it is built from.
//! Hosts hand it back from the cook's packs or what they kept; where none
//! holds it, its code builds it in place and the host keeps it. Its handle
//! is reserved at once and names it once it is here: until then what draws
//! with it finds it missing.

use engine::Records;
use engine::host::RecordKey;
use engine::prelude::*;
use engine::scene::{MeshStreams, decode_mesh_buffer, encode_mesh_buffer_with};

/// The key of the record of `kind` built from `identity`, every input its
/// bytes depend on, by the Stress Lab's code.
pub(crate) fn key(kind: &str, identity: &[u8]) -> RecordKey {
    RecordKey::new(&crate::SOURCE, kind, identity)
}

/// A mesh `make` builds, as the record of `kind` on `identity`: its handle,
/// reserved until the record is here.
pub(crate) fn mesh(
    world: &mut World,
    kind: &str,
    identity: &[u8],
    make: impl FnOnce() -> engine::Result<Mesh> + Send + Sync + 'static,
) -> engine::Result<MeshHandle> {
    mesh_then(world, kind, identity, make, |_, _, _| Ok(()))
}

/// [`mesh`], with `then` taking the world, the handle and the mesh once it
/// is here, before the handle names it: what needs the mesh's own
/// vertices spawns there.
pub(crate) fn mesh_then(
    world: &mut World,
    kind: &str,
    identity: &[u8],
    make: impl FnOnce() -> engine::Result<Mesh> + Send + Sync + 'static,
    then: impl FnOnce(&mut World, MeshHandle, &Mesh) -> engine::Result<()> + Send + Sync + 'static,
) -> engine::Result<MeshHandle> {
    let handle = world.resource_mut::<AssetLibrary>().reserve_mesh()?;
    world.resource_mut::<Records>().request(
        key(kind, identity),
        move || {
            make()
                .map(|mesh| mesh_bytes(&mesh))
                .map_err(|error| error.message)
        },
        move |world, bytes| {
            let mesh = decode_mesh_buffer(bytes).map_err(|error| error.message)?;
            then(world, handle, &mesh).map_err(|error| error.message)?;
            world
                .resource_mut::<AssetLibrary>()
                .fill_mesh(handle, mesh)
                .map_err(|error| error.to_string())
        },
    );
    Ok(handle)
}

/// `mesh` as an engine mesh buffer, with its colors, second coordinates
/// and joints.
fn mesh_bytes(mesh: &Mesh) -> Vec<u8> {
    encode_mesh_buffer_with(
        mesh.vertices(),
        mesh.indices(),
        MeshStreams {
            colors: mesh.colors(),
            uv1: mesh.uv1(),
            joints: mesh.joints(),
            weights: mesh.weights(),
        },
    )
}

/// An RGBA8 texture `make` builds as its width, height and pixels, as the
/// record of `kind` on `identity`: its handle, which materials may name at
/// once, reserved until the record is here.
pub(crate) fn texture(
    world: &mut World,
    kind: &str,
    identity: &[u8],
    make: impl FnOnce() -> (u32, u32, Vec<u8>) + Send + Sync + 'static,
) -> engine::Result<TextureHandle> {
    let handle = world.resource_mut::<AssetLibrary>().reserve_texture()?;
    world.resource_mut::<Records>().request(
        key(kind, identity),
        move || {
            let (width, height, pixels) = make();
            let mut bytes = Vec::with_capacity(8 + pixels.len());
            bytes.extend_from_slice(&width.to_le_bytes());
            bytes.extend_from_slice(&height.to_le_bytes());
            bytes.extend_from_slice(&pixels);
            Ok(bytes)
        },
        move |world, bytes| {
            let size = |at: usize| {
                bytes
                    .get(at..at + 4)
                    .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            };
            let (Some(width), Some(height)) = (size(0), size(4)) else {
                return Err("a texture record starts with its width and height".into());
            };
            let texture = Texture::from_rgba(width, height, bytes[8..].to_vec())
                .map_err(|error| error.to_string())?;
            world
                .resource_mut::<AssetLibrary>()
                .fill_texture(handle, texture)
                .map_err(|error| error.to_string())
        },
    );
    Ok(handle)
}
