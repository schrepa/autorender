//! The worlds' generated content as records ([`engine::Records`]): a mesh a
//! world's code builds, kept by the hash of the Stress Lab's code
//! ([`crate::SOURCE`]), its kind and the inputs it is built from, and sets
//! of textures it generates, which load as the cook compresses them for the
//! device. Hosts hand each record back from the cook's packs or what they
//! kept; where none holds it, its code builds it in place and the host
//! keeps it. Its handle is reserved at once and names it once it is here:
//! until then what draws with it finds it missing.

use std::sync::Arc;

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

/// `mesh` as an engine mesh buffer, with its colors, second coordinates,
/// joints and levels of detail of its own.
fn mesh_bytes(mesh: &Mesh) -> Vec<u8> {
    encode_mesh_buffer_with(
        mesh.vertices(),
        mesh.indices(),
        MeshStreams {
            colors: mesh.colors(),
            uv1: mesh.uv1(),
            joints: mesh.joints(),
            weights: mesh.weights(),
            lods: mesh.lods().map(AsRef::as_ref),
        },
    )
}

/// The textures `textures` of the set `name`, which `make` generates (one
/// source each, in order), asked of `records`: their handles in `assets`,
/// which materials may name at once, reserved until their records are here.
pub(crate) fn textures(
    (records, assets): (&mut Records, &mut AssetLibrary),
    name: String,
    textures: &[(&str, TextureUsage)],
    make: impl Fn() -> Result<Vec<TextureSource>, String> + Send + Sync + 'static,
) -> engine::Result<Vec<TextureHandle>> {
    let set = TextureSet {
        source: crate::SOURCE,
        name,
        textures: textures
            .iter()
            .map(|&(texture, usage)| (texture.to_owned(), usage))
            .collect(),
        make: Arc::new(make),
    };
    records.textures(assets, set).map_err(engine::Error::from)
}
