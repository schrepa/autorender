use std::sync::Arc;

use engine::host::RecordKey;
use engine::prelude::*;
use engine::scene::{MeshStreams, decode_mesh_buffer, encode_mesh_buffer_with};
use engine_core::RecordExchange;

pub(crate) fn key(kind: &'static str, identity: &[u8]) -> RecordKey {
    RecordKey::new(&crate::SOURCE.hash(), kind, identity)
}

pub(crate) fn mesh(
    world: &mut World,
    kind: &'static str,
    identity: &[u8],
    make: impl FnOnce() -> engine::Result<Mesh> + Send + Sync + 'static,
) -> engine::Result<MeshHandle> {
    let handle = world.resource_mut::<AssetLibrary>().reserve_mesh()?;
    world.resource_mut::<RecordExchange>().request(
        key(kind, identity),
        move || {
            make()
                .map(|mesh| mesh_bytes(&mesh))
                .map_err(|error| error.to_string())
        },
        move |world, bytes| {
            let mesh = decode_mesh_buffer(bytes).map_err(|error| error.to_string())?;
            world
                .resource_mut::<AssetLibrary>()
                .fill_mesh(handle, mesh)
                .map_err(|error| error.to_string())
        },
    );
    Ok(handle)
}

fn mesh_bytes(mesh: &Mesh) -> Vec<u8> {
    encode_mesh_buffer_with(
        mesh.vertices(),
        mesh.indices(),
        MeshStreams {
            colors: mesh.colors(),
            uv1: mesh.uv1(),
            lods: mesh.lods().map(AsRef::as_ref),
        },
    )
}

pub(crate) fn textures(
    (records, assets): (&mut RecordExchange, &mut AssetLibrary),
    name: String,
    textures: &[(&str, TextureUsage)],
    make: impl Fn() -> Result<Vec<TextureSource>, String> + Send + Sync + 'static,
) -> engine::Result<Vec<TextureHandle>> {
    let set = TextureSet {
        source: crate::SOURCE.hash(),
        name,
        textures: textures
            .iter()
            .map(|&(texture, usage)| (texture.to_owned(), usage))
            .collect(),
        make: MakeSources::Here(Arc::new(make)),
    };
    records.textures(assets, set)
}
