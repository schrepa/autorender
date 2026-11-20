use std::sync::Arc;

use engine::{Error, ErrorCode, Expected};
use engine_core::MeshFile;
use engine_meshpack::POSITION;
use glam::Vec3;
use serde::de::DeserializeOwned;

fn text<'a>(bytes: &'a [u8], at: &str) -> Result<&'a str, Error> {
    std::str::from_utf8(bytes)
        .map_err(|_| Error::new(ErrorCode::InvalidRecord, at.to_owned()).expected(Expected::Utf8))
}

pub fn json<T: DeserializeOwned>(bytes: Arc<[u8]>, at: &str) -> Result<T, Error> {
    engine::bricks::parse_json(text(&bytes, at)?).map_err(|error| error.within(at))
}
pub struct Triangles {
    pub positions: Vec<Vec3>,
    pub indices: Vec<u32>,
}

pub fn triangles(bytes: Arc<[u8]>, at: &str) -> Result<Triangles, Error> {
    let mesh = MeshFile::read(bytes, at)?;
    Ok(Triangles {
        positions: (mesh.floats3(POSITION, at)?.into_iter())
            .map(Vec3::from_array)
            .collect(),
        indices: mesh.index_values(),
    })
}
