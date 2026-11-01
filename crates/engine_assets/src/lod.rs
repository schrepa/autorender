use crate::Mesh;

pub fn takes_levels(mesh: &Mesh) -> bool {
    mesh.lods().is_none()
}

pub fn lod_input(mesh: &Mesh) -> Option<Vec<u8>> {
    if !takes_levels(mesh) {
        return None;
    }
    Some(engine_lodbuild::input_bytes(mesh.vertices(), mesh.indices()))
}
