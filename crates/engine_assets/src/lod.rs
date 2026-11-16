use crate::Mesh;

pub fn takes_levels(mesh: &Mesh) -> bool {
    mesh.lods().is_none()
}

pub fn authored_input(meshes: &[&Mesh]) -> Vec<u8> {
    let meshes: Vec<_> = meshes
        .iter()
        .map(|mesh| (mesh.vertices(), mesh.indices()))
        .collect();
    engine_lodbuild::authored_bytes(&meshes)
}

pub fn lod_input(mesh: &Mesh) -> Option<Vec<u8>> {
    if !takes_levels(mesh) {
        return None;
    }
    Some(engine_lodbuild::input_bytes(
        mesh.vertices(),
        mesh.indices(),
    ))
}
