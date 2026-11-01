use super::*;

fn spline_mesh_view(world: &World, entity: Option<Entity>, context: &Context<'_>) -> Option<Value> {
    let scene = context.scene;
    let entity = entity?;
    let spline = world.get::<engine_core::SplineMesh>(entity)?;
    let chunks = world.get::<engine_core::SplineChunks>(entity);
    let mesh = scene.and_then(|scene| {
        scene
            .meshes()
            .find(|(_, handle)| *handle == spline.mesh)
            .map(|(id, _)| id.to_owned())
    });
    let material = scene.and_then(|scene| {
        scene
            .materials()
            .find(|(_, handle)| *handle == spline.material)
            .map(|(id, _)| id.to_owned())
    });
    Some(json!(SplineMeshView {
        points: spline.points.iter().map(|point| point.to_array()).collect(),
        mesh,
        material,
        length: chunks.map(|chunks| chunks.length),
        copies: chunks.map(|chunks| chunks.copies),
        chunks: chunks.map(|chunks| chunks.entities.len()),
        vertices: chunks.map(|chunks| chunks.vertices),
        error: chunks.and_then(|chunks| chunks.error.clone()),
    }))
}
