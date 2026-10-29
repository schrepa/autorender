import package::shadows::sample::{SHADOW_MAPS, map_visibility, shadow};
import package::sky::air::cloud_passing;
import package::terrain::horizons::terrain_visibility;

fn sun_visibility(world: vec3<f32>, normal: vec3<f32>, dx: vec3<f32>, dy: vec3<f32>,
        distance: f32, on_terrain: bool) -> f32 {
    if !SHADOW_MAPS || shadow.state.x == 0.0 {
        return cloud_passing(world);
    }
    let terrain = terrain_visibility(world, on_terrain);
    if terrain <= 0.0 {
        return 0.0;
    }
    let visible = min(terrain, map_visibility(world, normal, dx, dy));
    if visible == 0.0 {
        return 0.0;
    }
    return visible * cloud_passing(world);
}
