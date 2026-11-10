import package::engine::shading::SurfacePoint;

fn derive(point: SurfacePoint) -> SurfacePoint {
    var result = point;
    result.roughness = max(point.roughness, 0.03);
    result.alpha = result.roughness * result.roughness;
    result.f0 = mix(vec3<f32>(0.04), point.base_color, point.metalness);
    result.diffuse_color = point.base_color * (1.0 - point.metalness);
    return result;
}
