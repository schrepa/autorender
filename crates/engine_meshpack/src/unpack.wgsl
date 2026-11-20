struct MeshBase {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec2<f32>,
    @location(2) uv0: vec2<f32>,
}

struct MeshLightmap {
    @location(3) uv1: vec2<f32>,
}

struct MeshColor {
    @location(4) color: vec4<f32>,
}

struct MeshSkin {
    @location(5) joints: vec4<u32>,
    @location(6) weights: vec4<f32>,
}

fn mesh_normal(packed: vec2<f32>) -> vec3<f32> {
    var normal = vec3<f32>(packed, 1.0 - abs(packed.x) - abs(packed.y));
    let fold = max(-normal.z, 0.0);
    normal.x += select(fold, -fold, normal.x >= 0.0);
    normal.y += select(fold, -fold, normal.y >= 0.0);
    return normalize(normal);
}
