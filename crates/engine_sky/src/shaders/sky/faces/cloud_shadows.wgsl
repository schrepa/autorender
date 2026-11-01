import package::face::sun::sun_direction;
import package::generated::sky::CLOUD_SHADOW_CELLS;
import package::sky::piece::sky_piece;

@group(0) @binding(0) var<storage, read> cloud_shadows: array<u32, CLOUD_SHADOW_CELLS * CLOUD_SHADOW_CELLS / 2u>;

fn cloud_shadow_cell(x: u32, z: u32) -> f32 {
    let index = z * CLOUD_SHADOW_CELLS + x;
    return unpack2x16float(cloud_shadows[index / 2u])[index % 2u];
}

fn cloud_shadow(world: vec3<f32>) -> f32 {
    let strength = sky_piece.cloud_shadow.w;
    if strength <= 0.0 || sun_direction().y <= 0.01 {
        return 1.0;
    }
    let rise = max(sky_piece.cloud_layer.x - world.y, 0.0);
    let base = world.xz + sun_direction().xz * (rise / sun_direction().y);
    let size = f32(CLOUD_SHADOW_CELLS);
    let cells = base * sky_piece.cloud_shadow.z - 0.5;
    let local = cells - sky_piece.cloud_shadow.xy;
    if any(local < vec2<f32>(0.0)) || any(local > vec2<f32>(size - 1.0)) {
        return 1.0;
    }
    let count = i32(CLOUD_SHADOW_CELLS);
    let first = vec2<i32>(floor(cells));
    let a = vec2<u32>(first & vec2<i32>(count - 1));
    let b = vec2<u32>((first + vec2<i32>(1)) & vec2<i32>(count - 1));
    let f = cells - floor(cells);
    let passing = mix(
        mix(cloud_shadow_cell(a.x, a.y), cloud_shadow_cell(b.x, a.y), f.x),
        mix(cloud_shadow_cell(a.x, b.y), cloud_shadow_cell(b.x, b.y), f.x), f.y);
    let edge = min(local, vec2<f32>(size - 1.0) - local) / (0.1 * size);
    let fade = smoothstep(0.0, 1.0, saturate(min(edge.x, edge.y)));
    return mix(1.0, passing, strength * fade);
}
