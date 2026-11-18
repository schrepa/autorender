import package::engine::view::stable_normalize;
import package::level::sky::{SkyCorner, sky_corner, sky_pictured, sky_sunned};

@group(1) @binding(0) var sky_picture_map: texture_2d<f32>;
@group(1) @binding(1) var sky_picture_sampler: sampler;

@vertex fn vs_sky_picture(@builtin(vertex_index) vertex: u32) -> SkyCorner {
    return sky_corner(vertex);
}

@fragment fn fs_sky_picture(corner: SkyCorner) -> @location(0) vec4<f32> {
    let toward = stable_normalize(corner.direction);
    let color = sky_pictured(sky_picture_map, sky_picture_sampler, toward);
    return vec4<f32>(sky_sunned(color, toward), 1.0);
}
