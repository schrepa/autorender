// World position of a depth-buffer pixel; w is 0 where only sky is (the
// cleared depth, 0 with reversed depth).
fn world_at(pixel: vec2<i32>) -> vec4<f32> {
    let depth = textureLoad(depth_texture, pixel, 0);
    let uv = (vec2<f32>(pixel) + 0.5) * post.size.zw;
    let clip = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let world = post.inverse_view_projection * clip;
    return vec4<f32>(world.xyz / world.w, select(0.0, 1.0, depth > 0.0));
}

// Reflection rays walk at most this many cells of the depth pyramid.
const REFLECTION_STEPS: u32 = 96u;
// A reflected surface counts where the ray passes within this distance of
// it, in world units, plus this share of the ray's length.
const REFLECTION_THICKNESS: f32 = 0.3;
const REFLECTION_THICKNESS_SHARE: f32 = 0.05;
// Rays heading toward the camera reach surfaces from their hidden side: the
// visible surface stands in for it through this much more depth (a solid
// object's), fading over it.
const REFLECTION_SOLID: f32 = 1.0;
// A roughness lobe's half-width in radians per GGX alpha: the Phong lobe
// matching it narrows to 0.244 of its peak there (as in cone-traced
// reflections).
const REFLECTION_CONE: f32 = 1.19;
const FLOAT_MAX: f32 = 0x1.fffffep+127f;
// The pixel of each 2x2 block the half-resolution reflection passes trace:
// the first without temporal antialiasing; with it, another each frame
// (`post.reflections.w`), so the resolve's history gathers every block.
const BLOCK_PIXELS = array<vec2<i32>, 4>(
    vec2<i32>(0, 0), vec2<i32>(1, 1), vec2<i32>(1, 0), vec2<i32>(0, 1),
);

// The full-resolution pixel half-resolution texel `texel` traces.
fn traced_pixel(texel: vec2<i32>) -> vec2<i32> {
    let last = vec2<i32>(post.size.xy) - 1;
    return min(texel * 2 + BLOCK_PIXELS[u32(post.reflections.w) & 3u], last);
}

// The nearest depth (the largest, reversed) of cell `cell` of mip `mip`:
// mip 0 is the depth buffer, mip m > 0 the pyramid's level m - 1, whose
// texels cover 2^m pixels a side (the pyramid repeats the buffer's edge).
// `pyramid` is the pyramid's first level's size: each level halves it, as
// mip levels do, so the march asks the texture its size once.
fn nearest_depth_at(cell: vec2<i32>, mip: i32, pyramid: vec2<u32>) -> f32 {
    if mip == 0 {
        let last = vec2<i32>(post.size.xy) - 1;
        return textureLoad(depth_texture, clamp(cell, vec2<i32>(0), last), 0);
    }
    let last = vec2<i32>(max(pyramid >> vec2<u32>(u32(mip - 1)), vec2<u32>(1u))) - 1;
    return textureLoad(nearest_depth, clamp(cell, vec2<i32>(0), last), mip - 1).r;
}

// Pixel position (rows downward) and depth of world point `world`, and its
// clip w.
fn screen_point(world: vec3<f32>) -> vec4<f32> {
    let clip = post.view_projection * vec4<f32>(world, 1.0);
    let ndc = clip.xyz / clip.w;
    return vec4<f32>((ndc.x * 0.5 + 0.5) * post.size.x, (0.5 - ndc.y * 0.5) * post.size.y,
        ndc.z, clip.w);
}

// The unit normal at octahedral coordinates `encoded` in [-1, 1].
fn unfolded_normal(encoded: vec2<f32>) -> vec3<f32> {
    let z = 1.0 - abs(encoded.x) - abs(encoded.y);
    let fold = max(-z, 0.0);
    let shift = vec2<f32>(select(-fold, fold, encoded.x >= 0.0),
        select(-fold, fold, encoded.y >= 0.0));
    return normalize(vec3<f32>(encoded - shift, z));
}

// The world point at pixel position `pixel` (continuous) and depth `depth`.
fn world_point(pixel: vec2<f32>, depth: f32) -> vec3<f32> {
    let uv = pixel * post.size.zw;
    let world = post.inverse_view_projection
        * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    return world.xyz / world.w;
}

// Screen-space reflections at half resolution: from each
// reflective pixel (the glow target's second channel), a ray along the
// mirror direction of its depth normal crosses the screen as a straight
// line in pixels and depth. It walks the pyramid of nearest depths: a cell
// the ray crosses in front of everything is skipped and the next one is
// coarser; a cell it may meet is refined, down to single pixels (FidelityFX
// SSSR's hierarchical march), so every pixel along the ray near a surface is
// visited and thin objects are found alike from pixel to pixel. The first
// pixel the ray passes behind is its hit, reflected where the ray passes
// close to that surface (or, heading toward the camera, within a solid
// object's depth of it), fading toward the screen edges, the ray's end and
// the surface's thickness; misses weigh nothing.
fn reflection_at(half_texel: vec2<i32>) -> vec4<f32> {
    let last = vec2<i32>(post.size.xy) - 1;
    let pixel = traced_pixel(half_texel);
    let reflectivity = textureLoad(glow, pixel, 0).g;
    if reflectivity < 0.01 {
        return vec4<f32>(0.0);
    }
    let center = world_at(pixel);
    if center.w == 0.0 {
        return vec4<f32>(0.0);
    }
    let p = center.xyz;
    // The normal the lighting shaded (with normal maps and ripples) and the
    // roughness, bent back toward the eye where bumps turn it past the view
    // so the ray leaves the surface.
    let shading = textureLoad(surface_texture, pixel, 0);
    let roughness = shading.z;
    let to_eye = normalize(post.eye.xyz - p);
    var normal = unfolded_normal(shading.xy * 2.0 - 1.0);
    normal = normalize(normal + max(0.05 - dot(normal, to_eye), 0.0) * to_eye);
    let direction = reflect(-to_eye, normal);
    // The ray's segment: `distance` long, ending nine tenths of the way to
    // the eye plane when it heads toward the camera.
    let start = screen_point(p);
    let rows = transpose(post.view_projection);
    let w_rate = dot(rows[3].xyz, direction);
    var reach = post.reflections.x;
    if w_rate < 0.0 {
        reach = min(reach, 0.9 * start.w / -w_rate);
    }
    let end = screen_point(p + direction * reach);
    if !(start.w > 0.0 && end.w > 0.0) {
        return vec4<f32>(0.0);
    }
    let origin = start.xyz;
    let ray = end.xyz - start.xyz;
    let moving = abs(ray.xy) > vec2<f32>(1.0e-9);
    let forward = ray.xy >= vec2<f32>(0.0);
    // Cell edges ahead of the ray, and a little past them so the next cell
    // starts inside.
    let ahead = select(vec2<f32>(0.0), vec2<f32>(1.0), forward);
    let past = select(vec2<f32>(-0.005), vec2<f32>(0.005), forward);
    let top = i32(textureNumLevels(nearest_depth));
    let pyramid = textureDimensions(nearest_depth);
    // Leave the starting pixel first: a surface does not reflect itself.
    var t = FLOAT_MAX;
    let exits = select(vec2<f32>(FLOAT_MAX),
        (floor(origin.xy) + ahead + past - origin.xy) / select(vec2<f32>(1.0), ray.xy, moving),
        moving);
    t = min(exits.x, exits.y);
    var point = origin + ray * t;
    var mip = 0;
    for (var step = 0u; step < REFLECTION_STEPS; step++) {
        if mip < 0 || t > 1.0 || any(point.xy < vec2<f32>(0.0))
                || any(point.xy >= post.size.xy) {
            break;
        }
        let size = exp2(f32(mip));
        let cell = floor(point.xy / size);
        let surface = nearest_depth_at(vec2<i32>(cell), mip, pyramid);
        let edges = select(vec2<f32>(FLOAT_MAX),
            ((cell + ahead) * size + past - origin.xy) / select(vec2<f32>(1.0), ray.xy, moving),
            moving);
        let leave = min(edges.x, edges.y);
        // Reversed depth: the ray meets the cell's nearest depth only while
        // it heads away from the eye.
        var meet = FLOAT_MAX;
        if ray.z < 0.0 {
            meet = (surface - origin.z) / ray.z;
        }
        // In front of everything in the cell: advance to where it leaves the
        // cell (coarser next) or meets its nearest depth (finer next).
        let above = surface < point.z;
        let skipped = above && leave < meet;
        if above {
            t = min(leave, meet);
            point = origin + ray * t;
        }
        if skipped {
            mip = min(mip + 1, top);
        } else {
            mip -= 1;
        }
    }
    if mip >= 0 || t > 1.0 {
        return vec4<f32>(0.0);
    }
    let hit = vec2<i32>(floor(point.xy));
    if any(hit < vec2<i32>(0)) || any(hit > last) || all(abs(point.xy - origin.xy) < vec2<f32>(2.0)) {
        return vec4<f32>(0.0);
    }
    let surface = world_at(hit);
    if surface.w == 0.0 {
        return vec4<f32>(0.0);
    }
    // How far behind that surface the ray passes, against a thickness that
    // grows with the ray's length.
    let ray_point = world_point(point.xy, point.z);
    let travelled = distance(ray_point, p);
    let behind = distance(ray_point, surface.xyz);
    let thickness = REFLECTION_THICKNESS + REFLECTION_THICKNESS_SHARE * travelled;
    var confidence = 1.0 - smoothstep(0.0, thickness, behind);
    confidence *= confidence;
    if ray.z > 0.0 {
        // Heading toward the camera (reversed depth grows), the ray reaches
        // what it meets from the side the camera cannot see, and crosses
        // much of the world per pixel: the surface in front counts fully
        // through the thickness and the ray's length across a pixel there,
        // then fades over a solid object's depth.
        let pixels = max(abs(ray.x), abs(ray.y));
        let stride = distance(
            world_point(point.xy + ray.xy / pixels, point.z + ray.z / pixels), ray_point);
        let near = thickness + stride;
        confidence = 1.0 - smoothstep(near, near + REFLECTION_SOLID, behind);
    }
    let uv = point.xy * post.size.zw;
    let edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    let fraction = min(travelled / max(post.reflections.x, 1.0e-6), 1.0);
    let fade = saturate(edge * 10.0) * (1.0 - fraction * fraction) * confidence;
    // The light reflected over the roughness lobe: the cone's width where it
    // lands, in pixels, picks the blurred scene level read, mirrors reading
    // the scene itself.
    let width = 2.0 * travelled * REFLECTION_CONE * roughness * roughness * post.eye.w
        / max(distance(ray_point, post.eye.xyz), 1.0e-4);
    let level = log2(max(width, 1.0));
    var color = textureSampleLevel(scene, linear, uv, 0.0).rgb;
    if level > 0.0 {
        let blurred = textureSampleLevel(scene_levels, linear, uv, max(level - 1.0, 0.0)).rgb;
        color = mix(color, blurred, saturate(level));
    }
    return vec4<f32>(color, reflectivity * fade);
}

