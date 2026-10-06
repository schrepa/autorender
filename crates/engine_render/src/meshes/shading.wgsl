// Surface shading every mesh pass shares: material texels, normals, the
// sun, environment and local lights, fog and grading, and the targets
// each kind of pass writes.

// Wave normals of liquid surfaces (decision 019): three directional waves
// scrolling with presentation time tilt the normal by their slopes.
fn rippled(normal: vec3<f32>, world: vec3<f32>, strength: f32) -> vec3<f32> {
    let time = camera.position.w;
    let p = world.xz;
    let a = dot(p, vec2<f32>(0.8, 0.6)) * 0.7 + time * 1.1;
    let b = dot(p, vec2<f32>(-0.6, 0.8)) * 1.6 + time * 1.7;
    let c = dot(p, vec2<f32>(0.1, -1.0)) * 3.3 + time * 2.3;
    let slope = vec2<f32>(0.8, 0.6) * (0.126 * cos(a))
        + vec2<f32>(-0.6, 0.8) * (0.112 * cos(b))
        + vec2<f32>(0.1, -1.0) * (0.083 * cos(c));
    return normalize(normal - vec3<f32>(slope.x, 0.0, slope.y) * strength);
}

// What a shaded surface point needs: its material, normals and view.
struct SurfacePoint {
    albedo: vec3<f32>,
    // Coverage: color alpha, times base color texture alpha with maps.
    opacity: f32,
    // Emitted light as a multiple of the albedo, masked by texture alpha.
    emission: f32,
    // glTF emission, and how much indirect light reaches the point.
    emissive: vec3<f32>,
    occlusion: f32,
    roughness: f32,
    alpha: f32,
    f0: vec3<f32>,
    diffuse_color: vec3<f32>,
    normal: vec3<f32>,
    geometric: vec3<f32>,
    view: vec3<f32>,
    n_v: f32,
    world: vec3<f32>,
    // How `world` changes per pixel, for shadow receiver planes.
    world_dx: vec3<f32>,
    world_dy: vec3<f32>,
    distance: f32,
    // Where the sun's shadow is looked up from `world`: zero but for
    // impostors (decision 048), whose rebuilt surfaces look from outside
    // their own meshes' shadows.
    shadow_offset: vec3<f32>,
    // Whether the point is terrain's (decision 052), which lies on the
    // ground of terrain's horizon maps.
    on_terrain: bool,
    // Light from surfaces around the point where they hide the sky, per
    // unit diffuse color: terrain's (decision 055), zero elsewhere.
    bounce: vec3<f32>,
    // The share of the sun the point's own mesh hides: impostors' (decision
    // 060), zero elsewhere.
    sun_shade: f32,
}

// Shaded light of a surface: graded color (and its linear light before the
// grade, which reflection probes capture), the ambient share of its light
// (for ambient occlusion), its glow weight, its reflectivity, shading normal
// and roughness for screen-space reflections, and its coverage.
struct Shaded {
    color: vec3<f32>,
    linear: vec3<f32>,
    ambient: f32,
    glow: f32,
    reflect: f32,
    normal: vec3<f32>,
    roughness: f32,
    opacity: f32,
}

// Texture coordinates of map `index` (base color, metallic-roughness,
// normal, occlusion, emissive) through its transform, from the first or
// second set as the map reads.
fn map_uv(index: u32, uv: vec2<f32>, uv1: vec2<f32>) -> vec2<f32> {
    let second = ((u32(material_maps.params.w) >> index) & 1u) == 1u;
    let point = vec3<f32>(select(uv, uv1, second), 1.0);
    return vec2<f32>(dot(material_maps.transforms[index * 2u].xyz, point),
        dot(material_maps.transforms[index * 2u + 1u].xyz, point));
}

// Screen variance of the pixel footprint and the largest squared roughness
// specular antialiasing adds.
const SPECULAR_VARIANCE: f32 = 0.15;
const SPECULAR_LIMIT: f32 = 0.2;

// Normal mapping without stored tangents (decision 031): the tangent frame
// from screen derivatives of the position and of the normal map's texture
// coordinates. glTF normal maps point green up the image, toward decreasing v.
fn mapped_normal(normal: vec3<f32>, dp_dx: vec3<f32>, dp_dy: vec3<f32>, uv_dx: vec2<f32>,
        uv_dy: vec2<f32>, sampled: vec3<f32>, scale: f32) -> vec3<f32> {
    let dp_dy_perp = cross(dp_dy, normal);
    let dp_dx_perp = cross(normal, dp_dx);
    let tangent = dp_dy_perp * uv_dx.x + dp_dx_perp * uv_dy.x;
    let bitangent = dp_dy_perp * uv_dx.y + dp_dx_perp * uv_dy.y;
    let frame = inverseSqrt(max(max(dot(tangent, tangent), dot(bitangent, bitangent)), 1.0e-30));
    let deflection = sampled.xy * (scale * frame);
    return stable_normalize(tangent * deflection.x - bitangent * deflection.y + normal * sampled.z);
}

// What a surface point reads from its material's maps: the base color
// texel, the roughness and metalness factors, the occlusion and emissive
// texels, and the normal map's normal with its coordinates' screen
// derivatives. Materials without maps read the base color alone.
struct MaterialTexels {
    base: vec4<f32>,
    factors: vec2<f32>,
    occlusion: f32,
    emissive: vec3<f32>,
    normal: vec3<f32>,
    normal_dx: vec2<f32>,
    normal_dy: vec2<f32>,
    // What masked tests scale the base color's alpha by.
    coverage: f32,
    // World-mapped normal maps' normal (decision 093), world space.
    world_normal: vec3<f32>,
}

fn untextured() -> MaterialTexels {
    var texels: MaterialTexels;
    texels.factors = vec2<f32>(1.0);
    texels.occlusion = 1.0;
    texels.normal = vec3<f32>(0.0, 0.0, 1.0);
    texels.coverage = 1.0;
    return texels;
}

// Masked coverage: alpha averaged down a texture's mip chain falls under the
// cutoff, so thin cut-outs such as needles and leaves thin out with
// distance. Masked tests scale alpha up by this much per level below the
// base (Golus's mip scale), keeping their coverage.
const MASK_MIP_SCALE: f32 = 0.25;

// What masked tests scale alpha by where the base color texture's
// coordinates change by `dx` across and `dy` down a pixel.
fn mask_coverage(dx: vec2<f32>, dy: vec2<f32>) -> f32 {
    let size = vec2<f32>(textureDimensions(base_color_texture));
    let x = dx * size;
    let y = dy * size;
    return 1.0 + MASK_MIP_SCALE * max(0.5 * log2(max(dot(x, x), dot(y, y))), 0.0);
}

// The texels of texture coordinates `uv` and `uv1` with implicit
// derivatives, so callers sample in uniform control flow.
fn material_texels(uv: vec2<f32>, uv1: vec2<f32>) -> MaterialTexels {
    var texels = untextured();
    // RGBA8UnormSrgb sampling decodes source RGB; alpha masks emission and,
    // with maps, is coverage.
    var base_uv = uv;
    if material_flag(1u, MATERIAL_MAPS) {
        base_uv = map_uv(0u, uv, uv1);
    }
    texels.base = textureSample(base_color_texture, base_color_sampler, base_uv);
    if ALPHA_MASK {
        texels.coverage = mask_coverage(dpdx(base_uv), dpdy(base_uv));
    }
    if material_flag(1u, MATERIAL_MAPS) {
        // glTF: green is roughness and blue metalness, red packed occlusion.
        var metallic_roughness = vec4<f32>(1.0);
        if material_flag(128u, METALLIC_ROUGHNESS_MAP) {
            metallic_roughness = textureSample(metallic_roughness_texture,
                metallic_roughness_sampler, map_uv(1u, uv, uv1));
        }
        texels.factors = metallic_roughness.gb;
        if material_flag(1024u, PACKED_OCCLUSION) {
            texels.occlusion = metallic_roughness.r;
        } else if material_flag(256u, OCCLUSION_MAP) {
            texels.occlusion = textureSample(occlusion_texture, occlusion_sampler,
                map_uv(3u, uv, uv1)).r;
        }
        texels.emissive = vec3<f32>(1.0);
        if material_flag(512u, EMISSIVE_MAP) {
            texels.emissive = textureSample(emissive_texture, emissive_sampler,
                map_uv(4u, uv, uv1)).rgb;
        }
    }
    if material_flag(2u, NORMAL_MAP) {
        let normal_uv = map_uv(2u, uv, uv1);
        texels.normal = textureSample(normal_texture, normal_sampler, normal_uv).xyz * 2.0 - 1.0;
        texels.normal_dx = dpdx(normal_uv);
        texels.normal_dy = dpdy(normal_uv);
    }
    return texels;
}

// Coordinates on axis `axis`'s plane of world-mapped maps (decision 093),
// as terrain's maps lie (`meshes/terrain.wgsl`'s `terrain_project`): x
// sees (z, -y), y sees (x, z) and z sees (x, -y), so on the sides the maps'
// up is the world's.
fn world_plane(p: vec3<f32>, axis: u32) -> vec2<f32> {
    if axis == 0u {
        return vec2<f32>(p.z, -p.y);
    }
    if axis == 2u {
        return vec2<f32>(p.x, -p.y);
    }
    return p.xz;
}

// The shares of the three projections a surface of `normal` blends:
// sharpened toward the axis it faces most, those barely showing skipped.
fn world_shares(normal: vec3<f32>) -> vec3<f32> {
    let sharp = pow(abs(normal), vec3<f32>(4.0));
    let shares = max(sharp / max(sharp.x + sharp.y + sharp.z, 1.0e-6) - 0.03, vec3<f32>(0.0));
    return shares / max(shares.x + shares.y + shares.z, 1.0e-6);
}

// Map `index` of `map` at `world` (changing by `dx` across and `dy` down a
// pixel) on axis `axis`'s plane, one repeat every `emissive.w` meters,
// through its transform.
fn world_map_texel(map: texture_2d<f32>, map_sampler: sampler, index: u32, world: vec3<f32>,
        dx: vec3<f32>, dy: vec3<f32>, axis: u32) -> vec4<f32> {
    let scale = 1.0 / max(material_maps.emissive.w, 1.0e-6);
    let uv = world_plane(world, axis) * scale;
    let grad_x = world_plane(dx, axis) * scale;
    let grad_y = world_plane(dy, axis) * scale;
    return textureSampleGrad(map, map_sampler, map_uv(index, uv, uv),
        map_gradient(index, grad_x, grad_x), map_gradient(index, grad_y, grad_y));
}

// A world-mapped normal map's tangent-space `sampled` normal on axis
// `axis`'s plane, deflections scaled by `scale`, added to the surface's
// `normal` in that plane (whiteout blending), as terrain's are.
fn world_plane_normal(sampled: vec3<f32>, axis: u32, normal: vec3<f32>, scale: f32)
        -> vec3<f32> {
    let n = vec3<f32>(sampled.xy * scale, sampled.z);
    if axis == 0u {
        return stable_normalize(vec3<f32>(abs(n.z) * normal.x, n.y + normal.y, n.x + normal.z));
    }
    if axis == 2u {
        return stable_normalize(vec3<f32>(n.x + normal.x, n.y + normal.y, abs(n.z) * normal.z));
    }
    return stable_normalize(vec3<f32>(n.x + normal.x, abs(n.z) * normal.y, normal.z - n.y));
}

// The texels of a world-mapped material (decision 093) at `world` on a
// surface of `normal`, changing by `dx` across and `dy` down a pixel: each
// map blended over the projections the surface faces, its normal map's
// normal in world space.
fn world_texels(world: vec3<f32>, normal: vec3<f32>, dx: vec3<f32>, dy: vec3<f32>)
        -> MaterialTexels {
    var texels = untextured();
    let shares = world_shares(normal);
    var base = vec4<f32>(0.0);
    var metallic_roughness = vec4<f32>(0.0);
    var occlusion = 0.0;
    var emissive = vec3<f32>(0.0);
    var mapped = vec3<f32>(0.0);
    for (var axis = 0u; axis < 3u; axis++) {
        let share = shares[axis];
        if share <= 0.0 {
            continue;
        }
        base += share * world_map_texel(base_color_texture, base_color_sampler, 0u, world, dx, dy,
            axis);
        if material_flag(128u, METALLIC_ROUGHNESS_MAP) {
            metallic_roughness += share * world_map_texel(metallic_roughness_texture,
                metallic_roughness_sampler, 1u, world, dx, dy, axis);
        }
        if material_flag(256u, OCCLUSION_MAP) {
            occlusion += share * world_map_texel(occlusion_texture, occlusion_sampler, 3u, world,
                dx, dy, axis).r;
        }
        if material_flag(512u, EMISSIVE_MAP) {
            emissive += share * world_map_texel(emissive_texture, emissive_sampler, 4u, world, dx,
                dy, axis).rgb;
        }
        if material_flag(2u, NORMAL_MAP) {
            let sampled = world_map_texel(normal_texture, normal_sampler, 2u, world, dx, dy, axis)
                .xyz * 2.0 - 1.0;
            mapped += share * world_plane_normal(sampled, axis, normal, material_maps.params.x);
        }
    }
    texels.base = base;
    if material_flag(128u, METALLIC_ROUGHNESS_MAP) {
        texels.factors = metallic_roughness.gb;
    }
    if material_flag(1024u, PACKED_OCCLUSION) {
        texels.occlusion = metallic_roughness.r;
    } else if material_flag(256u, OCCLUSION_MAP) {
        texels.occlusion = occlusion;
    }
    texels.emissive = vec3<f32>(1.0);
    if material_flag(512u, EMISSIVE_MAP) {
        texels.emissive = emissive;
    }
    if material_flag(2u, NORMAL_MAP) {
        texels.world_normal = mapped;
    }
    return texels;
}

// How the texture coordinates of both sets change per pixel across and
// down.
struct UvGradients {
    dx: vec2<f32>,
    dy: vec2<f32>,
    dx1: vec2<f32>,
    dy1: vec2<f32>,
}

// How map `index`'s coordinates change for changes `d` and `d1` of the first
// and second sets: its transform's linear part.
fn map_gradient(index: u32, d: vec2<f32>, d1: vec2<f32>) -> vec2<f32> {
    let second = ((u32(material_maps.params.w) >> index) & 1u) == 1u;
    let change = vec3<f32>(select(d, d1, second), 0.0);
    return vec2<f32>(dot(material_maps.transforms[index * 2u].xyz, change),
        dot(material_maps.transforms[index * 2u + 1u].xyz, change));
}

// `material_texels` with explicit derivatives, for pixels shading triangles
// their neighbors may not share (decision 045).
fn material_texels_grad(uv: vec2<f32>, uv1: vec2<f32>, gradients: UvGradients)
        -> MaterialTexels {
    var texels = untextured();
    var base_uv = uv;
    var base_dx = gradients.dx;
    var base_dy = gradients.dy;
    if material_flag(1u, MATERIAL_MAPS) {
        base_uv = map_uv(0u, uv, uv1);
        base_dx = map_gradient(0u, gradients.dx, gradients.dx1);
        base_dy = map_gradient(0u, gradients.dy, gradients.dy1);
    }
    texels.base = textureSampleGrad(base_color_texture, base_color_sampler, base_uv, base_dx,
        base_dy);
    if ALPHA_MASK {
        texels.coverage = mask_coverage(base_dx, base_dy);
    }
    if material_flag(1u, MATERIAL_MAPS) {
        var metallic_roughness = vec4<f32>(1.0);
        if material_flag(128u, METALLIC_ROUGHNESS_MAP) {
            metallic_roughness = textureSampleGrad(metallic_roughness_texture,
                metallic_roughness_sampler, map_uv(1u, uv, uv1),
                map_gradient(1u, gradients.dx, gradients.dx1),
                map_gradient(1u, gradients.dy, gradients.dy1));
        }
        texels.factors = metallic_roughness.gb;
        if material_flag(1024u, PACKED_OCCLUSION) {
            texels.occlusion = metallic_roughness.r;
        } else if material_flag(256u, OCCLUSION_MAP) {
            texels.occlusion = textureSampleGrad(occlusion_texture, occlusion_sampler,
                map_uv(3u, uv, uv1), map_gradient(3u, gradients.dx, gradients.dx1),
                map_gradient(3u, gradients.dy, gradients.dy1)).r;
        }
        texels.emissive = vec3<f32>(1.0);
        if material_flag(512u, EMISSIVE_MAP) {
            texels.emissive = textureSampleGrad(emissive_texture, emissive_sampler,
                map_uv(4u, uv, uv1), map_gradient(4u, gradients.dx, gradients.dx1),
                map_gradient(4u, gradients.dy, gradients.dy1)).rgb;
        }
    }
    if material_flag(2u, NORMAL_MAP) {
        texels.normal_dx = map_gradient(2u, gradients.dx, gradients.dx1);
        texels.normal_dy = map_gradient(2u, gradients.dy, gradients.dy1);
        texels.normal = textureSampleGrad(normal_texture, normal_sampler, map_uv(2u, uv, uv1),
            texels.normal_dx, texels.normal_dy).xyz * 2.0 - 1.0;
    }
    return texels;
}

// The shading normal of a point whose geometric normal is `geometric`: bent
// by the normal map's `sampled` normal along the tangent frame of the world
// position's and the map's coordinates' screen derivatives, then by
// `ripples`.
fn shading_normal(geometric: vec3<f32>, world: vec3<f32>, world_dx: vec3<f32>,
        world_dy: vec3<f32>, sampled: vec3<f32>, normal_dx: vec2<f32>, normal_dy: vec2<f32>,
        ripples: f32) -> vec3<f32> {
    var normal = geometric;
    if material_flag(2u, NORMAL_MAP) {
        normal = mapped_normal(geometric, world_dx, world_dy, normal_dx, normal_dy, sampled,
            material_maps.params.x);
    }
    if ripples > 0.0 {
        normal = rippled(normal, world, ripples);
    }
    return normal;
}

// The geometric normal of interpolated normal `surface_normal`: back faces
// of double-sided materials are lit from their own side.
fn geometric_normal(surface_normal: vec3<f32>, front: bool) -> vec3<f32> {
    let geometric = stable_normalize(surface_normal);
    if material_flag(8u, DOUBLE_SIDED) && !front {
        return -geometric;
    }
    return geometric;
}

// The point of `texels` sampled there, before specular antialiasing. `color`
// is linear color and opacity; `front` whether the surface faces the camera.
fn textured_point(color: vec4<f32>, surface_normal: vec3<f32>, world: vec3<f32>,
        world_dx: vec3<f32>, world_dy: vec3<f32>, texels: MaterialTexels, packed: u32,
        front: bool) -> SurfacePoint {
    // Roughness, metalness, emission over 16 and ripples.
    let surface = unpack4x8unorm(packed);
    var point: SurfacePoint;
    point.albedo = texels.base.rgb * color.rgb;
    point.opacity = color.a;
    point.emission = surface.z * 16.0 * texels.base.a;
    point.emissive = vec3<f32>(0.0);
    point.occlusion = 1.0;
    var roughness = surface.x;
    var metalness = surface.y;
    if material_flag(1u, MATERIAL_MAPS) {
        point.opacity *= texels.base.a;
        roughness *= texels.factors.x;
        metalness *= texels.factors.y;
        point.occlusion = 1.0 + material_maps.params.y * (texels.occlusion - 1.0);
        point.emissive = material_maps.emissive.rgb * texels.emissive;
    }
    point.roughness = max(roughness, 0.03);
    point.alpha = point.roughness * point.roughness;
    point.f0 = mix(vec3<f32>(0.04), point.albedo, metalness);
    point.diffuse_color = point.albedo * (1.0 - metalness);
    point.geometric = geometric_normal(surface_normal, front);
    point.normal = shading_normal(point.geometric, world, world_dx, world_dy, texels.normal,
        texels.normal_dx, texels.normal_dy, surface.w);
    if material_flag(4096u, WORLD_MAPPING) && material_flag(2u, NORMAL_MAP)
        && dot(texels.world_normal, texels.world_normal) > 0.0 {
        // World-mapped normal maps (decision 093) shade their own normal,
        // turned to the side the surface shows.
        var mapped = stable_normalize(texels.world_normal);
        if dot(mapped, point.geometric) < 0.0 {
            mapped = -mapped;
        }
        point.normal = mapped;
    }
    if material_flag(8192u, TOP_LAYER) {
        // The top layer (decision 093) settles on faces turned up, broken
        // by noise and gathering in the darker hollows of the base color.
        let breakup = variety_noise(world.xz * 2.3 + vec2<f32>(world.y * 0.7, 3.1)) * 0.6
            + variety_noise(world.xz * 0.55 - vec2<f32>(world.y * 0.4, 7.7)) * 0.4;
        let hollow = 1.0 - dot(texels.base.rgb, vec3<f32>(0.299, 0.587, 0.114));
        let lift = point.normal.y + 0.5 * (breakup - 0.5) + 0.3 * (hollow - 0.5);
        let reach = material_maps.top.w;
        let amount = smoothstep(1.0 - reach, 1.2 - reach, lift) * step(1.0e-4, reach);
        point.albedo = mix(point.albedo, material_maps.top.rgb, amount);
        point.diffuse_color = mix(point.diffuse_color, material_maps.top.rgb, amount);
        point.f0 = mix(point.f0, vec3<f32>(0.04), amount);
        roughness = mix(roughness, material_maps.layer.x, amount);
        point.roughness = max(roughness, 0.03);
        point.alpha = point.roughness * point.roughness;
        point.normal = stable_normalize(mix(point.normal, point.geometric, 0.5 * amount));
    }
    let to_eye = camera.position.xyz - world;
    point.distance = length(to_eye);
    point.view = stable_normalize(to_eye);
    point.n_v = max(dot(point.normal, point.view), 1.0e-4);
    point.world = world;
    point.world_dx = world_dx;
    point.world_dy = world_dy;
    return point;
}

// Specular antialiasing (Kaplanyan et al.'s kernel, with Filament's variance
// and limit): normals turning within a pixel (by `turn_dx` across and
// `turn_dy` down), over fine bumps or across facet edges, widen the
// highlight lobe, so they shade as the rougher surface they average to
// instead of sparkling or streaking.
fn widened(point: SurfacePoint, turn_dx: vec3<f32>, turn_dy: vec3<f32>) -> SurfacePoint {
    var result = point;
    let widening = min(2.0 * SPECULAR_VARIANCE * (dot(turn_dx, turn_dx) + dot(turn_dy, turn_dy)),
        SPECULAR_LIMIT);
    result.alpha = sqrt(saturate(point.alpha * point.alpha + widening));
    result.roughness = sqrt(result.alpha);
    return result;
}

// Sampling uses implicit derivatives, so callers build the point in uniform
// control flow; what follows may not be. `color` is linear color and
// opacity; `front` whether the surface faces the camera.
fn surface_point(color: vec4<f32>, surface_normal: vec3<f32>, world: vec3<f32>,
        uv: vec2<f32>, uv1: vec2<f32>, packed: u32, front: bool) -> SurfacePoint {
    let world_dx = dpdx(world);
    let world_dy = dpdy(world);
    var texels: MaterialTexels;
    if material_flag(4096u, WORLD_MAPPING) {
        texels = world_texels(world, stable_normalize(surface_normal), world_dx, world_dy);
    } else {
        texels = material_texels(uv, uv1);
    }
    let point = textured_point(color, surface_normal, world, world_dx, world_dy, texels, packed,
        front);
    let result = widened(point, dpdx(point.normal), dpdy(point.normal));
    if ALPHA_MASK && result.opacity * texels.coverage < material_maps.params.z {
        discard;
    }
    return result;
}

// Where the ray from `position` along `direction` leaves the box `extent`
// around `center`, seen from `center`: the direction to read a probe's map
// along so reflections land on its walls (parallax correction).
fn box_projected(position: vec3<f32>, direction: vec3<f32>, center: vec3<f32>,
        extent: vec3<f32>) -> vec3<f32> {
    let safe = select(direction, vec3<f32>(1.0e-6), abs(direction) < vec3<f32>(1.0e-6));
    let exits = max((center - extent - position) / safe, (center + extent - position) / safe);
    return position + direction * min(exits.x, min(exits.y, exits.z)) - center;
}

// The environment's light reflected toward `reflected` off a surface of
// `normal` and `roughness` at `world`: the reflection probes whose boxes
// hold it (decision 043), smallest first, each fading out toward its faces
// over its blend distance, and the sky's map for what they leave; each map
// at the level blurred by that roughness, read along the lobe's dominant
// direction, which leans toward the normal as the surface roughens (Lagarde
// and de Rousiers).
fn environment_light(reflected: vec3<f32>, normal: vec3<f32>, roughness: f32,
        world: vec3<f32>) -> vec3<f32> {
    let smoothness = 1.0 - roughness;
    let lean = smoothness * (sqrt(smoothness) + roughness);
    let direction = normalize(mix(normal, reflected, lean));
    let level = roughness * f32(textureNumLevels(environment_maps) - 1u);
    var light = vec3<f32>(0.0);
    var covered = 0.0;
    let header = probe_words[0];
    if header.w != 0u {
        let grid = probe_words[1];
        let cell = vec2<i32>(floor((world.xz - bitcast<vec2<f32>>(header.xy))
            / bitcast<f32>(header.z)));
        if all(cell >= vec2<i32>(0)) && all(vec2<u32>(cell) < grid.xy) {
            let list = probe_words[grid.z + u32(cell.y) * grid.x + u32(cell.x)];
            for (var n = 0u; n < list.y && covered < 0.999; n++) {
                let entry = grid.w + (list.x + n) * 3u;
                // Interiors not captured yet (decision 079) reflect nothing.
                if probe_words[entry + 2u].w == 0u {
                    continue;
                }
                let first = probe_words[entry];
                let second = probe_words[entry + 1u];
                let center = bitcast<vec3<f32>>(first.xyz);
                let blend = bitcast<f32>(first.w);
                let extent = bitcast<vec3<f32>>(second.xyz);
                let inside = extent - abs(world - center);
                let depth = min(inside.x, min(inside.y, inside.z));
                if depth <= 0.0 {
                    continue;
                }
                let weight = select(1.0, saturate(depth / blend), blend > 0.0) * (1.0 - covered);
                let along = box_projected(world, direction, center, extent);
                light += weight * textureSampleLevel(environment_maps, environment_sampler, along,
                    second.w, level).rgb;
                covered += weight;
            }
        }
    }
    if covered < 0.999 {
        light += (1.0 - covered)
            * textureSampleLevel(environment_maps, environment_sampler, direction, 0, level).rgb;
    }
    return light;
}

// The shares of the sky's light (x) and the sun's (y) reaching `world`
// (decision 079): inside interiors' boxes, theirs, blending in over each
// box's blend distance inside its faces; all of both elsewhere.
fn interior_light(world: vec3<f32>) -> vec2<f32> {
    var light = vec2<f32>(1.0);
    let header = probe_words[0];
    if header.w == 0u {
        return light;
    }
    let grid = probe_words[1];
    let cell = vec2<i32>(floor((world.xz - bitcast<vec2<f32>>(header.xy))
        / bitcast<f32>(header.z)));
    if any(cell < vec2<i32>(0)) || any(vec2<u32>(cell) >= grid.xy) {
        return light;
    }
    let list = probe_words[grid.z + u32(cell.y) * grid.x + u32(cell.x)];
    if list.z == 0u {
        return light;
    }
    for (var n = 0u; n < list.y; n++) {
        let entry = grid.w + (list.x + n) * 3u;
        let shares = probe_words[entry + 2u];
        if shares.z == 0u {
            continue;
        }
        let first = probe_words[entry];
        let center = bitcast<vec3<f32>>(first.xyz);
        let blend = bitcast<f32>(first.w);
        let extent = bitcast<vec3<f32>>(probe_words[entry + 1u].xyz);
        let inside = extent - abs(world - center);
        let depth = min(inside.x, min(inside.y, inside.z));
        if depth <= 0.0 {
            continue;
        }
        let weight = select(1.0, saturate(depth / blend), blend > 0.0);
        light = min(light, mix(vec2<f32>(1.0), bitcast<vec2<f32>>(shares.xy), weight));
    }
    return light;
}

// Lit, fogged and graded color of `point` with `local` light added.
fn finish(point: SurfacePoint, local: vec3<f32>) -> Shaded {
    let normal = point.normal;
    let view = point.view;
    let sun = lighting.sun.xyz;
    let n_l = dot(normal, sun);
    let specular = lighting.sun_color.w;
    // Wrapped diffuse softens the terminator of flat low-poly faces. Where
    // a face itself turns from the sun, its mesh's own shadow darkens it
    // whatever its normal says, so the wrap fades out over the faces about
    // the sun's edge instead of stepping down at every one of them.
    // Impostors' cards are not their surfaces' faces.
    var facing = 1.0;
    let face = cross(point.world_dx, point.world_dy);
    if dot(face, face) > 1.0e-20 && all(point.shadow_offset == vec3<f32>(0.0)) {
        let toward = select(-1.0, 1.0, dot(face, point.geometric) >= 0.0);
        facing = smoothstep(-0.05, 0.1, dot(normalize(face) * toward, sun));
    }
    let lambert = max(n_l, 0.0);
    let wrapped = lambert + (max((n_l + 0.25) / 1.25, 0.0) - lambert) * facing;
    var highlight = vec3<f32>(0.0);
    if n_l > 0.0 {
        let half_vector = normalize(view + sun);
        let n_h = max(dot(normal, half_vector), 0.0);
        highlight = fresnel(point.f0, max(dot(view, half_vector), 0.0))
            * (distribution(n_h, point.alpha) * visibility(point.n_v, n_l, point.alpha) * n_l);
    }
    // The environment mirrored in the surface, blurred by its roughness,
    // weighted by the environment BRDF.
    let mirrored = environment_light(reflect(-view, normal), normal, point.roughness,
        point.world);
    let brdf = environment_brdf(point.f0, point.roughness, point.n_v);
    let reflection = mirrored * brdf * specular;
    // Rim light outlines walls against the sky; level surfaces seen at
    // grazing angles (distant ground and roofs) take none.
    let grazing = 1.0 - max(dot(point.geometric, view), 0.0);
    let rim = (grazing * grazing) * (grazing * grazing) * lighting.sun.w
        * (1.0 - abs(point.geometric.y));
    // Occlusion maps darken indirect light only (glTF), as do the covers of
    // placed objects over the point (decision 059).
    let covered_sky = occupancy_visibility(point.world);
    // Inside interiors (decision 079) only their shares of the sky and sun
    // reach; what they reflect is their probe's own capture.
    let interior = interior_light(point.world);
    let sky_light = (point.diffuse_color * (sky_average(normal) * lighting.zenith.w
        + lighting.horizon.rgb * rim) * interior.x + reflection) * point.occlusion * covered_sky
        + point.diffuse_color * point.bounce * covered_sky * interior.x;
    let sun_light = (point.diffuse_color * wrapped + highlight * specular) * lighting.sun_color.rgb
        * sun_visibility(point.world + point.shadow_offset, point.geometric, point.world_dx,
            point.world_dy, point.distance, point.on_terrain) * (1.0 - point.sun_shade)
        * interior.y;
    let emitted = (point.albedo * point.emission + point.emissive) * lighting.effects.x;
    let lit = sky_light + sun_light + emitted + local;
    let through = atmospheric(lit, camera.position.xyz, point.world, -view);
    let fogged = through.rgb;
    let covered = through.a;
    var shaded: Shaded;
    shaded.opacity = point.opacity;
    shaded.normal = normal;
    shaded.roughness = point.roughness;
    shaded.linear = fogged;
    shaded.color = grade(fogged);
    shaded.ambient = saturate(luminance(sky_light) / max(luminance(lit), 1.0e-6)) * (1.0 - covered);
    shaded.glow = glow_weight(fogged);
    // Reflectivity for screen-space reflections: the environment BRDF's
    // share, fading out toward the roughness limit and through fog.
    if lighting.effects.w != 0.0 {
        let smooth_enough = saturate(2.0 * (1.0 - point.roughness / lighting.effects.z));
        shaded.reflect = saturate(luminance(brdf) * specular) * smooth_enough * (1.0 - covered);
    }
    return shaded;
}

// What a fragment shades: linear color and opacity, world normal and
// position, both texture coordinate sets, the packed surface and the pixel.
struct Fragment {
    color: vec4<f32>,
    normal: vec3<f32>,
    world: vec3<f32>,
    uv: vec2<f32>,
    uv1: vec2<f32>,
    surface: u32,
    pixel: vec2<f32>,
}

// Meshes without a second stream read their first coordinates as second.
fn fragment(input: VertexOutput) -> Fragment {
    var values: Fragment;
    values.color = output_color(input.material);
    values.normal = input.normal_u.xyz;
    values.world = input.world_v.xyz;
    values.uv = vec2<f32>(input.normal_u.w, input.world_v.w);
    values.uv1 = values.uv;
    values.surface = input.material.z;
    values.pixel = input.clip_position.xy;
    return values;
}

// Vertex colors multiply the color and opacity (glTF `COLOR_0`).
fn extended_fragment(input: ExtendedOutput) -> Fragment {
    var values: Fragment;
    values.color = output_color(input.material) * input.color;
    values.normal = input.normal_u.xyz;
    values.world = input.world_v.xyz;
    values.uv = vec2<f32>(input.normal_u.w, input.world_v.w);
    values.uv1 = input.uv1;
    values.surface = input.material.z;
    values.pixel = input.clip_position.xy;
    return values;
}

fn shade(input: Fragment, front: bool) -> Shaded {
    return finish(surface_point(input.color, input.normal, input.world, input.uv, input.uv1,
        input.surface, front), vec3<f32>(0.0));
}

// Light of the local lights of this pixel's cluster at `point`: windowed
// inverse-square falloff, spot cones and the same GGX highlights as the sun,
// fading out over the last tenth of the clustered depth range.
fn local_light(point: SurfacePoint, pixel: vec2<f32>) -> vec3<f32> {
    let depth = dot(clusters.depth.xyz, point.world) + clusters.depth.w;
    let end = clusters.projection.w;
    if depth >= end {
        return vec3<f32>(0.0);
    }
    let counts = clusters.counts;
    let tile = min(vec2<u32>(max(pixel, vec2<f32>(0.0)) / clusters.grid.xy),
        counts.xy - vec2<u32>(1u));
    let slice = u32(clamp(floor(log2(max(depth, 1.0e-6)) * clusters.grid.z + clusters.grid.w),
        0.0, f32(counts.z - 1u)));
    let cluster = (slice * counts.y + tile.y) * counts.x + tile.x;
    let range = cluster_ranges[cluster];
    let specular = lighting.sun_color.w;
    var sum = vec3<f32>(0.0);
    for (var n = 0u; n < range.y; n++) {
        let light = lights[cluster_lights[range.x + n]];
        let to_light = light.position - point.world;
        let d2 = dot(to_light, to_light);
        let r2 = light.range * light.range;
        if d2 >= r2 {
            continue;
        }
        let l = to_light * inverseSqrt(max(d2, 1.0e-8));
        let n_l = dot(point.normal, l);
        if n_l <= 0.0 {
            continue;
        }
        let ratio = d2 / r2;
        let window = saturate(1.0 - ratio * ratio);
        var falloff = window * window / max(d2, 0.01);
        if light.cos_outer > -1.5 {
            falloff *= smoothstep(light.cos_outer, light.cos_inner, dot(-l, light.direction));
        }
        let half_vector = normalize(point.view + l);
        let n_h = max(dot(point.normal, half_vector), 0.0);
        let highlight = fresnel(point.f0, max(dot(point.view, half_vector), 0.0))
            * (distribution(n_h, point.alpha) * visibility(point.n_v, n_l, point.alpha));
        sum += (point.diffuse_color + highlight * specular) * light.color * (falloff * n_l);
    }
    return sum * saturate((end - depth) / (0.1 * end));
}

fn shade_lit(input: Fragment, front: bool) -> Shaded {
    let point = surface_point(input.color, input.normal, input.world, input.uv, input.uv1,
        input.surface, front);
    return finish(point, local_light(point, input.pixel));
}

fn temporal_output(input: TemporalOutput) -> VertexOutput {
    var output: VertexOutput;
    output.clip_position = input.clip_position;
    output.normal_u = input.normal_u;
    output.world_v = input.world_v;
    output.material = input.material;
    return output;
}

fn extended_temporal_output(input: ExtendedTemporalOutput) -> ExtendedOutput {
    var output: ExtendedOutput;
    output.clip_position = input.clip_position;
    output.normal_u = input.normal_u;
    output.world_v = input.world_v;
    output.material = input.material;
    output.color = input.color;
    output.uv1 = input.uv1;
    return output;
}

// Octahedral coordinates of unit `normal` (as `gpu_data::octahedral`, which
// `octahedral_normal` unfolds).
fn octahedral_coordinates(normal: vec3<f32>) -> vec2<f32> {
    let n = normal / (abs(normal.x) + abs(normal.y) + abs(normal.z));
    let sign = select(vec2<f32>(-1.0), vec2<f32>(1.0), n.xy >= vec2<f32>(0.0));
    return select(n.xy, (vec2<f32>(1.0) - abs(n.yx)) * sign, n.z < 0.0);
}

// The surface target's texel: the shading normal, octahedral in [0, 1], and
// the roughness.
fn surface_texel(shaded: Shaded) -> vec4<f32> {
    return vec4<f32>(octahedral_coordinates(shaded.normal) * 0.5 + 0.5, shaded.roughness, 1.0);
}

fn scene_targets(shaded: Shaded) -> SceneTargets {
    var targets: SceneTargets;
    targets.color = vec4<f32>(shaded.color, shaded.ambient);
    targets.glow = vec2<f32>(shaded.glow, shaded.reflect);
    targets.surface = surface_texel(shaded);
    return targets;
}

// `clip_position` is the fragment's, `previous` the same point's previous
// clip position: x, y and w.
fn temporal_targets(shaded: Shaded, clip_position: vec4<f32>, previous: vec3<f32>)
        -> TemporalTargets {
    var targets: TemporalTargets;
    targets.color = vec4<f32>(shaded.color, shaded.ambient);
    targets.glow = vec2<f32>(shaded.glow, shaded.reflect);
    targets.surface = surface_texel(shaded);
    // This pixel center without the jitter, and the same point last frame.
    let current = vec2<f32>(clip_position.x * camera.jitter.z - 1.0,
        1.0 - clip_position.y * camera.jitter.w) - camera.jitter.xy;
    targets.motion = (previous.xy / previous.z - current) * vec2<f32>(0.5, -0.5);
    // A point behind the previous camera has no position there: send its
    // history lookup out of the viewport.
    if !(previous.z > 0.0) {
        targets.motion = vec2<f32>(2.0);
    }
    return targets;
}

// The share of light glass lets through (decision 038); 0 for materials
// without transmission.
fn transmitted_share() -> f32 {
    if TRANSMISSION {
        return material_maps.volume.x;
    }
    return 0.0;
}

// A transparent surface's point: glass diffuses only the light it does not
// let through.
fn transparent_point(input: Fragment, front: bool) -> SurfacePoint {
    var point = surface_point(input.color, input.normal, input.world, input.uv, input.uv1,
        input.surface, front);
    point.diffuse_color *= 1.0 - transmitted_share();
    return point;
}

// Directions of the blur taps of rough glass, alternating full and half
// radius.
const GLASS_TAPS = array<vec2<f32>, 8>(
    vec2<f32>(1.0, 0.0), vec2<f32>(0.3535534, 0.3535534),
    vec2<f32>(0.0, 1.0), vec2<f32>(-0.3535534, 0.3535534),
    vec2<f32>(-1.0, 0.0), vec2<f32>(-0.3535534, -0.3535534),
    vec2<f32>(0.0, -1.0), vec2<f32>(0.3535534, -0.3535534),
);

// Weighted blending of a transparent surface (decision 026). Glass (decision
// 038) with refraction covers what is behind it with the opaque scene seen
// through it: along the ray its index of refraction bends through its
// thickness, blurred by its roughness and tinted by its color. Without
// refraction its transmitted share lets what is behind show through.
fn transparent_targets(shaded: Shaded, point: SurfacePoint, input: Fragment)
        -> TransparentTargets {
    let distance = length(camera.position.xyz - input.world);
    let share = transmitted_share();
    if share <= 0.0 {
        return weighted(shaded.color, shaded.opacity, distance);
    }
    // Light glass reflects instead of letting through.
    let reflectance = saturate(luminance(environment_brdf(point.f0, point.roughness,
        point.n_v)));
    let through = share * (1.0 - reflectance);
    if lighting.refraction.x <= 0.0 {
        let coverage = max(shaded.opacity * (1.0 - through), 1.0e-3);
        return weighted(min(shaded.color / coverage, vec3<f32>(1.0)), coverage, distance);
    }
    let bent = refract(-point.view, point.normal, 1.0 / material_maps.volume.y);
    let exit = point.world + bent * (material_maps.volume.z * lighting.refraction.x);
    let clip = camera.view_projection * vec4<f32>(exit, 1.0);
    let size = vec2<f32>(textureDimensions(refraction_source));
    // A ray leaving behind the eye shows what is straight behind the pixel.
    var uv = input.pixel / size;
    if clip.w > 1.0e-6 {
        uv = vec2<f32>(clip.x, -clip.y) / clip.w * 0.5 + 0.5;
    }
    var behind = textureSampleLevel(refraction_source, refraction_sampler, uv, 0.0).rgb;
    let radius = point.roughness * lighting.refraction.y / size;
    if radius.x * size.x >= 0.5 {
        var sum = behind;
        for (var tap = 0u; tap < 8u; tap++) {
            sum += textureSampleLevel(refraction_source, refraction_sampler,
                uv + GLASS_TAPS[tap] * radius, 0.0).rgb;
        }
        behind = sum / 9.0;
    }
    return weighted(shaded.color + behind * point.albedo * through, shaded.opacity, distance);
}
