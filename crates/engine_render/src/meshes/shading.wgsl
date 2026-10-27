// Surface shading every mesh pass shares: material texels, normals, the
// sun, environment and local lights, fog and grading, and the targets
// each kind of pass writes.

// Wave normals of liquid surfaces: three directional waves
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
    // Roughness along the view (and its square, GGX's alpha), and how much
    // smoother than that the point is across it: water's, as its ellipse
    // of slopes projects into the view (`water_light`); zero elsewhere,
    // whose slopes spread alike every way.
    roughness: f32,
    alpha: f32,
    smoother_across: f32,
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
    // impostors, whose rebuilt surfaces look from outside
    // their own meshes' shadows.
    shadow_offset: vec3<f32>,
    // Whether the point is terrain's, which lies on the
    // ground of terrain's horizon maps.
    on_terrain: bool,
    // Light from surfaces around the point where they hide the sky, per
    // unit diffuse color: terrain's, zero elsewhere.
    bounce: vec3<f32>,
    // How the normal the sky lights the point around turns from its own,
    // toward the middle of the sky it sees: terrain's, zero elsewhere.
    sky_turn: vec3<f32>,
    // The share of the sun the point's own mesh hides: impostors',
    // zero elsewhere.
    sun_shade: f32,
    // How much of the point is leaf, needle or blade (`leafiness`): zero
    // but for masked foliage, its cards and ground cover.
    leaf: f32,
    // How much of the point lies inside a crown, which the sun's light
    // reaches into (scene.wgsl's `crown_light`): masked foliage's leaf
    // share and its cards'; zero elsewhere, blades too.
    crown: f32,
    // What lies behind the point's surface, which screen-space
    // reflections' rays passing behind it meet (`post.wgsl`'s walk): an
    // object's body (`BODY_SOLID`, zero, as every point but these takes);
    // a sheet a few centimeters thick (`BODY_THIN`: masked surfaces,
    // foliage and blades); or the ground or water, solid below their
    // surface (`BODY_GROUND`).
    body: u32,
}

// What lies behind a surface (`SurfacePoint`'s `body`), which the surface
// target's last word holds above its mirrored share's MIRRORED_BITS
// (`surface_texel`).
const BODY_SOLID: u32 = 0u;
const BODY_THIN: u32 = 1u;
const BODY_GROUND: u32 = 2u;
const MIRRORED_BITS: u32 = 14u;

// Shaded light of a surface: graded color (and its linear light before the
// grade, which reflection probes capture), the ambient share of its light
// (for ambient occlusion), its glow weight; for screen-space reflections its
// reflectivity, the share of its light its mirrored environment makes up,
// which they replace, its shading normal, its roughness along the view and
// across it and what lies behind it (`SurfacePoint`'s `body`); and its
// coverage.
struct Shaded {
    color: vec3<f32>,
    linear: vec3<f32>,
    ambient: f32,
    glow: f32,
    reflect: f32,
    mirrored: f32,
    normal: vec3<f32>,
    roughness: f32,
    across: f32,
    body: u32,
    opacity: f32,
}

// The fragment being shaded: its vertex stage's material word, whose top
// byte is its slot's origin key's (`origin_key`), its pixel and, for
// placed objects blending into the ground (`GROUNDING`), its slot.
// Fragments outside the mesh entry points keep zero.
var<private> fragment_plant: u32;
var<private> fragment_pixel: vec2<f32>;
var<private> fragment_slot: u32;

// How much of a surface of linear color `albedo` reads as leaf: green over
// its other channels; bark's browns and greys read none.
fn leafiness(albedo: vec3<f32>) -> f32 {
    return saturate(4.0 * (albedo.g - max(albedo.r, albedo.b)) / max(albedo.g, 1.0e-4));
}

// The tint the plant of word `plant` gives `leaf` of a surface,
// `foliage.y` of the way (0 off): its hash picks a place along a gradient
// from drier and yellower to deeper and bluer green.
fn plant_tint(plant: u32, leaf: f32) -> vec3<f32> {
    let strength = lighting.foliage.y * leaf;
    if strength <= 0.0 {
        return vec3<f32>(1.0);
    }
    let along = f32(plant >> 24u) / 255.0;
    let gradient = mix(vec3<f32>(1.16, 1.05, 0.64), vec3<f32>(0.80, 0.97, 1.08), along);
    return mix(vec3<f32>(1.0), gradient, strength);
}

// How far the sun's light bends toward a leaf's normal passing through it,
// and how tightly it gathers about the sun seen through it.
const TRANSMISSION_BEND: f32 = 0.3;
const TRANSMISSION_POWER: f32 = 4.0;

// Sunlight `point`'s leaf share passes toward the eye, per unit of the
// sun's color, `foliage.x` of it (0 off): strongest looking at the sun
// through the leaf, its direction bent a little by the leaf's normal
// (Barré-Brisebois and Bouchard's translucency), tinted by its color.
fn transmitted(point: SurfacePoint) -> vec3<f32> {
    let strength = lighting.foliage.x * point.leaf;
    if strength <= 0.0 {
        return vec3<f32>(0.0);
    }
    let bent = lighting.sun.xyz + point.normal * TRANSMISSION_BEND;
    let through = saturate(dot(point.view, -normalize(bent)));
    return point.diffuse_color * (strength * pow(through, TRANSMISSION_POWER));
}

// The share of ambient occlusion `point` keeps: leaves passing light take
// half of it off, as light reaches into crowns and turf.
fn foliage_occlusion(point: SurfacePoint) -> f32 {
    return 1.0 - 0.5 * saturate(lighting.foliage.x) * point.leaf;
}

// The coverage masked fragments keep pixels above: the material's `cutoff`,
// or on temporally antialiased frames with hashed cutoffs (`foliage.z`) a
// threshold turning across pixels and frames, so frames average soft
// partial coverage over alpha's falloff (Wyman and McGuire's hashed alpha
// testing).
fn alpha_cutoff(cutoff: f32) -> f32 {
    if lighting.foliage.z == 0.0 {
        return cutoff;
    }
    return max(fade_noise(fragment_pixel), 1.0 / 255.0);
}

// Whether a masked fragment of `opacity` leaves its pixel: under the
// material's cutoff once its texels' `coverage` scales it.
fn masked_out(opacity: f32, coverage: f32) -> bool {
    return opacity * coverage < alpha_cutoff(material_maps.params.z);
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

// Normal mapping without stored tangents: the tangent frame
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
    // World-mapped normal maps' normal, world space, and the variance of the
    // slopes their filtering averaged away (`hidden_slopes`); none for the
    // others.
    world_normal: vec3<f32>,
    slopes: f32,
    // The metallic-roughness map's alpha, heights where its maps keep them;
    // 1 without one.
    height: f32,
}

fn untextured() -> MaterialTexels {
    var texels: MaterialTexels;
    texels.factors = vec2<f32>(1.0);
    texels.occlusion = 1.0;
    texels.normal = vec3<f32>(0.0, 0.0, 1.0);
    texels.coverage = 1.0;
    texels.height = 1.0;
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
// derivatives, so callers sample in uniform control flow, their mips biased
// by `bias` (`material_bias` for frames, 0 for bakes).
fn material_texels(uv: vec2<f32>, uv1: vec2<f32>, bias: f32) -> MaterialTexels {
    var texels = untextured();
    // RGBA8UnormSrgb sampling decodes source RGB; alpha masks emission and,
    // with maps, is coverage.
    var base_uv = uv;
    if material_flag(VARIANT_MAPS, MATERIAL_MAPS) {
        base_uv = map_uv(0u, uv, uv1);
    }
    texels.base = textureSampleBias(base_color_texture, base_color_sampler, base_uv, bias);
    if ALPHA_MASK {
        // The level the bias samples.
        let detail = exp2(bias);
        texels.coverage = mask_coverage(dpdx(base_uv) * detail, dpdy(base_uv) * detail);
    }
    if material_flag(VARIANT_MAPS, MATERIAL_MAPS) {
        // glTF: green is roughness and blue metalness, red packed occlusion.
        var metallic_roughness = vec4<f32>(1.0);
        if material_flag(VARIANT_METALLIC_ROUGHNESS_MAP, METALLIC_ROUGHNESS_MAP) {
            metallic_roughness = textureSampleBias(metallic_roughness_texture,
                metallic_roughness_sampler, map_uv(1u, uv, uv1), bias);
        }
        texels.factors = metallic_roughness.gb;
        texels.height = metallic_roughness.a;
        if material_flag(VARIANT_PACKED_OCCLUSION, PACKED_OCCLUSION) {
            texels.occlusion = metallic_roughness.r;
        } else if material_flag(VARIANT_OCCLUSION_MAP, OCCLUSION_MAP) {
            texels.occlusion = textureSampleBias(occlusion_texture, occlusion_sampler,
                map_uv(3u, uv, uv1), bias).r;
        }
        texels.emissive = vec3<f32>(1.0);
        if material_flag(VARIANT_EMISSIVE_MAP, EMISSIVE_MAP) {
            texels.emissive = textureSampleBias(emissive_texture, emissive_sampler,
                map_uv(4u, uv, uv1), bias).rgb;
        }
    }
    if material_flag(VARIANT_NORMAL_MAP, NORMAL_MAP) {
        let normal_uv = map_uv(2u, uv, uv1);
        texels.normal = textureSampleBias(normal_texture, normal_sampler, normal_uv, bias).xyz
            * 2.0 - 1.0;
        texels.normal_dx = dpdx(normal_uv);
        texels.normal_dy = dpdy(normal_uv);
    }
    return texels;
}

// Coordinates on axis `axis`'s plane of world-mapped maps, terrain's
// among them: x sees (z, -y), y sees (x, z) and z sees (x, -y), so on the
// sides the maps' up is the world's.
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

// `map`'s mean texel: its mip chain's last level.
fn map_mean(map: texture_2d<f32>, map_sampler: sampler) -> vec4<f32> {
    return textureSampleLevel(map, map_sampler, vec2<f32>(0.5), f32(textureNumLevels(map) - 1u));
}

// Map `index` of `map` at `world` (changing by `dx` across and `dy` down a
// pixel) on axis `axis`'s plane, one repeat every `emissive.w` meters,
// through its transform, as filtering at the pixel reads it at every
// distance: through its mipmaps and anisotropy by the pixel's derivatives.
// Where the pixel spans a whole repeat on its shorter side, filtering reads
// the last level, its stored mean, which stands in without a filtered read.
fn world_map_texel(map: texture_2d<f32>, map_sampler: sampler, index: u32, world: vec3<f32>,
        dx: vec3<f32>, dy: vec3<f32>, axis: u32) -> vec4<f32> {
    let scale = 1.0 / max(material_maps.emissive.w, 1.0e-6);
    let uv = world_plane(world, axis) * scale;
    let across = world_plane(dx, axis) * scale;
    let down = world_plane(dy, axis) * scale;
    let grad_x = map_gradient(index, across, across);
    let grad_y = map_gradient(index, down, down);
    if min(dot(grad_x, grad_x), dot(grad_y, grad_y)) >= 1.0 { // filter: the map's last level, its mean, which filtering reads there
        return map_mean(map, map_sampler);
    }
    return textureSampleGrad(map, map_sampler, map_uv(index, uv, uv), grad_x, grad_y);
}

// The most a unit normal stored in eight bits a channel rounds shorter:
// half a step of 2/255 on each of its three channels, all along it.
const NORMAL_ROUNDING: f32 = 1.7320508 / 255.0;

// The variance of the slopes (meters a meter) a normal map's filtered texel
// `sampled` hides, its deflections scaled by `scale`: averaging normals
// whose slopes spread by a variance σ² leaves their mean shorter than unit
// by about σ²/2 of its length (Toksvig 2005), past what rounding its texels
// to eight bits a channel shortens it.
fn hidden_slopes(sampled: vec3<f32>, scale: f32) -> f32 {
    let mean = max(length(sampled), 1.0e-4);
    return 2.0 * max(1.0 - mean - NORMAL_ROUNDING, 0.0) / mean * scale * scale;
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

// The texels of a world-mapped material at `world` on a
// surface of `normal`, changing by `dx` across and `dy` down a pixel: each
// map blended over the projections the surface faces as filtering at the
// pixel reads it (`world_map_texel`), its normal map's normal in world
// space and the variance of the slopes its filtering averaged away.
fn world_texels(world: vec3<f32>, normal: vec3<f32>, dx: vec3<f32>, dy: vec3<f32>)
        -> MaterialTexels {
    var texels = untextured();
    let shares = world_shares(normal);
    var base = vec4<f32>(0.0);
    var metallic_roughness = vec4<f32>(0.0);
    var occlusion = 0.0;
    var emissive = vec3<f32>(0.0);
    var mapped = vec3<f32>(0.0);
    var hidden = 0.0;
    for (var axis = 0u; axis < 3u; axis++) {
        let share = shares[axis];
        if share <= 0.0 {
            continue;
        }
        base += share * world_map_texel(base_color_texture, base_color_sampler, 0u, world, dx, dy,
            axis);
        if material_flag(VARIANT_METALLIC_ROUGHNESS_MAP, METALLIC_ROUGHNESS_MAP) {
            metallic_roughness += share * world_map_texel(metallic_roughness_texture,
                metallic_roughness_sampler, 1u, world, dx, dy, axis);
        }
        if material_flag(VARIANT_OCCLUSION_MAP, OCCLUSION_MAP) {
            occlusion += share * world_map_texel(occlusion_texture, occlusion_sampler, 3u, world,
                dx, dy, axis).r;
        }
        if material_flag(VARIANT_EMISSIVE_MAP, EMISSIVE_MAP) {
            emissive += share * world_map_texel(emissive_texture, emissive_sampler, 4u, world, dx,
                dy, axis).rgb;
        }
        if material_flag(VARIANT_NORMAL_MAP, NORMAL_MAP) {
            let sampled = world_map_texel(normal_texture, normal_sampler, 2u, world, dx, dy, axis)
                .xyz * 2.0 - 1.0;
            mapped += share * world_plane_normal(sampled, axis, normal, material_maps.params.x);
            hidden += share * hidden_slopes(sampled, material_maps.params.x);
        }
    }
    texels.base = base;
    if material_flag(VARIANT_METALLIC_ROUGHNESS_MAP, METALLIC_ROUGHNESS_MAP) {
        texels.factors = metallic_roughness.gb;
        texels.height = metallic_roughness.a;
    }
    if material_flag(VARIANT_PACKED_OCCLUSION, PACKED_OCCLUSION) {
        texels.occlusion = metallic_roughness.r;
    } else if material_flag(VARIANT_OCCLUSION_MAP, OCCLUSION_MAP) {
        texels.occlusion = occlusion;
    }
    texels.emissive = vec3<f32>(1.0);
    if material_flag(VARIANT_EMISSIVE_MAP, EMISSIVE_MAP) {
        texels.emissive = emissive;
    }
    if material_flag(VARIANT_NORMAL_MAP, NORMAL_MAP) {
        texels.world_normal = mapped;
        texels.slopes = hidden;
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
// their neighbors may not share.
fn material_texels_grad(uv: vec2<f32>, uv1: vec2<f32>, gradients: UvGradients)
        -> MaterialTexels {
    var texels = untextured();
    var base_uv = uv;
    var base_dx = gradients.dx;
    var base_dy = gradients.dy;
    if material_flag(VARIANT_MAPS, MATERIAL_MAPS) {
        base_uv = map_uv(0u, uv, uv1);
        base_dx = map_gradient(0u, gradients.dx, gradients.dx1);
        base_dy = map_gradient(0u, gradients.dy, gradients.dy1);
    }
    texels.base = textureSampleGrad(base_color_texture, base_color_sampler, base_uv, base_dx,
        base_dy);
    if ALPHA_MASK {
        texels.coverage = mask_coverage(base_dx, base_dy);
    }
    if material_flag(VARIANT_MAPS, MATERIAL_MAPS) {
        var metallic_roughness = vec4<f32>(1.0);
        if material_flag(VARIANT_METALLIC_ROUGHNESS_MAP, METALLIC_ROUGHNESS_MAP) {
            metallic_roughness = textureSampleGrad(metallic_roughness_texture,
                metallic_roughness_sampler, map_uv(1u, uv, uv1),
                map_gradient(1u, gradients.dx, gradients.dx1),
                map_gradient(1u, gradients.dy, gradients.dy1));
        }
        texels.factors = metallic_roughness.gb;
        texels.height = metallic_roughness.a;
        if material_flag(VARIANT_PACKED_OCCLUSION, PACKED_OCCLUSION) {
            texels.occlusion = metallic_roughness.r;
        } else if material_flag(VARIANT_OCCLUSION_MAP, OCCLUSION_MAP) {
            texels.occlusion = textureSampleGrad(occlusion_texture, occlusion_sampler,
                map_uv(3u, uv, uv1), map_gradient(3u, gradients.dx, gradients.dx1),
                map_gradient(3u, gradients.dy, gradients.dy1)).r;
        }
        texels.emissive = vec3<f32>(1.0);
        if material_flag(VARIANT_EMISSIVE_MAP, EMISSIVE_MAP) {
            texels.emissive = textureSampleGrad(emissive_texture, emissive_sampler,
                map_uv(4u, uv, uv1), map_gradient(4u, gradients.dx, gradients.dx1),
                map_gradient(4u, gradients.dy, gradients.dy1)).rgb;
        }
    }
    if material_flag(VARIANT_NORMAL_MAP, NORMAL_MAP) {
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
    if material_flag(VARIANT_NORMAL_MAP, NORMAL_MAP) {
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
    if material_flag(VARIANT_DOUBLE_SIDED, DOUBLE_SIDED) && !front {
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
    point.opacity = surface_opacity(color, texels);
    point.emission = surface.z * 16.0 * texels.base.a;
    point.emissive = vec3<f32>(0.0);
    point.occlusion = 1.0;
    var roughness = surface.x;
    var metalness = surface.y;
    if material_flag(VARIANT_MAPS, MATERIAL_MAPS) {
        roughness *= texels.factors.x;
        metalness *= texels.factors.y;
        point.occlusion = 1.0 + material_maps.params.y * (texels.occlusion - 1.0);
        point.emissive = material_maps.emissive.rgb * texels.emissive;
    }
    if texels.slopes > 0.0 {
        // The slopes world-mapped normal maps' filtering averaged away
        // roughen the surface instead, their variance widening its
        // highlights as a microfacet distribution's does.
        let alpha = roughness * roughness;
        roughness = sqrt(sqrt(min(alpha * alpha + texels.slopes, 1.0)));
    }
    point.roughness = max(roughness, 0.03);
    point.alpha = point.roughness * point.roughness;
    if ALPHA_MASK || material_flag(VARIANT_FOLIAGE, FOLIAGE) {
        point.leaf = leafiness(point.albedo);
        point.crown = point.leaf;
        point.albedo *= plant_tint(fragment_plant, point.leaf);
        // Cutouts and leaves are sheets.
        point.body = BODY_THIN;
    }
    point.f0 = mix(vec3<f32>(0.04), point.albedo, metalness);
    point.diffuse_color = point.albedo * (1.0 - metalness);
    point.geometric = geometric_normal(surface_normal, front);
    point.normal = shading_normal(point.geometric, world, world_dx, world_dy, texels.normal,
        texels.normal_dx, texels.normal_dy, surface.w);
    if material_flag(VARIANT_WORLD_MAPPING, WORLD_MAPPING) && material_flag(VARIANT_NORMAL_MAP, NORMAL_MAP)
        && dot(texels.world_normal, texels.world_normal) > 0.0 {
        // World-mapped normal maps shade their own normal,
        // turned to the side the surface shows.
        var mapped = stable_normalize(texels.world_normal);
        if dot(mapped, point.geometric) < 0.0 {
            mapped = -mapped;
        }
        point.normal = mapped;
    }
    if material_flag(VARIANT_TOP_LAYER, TOP_LAYER) {
        // The top layer settles on faces turned up.
        let breakup = variety_noise(world.xz * 2.3 + vec2<f32>(world.y * 0.7, 3.1)) * 0.6
            + variety_noise(world.xz * 0.55 - vec2<f32>(world.y * 0.4, 7.7)) * 0.4;
        let amount = top_layer_amount(point.normal.y, breakup, texels);
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

// How much of the top layer settles on a face turned up by
// `up` (its normal's y): broken by noise `breakup` (0 to 1) and gathering
// in the low parts of the material's height map (its metallic-roughness
// map's alpha) and the darker hollows of its base color.
fn top_layer_amount(up: f32, breakup: f32, texels: MaterialTexels) -> f32 {
    var hollow = 1.0 - dot(texels.base.rgb, vec3<f32>(0.299, 0.587, 0.114));
    if material_flag(VARIANT_METALLIC_ROUGHNESS_MAP, METALLIC_ROUGHNESS_MAP) {
        hollow = mix(hollow, 1.0 - texels.height, 0.6);
    }
    let lift = up + 0.5 * (breakup - 0.5) + 0.4 * (hollow - 0.5);
    let reach = material_maps.top.w;
    return smoothstep(1.0 - reach, 1.2 - reach, lift) * step(1.0e-4, reach);
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

// The texels of a surface point at `world`, which changes by `world_dx`
// across and `world_dy` down a pixel, of interpolated normal
// `surface_normal` and texture coordinates `uv` and `uv1`: world-mapped
// materials' over the projections it faces, the others' at its
// coordinates, as finely as the display shows them.
fn surface_texels(surface_normal: vec3<f32>, world: vec3<f32>, world_dx: vec3<f32>,
        world_dy: vec3<f32>, uv: vec2<f32>, uv1: vec2<f32>) -> MaterialTexels {
    var texels: MaterialTexels;
    if material_flag(VARIANT_WORLD_MAPPING, WORLD_MAPPING) {
        let detail = material_detail();
        texels = world_texels(world, stable_normalize(surface_normal), world_dx * detail,
            world_dy * detail);
    } else {
        texels = material_texels(uv, uv1, material_bias());
    }
    return texels;
}

// The coverage of a surface of linear color and opacity `color` sampling
// `texels`: its opacity, times its base color texel's alpha with maps.
fn surface_opacity(color: vec4<f32>, texels: MaterialTexels) -> f32 {
    if material_flag(VARIANT_MAPS, MATERIAL_MAPS) {
        return color.a * texels.base.a;
    }
    return color.a;
}

// Sampling uses implicit derivatives, so callers build the point in uniform
// control flow; what follows may not be. `color` is linear color and
// opacity; `front` whether the surface faces the camera. Masked surfaces
// leave the pixels their alpha test fails; their colors after the depth
// pre-pass (`PRETESTED`), which draw only where its test passed, test
// nothing.
fn surface_point(color: vec4<f32>, surface_normal: vec3<f32>, world: vec3<f32>,
        uv: vec2<f32>, uv1: vec2<f32>, packed: u32, front: bool) -> SurfacePoint {
    let world_dx = dpdx(world);
    let world_dy = dpdy(world);
    let texels = surface_texels(surface_normal, world, world_dx, world_dy, uv, uv1);
    var point = textured_point(color, surface_normal, world, world_dx, world_dy, texels, packed,
        front);
    if GROUNDING {
        point = grounded(point);
    }
    let result = widened(point, dpdx(point.normal), dpdy(point.normal));
    if ALPHA_MASK && !PRETESTED && masked_out(result.opacity, texels.coverage) {
        discard;
    }
    return result;
}

// `point` of a placed object blending into the ground (`GROUNDING`),
// from its slot's ground and pose: below its reach above the plane through
// the ground under its sunk base, its color, roughness and shading normal
// turn toward the ground's there, all the way where it enters the ground,
// so no seam shows where the two meet. Objects whose ground's color is not
// read yet keep theirs.
fn grounded(point: SurfacePoint) -> SurfacePoint {
    let ground = grounds[fragment_slot];
    let reach = bitcast<f32>(ground.z);
    if !(reach > 0.0) || ground.w == 0u {
        return point;
    }
    let pose = poses[fragment_slot];
    let origin = vec3<f32>(pose.row_0.w, pose.row_1.w, pose.row_2.w);
    let tilt = unpack2x16snorm(ground.x);
    let up = sqrt(max(1.0 - dot(tilt, tilt), 0.01));
    let above = point.world.y - origin.y - bitcast<f32>(ground.y)
        + dot(tilt, point.world.xz - origin.xz) / up;
    if above >= reach {
        return point;
    }
    let amount = 1.0 - smoothstep(0.0, reach, above);
    // The ground's color as its linear channels' square roots, and its
    // roughness.
    let texel = unpack4x8unorm(ground.w);
    let albedo = texel.rgb * texel.rgb;
    var result = point;
    result.albedo = mix(point.albedo, albedo, amount);
    result.diffuse_color = mix(point.diffuse_color, albedo, amount);
    result.f0 = mix(point.f0, vec3<f32>(0.04), amount);
    result.roughness = mix(point.roughness, max(texel.a, 0.03), amount);
    result.alpha = result.roughness * result.roughness;
    result.normal = stable_normalize(mix(point.normal, vec3<f32>(tilt.x, up, tilt.y), amount));
    result.n_v = max(dot(result.normal, result.view), 1.0e-4);
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

// What the environment gives a surface point: the light it reflects toward
// the eye, and the shares of the sky's light (x) and the sun's (y) reaching
// it.
struct EnvironmentLight {
    reflected: vec3<f32>,
    interior: vec2<f32>,
}

// The lobe a surface reflects its environment over: its dominant direction
// (`dominant_direction`), which the maps are read along; the level of the
// environment maps its roughness blurs them to; and how wide it is against
// the texels of the maps' first level, in levels (`lobe_texels`).
struct EnvironmentLobe {
    direction: vec3<f32>,
    level: f32,
    texels: f32,
}

// The lobe `point` reflects its environment over.
fn environment_lobe(point: SurfacePoint) -> EnvironmentLobe {
    let levels = textureNumLevels(environment_maps);
    return EnvironmentLobe(dominant_direction(point), point.roughness * f32(levels - 1u),
        lobe_texels(point.alpha));
}

// How wide a lobe of reflection of GGX's `alpha` is against the texels of
// the environment maps' first level, in levels, as their blur measures its
// samples against the texels (`environment.wgsl`'s `prefiltered`): half the
// base-2 logarithm of its solid angle over a texel's, negative while it is
// narrower than one. Its solid angle is the reciprocal of its peak density
// with the view along the normal, as the maps assume (GGX's D over 4 there,
// 1 / (4 pi alpha^2)); a texel's is the sphere's over the cube's texels.
fn lobe_texels(alpha: f32) -> f32 {
    let size = f32(textureDimensions(environment_maps).x);
    let lobe = 4.0 * PI * alpha * alpha;
    let texel = 4.0 * PI / (6.0 * size * size);
    return 0.5 * log2(max(lobe / texel, 1.0e-12));
}

// The environment around `world` reflected over `lobe`, from one walk of the
// reflection probes whose boxes hold it, smallest first, each fading in over
// its blend distance inside its faces: those probes' maps at the lobe's
// level, each weighted by what the smaller ones leave, and `reflected_sky`,
// the sky's light the lobe reflects, for what they all leave. Inside
// interiors' boxes only their shares of the sky and sun reach; all of both
// elsewhere.
fn environment_light(lobe: EnvironmentLobe, world: vec3<f32>, reflected_sky: vec3<f32>)
        -> EnvironmentLight {
    var light = vec3<f32>(0.0);
    var covered = 0.0;
    var shares = vec2<f32>(1.0);
    let header = probe_words[0];
    if PROBES && header.w != 0u {
        let grid = probe_words[1];
        let cell = vec2<i32>(floor((world.xz - bitcast<vec2<f32>>(header.xy))
            / bitcast<f32>(header.z)));
        if all(cell >= vec2<i32>(0)) && all(vec2<u32>(cell) < grid.xy) {
            let list = probe_words[grid.z + u32(cell.y) * grid.x + u32(cell.x)];
            // Interiors' shares take every probe of the list; reflections
            // end once covered.
            let interiors = list.z != 0u;
            for (var n = 0u; n < list.y && (covered < 0.999 || interiors); n++) {
                let entry = grid.w + (list.x + n) * PROBE_ENTRY_WORDS;
                let interior = probe_words[entry + 2u];
                // Interiors not captured yet reflect nothing.
                let reflects = interior.w != 0u && covered < 0.999;
                let shades = interiors && interior.z != 0u;
                if !reflects && !shades {
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
                let fade = select(1.0, saturate(depth / blend), blend > 0.0);
                if reflects {
                    let weight = fade * (1.0 - covered);
                    let along = box_projected(world, lobe.direction, center, extent);
                    light += weight * textureSampleLevel(environment_maps, environment_sampler,
                        along, second.w, lobe.level).rgb;
                    covered += weight;
                }
                if shades {
                    shares = min(shares, mix(vec2<f32>(1.0), bitcast<vec2<f32>>(interior.xy),
                        fade));
                }
            }
        }
    }
    if covered < 0.999 {
        light += (1.0 - covered) * reflected_sky;
    }
    return EnvironmentLight(light, shares);
}

// The sky's light `lobe` reflects as the sky's environment map holds it, at
// the lobe's level.
fn sky_map_light(lobe: EnvironmentLobe) -> vec3<f32> {
    return textureSampleLevel(environment_maps, environment_sampler, lobe.direction, 0,
        lobe.level).rgb;
}

// The sky's light `lobe` reflects from `world`: exact, as a mirror there
// shows it (`mirrored_along`), while the lobe is no wider than a texel of
// the sky's environment map, which could only coarsen it; the map, at the
// level its roughness blurs it to, takes over as the lobe widens past its
// texels, all of it once the lobe spans two, a level up. So calm water takes
// the exact sky alone, and every roughness keeps the sky's mean.
fn mirrored_sky(lobe: EnvironmentLobe, world: vec3<f32>) -> vec3<f32> {
    let share = saturate(lobe.texels);
    if share >= 1.0 {
        return sky_map_light(lobe);
    }
    let exact = mirrored_along(world, lobe.direction);
    if share <= 0.0 {
        return exact;
    }
    return mix(exact, sky_map_light(lobe), share);
}

// Whether `world` lies inside an interior probe's box. Lights keep to
// the boxes themselves, not to the fade of the sky's and sun's shares over
// their blend distances, which may reach well into a cave's own walls.
fn in_interior(world: vec3<f32>) -> bool {
    let header = probe_words[0];
    if !PROBES || header.w == 0u {
        return false;
    }
    let grid = probe_words[1];
    let cell = vec2<i32>(floor((world.xz - bitcast<vec2<f32>>(header.xy))
        / bitcast<f32>(header.z)));
    if any(cell < vec2<i32>(0)) || any(vec2<u32>(cell) >= grid.xy) {
        return false;
    }
    let list = probe_words[grid.z + u32(cell.y) * grid.x + u32(cell.x)];
    let count = select(0u, list.y, list.z != 0u);
    for (var n = 0u; n < count; n++) {
        let entry = grid.w + (list.x + n) * PROBE_ENTRY_WORDS;
        if probe_words[entry + 2u].z == 0u {
            continue;
        }
        let center = bitcast<vec3<f32>>(probe_words[entry].xyz);
        let extent = bitcast<vec3<f32>>(probe_words[entry + 1u].xyz);
        if all(abs(world - center) < extent) {
            return true;
        }
    }
    return false;
}

// Whether the sun's highlight spreads over its disk (`lighting.highlights.y`,
// the sine of its angular radius); otherwise the sun is a point.
fn sized_highlights() -> bool {
    return lighting.highlights.y != 0.0;
}

// The direction a surface's lobe of reflection leans to, along which its
// highlights and its environment are read: the view mirrored, toward the
// normal as the surface roughens (Lagarde and de Rousiers).
fn dominant_direction(point: SurfacePoint) -> vec3<f32> {
    let smoothness = 1.0 - point.roughness;
    let lean = smoothness * (sqrt(smoothness) + point.roughness);
    return normalize(mix(point.normal, reflect(-point.view, point.normal), lean));
}

// The share of a lobe of `alpha` a light keeps when it widens the lobe by
// half its angular size, `sine` the sine of its angular radius: the lobe's
// peak over the widened lobe's, squared over a disk (Karis).
fn disk_energy(alpha: f32, sine: f32) -> f32 {
    let kept = alpha / saturate(alpha + 0.5 * sine);
    return kept * kept;
}

// The share of a local light's highlight light `light` the cap keeps
// (`lighting.highlights.z`, an exposed brightness; 0 caps none): all of it
// up to half the cap, then less and less, so the brightest lamps'
// highlights ease toward the cap in their own color instead of blooming
// white, and dim lamps' highlights keep all they had.
fn cap_share(light: vec3<f32>) -> f32 {
    let cap = lighting.highlights.z;
    let bright = luminance(light) * lighting.horizon.w;
    let knee = 0.5 * cap;
    if cap <= 0.0 || bright <= knee {
        return 1.0;
    }
    let over = bright - knee;
    return (knee + over / (1.0 + over / knee)) / bright;
}

// The sun's highlight at `point` over the sun's disk (Karis's representative
// point): lit from the direction within the disk nearest the lobe's
// dominant direction, the lobe widened by half the disk's angular size and
// renormalized. `n_l` is the sun's cosine at the point.
fn sun_highlight(point: SurfacePoint, sun: vec3<f32>, n_l: f32) -> vec3<f32> {
    let cosine = lighting.highlights.x;
    let sine = lighting.highlights.y;
    let direction = dominant_direction(point);
    let along = dot(direction, sun);
    let across = direction - sun * along;
    let off = length(across);
    var toward = direction;
    if along < cosine {
        toward = select(sun, sun * cosine + across * (sine / off), off > 1.0e-6);
    }
    let half_vector = normalize(point.view + toward);
    let n_h = max(dot(point.normal, half_vector), 0.0);
    let lobe = distribution(n_h, point.alpha) * visibility(point.n_v, n_l, point.alpha)
        * disk_energy(point.alpha, sine);
    return fresnel(point.f0, max(dot(point.view, half_vector), 0.0)) * (lobe * n_l);
}

// The share of the roughness past which surfaces reflect only the sky
// (`lighting.effects.z`) over which, below it, their screen-space
// reflections fade out, so no edge shows where their roughness across the
// view crosses it.
const REFLECTION_FADE: f32 = 0.2;

// Lit, fogged and graded color of `point` with `local` light added, the sky
// it mirrors exact where it is smooth (`mirrored_sky`).
fn finish(point: SurfacePoint, local: vec3<f32>) -> Shaded {
    let lobe = environment_lobe(point);
    return finish_reflecting(point, local, lobe, mirrored_sky(lobe, point.world));
}

// `finish` in terrain's passes, whose stages hold no more sampled textures
// and so none of the sky's tables or the clouds' domes
// (`bindings::CAMERA_TERRAIN`): they mirror the sky's environment map at
// every roughness.
fn finish_ground(point: SurfacePoint, local: vec3<f32>) -> Shaded {
    let lobe = environment_lobe(point);
    return finish_reflecting(point, local, lobe, sky_map_light(lobe));
}

// Lit, fogged and graded color of `point` with `local` light added, its
// environment reflected over `lobe`, `reflected_sky` the sky's light it
// reflects there.
fn finish_reflecting(point: SurfacePoint, local: vec3<f32>, lobe: EnvironmentLobe,
        reflected_sky: vec3<f32>) -> Shaded {
    let normal = point.normal;
    let view = point.view;
    let sun = lighting.sun.xyz;
    let n_l = dot(normal, sun);
    let specular = lighting.sun_color.w;
    // Wrapped diffuse softens the terminator of flat low-poly faces. Where
    // a face itself turns from the sun, its mesh's own shadow darkens it
    // whatever its normal says, so the wrap fades out over the faces about
    // the sun's edge instead of stepping down at every one of them.
    // Impostors' cards are not their surfaces' faces, nor are terrain's
    // triangles, its levels of detail, its ground's: its normal maps and
    // horizon maps shade it.
    var facing = 1.0;
    let face = cross(point.world_dx, point.world_dy); // filter: none, the face's own plane across the pixel, a direction
    if dot(face, face) > 1.0e-20 && all(point.shadow_offset == vec3<f32>(0.0))
        && !point.on_terrain {
        let toward = select(-1.0, 1.0, dot(face, point.geometric) >= 0.0);
        facing = smoothstep(-0.05, 0.1, dot(normalize(face) * toward, sun));
    }
    let lambert = max(n_l, 0.0);
    let wrapped = lambert + (max((n_l + 0.25) / 1.25, 0.0) - lambert) * facing;
    var highlight = vec3<f32>(0.0);
    if n_l > 0.0 {
        if sized_highlights() {
            highlight = sun_highlight(point, sun, n_l);
        } else {
            let half_vector = normalize(view + sun);
            let n_h = max(dot(normal, half_vector), 0.0);
            highlight = fresnel(point.f0, max(dot(view, half_vector), 0.0))
                * (distribution(n_h, point.alpha) * visibility(point.n_v, n_l, point.alpha) * n_l);
        }
    }
    // The environment mirrored in the surface, blurred by its roughness,
    // weighted by the environment BRDF.
    let environment = environment_light(lobe, point.world, reflected_sky);
    let brdf = environment_brdf(point.f0, point.roughness, point.n_v);
    let reflection = environment.reflected * brdf * specular;
    // Rim light outlines walls against the sky; level surfaces seen at
    // grazing angles (distant ground and roofs) take none.
    let grazing = 1.0 - max(dot(point.geometric, view), 0.0);
    let rim = (grazing * grazing) * (grazing * grazing) * lighting.sun.w
        * (1.0 - abs(point.geometric.y));
    // Occlusion maps darken indirect light only (glTF), as do the covers of
    // placed objects over the point.
    let covered_sky = occupancy_visibility(point.world);
    // Inside interiors only their shares of the sky and sun
    // reach; what they reflect is their probe's own capture.
    let interior = environment.interior;
    // The sky's light around the normal turned toward the sky the point
    // sees. What stands on terrains takes instead, where their probes
    // relight, the sky's own light by the share of it the ground leaves
    // along its normal and, in place of the sky's ground color, the light
    // the ground around sends it, which occlusion maps darken as indirect
    // light (terrain's own surfaces take theirs from their sky maps).
    let skyward = normalize(normal + point.sky_turn);
    var ambient = sky_average(skyward);
    var bounce = point.bounce;
    if !point.on_terrain {
        let around = bounce_light(point.world, normal, false);
        ambient = mix(ambient, sky_above(skyward) * around.sky, around.weight);
        bounce += around.light * (around.weight * point.occlusion);
    }
    let sky_light = (point.diffuse_color * (ambient * lighting.zenith.w
        + lighting.horizon.rgb * rim) * interior.x + reflection) * point.occlusion
        * covered_sky + point.diffuse_color * bounce * covered_sky * interior.x;
    // The sun's light before shadows: where it, the point's sun shade or its
    // interior's share is zero, so is what shadows leave, which then goes
    // unlooked up.
    let sunlit = (point.diffuse_color * wrapped + highlight * specular + transmitted(point))
        * lighting.sun_color.rgb;
    var visible = 1.0;
    if any(sunlit != vec3<f32>(0.0)) && point.sun_shade != 1.0 && interior.y != 0.0 {
        crown_light = point.crown;
        visible = sun_visibility(point.world + point.shadow_offset, point.geometric,
            point.world_dx, point.world_dy, point.distance, point.on_terrain);
        crown_light = 0.0;
    }
    let sun_light = sunlit * visible * (1.0 - point.sun_shade) * interior.y;
    let emitted = (point.albedo * point.emission + point.emissive) * lighting.effects.x;
    let lit = sky_light + sun_light + emitted + local;
    let through = atmospheric(lit, camera.position.xyz, point.world, -view);
    let fogged = through.rgb;
    let covered = through.a;
    var shaded: Shaded;
    shaded.opacity = point.opacity;
    shaded.normal = normal;
    shaded.roughness = point.roughness;
    shaded.across = point.roughness - point.smoother_across;
    shaded.body = point.body;
    shaded.linear = fogged;
    shaded.color = grade(fogged);
    shaded.ambient = saturate(luminance(sky_light) / max(luminance(lit), 1.0e-6)) * (1.0 - covered)
        * foliage_occlusion(point);
    shaded.glow = glow_weight(fogged);
    // For screen-space reflections: the environment BRDF's share, by which
    // the scene they show is mirrored, and the share of the light the
    // mirrored environment makes up, which that scene replaces; both fading
    // out over the last REFLECTION_FADE of the roughness limit by the
    // roughness across the view, the narrow way of a mirrored lobe, and
    // through fog.
    if lighting.effects.w != 0.0 {
        let smooth_enough = saturate((1.0 - shaded.across / lighting.effects.z)
            / REFLECTION_FADE);
        let faded = smooth_enough * (1.0 - covered);
        shaded.reflect = saturate(luminance(brdf) * specular) * faded;
        shaded.mirrored = saturate(luminance(reflection * (point.occlusion * covered_sky))
            / max(luminance(lit), 1.0e-6)) * faded;
    }
    return shaded;
}

// What a fragment shades: linear color and opacity, world normal and
// position, both texture coordinate sets, the packed surface, the pixel and
// its slot's origin key (`origin_key`, shifted down).
struct Fragment {
    color: vec4<f32>,
    normal: vec3<f32>,
    world: vec3<f32>,
    uv: vec2<f32>,
    uv1: vec2<f32>,
    surface: u32,
    pixel: vec2<f32>,
    origin: u32,
}

// Linear color and opacity of a fragment's material word; placed objects
// blending into the ground (`placed_material`) note their slot.
fn fragment_color(material: vec4<u32>) -> vec4<f32> {
    if GROUNDING {
        fragment_slot = material.y;
        return unpack4x8unorm(material.x);
    }
    return output_color(material);
}

// Meshes without a second stream read their first coordinates as second.
fn fragment(input: VertexOutput) -> Fragment {
    // Placed objects fading leave the pixels their fade
    // hides.
    if FADE_LIST && fade_hidden(input.material.w, input.clip_position.xy) {
        discard;
    }
    fragment_plant = input.material.w;
    fragment_pixel = input.clip_position.xy;
    var values: Fragment;
    values.color = fragment_color(input.material);
    values.normal = input.normal_u.xyz;
    values.world = input.world_v.xyz;
    values.uv = vec2<f32>(input.normal_u.w, input.world_v.w);
    values.uv1 = values.uv;
    values.surface = input.material.z;
    values.pixel = input.clip_position.xy;
    values.origin = input.material.w >> 16u;
    return values;
}

// Vertex colors multiply the color and opacity (glTF `COLOR_0`).
fn extended_fragment(input: ExtendedOutput) -> Fragment {
    if FADE_LIST && fade_hidden(input.material.w, input.clip_position.xy) {
        discard;
    }
    fragment_plant = input.material.w;
    fragment_pixel = input.clip_position.xy;
    var values: Fragment;
    values.color = fragment_color(input.material) * input.color;
    values.normal = input.normal_u.xyz;
    values.world = input.world_v.xyz;
    values.uv = vec2<f32>(input.normal_u.w, input.world_v.w);
    values.uv1 = input.uv1;
    values.surface = input.material.z;
    values.pixel = input.clip_position.xy;
    values.origin = input.material.w >> 16u;
    return values;
}

// Masked surfaces' depth pre-pass in the opaque passes: a fragment's fade
// (`fragment` takes it) and alpha test alone, its texels and coverage as
// `surface_point` reads them, so the depth it leaves is the one its color
// then draws at (`PRETESTED`). Of its texels only the base color's alpha is
// read.
fn masked_depth(input: Fragment) {
    let texels = surface_texels(input.normal, input.world, dpdx(input.world),
        dpdy(input.world), input.uv, input.uv1);
    if masked_out(surface_opacity(input.color, texels), texels.coverage) {
        discard;
    }
}

// How much of their color the windows of unlit rooms keep: the rooms behind
// the glass are dark.
const UNLIT_ROOM: f32 = 0.12;
// The coolest and the warmest light of lit rooms.
const COOL_ROOM: vec3<f32> = vec3<f32>(0.78, 0.88, 1.0);
const WARM_ROOM: vec3<f32> = vec3<f32>(1.0, 0.72, 0.42);

// Light of the rooms of windows switching on through the evening
// (`lighting.windows`): on a material divided into rooms (`layer.y`
// across its base color texture, square in its texels), each room of each
// slot (`input.origin`) lights once the sun sinks below its own elevation,
// spread around dusk; a share stays dark, and lit rooms glow brighter or
// dimmer, cool to sodium-warm. Only light the point's texels emit; where
// they emit by their alpha, unlit rooms' windows darken.
fn room_light(point: SurfacePoint, input: Fragment) -> SurfacePoint {
    let rooms = material_maps.layer.y;
    if !WINDOWS || lighting.windows.z <= 0.0 || rooms <= 0.0 {
        return point;
    }
    var base_uv = input.uv;
    if material_flag(VARIANT_MAPS, MATERIAL_MAPS) {
        base_uv = map_uv(0u, input.uv, input.uv1);
    }
    let size = vec2<f32>(textureDimensions(base_color_texture));
    let room = vec2<i32>(floor(base_uv * vec2<f32>(rooms, rooms * size.y / size.x)));
    let key = pcg(input.origin ^ pcg(bitcast<u32>(room.x) ^ pcg(bitcast<u32>(room.y))));
    let switched = lighting.windows.y + lighting.windows.z * (2.0 * unit(key) - 1.0);
    var lit = point;
    lit.emission = 0.0;
    lit.emissive = vec3<f32>(0.0);
    if lighting.windows.x < switched && unit(pcg(key + 1u)) >= lighting.windows.w {
        let tint = mix(COOL_ROOM, WARM_ROOM, unit(pcg(key + 2u)));
        let brightness = 0.6 + 0.8 * unit(pcg(key + 3u));
        lit.emissive = (point.albedo * point.emission + point.emissive) * tint
            * (brightness / luminance(tint));
    } else {
        // The share of the point that is window: its texels' alpha.
        let window = point.emission / max(unpack4x8unorm(input.surface).z * 16.0, 1e-4);
        let kept = mix(1.0, UNLIT_ROOM, saturate(window));
        lit.albedo = point.albedo * kept;
        lit.diffuse_color = point.diffuse_color * kept;
    }
    return lit;
}

// Windows on the card of a far building whose material is divided into
// rooms (`IMPOSTOR_ROOMS`), its rooms finer than the card's texels: the
// share of rooms `room_light` lights at this sun's elevation lights the
// point's `window` share in the rooms' mean color and brightness, and the
// rest of that share darkens as unlit rooms' windows do, so far buildings
// switch on and go dark with the near ones.
fn far_rooms(point: SurfacePoint, window: f32) -> SurfacePoint {
    if !WINDOWS || lighting.windows.z <= 0.0 {
        return point;
    }
    let lit = saturate(0.5 - 0.5 * (lighting.windows.x - lighting.windows.y)
        / lighting.windows.z) * (1.0 - lighting.windows.w);
    let tint = mix(COOL_ROOM, WARM_ROOM, 0.5);
    var far = point;
    far.emission = 0.0;
    far.emissive = (point.albedo * point.emission + point.emissive) * tint
        * (lit / luminance(tint));
    let kept = mix(1.0, UNLIT_ROOM, window * (1.0 - lit));
    far.albedo = point.albedo * kept;
    far.diffuse_color = point.diffuse_color * kept;
    return far;
}

fn shade(input: Fragment, front: bool) -> Shaded {
    return finish(room_light(surface_point(input.color, input.normal, input.world, input.uv,
        input.uv1, input.surface, front), input), vec3<f32>(0.0));
}

// The share of a lobe of `alpha` a tube keeps when it widens the lobe along
// itself by half its angular length `angle` (Karis).
fn tube_energy(alpha: f32, angle: f32) -> f32 {
    return alpha / saturate(alpha + 0.5 * angle);
}

// Light `light` at `point` shining from its sphere and tube while `shaped`,
// and kept to its region while `kept`: interior lamps light only points
// inside interiors, exterior ones only points outside them (`inside` holds
// whether the point is, negative until looked up). Diffuse light comes from
// the tube's ends (Karis's line irradiance, a point's for spheres and
// points); the highlight from the point of the tube, then of the sphere
// around it, nearest the lobe's dominant direction, the lobe widened by
// half their angular sizes and renormalized (Karis) and capped; both
// windowed by the distance to the tube and faded by the cone.
fn shaped_light(point: SurfacePoint, light: Light, shaped: bool, kept: bool,
        inside: ptr<function, f32>) -> vec3<f32> {
    if kept && light.region != 0u {
        if *inside < 0.0 {
            *inside = select(0.0, 1.0, in_interior(point.world));
        }
        let in_box = *inside > 0.0;
        if (light.region == LIGHT_INTERIOR && !in_box)
            || (light.region == LIGHT_EXTERIOR && in_box) {
            return vec3<f32>(0.0);
        }
    }
    let radius = select(0.0, max(light.radius, lighting.highlights.w), shaped);
    let span = select(vec3<f32>(0.0), light.direction * light.length, shaped);
    let first = light.position - point.world;
    let last = first + span;
    let spanned = dot(span, span);
    // The range reaches from the tube's point nearest the surface.
    let nearest = first + span * saturate(-dot(first, span) / max(spanned, 1.0e-8));
    let d2 = dot(nearest, nearest);
    let r2 = light.range * light.range;
    if d2 >= r2 {
        return vec3<f32>(0.0);
    }
    let ratio = d2 / r2;
    let window = saturate(1.0 - ratio * ratio);
    let first_length = max(length(first), 1.0e-4);
    let last_length = max(length(last), 1.0e-4);
    let n_l = saturate(0.5 * (dot(point.normal, first) / first_length
        + dot(point.normal, last) / last_length));
    var falloff = window * window
        / max(0.5 * (first_length * last_length + dot(first, last)), 0.01);
    if light.cos_outer > -1.5 {
        let center = first * inverseSqrt(max(dot(first, first), 1.0e-8));
        falloff *= smoothstep(light.cos_outer, light.cos_inner, dot(-center, light.direction));
    }
    if falloff == 0.0 {
        return vec3<f32>(0.0);
    }
    let reflected = dominant_direction(point);
    var toward = first;
    if spanned > 1.0e-8 {
        let across = dot(reflected, span);
        toward = first + span * saturate((dot(first, reflected) * across - dot(first, span))
            / max(spanned - across * across, 1.0e-8));
    }
    if radius > 0.0 {
        let to_ray = reflected * dot(toward, reflected) - toward;
        toward += to_ray * saturate(radius / max(length(to_ray), 1.0e-6));
    }
    // Angular sizes as seen from the tube's middle.
    let distance = max(length(first + 0.5 * span), 1.0e-4);
    let l = normalize(toward);
    let n_l_highlight = dot(point.normal, l);
    var highlight = vec3<f32>(0.0);
    if n_l_highlight > 0.0 {
        let half_vector = normalize(point.view + l);
        let n_h = max(dot(point.normal, half_vector), 0.0);
        let energy = disk_energy(point.alpha, saturate(radius / distance))
            * tube_energy(point.alpha, saturate(sqrt(spanned) / distance));
        let lobe = distribution(n_h, point.alpha)
            * visibility(point.n_v, n_l_highlight, point.alpha) * energy;
        highlight = fresnel(point.f0, max(dot(point.view, half_vector), 0.0))
            * (lobe * n_l_highlight);
        highlight *= cap_share(highlight * (lighting.sun_color.w * falloff) * light.color);
    }
    return (point.diffuse_color * n_l + highlight * lighting.sun_color.w) * light.color
        * falloff;
}

// Light of the local lights of this pixel's cluster at `point`: windowed
// inverse-square falloff, spot cones and the same GGX highlights as the sun,
// fading out over the last tenth of the clustered depth range. While lights
// shine from their spheres and tubes or highlights are capped, every light,
// and otherwise lights kept to their regions, shade as `shaped_light`.
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
    let shaped = (clusters.lights.y & SHAPED_LIGHTS) != 0u;
    let kept = (clusters.lights.y & KEPT_REGIONS) != 0u;
    let capped = lighting.highlights.z > 0.0;
    var inside = -1.0;
    var sum = vec3<f32>(0.0);
    for (var n = 0u; n < range.y; n++) {
        let light = lights[cluster_lights[range.x + n]];
        if shaped || capped || (kept && light.region != 0u) {
            sum += shaped_light(point, light, shaped, kept, &inside);
            continue;
        }
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
        // Outside its cone a spot adds exactly nothing.
        if falloff == 0.0 {
            continue;
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
    let point = room_light(surface_point(input.color, input.normal, input.world, input.uv,
        input.uv1, input.surface, front), input);
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

// The surface target's texel (`post::SURFACE_FORMAT`, four 16-bit words):
// the shading normal, octahedral in [0, 1], a word each coordinate; its
// roughness along the view and across it, a byte each; and the share of
// its light its mirrored environment makes up, which screen-space
// reflections replace, in the last word's low MIRRORED_BITS bits, with
// what lies behind it above them.
fn surface_texel(shaded: Shaded) -> vec4<u32> {
    let normal = pack2x16unorm(octahedral_coordinates(shaded.normal) * 0.5 + 0.5);
    let roughness = pack4x8unorm(vec4<f32>(shaded.roughness, shaded.across, 0.0, 0.0));
    let most = f32((1u << MIRRORED_BITS) - 1u);
    let mirrored = u32(floor(saturate(shaded.mirrored) * most + 0.5));
    return vec4<u32>(normal & 0xffffu, normal >> 16u, roughness,
        mirrored | (shaded.body << MIRRORED_BITS));
}

// `pixel` is the fragment's.
fn scene_targets(shaded: Shaded, pixel: vec2<f32>) -> SceneTargets {
    var targets: SceneTargets;
    targets.color = vec4<f32>(stored(shaded.color, pixel), shaded.ambient);
    targets.glow = vec2<f32>(shaded.glow, shaded.reflect);
    targets.surface = surface_texel(shaded);
    return targets;
}

// `clip_position` is the fragment's, `previous` the same point's previous
// clip position: x, y and w.
fn temporal_targets(shaded: Shaded, clip_position: vec4<f32>, previous: vec3<f32>)
        -> TemporalTargets {
    var targets: TemporalTargets;
    targets.color = vec4<f32>(stored(shaded.color, clip_position.xy), shaded.ambient);
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

// The share of light glass lets through; 0 for materials
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

// Weighted blending of a transparent surface. Glass
// with refraction covers what is behind it with the opaque scene seen
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
    // The opaque color lies over the frame's render size of its target, from
    // its top-left corner: places in the view scale by its share, kept half
    // a texel within it.
    let reach = camera.region.zw - 0.5 / size;
    var uv = input.pixel / size;
    if clip.w > 1.0e-6 {
        uv = (vec2<f32>(clip.x, -clip.y) / clip.w * 0.5 + 0.5) * camera.region.zw;
    }
    uv = min(uv, reach);
    var behind = textureSampleLevel(refraction_source, refraction_sampler, uv, 0.0).rgb;
    let radius = point.roughness * lighting.refraction.y / size;
    if radius.x * size.x >= 0.5 {
        var sum = behind;
        for (var tap = 0u; tap < 8u; tap++) {
            sum += textureSampleLevel(refraction_source, refraction_sampler,
                min(uv + GLASS_TAPS[tap] * radius, reach), 0.0).rgb;
        }
        behind = sum / 9.0;
    }
    return weighted(shaded.color + behind * point.albedo * through, shaded.opacity, distance);
}
