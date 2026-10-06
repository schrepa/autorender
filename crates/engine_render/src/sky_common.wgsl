// A physically based sky (decision 088), after Hillaire, "A Scalable and
// Production Ready Sky and Atmosphere Rendering Technique" (2020): air over
// a planet, integrated into a transmittance table (to the top of the
// atmosphere by height and zenith angle), a table of multiple scattering's
// isotropic light (by height and the sun's angle) and a sky-view table (the
// sky's light around the camera). Distances are kilometers; the planet's
// center is the origin and the camera stands over its north pole, so world
// directions are the sky's. These are the shared declarations: the tables'
// builds are `sky.wgsl`, and the sky's passes sample them.

// WGSL `SkyUniform` (`sky.rs`).
struct SkyUniform {
    // Rayleigh scattering per km (rgb) and scale height (km).
    rayleigh: vec4<f32>,
    // Mie scattering and absorption per km, scale height (km), phase g.
    mie: vec4<f32>,
    // Ozone absorption per km at its densest (rgb), the sun's intensity.
    ozone: vec4<f32>,
    // Ground albedo (rgb), the sun's angular radius (radians).
    ground: vec4<f32>,
    // The planet's and the atmosphere's radius (km), how far exposure
    // adapts (0 to 1), and the camera's height above the planet's ground
    // (km).
    radii: vec4<f32>,
    // Direction toward the sun, and the authored exposure.
    sun: vec4<f32>,
}

const SKY_PI: f32 = 3.14159265358979;
const SKY_VIEW_SIZE: vec2<f32> = vec2<f32>(192.0, 108.0);
const SKY_MULTIPLE_SIZE: f32 = 32.0;

/// The distance along a ray from `origin` in `direction` (unit) to a sphere
/// of `radius` around the origin: the nearest hit ahead, or -1.
fn sphere_hit(origin: vec3<f32>, direction: vec3<f32>, radius: f32) -> f32 {
    let b = dot(origin, direction);
    let c = dot(origin, origin) - radius * radius;
    let discriminant = b * b - c;
    if discriminant < 0.0 {
        return -1.0;
    }
    let root = sqrt(discriminant);
    let near = -b - root;
    let far = -b + root;
    if near >= 0.0 {
        return near;
    }
    if far >= 0.0 {
        return far;
    }
    return -1.0;
}

// Table coordinates whose ends are texel centers.
fn to_sub_uv(u: f32, size: f32) -> f32 {
    return (u + 0.5 / size) * (size / (size + 1.0));
}

fn from_sub_uv(u: f32, size: f32) -> f32 {
    return (u - 0.5 / size) * (size / (size - 1.0));
}

/// The transmittance table's coordinates of radius `r` (km from the
/// center) and zenith cosine `mu` (Bruneton's parametrization).
fn transmittance_uv(air: SkyUniform, r: f32, mu: f32) -> vec2<f32> {
    let bottom = air.radii.x;
    let top = air.radii.y;
    let h = sqrt(max(top * top - bottom * bottom, 0.0));
    let rho = sqrt(max(r * r - bottom * bottom, 0.0));
    let discriminant = r * r * (mu * mu - 1.0) + top * top;
    let d = max(0.0, -r * mu + sqrt(max(discriminant, 0.0)));
    let d_min = top - r;
    let d_max = rho + h;
    return vec2<f32>((d - d_min) / max(d_max - d_min, 1e-6), rho / max(h, 1e-6));
}

/// The radius and zenith cosine of transmittance table coordinates `uv`.
fn transmittance_parameters(air: SkyUniform, uv: vec2<f32>) -> vec2<f32> {
    let bottom = air.radii.x;
    let top = air.radii.y;
    let h = sqrt(max(top * top - bottom * bottom, 0.0));
    let rho = h * uv.y;
    let r = sqrt(rho * rho + bottom * bottom);
    let d_min = top - r;
    let d_max = rho + h;
    let d = d_min + uv.x * (d_max - d_min);
    var mu = 1.0;
    if d > 0.0 {
        mu = clamp((h * h - rho * rho - d * d) / (2.0 * r * d), -1.0, 1.0);
    }
    return vec2<f32>(r, mu);
}

/// The share of light through the air from radius `r` along zenith cosine
/// `mu` to the top of the atmosphere, from `table`; none through the
/// planet.
fn air_transmittance(table: texture_2d<f32>, table_sampler: sampler, air: SkyUniform, r: f32, mu: f32)
        -> vec3<f32> {
    let origin = vec3<f32>(0.0, r, 0.0);
    let direction = vec3<f32>(sqrt(max(1.0 - mu * mu, 0.0)), mu, 0.0);
    if sphere_hit(origin, direction, air.radii.x) >= 0.0 {
        return vec3<f32>(0.0);
    }
    return textureSampleLevel(table, table_sampler, transmittance_uv(air, r, mu), 0.0).rgb;
}

/// The sky-view table's coordinates of a view at zenith cosine `view_cos`
/// whose azimuth is `light_cos`'s cosine from the sun's, from radius `r`;
/// `ground` when the view meets the planet.
fn sky_view_uv(air: SkyUniform, ground: bool, view_cos: f32, light_cos: f32, r: f32) -> vec2<f32> {
    let bottom = air.radii.x;
    let horizon = sqrt(max(r * r - bottom * bottom, 0.0));
    let beta = acos(clamp(horizon / r, -1.0, 1.0));
    let zenith_horizon = SKY_PI - beta;
    let angle = acos(clamp(view_cos, -1.0, 1.0));
    var uv: vec2<f32>;
    if !ground {
        let coord = 1.0 - sqrt(max(1.0 - angle / zenith_horizon, 0.0));
        uv.y = coord * 0.5;
    } else {
        let coord = sqrt(max((angle - zenith_horizon) / max(beta, 1e-6), 0.0));
        uv.y = coord * 0.5 + 0.5;
    }
    uv.x = sqrt(clamp(-light_cos * 0.5 + 0.5, 0.0, 1.0));
    return vec2<f32>(to_sub_uv(uv.x, SKY_VIEW_SIZE.x), to_sub_uv(uv.y, SKY_VIEW_SIZE.y));
}

/// A view's zenith cosine and its azimuth's cosine from the sun's, at
/// sky-view table coordinates `uv` from radius `r`.
fn sky_view_parameters(air: SkyUniform, sub_uv: vec2<f32>, r: f32) -> vec2<f32> {
    let uv = vec2<f32>(from_sub_uv(sub_uv.x, SKY_VIEW_SIZE.x), from_sub_uv(sub_uv.y, SKY_VIEW_SIZE.y));
    let bottom = air.radii.x;
    let horizon = sqrt(max(r * r - bottom * bottom, 0.0));
    let beta = acos(clamp(horizon / r, -1.0, 1.0));
    let zenith_horizon = SKY_PI - beta;
    var view_cos: f32;
    if uv.y < 0.5 {
        var coord = 1.0 - 2.0 * uv.y;
        coord = 1.0 - coord * coord;
        view_cos = cos(zenith_horizon * coord);
    } else {
        var coord = uv.y * 2.0 - 1.0;
        coord = coord * coord;
        view_cos = cos(zenith_horizon + beta * coord);
    }
    let x = uv.x * uv.x;
    return vec2<f32>(view_cos, -(x * 2.0 - 1.0));
}

/// The camera's radius from the planet's center (km).
fn air_radius(air: SkyUniform) -> f32 {
    return air.radii.x + max(air.radii.w, 0.0005);
}

/// The sky's light looking along `direction` (unit) from `view_table`,
/// with the sun's disc in `transmittance` when `disc`.
fn air_sky(view_table: texture_2d<f32>, transmittance: texture_2d<f32>, table_sampler: sampler,
        air: SkyUniform, direction: vec3<f32>, disc: bool) -> vec3<f32> {
    let r = air_radius(air);
    let origin = vec3<f32>(0.0, r, 0.0);
    let ground = sphere_hit(origin, direction, air.radii.x) >= 0.0;
    let sun = normalize(air.sun.xyz);
    let across = vec2<f32>(direction.x, direction.z);
    let sun_across = vec2<f32>(sun.x, sun.z);
    var light_cos = 1.0;
    if dot(across, across) > 1e-8 && dot(sun_across, sun_across) > 1e-8 {
        light_cos = dot(normalize(across), normalize(sun_across));
    }
    let uv = sky_view_uv(air, ground, direction.y, light_cos, r);
    var light = textureSampleLevel(view_table, table_sampler, uv, 0.0).rgb;
    if disc && !ground {
        let angle = acos(clamp(dot(direction, sun), -1.0, 1.0));
        let radius = air.ground.w;
        if angle < radius {
            // Limb darkening toward the disc's edge.
            let edge = angle / radius;
            let limb = 1.0 - 0.6 * (1.0 - sqrt(max(1.0 - edge * edge, 0.0)));
            light += air.ozone.w * air_transmittance(transmittance, table_sampler, air, r, direction.y)
                * (6.0 * limb);
        }
    }
    return light;
}
