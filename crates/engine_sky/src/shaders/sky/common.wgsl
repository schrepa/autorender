import package::sky::piece::sky_air;

const SKY_PI: f32 = 3.14159265358979;
const SKY_VIEW_SIZE: vec2<f32> = vec2<f32>(192.0, 108.0);

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

fn to_sub_uv(u: f32, size: f32) -> f32 {
    return (u + 0.5 / size) * (size / (size + 1.0));
}

fn from_sub_uv(u: f32, size: f32) -> f32 {
    return (u - 0.5 / size) * (size / (size - 1.0));
}

fn transmittance_uv(r: f32, mu: f32) -> vec2<f32> {
    let bottom = sky_air.radii.x;
    let top = sky_air.radii.y;
    let h = sqrt(max(top * top - bottom * bottom, 0.0));
    let rho = sqrt(max(r * r - bottom * bottom, 0.0));
    let discriminant = r * r * (mu * mu - 1.0) + top * top;
    let d = max(0.0, -r * mu + sqrt(max(discriminant, 0.0)));
    let d_min = top - r;
    let d_max = rho + h;
    return vec2<f32>((d - d_min) / max(d_max - d_min, 1e-6), rho / max(h, 1e-6));
}

fn air_transmittance(table: texture_2d<f32>, table_sampler: sampler, r: f32, mu: f32)
        -> vec3<f32> {
    let origin = vec3<f32>(0.0, r, 0.0);
    let direction = vec3<f32>(sqrt(max(1.0 - mu * mu, 0.0)), mu, 0.0);
    if sphere_hit(origin, direction, sky_air.radii.x) >= 0.0 {
        return vec3<f32>(0.0);
    }
    return textureSampleLevel(table, table_sampler, transmittance_uv(r, mu), 0.0).rgb;
}

fn sunlight_over(table: texture_2d<f32>, table_sampler: sampler,
        across: vec2<f32>, height: f32, sun: vec3<f32>) -> vec3<f32> {
    let bottom = sky_air.radii.x;
    let place = vec3<f32>(across.x, bottom + max(height, 0.0005), across.y);
    let r = length(place);
    let mu = dot(place / r, sun);
    let ratio = bottom / r;
    let limb = -sqrt(max(1.0 - ratio * ratio, 0.0));
    let above = (mu - limb) / max(ratio, 1.0e-3);
    let disc = smoothstep(-sky_air.ground.w, sky_air.ground.w, above);
    if disc <= 0.0 {
        return vec3<f32>(0.0);
    }
    return sky_air.ozone.w * disc
        * air_transmittance(table, table_sampler, r, max(mu, limb + 2.0e-3));
}

fn sky_view_uv(ground: bool, view_cos: f32, light_cos: f32, r: f32) -> vec2<f32> {
    let bottom = sky_air.radii.x;
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

fn sky_view_parameters(sub_uv: vec2<f32>, r: f32) -> vec2<f32> {
    let uv = vec2<f32>(from_sub_uv(sub_uv.x, SKY_VIEW_SIZE.x), from_sub_uv(sub_uv.y, SKY_VIEW_SIZE.y));
    let bottom = sky_air.radii.x;
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

fn air_radius() -> f32 {
    return sky_air.radii.x + max(sky_air.radii.w, 0.0005);
}

fn air_sky(view_table: texture_2d<f32>, table_sampler: sampler,
        direction: vec3<f32>) -> vec3<f32> {
    let r = air_radius();
    let origin = vec3<f32>(0.0, r, 0.0);
    let ground = sphere_hit(origin, direction, sky_air.radii.x) >= 0.0;
    let sun = normalize(sky_air.sun.xyz);
    let across = vec2<f32>(direction.x, direction.z);
    let sun_across = vec2<f32>(sun.x, sun.z);
    var light_cos = 1.0;
    if dot(across, across) > 1e-8 && dot(sun_across, sun_across) > 1e-8 {
        light_cos = dot(normalize(across), normalize(sun_across));
    }
    let uv = sky_view_uv(ground, direction.y, light_cos, r);
    return textureSampleLevel(view_table, table_sampler, uv, 0.0).rgb;
}

fn air_disc(transmittance: texture_2d<f32>, table_sampler: sampler,
        direction: vec3<f32>) -> vec3<f32> {
    let r = air_radius();
    let angle = acos(clamp(dot(direction, normalize(sky_air.sun.xyz)), -1.0, 1.0));
    let radius = sky_air.ground.w;
    if angle >= radius
        || sphere_hit(vec3<f32>(0.0, r, 0.0), direction, sky_air.radii.x) >= 0.0 {
        return vec3<f32>(0.0);
    }
    let edge = angle / radius;
    let limb = 1.0 - 0.6 * (1.0 - sqrt(max(1.0 - edge * edge, 0.0)));
    return sky_air.ozone.w * air_transmittance(transmittance, table_sampler, r, direction.y)
        * (6.0 * limb);
}
