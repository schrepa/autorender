// The sky behind all geometry: one triangle over the viewport at the far
// plane, drawn after the meshes so depth testing leaves only uncovered pixels.
struct SkyOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}

@vertex
fn vs_sky(@builtin(vertex_index) index: u32) -> SkyOutput {
    let ndc = vec2<f32>(f32((index << 1u) & 2u) * 2.0 - 1.0, f32(index & 2u) * 2.0 - 1.0);
    var output: SkyOutput;
    // The far plane: depth 0 with reversed depth.
    output.clip_position = vec4<f32>(ndc, 0.0, 1.0);
    output.ndc = ndc;
    return output;
}

// The clouds' newer and older shown domes, their
// sampler and their uniform: clear ones without clouds.
@group(1) @binding(38) var sky_clouds: texture_2d<f32>;
@group(1) @binding(39) var sky_cloud_sampler: sampler;
@group(1) @binding(44) var sky_clouds_old: texture_2d<f32>;
@group(1) @binding(45) var<uniform> sky_domes: CloudDomes;
// The physically based sky: its uniform, sky-view and
// transmittance tables and their sampler.
@group(1) @binding(40) var<uniform> sky_air: SkyUniform;
@group(1) @binding(41) var sky_view_table: texture_2d<f32>;
@group(1) @binding(42) var sky_transmittance_table: texture_2d<f32>;
@group(1) @binding(43) var sky_table_sampler: sampler;

// Whether the sky is the physically based one.
fn physical_sky() -> bool {
    return lighting.grade.z > 0.5;
}

// The sky's light along `ray` from its tables, with the sun's disc when
// `disc`.
fn air_light_along(ray: vec3<f32>, disc: bool) -> vec3<f32> {
    return air_sky(sky_view_table, sky_transmittance_table, sky_table_sampler, sky_air, ray, disc);
}

// A dome at `uv` through four filtered taps half a texel apart: a tent
// over the texels around, which softens the noise of the march's per-ray
// offsets where thin, brightly lit edges change from texel to texel.
fn dome_at(dome: texture_2d<f32>, uv: vec2<f32>) -> vec4<f32> {
    let half_texel = 0.5 / vec2<f32>(textureDimensions(dome));
    return 0.25 * (textureSampleLevel(dome, sky_cloud_sampler, uv - half_texel, 0.0)
        + textureSampleLevel(dome, sky_cloud_sampler, uv + half_texel, 0.0)
        + textureSampleLevel(dome, sky_cloud_sampler, uv + vec2<f32>(half_texel.x, -half_texel.y), 0.0)
        + textureSampleLevel(dome, sky_cloud_sampler, uv + vec2<f32>(-half_texel.x, half_texel.y), 0.0));
}

// The clouds along `ray`: their light and the share of the sky passing,
// the newer dome blending in over the older, each looked up where it sees
// what the ray does.
fn sky_cloud(ray: vec3<f32>) -> vec4<f32> {
    if sky_domes.older.w < 0.5 {
        return dome_at(sky_clouds, cloud_dome_uv(ray));
    }
    let newer = dome_at(sky_clouds,
        cloud_dome_uv(cloud_dome_direction(sky_domes, sky_domes.newer.xyz, ray)));
    let older = dome_at(sky_clouds_old,
        cloud_dome_uv(cloud_dome_direction(sky_domes, sky_domes.older.xyz, ray)));
    return mix(older, newer, sky_domes.newer.w);
}

// The sky along `ray` behind its clouds, with the night sky's; under the
// sea, the water's light. While the view marches its own clouds, they
// cover it after it draws.
fn clouded_sky(ray: vec3<f32>) -> vec3<f32> {
    if lighting.underwater.w > 0.0 {
        return sky(ray);
    }
    var light: vec3<f32>;
    if physical_sky() {
        light = air_light_along(ray, true);
    } else {
        light = sky(ray);
    }
    light = with_night_sky(light, ray);
    if sky_domes.view.x > 0.5 {
        return light;
    }
    return behind_clouds(light, sky_cloud(ray), ray);
}

// The night sky's stars: cells along each side of each face of the cube
// they are hashed on, the share of cells holding one (fewer toward the
// galaxy's poles, more along its plane), the light of the brightest, and
// the sky's luminance above which none shows (they fade in below it, fully
// from two fifths of it).
const STAR_CELLS: f32 = 160.0;
const STAR_SHARE: f32 = 0.4;
const STAR_LIGHT: f32 = 0.25;
const STAR_SKY: f32 = 0.1;
// The Milky Way: the light of its brightest stretch, toward the galactic
// centre; how many cells of its noise's coarsest octave go around the sky
// (about 2.8 degrees each); and the galactic latitude (its sine) past
// which it leaves no glow.
const GALAXY_LIGHT: f32 = 0.02;
const GALAXY_CELLS: f32 = 128.0;
const GALAXY_REACH: f32 = 0.55;
// The faint stars crowding the Milky Way: cells along each side of a face
// (three times the stars'), the share of them holding one where the band is
// brightest, and the light of the brightest of them.
const GRAIN_CELLS: f32 = 480.0;
const GRAIN_SHARE: f32 = 0.7;
const GRAIN_LIGHT: f32 = 0.05;
// The share of light a sea-level atmosphere takes away per air mass, by
// channel (as optical depth): what the analytic sky's stars and moon lose
// toward the horizon.
const AIR_EXTINCTION: vec3<f32> = vec3<f32>(0.07, 0.15, 0.28);
// The moon's usual angular radius (radians: twice its true quarter degree,
// the size the eye reads it at), which the night sky's moon size scales,
// and the light of its full face.
const MOON_RADIUS: f32 = 0.0091;
const MOON_LIGHT: f32 = 0.6;
// The brightest the moon's face shows after the exposure: in a dark sky
// the eye adapts to it, so its seas stay apart from its highlands rather
// than all of it burning white.
const MOON_SHOWN: f32 = 0.9;
// The north celestial pole in the galaxy's frame (the J2000 equatorial
// pole by the IAU's galactic coordinates), which the moon's north faces.
const CELESTIAL_POLE: vec3<f32> = vec3<f32>(-0.4838350, 0.7469822, 0.4559838);
// The moon's seas as the earth sees them: each one's centre in
// selenographic longitude (degrees east of the near side's centre) and
// latitude, its angular radius on the moon (degrees) and how much darker
// than the highlands it is; Oceanus Procellarum and Mare Frigoris as
// overlapping discs.
const MOON_SEAS = array<vec4<f32>, 15>(
    vec4<f32>(-16.0, 33.0, 18.0, 0.45),
    vec4<f32>(17.5, 28.0, 11.0, 0.42),
    vec4<f32>(31.0, 8.5, 13.0, 0.45),
    vec4<f32>(59.0, 17.0, 8.0, 0.5),
    vec4<f32>(51.0, -8.0, 12.0, 0.4),
    vec4<f32>(35.0, -15.0, 6.0, 0.4),
    vec4<f32>(-17.0, -21.0, 11.0, 0.35),
    vec4<f32>(-39.0, -24.0, 7.0, 0.4),
    vec4<f32>(-50.0, 12.0, 22.0, 0.42),
    vec4<f32>(-62.0, 30.0, 14.0, 0.4),
    vec4<f32>(-40.0, -3.0, 12.0, 0.38),
    vec4<f32>(-15.0, 56.0, 7.0, 0.32),
    vec4<f32>(15.0, 57.0, 7.0, 0.32),
    vec4<f32>(4.0, 13.0, 5.0, 0.35),
    vec4<f32>(-28.0, 0.0, 8.0, 0.32),
);
// Its bright young craters: Tycho (whose rays cross the southern
// highlands), Copernicus, Kepler and Aristarchus, each's place, angular
// radius (degrees) and brightness above the highlands.
const MOON_CRATERS = array<vec4<f32>, 4>(
    vec4<f32>(-11.2, -43.3, 3.0, 0.3),
    vec4<f32>(-20.0, 9.6, 3.0, 0.15),
    vec4<f32>(-38.0, 8.1, 2.5, 0.1),
    vec4<f32>(-47.5, 23.7, 2.5, 0.15),
);
// The moon's glow in the air around it, as shares of its face's light: a
// narrow aureole and a wide halo (the forward lobes they fall off by).
const MOON_AUREOLE: f32 = 0.06;
const MOON_HALO: f32 = 0.012;
// Cirrus: how far along a view it fades out by (world units); its height
// and drift are the night sky's (`lighting.night_shape`).
const CIRRUS_REACH: f32 = 90000.0;

// `direction` turned by the unit quaternion `turn`.
fn turned(turn: vec4<f32>, direction: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(turn.xyz, direction);
    return direction + turn.w * t + cross(turn.xyz, t);
}

// Three values in [0, 1) hashed from `cell`.
fn sky_hash(cell: vec3<u32>) -> vec3<f32> {
    return vec3<f32>(dither_hash(cell) >> vec3<u32>(8u)) * (1.0 / 16777216.0);
}

// Smooth value noise in [0, 1] over the plane, its lattice salted by
// `salt`.
fn sky_noise(p: vec2<f32>, salt: u32) -> f32 {
    let cell = floor(p);
    let f = p - cell;
    let u = f * f * (3.0 - 2.0 * f);
    let base = vec2<u32>(vec2<i32>(cell) & vec2<i32>(0xffff));
    let a = sky_hash(vec3<u32>(base, salt)).x;
    let b = sky_hash(vec3<u32>(base + vec2<u32>(1u, 0u), salt)).x;
    let c = sky_hash(vec3<u32>(base + vec2<u32>(0u, 1u), salt)).x;
    let d = sky_hash(vec3<u32>(base + vec2<u32>(1u, 1u), salt)).x;
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Two independent values of smooth value noise in [0, 1] over the plane,
// its lattice repeating every `period` cells along x and salted by `salt`.
fn sky_noise_pair(p: vec2<f32>, period: i32, salt: u32) -> vec2<f32> {
    let cell = floor(p);
    let f = p - cell;
    let u = f * f * (3.0 - 2.0 * f);
    let left = u32(((i32(cell.x) % period) + period) % period);
    let right = (left + 1u) % u32(period);
    let low = u32(i32(cell.y) & 0xffff);
    let high = u32((i32(cell.y) + 1) & 0xffff);
    let a = sky_hash(vec3<u32>(left, low, salt)).xy;
    let b = sky_hash(vec3<u32>(right, low, salt)).xy;
    let c = sky_hash(vec3<u32>(left, high, salt)).xy;
    let d = sky_hash(vec3<u32>(right, high, salt)).xy;
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// The Milky Way along `galactic`, a direction in the galaxy's frame: a
// band along its plane, brightest and widest toward the centre, where the
// bulge rises, and bright again in the star clouds of Cygnus and Carina;
// mottled by star clouds; threaded with dark dust, thickest in the rift
// from Sagittarius to Cygnus and in the Coalsack; warmer toward the
// centre. rgb its light, a how crowded with faint stars the band is there
// (0 to 1).
fn milky_way(galactic: vec3<f32>) -> vec4<f32> {
    if abs(galactic.z) >= GALAXY_REACH {
        return vec4<f32>(0.0);
    }
    let latitude = asin(galactic.z);
    let longitude = atan2(galactic.y, galactic.x);
    let inward = 0.5 + 0.5 * galactic.x * inverseSqrt(max(dot(galactic.xy, galactic.xy), 1.0e-8));
    let toward = inward * inward;
    let side = abs(longitude) - 1.3;
    let along = 0.16 + 0.84 * toward + 0.3 * exp(-8.0 * side * side);
    let width = 0.06 + 0.05 * toward * inward;
    let across = latitude / width;
    var glow = along * (exp(-0.5 * across * across) + 0.12 * exp(-across * across / 18.0));
    let bulge = vec2<f32>(longitude / 0.16, latitude / 0.12);
    glow += 0.9 * exp(-0.5 * dot(bulge, bulge));
    // Cells of about the same angle along and across the plane, around it
    // seamlessly, in four octaves.
    let p = vec2<f32>(longitude, latitude) * (GALAXY_CELLS / 6.2831853);
    let period = i32(GALAXY_CELLS);
    let coarse = sky_noise_pair(p, period, 21u);
    let middle = sky_noise_pair(p * 2.0, period * 2, 22u);
    let fine = sky_noise_pair(p * 4.0, period * 4, 23u);
    let finest = sky_noise_pair(p * 8.0, period * 8, 24u);
    let clumps = coarse.x * 0.45 + middle.x * 0.25 + fine.x * 0.18 + finest.x * 0.12;
    let mottle = 0.3 + 1.4 * smoothstep(0.3, 0.7, clumps);
    // Dust in filaments (ridges of the noise) near the plane, thickest
    // along a meandering lane through the rift from Sagittarius to Cygnus.
    let veins = coarse.y * 0.45 + middle.y * 0.3 + fine.y * 0.25;
    let filaments = smoothstep(0.6, 0.9, 1.0 - abs(2.0 * veins - 1.0));
    let lane_across = (latitude + 0.015 - 0.012 * sin(3.0 * longitude)) / 0.035;
    let lane = exp(-0.5 * lane_across * lane_across);
    let rift = smoothstep(-0.3, -0.05, longitude) * (1.0 - smoothstep(1.35, 1.65, longitude));
    let near_plane = latitude / 0.1;
    var dust = max(lane * (0.35 + 0.45 * rift) * (0.5 + 0.5 * filaments),
        0.7 * filaments * exp(-0.5 * near_plane * near_plane));
    // The Coalsack, at longitude 301 and latitude -1 degrees.
    let coal = vec2<f32>(longitude + 1.026, latitude + 0.017) / 0.04;
    dust = max(dust, 0.8 * exp(-0.5 * dot(coal, coal)));
    let core = vec2<f32>(longitude / 0.6, latitude / 0.2);
    let tint = mix(vec3<f32>(0.84, 0.89, 1.0), vec3<f32>(1.0, 0.9, 0.78),
        exp(-0.5 * dot(core, core)));
    let shown = glow * mottle * (1.0 - 0.85 * dust);
    return vec4<f32>(tint * (GALAXY_LIGHT * shown), saturate(0.6 * shown));
}

// The light of a field of stars along `star`, a direction in the galaxy's
// frame: a star in `share` of the cells of a cube's faces, `cells` along
// each side (salted by `salt`), placed within its cell, its brightness a
// power `fall` of a uniform value so faint stars outnumber bright ones, up
// to `light`; bluish white to warm; drawn as a soft spot of `spread`
// radians (about a pixel) around its true direction, so it neither vanishes
// nor shimmers between pixels nor stretches toward the cube's edges.
fn star_field(star: vec3<f32>, cells: f32, share: f32, fall: f32, light: f32, salt: u32,
        spread: f32) -> vec3<f32> {
    let a = abs(star);
    var axis = 0u;
    var uv = star.yz / a.x;
    if a.y >= a.x && a.y >= a.z {
        axis = 1u;
        uv = star.xz / a.y;
    } else if a.z >= a.x {
        axis = 2u;
        uv = star.xy / a.z;
    }
    let negative = star[axis] < 0.0;
    let face = axis * 2u + select(0u, 1u, negative);
    let position = (uv * 0.5 + 0.5) * cells;
    let cell = floor(position);
    let hashed = sky_hash(vec3<u32>(vec2<u32>(cell), face + salt));
    if hashed.x >= share {
        return vec3<f32>(0.0);
    }
    let at = (cell + 0.25 + 0.5 * hashed.yz) / cells * 2.0 - 1.0;
    let toward = select(1.0, -1.0, negative);
    var place = vec3<f32>(toward, at.x, at.y);
    if axis == 1u {
        place = vec3<f32>(at.x, toward, at.y);
    } else if axis == 2u {
        place = vec3<f32>(at.x, at.y, toward);
    }
    let offset = star - normalize(place);
    let spot = exp(-dot(offset, offset) / (2.0 * spread * spread));
    let bright = pow(hashed.x / share, fall);
    let warm = fract(hashed.y * 7.31 + hashed.z * 3.17);
    let tint = mix(vec3<f32>(0.78, 0.86, 1.0), vec3<f32>(1.0, 0.86, 0.68), warm);
    return tint * (light * bright * spot);
}

// The stars' light along `star`, a direction in the galaxy's frame: more
// cells hold one toward the galaxy's plane, as more of its stars lie along
// the line of sight there.
fn star_light(star: vec3<f32>, spread: f32) -> vec3<f32> {
    let share = STAR_SHARE * lighting.night_shape.w * (0.6 + 0.8 * exp(-6.0 * star.z * star.z));
    return star_field(star, STAR_CELLS, share, 8.0, STAR_LIGHT, 0u, spread);
}

// A place on the moon at selenographic `longitude` and `latitude`
// (degrees) as a unit vector: x toward its east limb, y its north, z the
// centre of its near side, which faces the earth.
fn moon_place(longitude: f32, latitude: f32) -> vec3<f32> {
    let lon = radians(longitude);
    let lat = radians(latitude);
    return vec3<f32>(cos(lat) * sin(lon), sin(lat), cos(lat) * cos(lon));
}

// The moon's albedo at `place` on its near side (moon_place's frame), the
// highlands' 1: its seas darker, softly edged, with a little mottling; its
// bright craters, Tycho's rays reaching across the south.
fn moon_albedo(place: vec3<f32>, local: vec2<f32>) -> f32 {
    var seas = MOON_SEAS;
    // Shores wander: the distance to each sea's centre shifts with
    // low-frequency noise over the face, and fades over some 16 degrees.
    let wander = 0.16 * (sky_noise(place.xy * 2.5 + 11.0, 5u) - 0.5)
        + 0.08 * (sky_noise(place.xy * 6.0 + 3.0, 6u) - 0.5);
    var dark = 0.0;
    for (var index = 0u; index < 15u; index++) {
        let sea = seas[index];
        let reach = radians(sea.z);
        let along = acos(clamp(dot(place, moon_place(sea.x, sea.y)), -1.0, 1.0)) + wander;
        dark = max(dark, sea.w * (1.0 - smoothstep(reach - 0.14, reach + 0.14, along)));
    }
    var craters = MOON_CRATERS;
    var bright = 0.0;
    for (var index = 0u; index < 4u; index++) {
        let crater = craters[index];
        let apart = acos(clamp(dot(place, moon_place(crater.x, crater.y)), -1.0, 1.0));
        let spot = apart / radians(crater.z);
        bright += crater.w * exp(-spot * spot);
    }
    // Tycho's rays: thin streaks around it, fading over a third of a turn.
    let tycho = moon_place(-11.2, -43.3);
    let east = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), tycho));
    let north = cross(tycho, east);
    let around = atan2(dot(place, north), dot(place, east));
    let apart = acos(clamp(dot(place, tycho), -1.0, 1.0));
    let streaks = pow(0.5 + 0.5 * sin(around * 11.0 + 1.3), 8.0)
        + pow(0.5 + 0.5 * sin(around * 7.0 + 4.1), 10.0);
    bright += 0.12 * streaks * exp(-apart / 0.5) * smoothstep(0.03, 0.08, apart);
    let mottle = mix(0.9, 1.0, smoothstep(0.3, 0.7,
        0.65 * sky_noise(local * 2.2 + 7.0, 3u) + 0.35 * sky_noise(local * 5.3 + 2.0, 4u)));
    return mottle * (1.0 - dark) + bright;
}

// The moon along `ray`, standing along `moon` and lit by the sun along
// `sun`: its near side, its north toward the celestial pole, as a sphere
// lit where the sun reaches it, which gives its phase; its seas and bright
// craters where they are, and earthshine faint on the rest; rgb its light,
// a how much of the ray its disc covers (soft at the rim).
fn moon_disc(ray: vec3<f32>, moon: vec3<f32>, sun: vec3<f32>) -> vec4<f32> {
    let radius = MOON_RADIUS * lighting.moon.w;
    if dot(ray, moon) < cos(radius * 1.1) {
        return vec4<f32>(0.0);
    }
    // Celestial north in the world: the galaxy's frame turned back.
    let back = vec4<f32>(-lighting.star_turn.xyz, lighting.star_turn.w);
    let pole = turned(back, CELESTIAL_POLE);
    var up = pole - moon * dot(pole, moon);
    if dot(up, up) < 1.0e-8 {
        up = cross(moon, vec3<f32>(1.0, 0.0, 0.0));
    }
    up = normalize(up);
    let right = cross(moon, up);
    let local = vec2<f32>(dot(ray, right), dot(ray, up)) / radius;
    let r2 = dot(local, local);
    let cover = 1.0 - smoothstep(0.92, 1.0, sqrt(r2));
    if cover <= 0.0 {
        return vec4<f32>(0.0);
    }
    // The near half faces the eye: its east limb on the right.
    let facing = sqrt(max(1.0 - r2, 0.0));
    let normal = local.x * right + local.y * up - facing * moon;
    let lit = saturate(dot(normal, sun) * 2.0);
    let albedo = moon_albedo(vec3<f32>(local.x, local.y, facing), local);
    let shown = min(MOON_LIGHT, MOON_SHOWN / max(lighting.horizon.w, 1.0e-3));
    let light = vec3<f32>(1.0, 0.96, 0.9) * (albedo * (lit + 0.02) * shown);
    return vec4<f32>(light, cover);
}

// A forward-scattering lobe of anisotropy `g` (Henyey-Greenstein's), 1
// looking straight along the light, at `cosine` from it.
fn forward_lobe(cosine: f32, g: f32) -> f32 {
    let ahead = 1.0 - g;
    return ahead * ahead * ahead / pow(1.0 + g * g - 2.0 * g * cosine, 1.5);
}

// The moon's glow along `ray` in the air around the moon standing along
// `moon`: its light scattered forward, a narrow aureole and a wide halo, as
// strong as the share `lit` of its face the sun lights.
fn moon_glow(ray: vec3<f32>, moon: vec3<f32>, lit: f32) -> vec3<f32> {
    let cosine = dot(ray, moon);
    let glow = MOON_AUREOLE * forward_lobe(cosine, 0.97) + MOON_HALO * forward_lobe(cosine, 0.75);
    return vec3<f32>(1.0, 0.96, 0.9) * (MOON_LIGHT * lit * glow);
}

// The share of light from beyond the air that reaches the eye along `ray`,
// relative to the share straight up: through the physically based sky's
// air, or the analytic sky's by its air mass (Kasten and Young's formula);
// none from below the horizon.
fn beyond_air(ray: vec3<f32>) -> vec3<f32> {
    if physical_sky() {
        let r = air_radius(sky_air);
        let up = air_transmittance(sky_transmittance_table, sky_table_sampler, sky_air, r, 1.0);
        let along = air_transmittance(sky_transmittance_table, sky_table_sampler, sky_air, r,
            ray.y);
        return along / max(up, vec3<f32>(1.0e-4));
    }
    let elevation = degrees(asin(clamp(ray.y, 0.0, 1.0)));
    let mass = 1.0 / (max(ray.y, 0.0) + 0.50572 * pow(elevation + 6.07995, -1.6364));
    return exp(-AIR_EXTINCTION * (mass - 1.0)) * smoothstep(-0.01, 0.0, ray.y);
}

// How much cirrus covers along `ray`: fibrous noise on a layer its height
// above the eye, stretched along x and drifting with `seconds`, over
// `coverage` of the sky, fading toward the horizon.
fn cirrus_cover(ray: vec3<f32>, coverage: f32, seconds: f32) -> f32 {
    if ray.y <= 0.02 {
        return 0.0;
    }
    let along = lighting.night_shape.x / ray.y;
    let fade = 1.0 - smoothstep(0.35 * CIRRUS_REACH, CIRRUS_REACH, along);
    var p = (ray.xz * along + lighting.night_shape.yz * seconds)
        * vec2<f32>(1.0 / 7000.0, 1.0 / 1700.0);
    var sum = 0.0;
    var weight = 0.5;
    for (var octave = 0u; octave < 4u; octave++) {
        sum += sky_noise(p, 9u + octave) * weight;
        p = p * 2.03 + vec2<f32>(17.0, 31.0);
        weight *= 0.5;
    }
    let shape = saturate((sum / 0.9375 - (1.0 - coverage)) / 0.3);
    return shape * shape * fade * 0.6;
}

// The light cirrus sends along `ray`: the sun's (with the physically based
// sky, what reaches the layer where the ray meets it, over its own place
// on the planet, which outlasts sunset there and toward the sunset longest),
// scattered forward toward it; the moon's, faintly; and the sky's.
fn cirrus_light(ray: vec3<f32>) -> vec3<f32> {
    let sun = lighting.sun.xyz;
    var sunlight = lighting.sun_color.rgb;
    if physical_sky() {
        let height = lighting.night_shape.x;
        let along = height / max(ray.y, 0.02);
        sunlight = sunlight_over(sky_transmittance_table, sky_table_sampler, sky_air,
            ray.xz * (along * 0.001), sky_air.radii.w + height * 0.001, sun);
    }
    let toward = max(dot(ray, sun), 0.0);
    let forward = 0.25 + 1.4 * pow(toward, 6.0);
    let moonlight = vec3<f32>(0.8, 0.85, 1.0) * (0.02 * MOON_LIGHT * lighting.night.y
        * lighting.night.w * saturate(lighting.moon.y * 4.0));
    return sunlight * (0.3 * forward) + moonlight + 0.5 * (lighting.zenith.rgb + lighting.horizon.rgb);
}

// `light`, the sky's along `ray`, with the night sky's: stars and the Milky
// Way where the sky in front of them is dark, and the moon, all beyond the
// air, which dims and reddens them toward the horizon; the moon's glow in
// the dark air around it; then cirrus over all.
fn with_night_sky(light: vec3<f32>, ray: vec3<f32>) -> vec3<f32> {
    let night = lighting.night;
    if all(night.xyz == vec3<f32>(0.0)) {
        return light;
    }
    var dark = 1.0 - smoothstep(0.4 * STAR_SKY, STAR_SKY, luminance(light));
    // The physically based sky's sun is the real one: no stars while it is
    // up, all of them once it is six degrees down.
    if physical_sky() {
        dark *= 1.0 - smoothstep(-0.1, 0.0, lighting.sun.y);
    }
    var beyond = vec3<f32>(0.0);
    if night.x > 0.0 && dark > 0.0 {
        let galactic = turned(lighting.star_turn, ray);
        let spread = max(0.8 * camera.options.y, 1.0e-5);
        beyond = star_light(galactic, spread) * (night.x * dark);
        // The Milky Way needs a darker sky than the stars: under the
        // physically based sky the sun eight degrees down, fully from
        // fifteen; and a bright moon above the horizon washes it out.
        var deep = dark * (1.0 - 0.85 * night.w * saturate(lighting.moon.y * 4.0)
            * min(night.y, 1.0));
        if physical_sky() {
            deep *= 1.0 - smoothstep(-0.26, -0.14, lighting.sun.y);
        }
        if deep > 0.0 {
            let band = milky_way(galactic);
            var light = band.rgb;
            if band.a > 0.0 {
                light += star_field(galactic, GRAIN_CELLS, GRAIN_SHARE * band.a, 3.0,
                    GRAIN_LIGHT, 8u, spread);
            }
            beyond += light * (night.x * deep);
        }
    }
    var glow = vec3<f32>(0.0);
    if night.y > 0.0 {
        let disc = moon_disc(ray, lighting.moon.xyz, lighting.sun.xyz);
        beyond = mix(beyond, disc.rgb * night.y, disc.a);
        glow = moon_glow(ray, lighting.moon.xyz, night.w) * (night.y * dark);
    }
    var sky = light;
    if any(beyond + glow > vec3<f32>(0.0)) {
        sky += (beyond + glow) * beyond_air(ray);
    }
    if night.z > 0.0 {
        let cover = cirrus_cover(ray, night.z, camera.position.w);
        if cover > 0.0 {
            sky = mix(sky, cirrus_light(ray), cover);
        }
    }
    return sky;
}

// The view ray through unjittered `ndc`.
fn sky_ray(ndc: vec2<f32>) -> vec3<f32> {
    let far = camera.inverse_view_projection * vec4<f32>(ndc, 0.0, 1.0);
    return stable_normalize(far.xyz / far.w - camera.position.xyz);
}

@fragment
fn fs_sky(input: SkyOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(encoded(grade(clouded_sky(sky_ray(input.ndc))), input.clip_position.xy),
        1.0);
}

// Reflection probe captures: the sky's haze without the sun's
// disc, as the sky's environment map holds it (surfaces shade the sun's
// highlights themselves).
@fragment
fn fs_probe_sky(input: SkyOutput) -> @location(0) vec4<f32> {
    let ray = sky_ray(input.ndc);
    var light: vec3<f32>;
    if physical_sky() {
        light = air_light_along(ray, false);
    } else {
        light = haze(ray);
    }
    return vec4<f32>(behind_clouds(light, sky_cloud(ray), ray), 1.0);
}

// The sky's surface texel (`surface_texel`): facing up, fully rough both
// ways, mirroring nothing (it reflects nothing).
const SKY_SURFACE = vec4<u32>(32768u, 32768u, 65535u, 0u);

// Clouds in the view (`clouds.wgsl`'s `march_view`): each
// pixel's light and the share of the scene passing, marched up to what the
// pixel shows, over the depth buffer's pixels from its top-left corner,
// those over the frame's render size marched.
@group(1) @binding(46) var view_clouds: texture_2d<f32>;

// The view's clouds as a layer over the scene at the pixel at `position`
// (framebuffer): their own light, graded, times the share they cover, with
// the share passing as alpha; none under the sea.
fn view_cloud_layer(position: vec2<f32>) -> vec4<f32> {
    let cloud = textureLoad(view_clouds, vec2<i32>(position), 0);
    let covering = 1.0 - cloud.a;
    if covering <= 1.0e-4 || lighting.underwater.w > 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    return vec4<f32>(grade(cloud.rgb / covering) * covering, cloud.a);
}

// The view's clouds over the canvas, encoded as it is; pixels they leave
// clear keep the target as it was, unblended.
@fragment
fn fs_view_clouds(input: SkyOutput) -> @location(0) vec4<f32> {
    let layer = view_cloud_layer(input.clip_position.xy);
    let covering = 1.0 - layer.a;
    if covering <= 1.0e-4 {
        discard;
    }
    return vec4<f32>(encoded(layer.rgb / covering, input.clip_position.xy) * covering, layer.a);
}

// The view's clouds over the screen effects' scene target.
@fragment
fn fs_view_clouds_post(input: SkyOutput) -> @location(0) vec4<f32> {
    let layer = view_cloud_layer(input.clip_position.xy);
    if layer.a >= 1.0 - 1.0e-4 {
        discard;
    }
    if !dithering() {
        return layer;
    }
    let covering = 1.0 - layer.a;
    return vec4<f32>(stored(layer.rgb / covering, input.clip_position.xy) * covering, layer.a);
}

// Screen-effect frames: the sky has no ambient occlusion; its sun glows.
@fragment
fn fs_sky_post(input: SkyOutput) -> SceneTargets {
    let light = clouded_sky(sky_ray(input.ndc));
    var targets: SceneTargets;
    targets.color = vec4<f32>(stored(grade(light), input.clip_position.xy), 0.0);
    targets.glow = vec2<f32>(glow_weight(light), 0.0);
    targets.surface = SKY_SURFACE;
    return targets;
}

// Temporal frames: the unjittered sky reprojects by the camera's turn alone.
@fragment
fn fs_sky_temporal(input: SkyOutput) -> TemporalTargets {
    let ray = sky_ray(input.ndc);
    let light = clouded_sky(ray);
    var targets: TemporalTargets;
    targets.color = vec4<f32>(stored(grade(light), input.clip_position.xy), 0.0);
    targets.glow = vec2<f32>(glow_weight(light), 0.0);
    targets.surface = SKY_SURFACE;
    let previous = camera.previous_view_projection * vec4<f32>(ray, 0.0);
    targets.motion = (previous.xy / previous.w - input.ndc) * vec2<f32>(0.5, -0.5);
    if !(previous.w > 0.0) {
        targets.motion = vec2<f32>(2.0);
    }
    return targets;
}
