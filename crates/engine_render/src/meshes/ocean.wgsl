const OCEAN_TAU: f32 = 6.283185307179586;
const OCEAN_CREST_SHARE: f32 = 0.257;

struct OceanOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) relative: vec3<f32>,
    @location(1) still: vec3<f32>,
}

struct OceanTemporalOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) relative: vec3<f32>,
    @location(1) still: vec3<f32>,
    @location(2) previous: vec3<f32>,
}

fn ocean_radius(ring: f32) -> f32 {
    let cell = water.ocean.y;
    let first = water.ocean.w;
    if (ring <= first) {
        return ring * cell;
    }
    return first * cell * exp2(log2(water.ocean_grid.x) * (ring - first));
}

fn ocean_keep(number: f32, spacing: f32) -> f32 {
    return saturate(OCEAN_TAU / (number * 4.0 * spacing) - 1.0);
}

fn ocean_shown(number: f32, way: vec2<f32>, footprint: WaterFootprint) -> f32 {
    return water_shown(OCEAN_TAU / number, water_footprint_toward(footprint, way));
}

fn ocean_moved(still: vec2<f32>, spacing: f32) -> vec3<f32> {
    var moved = vec3<f32>(0.0);
    let count = u32(water.ocean_grid.w);
    for (var index = 0u; index < count; index++) {
        let wave = water.ocean_waves[index];
        let number = max(length(wave.xy), 1.0e-6);
        let keep = ocean_keep(number, spacing);
        if (keep <= 0.0) {
            break;
        }
        let phase = dot(wave.xy, still) + wave.w;
        let reach = water.ocean_reach[index / 4u][index % 4u] * keep;
        let along = wave.xy / number * (reach * cos(phase));
        moved += vec3<f32>(along.x, wave.z * keep * sin(phase), along.y);
    }
    return moved;
}

const OCEAN_CORNERS = array<vec2<u32>, 6>(
    vec2<u32>(0u, 0u), vec2<u32>(0u, 1u), vec2<u32>(1u, 0u),
    vec2<u32>(1u, 0u), vec2<u32>(0u, 1u), vec2<u32>(1u, 1u),
);

fn ocean_output(vertex: u32, sector: u32) -> OceanOutput {
    let per_sector = u32(water.ocean_grid.z);
    let around = u32(water.ocean.z);
    let quad = vertex / 6u;
    let corner = OCEAN_CORNERS[vertex % 6u];
    let segment = (sector * per_sector + quad % per_sector + corner.x) % around;
    let ring = quad / per_sector + corner.y;
    let angle = f32(segment) / f32(around) * OCEAN_TAU;
    let radius = min(ocean_radius(f32(ring)), water.ocean_grid.y);
    let still = vec2<f32>(cos(angle), sin(angle)) * radius;
    let spacing = max(water.ocean.y, radius * OCEAN_TAU / f32(around));
    let moved = ocean_moved(still, spacing);
    let relative = vec3<f32>(still.x + moved.x, water.ocean.x + moved.y, still.y + moved.z);
    var output: OceanOutput;
    output.clip_position = water.relative_view_projection * vec4<f32>(relative, 1.0);
    output.relative = relative;
    output.still = vec3<f32>(still, moved.y);
    return output;
}

@vertex
fn vs_ocean(@builtin(vertex_index) vertex: u32, @builtin(instance_index) sector: u32)
        -> OceanOutput {
    return ocean_output(vertex, sector);
}

@vertex
fn vs_ocean_temporal(@builtin(vertex_index) vertex: u32,
        @builtin(instance_index) sector: u32) -> OceanTemporalOutput {
    let output = ocean_output(vertex, sector);
    var temporal: OceanTemporalOutput;
    temporal.clip_position = output.clip_position;
    temporal.relative = output.relative;
    temporal.still = output.still;
    temporal.previous = (camera.previous_view_projection
        * vec4<f32>(eye_world(output.relative), 1.0)).xyw;
    return temporal;
}

fn shade_ocean_below(input: OceanOutput) -> WaterShade {
    let world = eye_world(input.relative);
    let footprint = water_footprint(dpdx(input.still.xy), dpdy(input.still.xy));
    let up = ocean_normal(input.still.xy, footprint).xyz;
    let ray = stable_normalize(input.relative);
    let bent = refract(ray, -up, 1.333);
    var light = underwater_light(reflect(ray, -up));
    if (dot(bent, bent) > 0.0) {
        let through = 1.0 - fresnel(vec3<f32>(0.02), max(dot(-ray, -up), 0.0)).r;
        light = mix(light, open_sky(normalize(bent)), through);
    }
    let lit = mix(light, underwater_light(ray), underwater_cover(length(input.relative)));
    var shaded: Shaded;
    shaded.linear = lit;
    shaded.color = grade(lit);
    shaded.ambient = 0.0;
    shaded.glow = glow_weight(lit);
    shaded.reflect = 0.0;
    shaded.normal = -up;
    shaded.roughness = 0.1;
    shaded.opacity = 1.0;
    var result: WaterShade;
    result.shaded = shaded;
    result.coverage = 1.0;
    result.behind = vec3<f32>(0.0);
    result.fade = 1.0;
    result.flow = vec2<f32>(0.0);
    return result;
}

fn ocean_heights(still: vec2<f32>, footprint: WaterFootprint) -> vec3<f32> {
    var heights = vec3<f32>(0.0);
    let count = u32(water.ocean_grid.w);
    for (var index = 0u; index < count; index++) {
        let wave = water.ocean_waves[index];
        let number = max(length(wave.xy), 1.0e-6);
        let kept = wave.z * ocean_shown(number, wave.xy / number, footprint);
        heights += vec3<f32>(kept * sin(dot(wave.xy, still) + wave.w), 0.5 * kept * kept,
            0.5 * wave.z * wave.z);
    }
    return heights;
}

fn ocean_normal(still: vec2<f32>, footprint: WaterFootprint) -> vec4<f32> {
    var rise = vec2<f32>(0.0);
    var stretch = vec3<f32>(1.0, 0.0, 1.0);
    let count = u32(water.ocean_grid.w);
    for (var index = 0u; index < count; index++) {
        let wave = water.ocean_waves[index];
        let number = max(length(wave.xy), 1.0e-6);
        let direction = wave.xy / number;
        let keep = ocean_shown(number, direction, footprint);
        let phase = dot(wave.xy, still) + wave.w;
        let reach = water.ocean_reach[index / 4u][index % 4u] * keep;
        rise += wave.xy * (wave.z * keep * cos(phase));
        stretch -= vec3<f32>(direction.x * wave.x, direction.x * wave.y, direction.y * wave.y)
            * (reach * sin(phase));
    }
    let along_x = vec3<f32>(stretch.x, rise.x, stretch.y);
    let along_z = vec3<f32>(stretch.y, rise.y, stretch.z);
    return vec4<f32>(normalize(cross(along_z, along_x)),
        stretch.x * stretch.z - stretch.y * stretch.y);
}

fn shade_ocean(input: OceanOutput, refracts: bool) -> WaterShade {
    if lighting.underwater.w > 0.0 {
        return shade_ocean_below(input);
    }
    let world = eye_world(input.relative);
    let still = input.still.xy;
    let view = stable_normalize(-input.relative);
    let world_dx = dpdx(world);
    let world_dy = dpdy(world);
    let footprint = water_footprint(dpdx(still), dpdy(still));
    let waved = ocean_normal(still, footprint);
    var hidden = vec3<f32>(0.0);
    let count = u32(water.ocean_grid.w);
    for (var index = 0u; index < count; index++) {
        let wave = water.ocean_waves[index];
        let number = max(length(wave.xy), 1.0e-6);
        let keep = ocean_shown(number, wave.xy / number, footprint);
        let slope = number * wave.z;
        hidden += water_spread(wave.xy / number) * ((1.0 - keep * keep) * slope * slope * 0.5);
    }
    let slopes = -waved.xz / max(waved.y, 1.0e-3);
    let normal = normalize(vec3<f32>(-slopes.x, 1.0, -slopes.y));
    let compression = waved.w;
    var through = 1.0e6;
    var below = 1.0e6;
    var behind: WaterBehind;
    behind.color = vec3<f32>(0.0);
    let pixel = input.clip_position.xy;
    if textureLoad(water_depth, vec2<i32>(pixel), 0) > 0.0 {
        let own = water_ground(pixel);
        below = max(input.relative.y - own.y, 0.0); // optics: what the view meets behind the pixel
        behind.ground = own;
        if refracts {
            behind = water_behind(input.relative, normal, view, pixel, own,
                vec3<f32>(0.0, 1.0, 0.0));
        }
        through = max(length(behind.ground - input.relative), 0.0); // optics: the view's way through the water
    }
    let transmit = water_transmitted(through, 0.0);
    let capping = max(WATER_FOAM_CAPPING, fwidth(compression) * material_detail());
    let foam_capping = saturate((water.foam.x - compression) / capping + 0.5); // filter: the whitecaps' edge spread over the pixel, keeping their share
    let foam_contact = water_contact(below, abs(dpdx(below)) + abs(dpdy(below)), footprint,
        water.foam.y * sqrt(WATER_GRAVITY / max(below, WATER_BREAK_THINNEST)), 1.0);
    let foam = water_foam(water_layers(1.0 - (1.0 - foam_capping) * (1.0 - foam_contact)),
        0.0);
    let roughness = water_roughness(hidden + water_turning(slopes), water_heading(view));
    let swell = ocean_heights(still, footprint);
    let peak = mix(OCEAN_CREST_SHARE,
        saturate(swell.x * inverseSqrt(max(2.0 * swell.y, 1.0e-12))),
        saturate(swell.y / max(swell.z, 1.0e-12)));
    var surface: WaterSurface;
    surface.relative = input.relative;
    surface.world_dx = world_dx;
    surface.world_dy = world_dy;
    surface.normal = normal;
    surface.roughness = roughness;
    surface.plane = vec3<f32>(0.0, 1.0, 0.0);
    surface.foam = foam.cover;
    surface.foam_shade = foam.shade;
    surface.foam_thin = foam.thin;
    surface.scatter = water.color.rgb;
    surface.transmit = transmit;
    surface.crest = peak;
    surface.crest_wall = water_extinction(0.0) * (WATER_CREST_WALL * sqrt(swell.z));
    let lit = water_light(surface);
    let fade = smoothstep(0.0, WATER_SHORE, below);
    var result: WaterShade;
    result.shaded = lit.shaded;
    result.coverage = max(1.0 - (1.0 - lit.reflectance) * luminance(transmit), foam.cover)
        * fade;
    result.behind = behind.color * transmit * ((1.0 - lit.reflectance) * (1.0 - foam.cover)
        * (1.0 - atmospheric_cover(camera.position.xyz, world)));
    result.fade = fade;
    result.flow = vec2<f32>(0.0);
    return water_mirroring(result);
}

fn ocean_temporal_output(input: OceanTemporalOutput) -> OceanOutput {
    var output: OceanOutput;
    output.clip_position = input.clip_position;
    output.relative = input.relative;
    output.still = input.still;
    return output;
}

@fragment
fn fs_ocean(input: OceanOutput) -> @location(0) vec4<f32> {
    let shade = shade_ocean(input, false);
    return vec4<f32>(encoded(shade.shaded.color, input.clip_position.xy) * shade.coverage,
        shade.coverage);
}

@fragment
fn fs_ocean_post(input: OceanOutput) -> SceneTargets {
    let shade = shade_ocean(input, true);
    var targets = scene_targets(shade.shaded, input.clip_position.xy);
    targets.color = water_covering(shade, input.clip_position.xy);
    return targets;
}

@fragment
fn fs_ocean_temporal(input: OceanTemporalOutput) -> TemporalTargets {
    let shade = shade_ocean(ocean_temporal_output(input), true);
    var targets = temporal_targets(shade.shaded, input.clip_position, input.previous);
    targets.color = water_covering(shade, input.clip_position.xy);
    return targets;
}
