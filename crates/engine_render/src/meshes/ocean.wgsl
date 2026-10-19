// The sea: an endless surface at a still level that
// Gerstner waves raise, drawn from rings of cells around the eye, a cell
// apart near it and then each ring wider by a cell's share of its
// circumference, out to the sea's reach. Vertices move by the waves longer
// than a few of their cells; pixels shade with every wave longer than a
// few pixels, and the slopes of shorter ones roughen the surface instead,
// so distant water glitters rather than aliasing. The CPU's queries
// (`engine_core::OceanWaves`) sum the same waves. The waves' tiles ripple
// it as they do terrains' water, and it is lit as they are
// (`water_light`): what lies under it shows through, bent where the frame
// copies the scene's color and dimmed channel by channel over the view's
// path to it (the light reaching it already dims through the water above,
// `atmospheric`). Foam gathers where the waves compress crests and along
// shores, and sunlight glows through crests seen against it.

const OCEAN_TAU: f32 = 6.283185307179586;

struct OceanOutput {
    @builtin(position) clip_position: vec4<f32>,
    // The point less the eye.
    @location(0) relative: vec3<f32>,
    // The still sea's point it moved from, less the eye (xz), and the
    // meters the waves raised it over the still sea.
    @location(1) still: vec3<f32>,
}

struct OceanTemporalOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) relative: vec3<f32>,
    @location(1) still: vec3<f32>,
    @location(2) previous: vec3<f32>,
}

// Meters from the eye to ring `ring`: a cell apart within the first rings,
// then each ring wider by the rings' growth.
fn ocean_radius(ring: f32) -> f32 {
    let cell = water.ocean.y;
    let first = water.ocean.w;
    if (ring <= first) {
        return ring * cell;
    }
    return first * cell * exp2(log2(water.ocean_grid.x) * (ring - first));
}

// How much of a wave `number` radians per meter samples `spacing` meters
// apart keep: waves shorter than eight samples fade, and those shorter
// than four are gone.
fn ocean_keep(number: f32, spacing: f32) -> f32 {
    return saturate(OCEAN_TAU / (number * 4.0 * spacing) - 1.0);
}

// Where the waves move the still sea's point `still` (less the eye),
// keeping those that samples `spacing` meters apart resolve.
fn ocean_moved(still: vec2<f32>, spacing: f32) -> vec3<f32> {
    var moved = vec3<f32>(0.0);
    let count = u32(water.ocean_grid.w);
    for (var index = 0u; index < count; index++) {
        let wave = water.ocean_waves[index];
        let number = max(length(wave.xy), 1.0e-6);
        let keep = ocean_keep(number, spacing);
        // Waves run longest first: the rest are shorter still.
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

// The corners of a cell's two triangles, split along the diagonal from
// its (1, 0) corner to its (0, 1) corner.
const OCEAN_CORNERS = array<vec2<u32>, 6>(
    vec2<u32>(0u, 0u), vec2<u32>(0u, 1u), vec2<u32>(1u, 0u),
    vec2<u32>(1u, 0u), vec2<u32>(0u, 1u), vec2<u32>(1u, 1u),
);

fn ocean_output(vertex: u32, sector: u32) -> OceanOutput {
    let per_sector = u32(water.ocean_grid.z);
    let around = u32(water.ocean.z);
    let quad = vertex / 6u;
    let corner = OCEAN_CORNERS[vertex % 6u];
    // The last sector's far side is the first's near one: the same angle.
    let segment = (sector * per_sector + quad % per_sector + corner.x) % around;
    let ring = quad / per_sector + corner.y;
    let angle = f32(segment) / f32(around) * OCEAN_TAU;
    let radius = min(ocean_radius(f32(ring)), water.ocean_grid.y);
    let still = vec2<f32>(cos(angle), sin(angle)) * radius;
    // Cells are a ring's share of its circumference across, at least the
    // finest.
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
        * vec4<f32>(output.relative + camera.position.xyz, 1.0)).xyw;
    return temporal;
}

// The sea's surface seen from under it: within Snell's
// window the sky shows through it, bent; beyond, the surface mirrors the
// water below. The water between veils it like everything else.
fn shade_ocean_below(input: OceanOutput) -> WaterShade {
    let world = input.relative + camera.position.xyz;
    let footprint = max(max(length(dpdx(input.still.xy)), length(dpdy(input.still.xy))),
        1.0e-4);
    let up = ocean_normal(input.still.xy, footprint).xyz;
    let ray = stable_normalize(input.relative);
    let bent = refract(ray, -up, 1.333);
    // Beyond the window the surface mirrors the water below it.
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
    return result;
}

// The surface's normal at the still sea's point `still` (less the eye)
// from every wave longer than a few of `footprint`-meter pixels (xyz), and
// how much the waves compress it there (w).
fn ocean_normal(still: vec2<f32>, footprint: f32) -> vec4<f32> {
    var rise = vec2<f32>(0.0);
    var stretch = vec3<f32>(1.0, 0.0, 1.0);
    let count = u32(water.ocean_grid.w);
    for (var index = 0u; index < count; index++) {
        let wave = water.ocean_waves[index];
        let number = max(length(wave.xy), 1.0e-6);
        let keep = saturate(OCEAN_TAU / (number * 2.0 * footprint) - 1.0);
        let phase = dot(wave.xy, still) + wave.w;
        let direction = wave.xy / number;
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

// The sea at `input`, refracting the scene's color when `refracts`.
fn shade_ocean(input: OceanOutput, refracts: bool) -> WaterShade {
    if lighting.underwater.w > 0.0 {
        return shade_ocean_below(input);
    }
    let world = input.relative + camera.position.xyz;
    let still = input.still.xy;
    let time = camera.position.w;
    let view = stable_normalize(-input.relative);
    let world_dx = dpdx(world);
    let world_dy = dpdy(world);
    // Meters of still sea a pixel spans.
    let footprint = max(max(length(dpdx(still)), length(dpdy(still))), 1.0e-4);
    // The waves' tiles ripple the sea's own waves.
    let ripples = water_waves_at(world.xz, world_dx.xz, world_dy.xz, vec3<f32>(1.0));
    let strength = water.ripples.x * water_gusts(world.xz, time);
    // Every wave longer than a few pixels tilts and stretches the surface;
    // shorter waves' slopes roughen it instead.
    let waved = ocean_normal(still, footprint);
    var hidden = 0.0;
    let count = u32(water.ocean_grid.w);
    for (var index = 0u; index < count; index++) {
        let wave = water.ocean_waves[index];
        let number = max(length(wave.xy), 1.0e-6);
        let keep = saturate(OCEAN_TAU / (number * 2.0 * footprint) - 1.0);
        let slope = number * wave.z;
        hidden += (1.0 - keep * keep) * slope * slope * 0.5;
    }
    let normal = normalize(waved.xyz + vec3<f32>(-ripples.x, 0.0, -ripples.y) * strength);
    let compression = waved.w;
    // The ground behind the sea, unless only the sky is: how deep it
    // stands under the surface, the meters the view travels through the
    // water to what it shows and, where the frame refracts the scene, its
    // color.
    var through = 1.0e6;
    var below = 1.0e6;
    var behind: WaterBehind;
    behind.color = vec3<f32>(0.0);
    let pixel = input.clip_position.xy;
    if textureLoad(water_depth, vec2<i32>(pixel), 0) > 0.0 {
        let own = water_ground(pixel);
        below = max(input.relative.y - own.y, 0.0);
        behind.ground = own;
        if refracts {
            behind = water_behind(input.relative, normal, view, pixel, own);
        }
        through = max(length(behind.ground - input.relative), 0.0);
    }
    let transmit = water_transmitted(through);
    // Foam on crests the waves compress and along shores, in patches and
    // streaks drawn out along the sea's wind, evening out with distance.
    let heading = water.origin.xy;
    let streaked = vec2<f32>(dot(world.xz, heading) * 0.35,
        dot(world.xz, vec2<f32>(-heading.y, heading.x)));
    let moving = streaked - vec2<f32>(time * 0.6, 0.0);
    let noise = water_noise(moving * 0.45).x * 0.55 + water_noise(moving * 1.6).x * 0.3
        + water_noise(world.xz * 4.3 + vec2<f32>(time * 0.3, 0.0)).x * 0.15;
    let grain = mix(0.5, noise, saturate(1.6 - footprint * 2.0));
    let shore = 1.0 - smoothstep(0.0, water.foam.y, below);
    let crest = smoothstep(0.0, 1.0, (water.foam.z - compression) / 0.12);
    // The densest foam where crests break hardest, thinning into lace.
    let lace = smoothstep(0.2, 0.75, grain + 0.35 * crest);
    let foam = water.foam.x * max(crest * lace, shore * smoothstep(0.25, 0.65, grain + 0.3 * shore));
    // Calm water is a mirror; the slopes of waves too short to draw spread
    // its highlights, as a rougher surface would, and so do the ripples
    // finer than the pixel.
    let roughness = water_roughness(2.0 * hidden + ripples.z * strength * strength);
    // Sunlight through crests seen against it.
    let crest_height = saturate(input.still.z / max(water.foam.w, 1.0e-3) * 0.5 + 0.5);
    let against = pow(saturate(dot(-view, lighting.sun.xyz)), 4.0) * saturate(lighting.sun.y * 4.0);
    let glow = (water.color.rgb * 3.0 + vec3<f32>(0.0, 0.04, 0.035)) * lighting.sun_color.rgb
        * (against * crest_height * crest_height * crest_height * (1.0 - foam));
    var surface: WaterSurface;
    surface.relative = input.relative;
    surface.world_dx = world_dx;
    surface.world_dy = world_dy;
    surface.normal = normal;
    surface.roughness = roughness;
    surface.foam = foam;
    surface.transmit = transmit;
    surface.glow = glow;
    let lit = water_light(surface);
    // Of what the surface lets through, foam covers what lies below; the
    // sea thins out to nothing where it meets the shore and around what
    // stands in it.
    let fade = smoothstep(0.0, WATER_SHORE, below);
    var result: WaterShade;
    result.shaded = lit.shaded;
    result.coverage = max(1.0 - (1.0 - lit.reflectance) * luminance(transmit), foam) * fade;
    result.behind = behind.color * transmit * ((1.0 - lit.reflectance) * (1.0 - foam)
        * (1.0 - atmospheric_cover(camera.position.xyz, world)));
    result.fade = fade;
    return result;
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
    return vec4<f32>(encoded(shade.shaded.color) * shade.coverage, shade.coverage);
}

@fragment
fn fs_ocean_post(input: OceanOutput) -> SceneTargets {
    let shade = shade_ocean(input, true);
    var targets = scene_targets(shade.shaded);
    targets.color = water_covering(shade);
    return targets;
}

@fragment
fn fs_ocean_temporal(input: OceanTemporalOutput) -> TemporalTargets {
    let shade = shade_ocean(ocean_temporal_output(input), true);
    var targets = temporal_targets(shade.shaded, input.clip_position, input.previous);
    targets.color = water_covering(shade);
    return targets;
}
