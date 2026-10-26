// The sea: an endless surface at a still level that
// Gerstner waves raise, drawn from rings of cells around the eye, a cell
// apart near it and then each ring wider by a cell's share of its
// circumference, out to the sea's reach. Vertices move by the waves longer
// than a few of their cells; pixels shade with every wave longer than the
// steady size (STEADY_PIXELS), and the slopes of shorter ones roughen the
// surface instead, as does how far the normal of those drawn turns across a
// pixel, so distant water glitters rather than aliasing. The CPU's queries
// (`engine_core::OceanWaves`) sum the same waves. The wind's tiles ripple
// it with the waves shorter than its own, sized by the same spectrum
// (`water_wind_bands`), and it is lit as terrains' water is
// (`water_light`): what lies under it shows through, bent where the frame
// copies the scene's color and dimmed channel by channel over the view's
// path to it (the light reaching it already dims through the water above,
// `atmospheric`). Its foam is white water where its crests fold and its
// waves run up its shores (`meshes/water_foam.wgsl`), floating on the still
// sea's points, and sunlight glows through crests seen against it.

const OCEAN_TAU: f32 = 6.283185307179586;
// The mean over a sea of how near the top of a crest each point stands,
// its height over √2 of the heights' root mean square (where a sine's
// crests stand) and at most 1, its heights spreading as a Gaussian's:
// (φ(0) − φ(√2))/√2 plus the share standing higher, ≈ 0.257. A pixel
// spanning many crests shows their mean.
const OCEAN_CREST_SHARE: f32 = 0.257;

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

// How much of a wave `number` radians per meter a pixel spanning
// `footprint` meters of the still sea draws: as samples a quarter of the
// steady size apart keep it (`ocean_keep`, `water_steady`), so waves
// shorter than twice the steady size fade and those shorter than it are
// gone, their slopes roughening the surface instead.
fn ocean_shown(number: f32, footprint: f32) -> f32 {
    return ocean_keep(number, 0.25 * water_steady() * footprint);
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
        * vec4<f32>(eye_world(output.relative), 1.0)).xyw;
    return temporal;
}

// The sea's surface seen from under it: within Snell's
// window the sky shows through it, bent; beyond, the surface mirrors the
// water below. The water between veils it like everything else.
fn shade_ocean_below(input: OceanOutput) -> WaterShade {
    let world = eye_world(input.relative);
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

// The sea's waves over the still sea's point `still` (less the eye) as a
// pixel spanning `footprint` meters of it draws them, as `ocean_normal`
// keeps them (`ocean_shown`): the meters they raise it over the still sea
// (x), the variance of the heights the pixel draws (y) and that of all
// their heights (z), a wave `a` high adding a²/2.
fn ocean_heights(still: vec2<f32>, footprint: f32) -> vec3<f32> {
    var heights = vec3<f32>(0.0);
    let count = u32(water.ocean_grid.w);
    for (var index = 0u; index < count; index++) {
        let wave = water.ocean_waves[index];
        let kept = wave.z * ocean_shown(max(length(wave.xy), 1.0e-6), footprint);
        heights += vec3<f32>(kept * sin(dot(wave.xy, still) + wave.w), 0.5 * kept * kept,
            0.5 * wave.z * wave.z);
    }
    return heights;
}

// The surface's normal at the still sea's point `still` (less the eye)
// from every wave a pixel spanning `footprint` meters of it draws
// (`ocean_shown`, xyz), and how much the waves compress it there (w).
fn ocean_normal(still: vec2<f32>, footprint: f32) -> vec4<f32> {
    var rise = vec2<f32>(0.0);
    var stretch = vec3<f32>(1.0, 0.0, 1.0);
    let count = u32(water.ocean_grid.w);
    for (var index = 0u; index < count; index++) {
        let wave = water.ocean_waves[index];
        let number = max(length(wave.xy), 1.0e-6);
        let keep = ocean_shown(number, footprint);
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
    let world = eye_world(input.relative);
    let still = input.still.xy;
    let time = camera.position.w;
    let view = stable_normalize(-input.relative);
    let world_dx = dpdx(world);
    let world_dy = dpdy(world);
    // Meters of still sea a pixel spans.
    let footprint = max(max(length(dpdx(still)), length(dpdy(still))), 1.0e-4);
    // Every wave longer than the steady size tilts and stretches the surface
    // (`ocean_shown`); shorter waves' slopes roughen it instead.
    let waved = ocean_normal(still, footprint);
    var hidden = 0.0;
    let count = u32(water.ocean_grid.w);
    for (var index = 0u; index < count; index++) {
        let wave = water.ocean_waves[index];
        let number = max(length(wave.xy), 1.0e-6);
        let keep = ocean_shown(number, footprint);
        let slope = number * wave.z;
        hidden += (1.0 - keep * keep) * slope * slope * 0.5;
    }
    // The wind's tiles ripple it with the waves shorter than its own
    // shortest (they run longest first), sized by the spectrum its own are
    // drawn from: its wind, gusting, over unlimited fetch and deep water, so
    // the sea and its ripples are one spectrum. Behind terrain upwind, under
    // its cliffs, that wind slows and its ripples calm (`terrain_shelter`).
    let shortest = OCEAN_TAU / max(length(water.ocean_waves[max(count, 1u) - 1u].xy), 1.0e-6);
    let wind = water_wind_speed(world, time);
    let gains = water_wind_bands(wind, WATER_OPEN_FETCH, WATER_OPEN_DEPTH, shortest);
    let ripples = water_wind_waves(world.xz, world_dx.xz, world_dy.xz, gains);
    let slopes = ripples.slopes - waved.xz / max(waved.y, 1.0e-3);
    let normal = normalize(vec3<f32>(-slopes.x, 1.0, -slopes.y));
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
        below = max(input.relative.y - own.y, 0.0); // optics: what the view meets behind the pixel
        behind.ground = own;
        if refracts {
            behind = water_behind(input.relative, normal, view, pixel, own,
                vec3<f32>(0.0, 1.0, 0.0));
        }
        through = max(length(behind.ground - input.relative), 0.0); // optics: the view's way through the water
    }
    let transmit = water_transmitted(through, 0.0);
    // Foam (`meshes/water_foam.wgsl`): white water where the sea's crests
    // fold, where its waves compress the surface past the whitecap threshold
    // its wind sets (`water.foam.x`) and where its tiles' slopes pass the
    // breaking slope, on their crests where the pixel draws them (their
    // heights' rank); and where its waves run up its shores fast enough to
    // tear air in, a wave as high as all of them (`water.foam.y`) running
    // √(g/h) times its height in water h deep. It floats on the still sea's
    // points, so it rides the waves, and its relief tilts the normal it is
    // lit by.
    let foam_capping = saturate((water.foam.x - compression) / WATER_FOAM_CAPPING + 0.5);
    let foam_crest = 1.0 / (1.0 + exp(-clamp(ripples.height
        * inverseSqrt(max(ripples.heights, 1.0e-12)) / WATER_LOGISTIC, -30.0, 30.0)));
    let foam_folding = water_folding(dot(gains, gains), ripples.hidden, foam_crest, vec2<f32>(0.0));
    let foam_contact = water_contact(below, abs(dpdx(below)) + abs(dpdy(below)), footprint,
        water.foam.y * sqrt(WATER_GRAVITY / max(below, WATER_BREAK_THINNEST)));
    let foam_floats = eye_world(vec3<f32>(still.x, 0.0, still.y)).xz;
    let froth = water_froth(foam_floats, dpdx(foam_floats), dpdy(foam_floats),
        vec2<f32>(1.0 - (1.0 - foam_capping) * (1.0 - foam_folding) * (1.0 - foam_contact), 0.0),
        vec2<f32>(0.0), 0.0, time, 0.0, detail_span(world, vec3<f32>(0.0, 1.0, 0.0)).y);
    let foam = froth.cover;
    let foam_tilt = water_foam_tilt(froth.height, normal, world_dx, world_dy);
    // Calm water is a mirror; the slopes of waves too short to draw spread
    // its highlights, as a rougher surface would, and so do the ripples
    // finer than the steady size and how far the normal of those drawn
    // turns across the pixel (`water_turning`).
    let roughness = water_roughness(hidden + ripples.hidden + water_turning(normal));
    // Its crests (`ocean_heights`): how near their tops it stands, the
    // height its waves and the tiles' raise it over √2 of the root of the
    // variance of the heights the pixel draws (where a sine's crests
    // stand), and their mean OCEAN_CREST_SHARE for the share of that
    // variance the pixel leaves undrawn; the sun's light crosses them
    // through walls as thick as WATER_CREST_WALL of the root mean square
    // height of all its waves, lit as terrains' water's crests are.
    let swell = ocean_heights(still, footprint);
    let resolved = swell.y + ripples.heights;
    let peak = mix(OCEAN_CREST_SHARE,
        saturate((swell.x + ripples.height) * inverseSqrt(max(2.0 * resolved, 1.0e-12))),
        saturate(resolved / max(swell.z + ripples.heights, 1.0e-12)));
    var surface: WaterSurface;
    surface.relative = input.relative;
    surface.world_dx = world_dx;
    surface.world_dy = world_dy;
    surface.normal = normal;
    surface.roughness = roughness;
    surface.plane = vec3<f32>(0.0, 1.0, 0.0);
    surface.foam = foam;
    surface.foam_tilt = foam_tilt;
    surface.foam_shade = froth.shade;
    surface.foam_thin = froth.thin;
    surface.scatter = water.color.rgb;
    surface.transmit = transmit;
    surface.crest = peak;
    surface.crest_wall = water_extinction(0.0)
        * (WATER_CREST_WALL * sqrt(swell.z + ripples.heights));
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
    result.wind = wind;
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
