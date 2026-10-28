// The sea: an endless surface at a still level that
// Gerstner waves raise, drawn from rings of cells around the eye, a cell
// apart near it and then each ring wider by a cell's share of its
// circumference, out to the sea's reach. Vertices move by the waves longer
// than a few of their cells; pixels shade with every wave spanning two
// pixels along its heading through their own footprint (`ocean_shown`,
// `water_footprint`), so waves running across the view stay drawn far along
// it, and the slopes of the others roughen the surface instead, along their
// headings, as does how far the normal of those drawn turns across a pixel,
// so distant water streaks and glitters rather than aliasing. The CPU's
// queries (`engine_core::OceanWaves`) sum the same waves. It is lit as
// terrains' water is (`water_light`): what lies under it shows through,
// bent where the frame copies the scene's color and dimmed channel by
// channel over the view's path to it (the light reaching it already dims
// through the water above, `atmospheric`). Its foam is white water where
// its crests fold and its waves run up its shores, smooth coverage
// (`meshes/water_foam.wgsl`), and sunlight glows through crests seen
// against it.

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

// How much of a wave `number` radians per meter heading along `way` (a unit
// vector along x and z) a pixel whose footprint on the still sea is
// `footprint` (`water_footprint`) draws: all of it while it spans two
// pixels along its heading, none where it spans one (`water_shown`,
// `water_footprint_toward`), its slopes roughening the surface instead.
fn ocean_shown(number: f32, way: vec2<f32>, footprint: WaterFootprint) -> f32 {
    return water_shown(OCEAN_TAU / number, water_footprint_toward(footprint, way));
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
    let footprint = water_footprint(dpdx(input.still.xy), dpdy(input.still.xy));
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
    result.flow = vec2<f32>(0.0);
    return result;
}

// The sea's waves over the still sea's point `still` (less the eye) as a
// pixel whose footprint on it is `footprint` (`water_footprint`) draws
// them, as `ocean_normal` keeps them (`ocean_shown`): the meters they raise
// it over the still sea (x), the variance of the heights the pixel draws
// (y) and that of all their heights (z), a wave `a` high adding a²/2.
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

// The surface's normal at the still sea's point `still` (less the eye)
// from every wave a pixel whose footprint on it is `footprint`
// (`water_footprint`) draws (`ocean_shown`, xyz), and how much the waves
// compress it there (w).
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

// The sea at `input`, refracting the scene's color when `refracts`.
fn shade_ocean(input: OceanOutput, refracts: bool) -> WaterShade {
    if lighting.underwater.w > 0.0 {
        return shade_ocean_below(input);
    }
    let world = eye_world(input.relative);
    let still = input.still.xy;
    let view = stable_normalize(-input.relative);
    let world_dx = dpdx(world);
    let world_dy = dpdy(world);
    // The pixel's footprint on the still sea (`water_footprint`).
    let footprint = water_footprint(dpdx(still), dpdy(still));
    // Every wave spanning two pixels along its heading tilts and stretches
    // the surface (`ocean_shown`); the others' slopes roughen it instead,
    // along the way each runs (`water_spread`).
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
    // Foam (`meshes/water_foam.wgsl`), fresh white water covering its share
    // of the surface: where the sea's crests fold, where its waves compress
    // the surface past the whitecap threshold its wind sets (`water.foam.x`);
    // and where its waves run up its shores fast enough to tear air in, a
    // wave as high as all of them (`water.foam.y`) running √(g/h) times its
    // height in water h deep. Its whitecaps thin over WATER_FOAM_CAPPING of
    // compression, or over as much as the compression changes across a pixel
    // of the display where that is more, so their edge draws no finer than a
    // pixel.
    let capping = max(WATER_FOAM_CAPPING, fwidth(compression) * material_detail());
    let foam_capping = saturate((water.foam.x - compression) / capping + 0.5); // filter: the whitecaps' edge spread over the pixel, keeping their share
    let foam_contact = water_contact(below, abs(dpdx(below)) + abs(dpdy(below)), footprint,
        water.foam.y * sqrt(WATER_GRAVITY / max(below, WATER_BREAK_THINNEST)), 1.0);
    let foam = water_foam(water_layers(1.0 - (1.0 - foam_capping) * (1.0 - foam_contact)),
        0.0);
    // Calm water is a mirror; the slopes of waves the pixel's footprint does
    // not draw spread its highlights, as a rougher surface would, and so does
    // how far the normal of those drawn turns across the pixel
    // (`water_turning`), along the view and across it (`water_roughness`).
    let roughness = water_roughness(hidden + water_turning(slopes), water_heading(view));
    // Its crests (`ocean_heights`): how near their tops it stands, the
    // height its waves raise it over √2 of the root of the variance of the
    // heights the pixel draws (where a sine's crests stand), and their mean
    // OCEAN_CREST_SHARE for the share of that variance the pixel leaves
    // undrawn; the sun's light crosses them through walls as thick as
    // WATER_CREST_WALL of the root mean square height of all its waves, lit
    // as terrains' water's crests are.
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
    // Of what the surface lets through, foam covers what lies below; the
    // sea thins out to nothing where it meets the shore and around what
    // stands in it.
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
