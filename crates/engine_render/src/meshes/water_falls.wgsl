// Waterfalls' jets (`water_falls.rs`): each jet's water through the air,
// from where it leaves its lip to where it meets the water or rock below,
// shown as its physics leaves it and as nothing else. Each vertex holds how
// far across the sheet it stands, how wide its water lies there and how far
// its edges have spread, the sheet's thickness, how white its water has
// turned and how far the air has broken it up, so all of it changes
// smoothly across the sheet. Its water covers the sheet but where its edges
// spread, thinning out across them (`fall_water`). Water that has taken in
// air, the air has broken up or its edges have spread into spray is white:
// its bubbles and drops scatter light along the view's way through the
// sheet, sending back what they do not let through (`fall_white`), so deep
// white water hides what lies behind it and scatters the sun's and sky's
// light, and thin white water glows with the sun behind it. Its clear water
// lets through what lies behind it, as the water's own clarity takes it,
// and mirrors the sky and the sun over it. Nothing moves the sheet, so its
// normal is the sheet's own. It shows only where it stands clear of the
// water under it, so it joins the river at its lip and the pool it lands
// in, and it thins out where it meets the rock, as the view's way through
// it shortens.
struct FallInput {
    @location(0) position: vec3<f32>,
    // The sheet's face, turned out from the rock it falls from.
    @location(1) normal: vec3<f32>,
    // How far across the sheet it stands from the jet's middle (meters,
    // either way), half the width of the water there, the meters each of
    // the water's edges has spread and the sheet's thickness (meters).
    @location(2) sheet: vec4<f32>,
    // How white its water has turned and how far the air has broken it up
    // (0 to 1 each).
    @location(3) state: vec2<f32>,
    // How much of the sky it sees (0 to 1), the ground around hiding the
    // rest.
    @location(4) sky: f32,
}

struct FallOutput {
    @builtin(position) clip_position: vec4<f32>,
    // The point less the eye.
    @location(0) relative: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) sheet: vec4<f32>,
    @location(3) state: vec2<f32>,
    @location(4) sky: f32,
}

struct FallTemporalOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) relative: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) sheet: vec4<f32>,
    @location(3) state: vec2<f32>,
    @location(4) sky: f32,
    @location(5) previous: vec3<f32>,
}

// Nothing moves the sheet: it stands where its vertices do.
fn fall_output(input: FallInput) -> FallOutput {
    var output: FallOutput;
    output.clip_position = camera.view_projection * vec4<f32>(input.position, 1.0);
    output.relative = input.position - camera.position.xyz;
    output.normal = input.normal;
    output.sheet = input.sheet;
    output.state = input.state;
    output.sky = input.sky;
    return output;
}

@vertex
fn vs_fall(input: FallInput) -> FallOutput {
    return fall_output(input);
}

// The sheet stands still: its motion is the camera's.
@vertex
fn vs_fall_temporal(input: FallInput) -> FallTemporalOutput {
    let output = fall_output(input);
    var temporal: FallTemporalOutput;
    temporal.clip_position = output.clip_position;
    temporal.relative = output.relative;
    temporal.normal = output.normal;
    temporal.sheet = output.sheet;
    temporal.state = output.state;
    temporal.sky = output.sky;
    temporal.previous = (camera.previous_view_projection * vec4<f32>(input.position, 1.0)).xyw;
    return temporal;
}

fn fall_temporal_output(input: FallTemporalOutput) -> FallOutput {
    var output: FallOutput;
    output.clip_position = input.clip_position;
    output.relative = input.relative;
    output.normal = input.normal;
    output.sheet = input.sheet;
    output.state = input.state;
    output.sky = input.sky;
    return output;
}

// White falling water: how much of the light falling on it it sends back
// where too deep to see through, its bubbles' and drops' faces scattering
// the light again and again and its water taking a little at each pass;
// how much light it scatters a meter where all of it has turned white,
// half of it air (falling and skimming water's white water holds 30 to 60 %
// air, Chanson), whether in bubbles or in drops 3 mm across, each taking
// twice its cross section out of the light, 3ε/2r a meter (as water's
// bubbles do, `water.wgsl`'s WATER_BUBBLES); and how much of what its
// bubbles and drops scatter goes on forward, the asymmetry g of those much
// larger than light's wavelengths (about 0.85, as clouds' droplets').
const FALL_ALBEDO: f32 = 0.8;
const FALL_SCATTERING: f32 = 500.0;
const FALL_FORWARD: f32 = 0.85;

// How much of the sheet's area its water covers `across` meters from the
// jet's middle (either way), its water lying `half` meters each side of it
// and its edges having spread `spread` meters (x), and how much of that
// water its spreading edges have broken into spray (y). The edges spread
// over a band twice as wide as they have spread, about the water's edge,
// its coherent core narrowing as they widen (Horeni): across the band its
// water thins out evenly from all of it to none, so the band holds the
// water its edges spread, all of it spray; a band wider than the water
// holds no more than the water's whole width spread across it. The band is
// no narrower than the pixel spans across the sheet, so the edge stays as
// sharp as the pixel resolves it, the pixel averaging the water and the
// spray it spans.
fn fall_water(across: f32, half: f32, spread: f32) -> vec2<f32> {
    let band = max(max(2.0 * spread, fwidth(across)), 1.0e-6); // filter: the water's edge as sharp as a pixel resolves it
    let inside = (half - abs(across)) / band;
    let covered = min(saturate(0.5 + inside), 2.0 * half / band);
    let spray = saturate(0.5 - inside) * min(2.0 * spread / band, 1.0);
    return vec2<f32>(covered, spray);
}

// How the sheet's white water `white` (0 to 1) passes light along the
// view's way through it, `path` meters: its bubbles and drops scatter
// FALL_SCATTERING of it a meter where all of it is white, so much of it
// that what they scatter runs on in two streams, forward and back (Bohren's
// two-stream, of light scattered and not taken): they let through
// 1 / (1 + (1 - g)τ/2) of it and send back the rest. The share of what lies
// behind they pass unscattered (x), and the shares of the light falling on
// the sheet they send back (y) and pass on scattered (z).
fn fall_white(path: f32, white: f32) -> vec3<f32> {
    let depth = FALL_SCATTERING * white * path;
    let passed = 1.0 / (1.0 + 0.5 * (1.0 - FALL_FORWARD) * depth);
    let unscattered = exp(-depth);
    return vec3<f32>(unscattered, 1.0 - passed, passed - unscattered);
}

// How much of a jet `thickness` meters thick shows at `world` (world
// meters) over the water the pass shows under it: none within its
// thickness of that water's surface, all of it from twice its thickness
// over it, and all of it where no water lies under it. That water is the
// water as a pixel there resolves level water seen from the detail eye
// (`water_pixel`), live or settled, none lying past where the samples
// holding its surface end (WATER_HELD).
fn fall_clear(world: vec3<f32>, thickness: f32) -> f32 {
    let at = water_pixel(world.xz - terrain.origin.xz,
        detail_resolved(detail_span(world, vec3<f32>(0.0, 1.0, 0.0)).xy));
    let surface = terrain.origin.y + at.surface;
    let under = select(0.0, 1.0, at.depth > 0.0 && at.held >= WATER_HELD);
    return mix(1.0, smoothstep(thickness, 2.0 * thickness, world.y - surface), under);
}

// A fall's light, how much of the pixel it shows and the share of what lies
// behind it it hides.
struct FallShade {
    shaded: Shaded,
    shown: f32,
    coverage: f32,
}

fn shade_fall(input: FallOutput) -> FallShade {
    let world = eye_world(input.relative);
    let thickness = input.sheet.w;
    // Within its thickness of the water under it, that water shows instead.
    let clear = fall_clear(world, thickness);
    if clear < 0.01 {
        discard;
    }
    let view = stable_normalize(-input.relative);
    let world_dx = dpdx(world);
    let world_dy = dpdy(world);
    let normal = normalize(input.normal);
    // The face the eye sees, and how squarely.
    let facing = dot(normal, view); // optics: the face of the sheet the eye sees
    let seen = select(-normal, normal, facing >= 0.0);
    let squarely = max(abs(facing), 0.05);
    // The view's way through the sheet, its thickness as squarely as the
    // eye sees it, as far as it runs before what stands behind the sheet:
    // where the sheet meets the rock, less and less of it lies before it.
    let crossing = thickness / squarely;
    let behind = water_ground(input.clip_position.xy);
    let held = saturate((length(behind) - length(input.relative)) / crossing); // optics: what the view meets behind the sheet
    let path = crossing * held;
    // Its water across the sheet (`fall_water`), and the share of it white:
    // water that has taken in air, the air has broken up, or its edges have
    // spread into spray.
    let shares = fall_water(input.sheet.x, input.sheet.y, input.sheet.z);
    let white = 1.0 - (1.0 - input.state.x) * (1.0 - input.state.y) * (1.0 - shares.y);
    let shown = shares.x * held * clear;
    if shown < 0.01 {
        discard;
    }
    // Its clear water lets through what lies behind it as the water's own
    // clarity takes it over the view's way (`water_transmitted`, by the
    // river's water's settings), as far as its white water's bubbles and
    // drops pass it unscattered (`fall_white`); they send back and pass on
    // scattered the rest.
    let scattering = fall_white(path, white);
    let clarity = water_transmitted(path, 0.0);
    let coverage = shares.x * clear * (1.0 - luminance(clarity) * scattering.x);
    // The sun's light on the sheet as its shadows leave it, through the
    // face it stands before: in front of the face the eye sees, its clear
    // water's body and its white water send it back; behind it, its white
    // water passes on what it scatters, so thin white water glows with the
    // sun behind it, as evenly every way as what it sends back.
    let sun = lighting.sun.xyz;
    let backlit = dot(sun, seen) < 0.0;
    var sunlight = vec3<f32>(0.0);
    if sun.y > 0.0 {
        sunlight = lighting.sun_color.rgb * sun_visibility(world, select(seen, -seen, backlit),
            world_dx, world_dy, length(input.relative), false);
    }
    var glow = vec3<f32>(0.0);
    if backlit {
        glow = sunlight * (FALL_ALBEDO * scattering.z * -dot(sun, seen));
    }
    var point: SurfacePoint;
    point.albedo = vec3<f32>(FALL_ALBEDO);
    point.opacity = 1.0;
    point.emission = 0.0;
    point.emissive = vec3<f32>(0.0);
    point.occlusion = input.sky;
    point.bounce = vec3<f32>(0.0);
    // Its clear water's face is as smooth as still water's
    // (WATER_SMOOTHEST): nothing moves the sheet. White water's faces turn
    // every way, so what they mirror is part of what it sends back.
    point.roughness = WATER_SMOOTHEST;
    point.alpha = point.roughness * point.roughness;
    point.f0 = vec3<f32>(0.02 * (1.0 - white));
    point.diffuse_color = vec3<f32>(FALL_ALBEDO * scattering.y);
    point.geometric = seen;
    point.normal = seen;
    point.distance = length(input.relative);
    point.view = view;
    point.n_v = max(abs(facing), 1.0e-4);
    point.world = world;
    point.world_dx = world_dx;
    point.world_dy = world_dy;
    point.shadow_offset = vec3<f32>(0.0);
    point.on_terrain = false;
    point.sun_shade = 0.0;
    // Clear water's body scatters back what it takes of the light the
    // surface lets into it, as the river's does (`water_body`,
    // `water_scattered`), along the way its white water's bubbles and drops
    // leave unscattered; its Fresnel reflection lies over it. The ground
    // around hides as much of the sky from it as it hides from the water.
    let reflectance = luminance(environment_brdf(point.f0, point.roughness, point.n_v));
    let body = water_body(0.0) * (vec3<f32>(1.0) - clarity)
        * water_scattered(view, sunlight, seen)
        * ((1.0 - reflectance) * scattering.x * input.sky);
    var shade: FallShade;
    shade.shaded = finish(point, body + glow);
    shade.shown = shown;
    shade.coverage = coverage;
    return shade;
}

// Fall shade `shade` at `pixel` with the shares screen-space reflections
// take of what the pixel shows (`finish`'s of its own light): what it
// mirrors makes up the less of all the pixel shows the more of the opaque
// scene behind it (`water_scene`, which frames drawing the effects' targets
// copy) it lets through, and the scene mirrored in its place goes as far as
// it shows.
fn fall_mirroring(shade: FallShade, pixel: vec2<f32>) -> FallShade {
    var result = shade;
    let own = luminance(shade.shaded.color) * shade.shown;
    let behind = luminance(textureLoad(water_scene, vec2<i32>(pixel), 0).rgb)
        * (1.0 - shade.coverage);
    result.shaded.mirrored *= own / max(own + behind, 1.0e-6);
    result.shaded.reflect *= shade.shown;
    return result;
}

// Falls blend premultiplied over what they fall past, as water does: their
// light as far as they show, what lies behind them as far as they let it
// through.
@fragment
fn fs_fall(input: FallOutput) -> @location(0) vec4<f32> {
    let shade = shade_fall(input);
    return vec4<f32>(encoded(shade.shaded.color, input.clip_position.xy) * shade.shown,
        shade.coverage);
}

@fragment
fn fs_fall_post(input: FallOutput) -> SceneTargets {
    let pixel = input.clip_position.xy;
    let shade = fall_mirroring(shade_fall(input), pixel);
    var targets = scene_targets(shade.shaded, pixel);
    targets.color = vec4<f32>(stored(shade.shaded.color, pixel) * shade.shown, shade.coverage);
    return targets;
}

@fragment
fn fs_fall_temporal(input: FallTemporalOutput) -> TemporalTargets {
    let pixel = input.clip_position.xy;
    let shade = fall_mirroring(shade_fall(fall_temporal_output(input)), pixel);
    var targets = temporal_targets(shade.shaded, input.clip_position, input.previous);
    targets.color = vec4<f32>(stored(shade.shaded.color, pixel) * shade.shown, shade.coverage);
    return targets;
}
