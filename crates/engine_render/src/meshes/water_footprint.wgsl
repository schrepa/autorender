// How a pixel reads water's patterns, terrains' water's (`water_flow.wgsl`,
// `water_stream.wgsl`, `water_foam.wgsl`) and the sea's (`ocean.wgsl`):
// through its own footprint (`water_footprint`), as every texture reads,
// each computed part drawn while it spans two pixels along the way it
// varies (`water_shown`), so waves running across the view stay drawn far
// along it, while those running along it average out where the footprint
// spans them. What it averages of their slopes hides in their covariance,
// as variances add in a microfacet distribution (LEAN mapping, Olano and
// Baker 2010), which roughens the water along and across the view as it
// projects there (Bruneton, Neyret and Holzschuch 2010), the more along the
// view the more the footprint stretches, so far water streaks and glints
// along the view; nothing fades by distance. Where nothing roughens it, the
// water is a calm mirror at its smoothest (`water_roughness`).

// The least and the most roughness water shades with, each way: as smooth
// as every surface may be (`meshes/shading.wgsl`), where GGX's lobe stays
// finite in f32 though rounding takes a unit half vector's cosine with the
// normal an ulp or two past 1; and the most its waves' hidden slopes
// roughen it.
const WATER_SMOOTHEST: f32 = 0.03;
const WATER_ROUGHEST: f32 = 0.6;

// Waves as a pixel reads them: their slopes (meters a meter), the
// covariance of the slopes the pixel hides (`water_spread`), their height
// (meters) and the variance of the heights it resolves, summed over sources
// as independent fields: slopes and heights add, and so do variances.
struct WaterRead {
    slopes: vec2<f32>,
    hidden: vec3<f32>,
    height: f32,
    heights: f32,
}

// The covariance of slopes (xx, zz and xz, along x and z) that slopes
// `slopes` (meters a meter) hold as one sample of their spread: their outer
// product, whose trace is their square.
fn water_spread(slopes: vec2<f32>) -> vec3<f32> {
    return vec3<f32>(slopes.x * slopes.x, slopes.y * slopes.y, slopes.x * slopes.y);
}

// The covariance of slopes whose variance `variance` spreads alike every
// way: half of it along each axis.
fn water_spread_even(variance: f32) -> vec3<f32> {
    return vec3<f32>(0.5 * variance, 0.5 * variance, 0.0);
}

// The variance along the unit vector `way` (along x and z) of slopes of
// covariance `spread`.
fn water_spread_toward(spread: vec3<f32>, way: vec2<f32>) -> f32 {
    return spread.x * way.x * way.x + spread.y * way.y * way.y + 2.0 * spread.z * way.x * way.y;
}

// A pixel's footprint as water's patterns filter it: how its point changes
// across a pixel of the display (`dx` and `dy`, its derivatives scaled by
// the render size's share of the display, `material_detail`, as material
// textures' mip bias samples them: meters along x and z, or a pattern's own
// units), by which its texture reads filter and its computed patterns draw
// (`water_footprint_toward`, `water_shown`); the meters its short side and
// its long side span (`across` and `along`, the singular values of those
// derivatives); and the meters it resolves (`detail_resolved` of its sides:
// its short side, or its long side over DETAIL_ANISOTROPY where it is more
// stretched, as a texture's levels and anisotropy resolve it), by which
// features varying every way draw, as a read of their texture would
// resolve them.
struct WaterFootprint {
    dx: vec2<f32>,
    dy: vec2<f32>,
    across: f32,
    along: f32,
    resolved: f32,
}

// The footprint of a pixel whose point changes by `dx` and `dy` a pixel.
fn water_footprint(dx: vec2<f32>, dy: vec2<f32>) -> WaterFootprint {
    let detail = material_detail();
    var footprint: WaterFootprint;
    footprint.dx = dx * detail;
    footprint.dy = dy * detail;
    let squares = dot(footprint.dx, footprint.dx) + dot(footprint.dy, footprint.dy);
    let area = abs(footprint.dx.x * footprint.dy.y - footprint.dx.y * footprint.dy.x);
    let sum = sqrt(squares + 2.0 * area);
    let difference = sqrt(max(squares - 2.0 * area, 0.0));
    footprint.across = 0.5 * (sum - difference);
    footprint.along = 0.5 * (sum + difference);
    footprint.resolved = detail_resolved(vec2<f32>(footprint.across, footprint.along));
    return footprint;
}

// The meters footprint `footprint` (`water_footprint`) spans along the unit
// vector `way` (along x and z, or a pattern's own axes): how far a pattern
// varying that way moves across the pixel, the length of the pixel's
// derivatives projected on it, from the footprint's short side, where `way`
// runs across the view, to its long side.
fn water_footprint_toward(footprint: WaterFootprint, way: vec2<f32>) -> f32 {
    return length(vec2<f32>(dot(footprint.dx, way), dot(footprint.dy, way)));
}

// How much of a computed pattern's part `meters` long along the way it
// varies a pixel draws, its footprint spanning `span` meters that way
// (`water_footprint_toward`; the size it resolves for parts varying every
// way): all of it while it spans two pixels or more, none where it spans
// one, so nothing finer than the Nyquist limit draws, as mipmaps keep a
// texture's; what it leaves of the part's slopes hides in their variance.
fn water_shown(meters: f32, span: f32) -> f32 {
    return saturate(meters / max(span, 1.0e-9) - 1.0);
}

// The covariance (`water_spread`) water's drawn slopes `slopes` (meters a
// meter) add to those it hides by turning across a pixel: what the waves it
// draws vary within the pixel, whose glints its samples would otherwise
// catch in one frame and miss in the next (geometric specular antialiasing,
// Kaplanyan, Hill, Patney and Lefohn 2016: twice the pixel footprint's
// variance SPECULAR_VARIANCE times the outer products of the slopes' change
// across and down the pixel, so a pixel drawn out along the view widens the
// lobe along the view; scaled to at most SPECULAR_LIMIT of their square
// before it joins the slopes' as Tokuyoshi and Kaplanyan 2019 clamp it),
// every surface's kernel (`widened`). Its derivatives need it called in
// uniform control flow.
fn water_turning(slopes: vec2<f32>) -> vec3<f32> {
    let turning = 2.0 * SPECULAR_VARIANCE
        * (water_spread(dpdx(slopes)) + water_spread(dpdy(slopes)));
    return turning * min(SPECULAR_LIMIT / max(turning.x + turning.y, 1.0e-12), 1.0);
}

// The roughness along and across the view (x and y) of water whose waves'
// slopes vary within a pixel by the covariance `hidden` (`water_spread`:
// those it hides and those it draws turning across it, `water_turning`),
// seen heading along `heading` (a unit vector along x and z: the view laid
// on the water, away from the eye): the ellipse of that covariance
// projected into the view, its variance along the heading and across it
// (Bruneton, Neyret and Holzschuch 2010), sets each way's GGX alpha as the
// root of twice that variance, as a round lobe's square alpha is its
// slopes' whole variance; and each way's roughness is its alpha's root,
// from WATER_SMOOTHEST, where none vary, to WATER_ROUGHEST.
fn water_roughness(hidden: vec3<f32>, heading: vec2<f32>) -> vec2<f32> {
    let variances = vec2<f32>(water_spread_toward(hidden, heading),
        water_spread_toward(hidden, vec2<f32>(-heading.y, heading.x)));
    return clamp(sqrt(sqrt(2.0 * max(variances, vec2<f32>(0.0)))), vec2<f32>(WATER_SMOOTHEST),
        vec2<f32>(WATER_ROUGHEST));
}
