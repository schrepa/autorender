// Foam on terrains' water and the sea: the water's own, from its state
// alone. The water carries the foam its flow churns up, settling or
// simulated (`water_sim.wgsl`'s `sim_churn`: where it loses energy, plunges
// or jumps, along its eddy lines and where it converges, fading to 1/e in
// WATER_FOAM_LIFE), drained as it ages; and it turns white, its foam fresh,
// where it breaks, where its crests fold (`water_folding`: where the wind's
// waves have folded, the foam lingering for seconds, and where other waves
// pass the breaking slope) and where it runs fast against what stands in it
// (`water_contact`). Each is a share of the surface, and one bubble texture
// shows them all (`water_froth`): falling water's map's bubbles and the
// walls between them, carried only as the water moves them, which a pixel
// covers where they pass the threshold its share sets, so a little foam
// shows as sparse lace, more fills the cells as froth and the most heaps
// solid, and bubbles finer than the steady size (STEADY_PIXELS) show the
// share itself.

// Falling water's map's foam (`water_fall_map`'s b and a): the cells of its
// bubbles across it (`engine_tables::falls::BUBBLES`), FALL_MAP_STRETCH
// times as many along it, which water reads that many times longer, so they
// stand round; and how far its bubbles' walls (x) and bubbles (y) spread
// about their means, about as its cellular noise leaves them.
const FALL_MAP_BUBBLES: f32 = 16.0;
const FALL_MAP_FOAM_SPREAD: vec2<f32> = vec2<f32>(0.34, 0.18);
// The meters across foam's bubbles: the cells a raft of foam drains into, a
// few centimeters across; and how many times finer its second octave of
// them is, as a foam's bubbles range in size, each octave half its grain.
const WATER_FOAM_BUBBLE: f32 = 0.0625;
const WATER_FOAM_FINER: f32 = 3.0;
// How long a set of foam's carried bubbles lives (`water_froth`): as long as
// the flow's shear takes to strain it by WATER_STRAIN as the waves', its bubbles
// drawn out by two fifths at either end of its life before it starts over
// (Neyret's latency); from WATER_FOAM_SHORTEST seconds, below which it would
// churn faster than white water does, to the life of the foam it shows
// (`engine_terrain::shallow_water::FOAM_LIFE`).
const WATER_FOAM_SHORTEST: f32 = 1.0;
const WATER_FOAM_LIFE: f32 = 20.0;
// The share of the surface below which foam shows nothing: less than an
// 8-bit target's step.
const WATER_FOAM_LEAST: f32 = 1.0 / 255.0;
// The scale of the logistic distribution of spread 1, √3/π: foam takes its
// grain as logistic, whose share past a threshold and mean over it have
// closed forms.
const WATER_LOGISTIC: f32 = 0.5513289;
// The share of the light under it a single layer of wet bubbles hides,
// each layer over it hiding as much of what those below let through; the
// share of white water's layers the foam the water carries keeps, its
// bubbles drained into the walls of dry foam as it ages (fresh white water
// is wet foam, round bubbles churning: Weta's foam, 2022); and how much of
// the light heaped foam's hollows lose to the bubbles around them.
const WATER_FOAM_FILM: f32 = 0.5;
const WATER_FOAM_DRAINED: f32 = 0.5;
const WATER_FOAM_HOLLOW: f32 = 0.35;
// The meters foam rides toward the waves' crests for each unit of their
// slope: a wave moves what floats on it by its slope over its wave number
// (Gerstner's), here running water's turbulent waves', half a meter long,
// so foam gathers on their crests (Vlachos, Portal 2).
const WATER_FOAM_RIDE: f32 = 0.08;
// The meters a second over which water running against what stands in it
// tears air in, from the first to the second (Ervine, McKeogh and Elsawy
// 1980: about the meter a second past which water draws air in).
const WATER_FOAM_TEARING: vec2<f32> = vec2<f32>(0.7, 1.3);
// The compression over which a whitecap thins from all foam to none about
// the threshold the sea's wind sets (`water.foam.x`): from its fresh middle
// to its aging edge.
const WATER_FOAM_CAPPING: f32 = 0.12;

// The share of a noise's variance a filter spanning `footprint` meters
// resolves, its features `size` meters across: about 1 / (1 + (footprint /
// size)^2), the rest averaging out within the span as filtering averages
// it.
fn water_resolved_share(footprint: f32, size: f32) -> f32 {
    let ratio = footprint / size;
    return 1.0 / (1.0 + ratio * ratio);
}

// The share of water's surface its folding crests whiten: where the wind's
// tiles' waves have folded (a) and where the waves no tile follows break
// (b), two independent shares, 1 − (1 − a)(1 − b). The tiles' waves have
// folded where the sum of each one's gain times its faded compression
// passes 1 (`whitecaps`: its mean over the pixel and the variance its reads
// average away, `water_wind_whitecaps`; none for water reading no tiles'
// whitecaps), their foam lingering where a crest broke as the crest runs
// on: a is the sum's share past 1, taken as logistic of its spread, which
// spans what the reads average and how much the sum changes across the
// steady size (`water_steady`), so its edge draws no finer than the steady
// size and far water whitens as near water does on average. The other
// waves' slopes, resolved and hidden, spread `spread` (their variance, σ²),
// of which the pixel hides `hidden`: a Gaussian surface's slopes, their
// sizes Rayleigh-distributed, pass the breaking slope s_b (WATER_STEEPEST)
// over b = exp(−s_b²/σ²) of it. Where the pixel resolves those waves, that
// share lies on their crests, the points standing nearest their tops
// (`crest`, taken as the share of the surface standing lower); where it
// hides them, it is the share itself. A share too small to show whitens
// nothing, however near its crest's top a point stands. Its derivatives
// need it called in uniform control flow.
fn water_folding(spread: f32, hidden: f32, crest: f32, whitecaps: vec2<f32>) -> f32 {
    let change = fwidth(whitecaps.x) * water_steady();
    let edge = sqrt(whitecaps.y + change * change / 12.0) * WATER_LOGISTIC;
    let folded = 1.0 / (1.0 + exp(-clamp((whitecaps.x - 1.0) / max(edge, 1.0e-6), -30.0, 30.0)));
    let variance = max(spread, 1.0e-6);
    let share = exp(-WATER_STEEPEST * WATER_STEEPEST / variance);
    let on_crests = saturate((crest - 1.0) / max(share, 1.0e-6) + 1.5)
        * saturate(share / WATER_FOAM_LEAST);
    let breaking = mix(share, on_crests, saturate(1.0 - hidden / variance));
    return 1.0 - (1.0 - folded) * (1.0 - breaking);
}

// How strongly water foams against what stands in it behind the pixel
// (Sea of Thieves' depth test), its surface `standing` meters over it,
// changing by `change` across a pixel spanning `footprint` meters, where it
// runs at `speed` meters a second (its current or its eddies, whichever is
// faster): water climbs what it runs against by its velocity head U²/2g
// (Bernoulli), whitest where it meets it, and tears air in there as it
// passes WATER_FOAM_TEARING, so calm water foams at nothing it touches. A
// band rising less than what stands behind the pixel rises across the
// steady size (`water_steady`) spreads its foam over that rise, keeping its
// share, so no line of foam draws finer than the steady size. Only solid
// ground counts: behind blades and twigs what the pixel sees jumps from
// pixel to pixel faster than any bank rises (WATER_GRADIENT_RANGE a meter).
fn water_contact(standing: f32, change: f32, footprint: f32, speed: f32) -> f32 {
    let tearing = smoothstep(WATER_FOAM_TEARING.x, WATER_FOAM_TEARING.y, speed);
    if tearing <= 0.0 {
        return 0.0;
    }
    let head = speed * speed / (2.0 * WATER_GRAVITY);
    let band = max(head, change * water_steady());
    let rise = change / max(footprint, 1.0e-6);
    let solid = 1.0 - smoothstep(WATER_GRADIENT_RANGE, 2.0 * WATER_GRADIENT_RANGE, rise);
    return tearing * solid * saturate(1.0 - standing / band) * (head / band);
}

// Foam's grain at `p` (meters along x and z) for a pixel changing it by
// `dx` and `dy`, read at the steady size (`water_steady`): falling water's
// map's bubbles' walls and bubbles (`water_fall_map`'s b and a),
// WATER_FOAM_BUBBLE meters across and, if `finer`, WATER_FOAM_FINER times
// finer, read at the three corners of the triangle of a grid of patches
// half the map across holding it, each read turned to run along `heading`
// about its corner's middle and shifted at random (salted by `salt`), and
// blended by their weights over the root of their squares' sum, which keeps
// their spread (tiling and blending, Heitz and Neyret), so no patch shows
// the map repeat. Turning each read about a middle nearby keeps it from
// shearing where the heading turns. The larger bubbles' walls (x) and
// bubbles (y) and the finer's (zw, none unless `finer`), each as its
// deviation from the map's means `mean`, in its spreads.
fn water_bubbles(p: vec2<f32>, heading: vec2<f32>, dx: vec2<f32>, dy: vec2<f32>,
        mean: vec2<f32>, finer: bool, salt: u32) -> vec4<f32> {
    let side = vec2<f32>(-heading.y, heading.x);
    // The meters across and along the map as foam reads it, and between the
    // corners of the patches' grid.
    let map = vec2<f32>(1.0, FALL_MAP_STRETCH) * (WATER_FOAM_BUBBLE * FALL_MAP_BUBBLES);
    let spacing = 0.5 * map.x;
    // At the steady size (`water_steady`).
    let steady = water_steady();
    let turned_dx = vec2<f32>(dot(dx, side), dot(dx, heading)) * steady / map;
    let turned_dy = vec2<f32>(dot(dy, side), dot(dy, heading)) * steady / map;
    let triangle = water_triangle(p / spacing);
    var corners = triangle.corners;
    let weights = triangle.weights;
    var grain = vec4<f32>(0.0);
    for (var corner = 0u; corner < 3u; corner++) {
        let skewed = vec2<f32>(corners[corner]);
        let middle = vec2<f32>(skewed.x, (skewed.y + 0.57735027 * skewed.x) * 0.8660254)
            * spacing;
        let turned = vec2<f32>(dot(p - middle, side), dot(p - middle, heading)) / map;
        let larger = textureSampleGrad(water_fall_map, water_wave_sampler,
            turned + water_corner_offset(corners[corner], salt), turned_dx, turned_dy).ba;
        var smaller = mean;
        if finer {
            smaller = textureSampleGrad(water_fall_map, water_wave_sampler,
                turned * WATER_FOAM_FINER
                    + water_corner_offset(corners[corner], salt ^ 0x5bd1e995u),
                turned_dx * WATER_FOAM_FINER, turned_dy * WATER_FOAM_FINER).ba;
        }
        grain += vec4<f32>(larger - mean, smaller - mean) * weights[corner];
    }
    return grain * inverseSqrt(dot(weights, weights))
        / vec4<f32>(FALL_MAP_FOAM_SPREAD, FALL_MAP_FOAM_SPREAD);
}

// Foam over a pixel: the share of the surface it hides (`cover`), the
// meters it heaps on average over the pixel (`height`, its relief), how
// thin it lies, from heaped (0) to a single layer of wet bubbles (1), and
// how much its hollows shade it.
struct WaterFroth {
    cover: f32,
    height: f32,
    thin: f32,
    shade: f32,
}

// Foam at `p` (meters along x and z, where what floats on the water lies),
// changing by `dx` and `dy` a pixel, its fresh white water and the foam the
// water carries each covering `amounts` of the surface (x and y, 0 to 1),
// where the water flows along `flow` (meters a second) shearing `shear` a
// second, at `time`, the place's clock running `stagger` cycles ahead, for
// a pixel spanning `seen` meters seen from the detail eye: one grain
// (`water_bubbles`) read at the steady size (`water_steady`), white water's
// round bubbles and the walls the foam the water carries drains into, two
// octaves of each alike. It moves only as the water moves it: carried along
// the flow in two sets half a life apart (`water_set`), each living as long
// as the flow's shear lets it before it shows drawn out (WATER_STRAIN), in
// whole octaves of lives, the next longer blended in over the upper half of
// the place's own octave as detail levels hand over; where the water
// carries it less than a bubble over a life it stands still, one set and no
// clock. The sets blend keeping their spread. Each kind covers the share of
// its grain past the threshold its amount sets, the grain taken as logistic
// of its spread, so bubbles finer than the steady size show the amount
// itself; the threshold's edge spans what the steady size averages of the
// grain (`water_resolved_share`) and how much it changes across the steady
// size, so foam never draws lace finer than it, nor thins out or shimmers
// as it recedes. Past its threshold it heaps a layer of its finer bubbles
// for each spread its grain stands over it, white water's whole and carried
// foam's drained, each layer hiding WATER_FOAM_FILM of what those below let
// through. Its relief hands over to none as its bubbles, seen from the
// detail eye, come to span the steady size. Its derivatives need it called
// in uniform control flow.
fn water_froth(p: vec2<f32>, dx: vec2<f32>, dy: vec2<f32>, amounts: vec2<f32>, flow: vec2<f32>,
        shear: f32, time: f32, stagger: f32, seen: f32) -> WaterFroth {
    // The share of each octave's variance the steady size resolves
    // (`water_steady`), its mipmaps averaging the rest.
    let footprint = max(length(dx), length(dy)) * water_steady();
    let kept = vec2<f32>(water_resolved_share(footprint, WATER_FOAM_BUBBLE),
        water_resolved_share(footprint, WATER_FOAM_BUBBLE / WATER_FOAM_FINER));
    let finer = kept.y > WATER_FOAM_LEAST;
    // Each kind's grain about its middle, in its spreads (x white water's, y
    // the carried foam's), and the share of its variance the steady size
    // resolves: read nowhere foam is too sparse to show, nor where the
    // steady size resolves too little of its grain to show.
    var grains = vec2<f32>(0.0);
    var resolved = 0.0;
    if max(amounts.x, amounts.y) > WATER_FOAM_LEAST && kept.x > WATER_FOAM_LEAST {
        // The map's channels' means, as its last level holds them.
        let mean = textureSampleLevel(water_fall_map, water_wave_sampler, vec2<f32>(0.5),
            f32(textureNumLevels(water_fall_map) - 1u)).ba;
        // The place's own life, and how many bubbles the water carries the
        // grain over it: less than one, and the grain stands still.
        let speed = length(flow);
        let life = clamp(WATER_STRAIN / max(shear, 1.0e-6), WATER_FOAM_SHORTEST,
            WATER_FOAM_LIFE);
        let moving = saturate(speed * life / WATER_FOAM_BUBBLE);
        var sum = vec4<f32>(0.0);
        var weights = 0.0;
        if moving < 1.0 {
            let still = 1.0 - moving;
            sum += water_bubbles(p, vec2<f32>(1.0, 0.0), dx, dy, mean, finer, 0x1656667bu)
                * still;
            weights += still * still;
        }
        if moving > 0.0 {
            // The octave of lives at or below the place's own, and the share
            // of the next longer blended in.
            let heading = flow / max(speed, 1.0e-6);
            let octave = log2(life / WATER_FOAM_SHORTEST);
            let below = floor(octave);
            let toward = saturate(2.0 * (octave - below) - 1.0);
            for (var band = 0u; band < 2u; band++) {
                let shown = select(1.0 - toward, toward, band == 1u) * moving;
                if shown <= 0.0 {
                    continue;
                }
                let cycle = WATER_FOAM_SHORTEST * exp2(below + f32(band));
                let salt = ((u32(below) + band) * 0x632be5abu) ^ 0x27d4eb2fu;
                for (var which = 0u; which < 2u; which++) {
                    let carried = water_set(time, stagger, cycle, which, salt);
                    let weight = shown * carried.weight;
                    if weight <= 0.0 {
                        continue;
                    }
                    sum += water_bubbles(p - flow * carried.seconds + carried.offset, heading,
                        dx, dy, mean, finer, salt) * weight;
                    weights += weight * weight;
                }
            }
        }
        let grain = sum * inverseSqrt(max(weights, 1.0e-12));
        grains = vec2<f32>(grain.y + grain.w, grain.x + grain.z) * inverseSqrt(2.0);
        resolved = 0.5 * (kept.x + select(0.0, kept.y, finer));
    }
    // What the steady size averages of each grain and how much the grain
    // changes across it, in its spreads squared and no more than all of it:
    // the threshold's edge spans both.
    let change = fwidth(grains) * water_steady();
    let spread = sqrt(clamp(vec2<f32>(1.0 - resolved) + change * change / 12.0,
        vec2<f32>(1.0e-6), vec2<f32>(1.0)));
    // Each kind's share past the threshold its amount sets, and how many
    // spreads its grain stands over it on average over the pixel, as the
    // logistic of the grain's spread gives them.
    let shows = amounts > vec2<f32>(WATER_FOAM_LEAST);
    let amount = clamp(amounts, vec2<f32>(WATER_FOAM_LEAST), vec2<f32>(1.0 - WATER_FOAM_LEAST));
    let past = clamp((grains / WATER_LOGISTIC + log(amount / (1.0 - amount))) / spread,
        vec2<f32>(-30.0), vec2<f32>(30.0));
    let covers = select(vec2<f32>(0.0), 1.0 / (1.0 + exp(-past)), shows);
    let over = select(vec2<f32>(0.0),
        spread * WATER_LOGISTIC * (max(past, vec2<f32>(0.0)) + log(1.0 + exp(-abs(past)))),
        shows);
    // The layers past its first each kind heaps where it covers, one a
    // spread, carried foam's drained; what they let through of what the
    // first does, and the share of the surface each kind hides.
    let layers = over / max(covers, vec2<f32>(1.0e-6)) * vec2<f32>(1.0, WATER_FOAM_DRAINED);
    let left = pow(vec2<f32>(1.0 - WATER_FOAM_FILM), layers);
    let hiding = covers * (1.0 - (1.0 - WATER_FOAM_FILM) * left);
    var froth: WaterFroth;
    froth.cover = 1.0 - (1.0 - hiding.x) * (1.0 - hiding.y);
    froth.thin = dot(hiding, left) / max(hiding.x + hiding.y, 1.0e-6);
    froth.shade = WATER_FOAM_HOLLOW * (1.0 - froth.thin);
    // Its relief, a finer bubble's radius a layer, which hands over to none
    // as its bubbles, seen from the detail eye, come to span the steady
    // size.
    froth.height = max(over.x, over.y * WATER_FOAM_DRAINED) * WATER_FOAM_BUBBLE
        / (2.0 * WATER_FOAM_FINER) * (1.0 - saturate(STEADY_PIXELS * seen / WATER_FOAM_BUBBLE));
    return froth;
}

// How foam's relief `height` (meters, as `water_froth` gives it) tilts the
// normal it is lit by off the surface's `normal`, at a point changing by
// `world_dx` and `world_dy` a pixel: a surface gradient from how its height
// changes across the pixel (Mikkelsen), as falls bend theirs. Its
// derivatives need it called in uniform control flow.
fn water_foam_tilt(height: f32, normal: vec3<f32>, world_dx: vec3<f32>,
        world_dy: vec3<f32>) -> vec3<f32> {
    let across_dy = cross(world_dy, normal);
    let across_dx = cross(normal, world_dx);
    let spanned = dot(world_dx, across_dy);
    return (dpdx(height) * across_dy + dpdy(height) * across_dx)
        * (sign(spanned) / max(abs(spanned), 1.0e-12));
}
