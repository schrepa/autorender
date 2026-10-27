// Foam on terrains' water and the sea: the water's own, from its state
// alone. The water carries the foam its flow churns up, settling or
// simulated (`water_sim.wgsl`'s `sim_churn`: where it loses energy, plunges
// or jumps, along its eddy lines and where it converges, fading to 1/e in
// WATER_FOAM_LIFE), drained as it ages; and it turns white, its foam fresh,
// where it breaks, where its crests fold (`water_folding`: where its waves
// pass the breaking slope) and where it runs fast against what stands in it
// (`water_contact`). Each is a share of the surface, and one bubble texture
// shows them all (`water_froth`): falling water's map's bubbles and the
// walls between them, carried only as the water moves them, which a pixel
// covers where they pass the threshold its share sets, so a little foam
// shows as sparse lace, more fills the cells as froth and the most heaps
// solid. A pixel reads them through its own footprint (`water_footprint`),
// whose levels resolve them across the view and whose anisotropy averages
// them along it, so far foam streaks along the view, and bubbles finer than
// it resolves show the share itself.

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
// the flow's shear takes to strain it by WATER_STRAIN, its bubbles drawn out
// by two fifths at either end of its life before it starts over (Neyret's
// latency); from WATER_FOAM_SHORTEST seconds, below which it would churn
// faster than white water does, to the life of the foam it shows
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
// (Gerstner's), here of waves half a meter long, whichever the surface
// draws, so foam gathers on their crests (Vlachos, Portal 2).
const WATER_FOAM_RIDE: f32 = 0.08;
// The meters a second over which water running against what stands in it
// tears air in, from the first to the second (Ervine, McKeogh and Elsawy
// 1980: about the meter a second past which water draws air in).
const WATER_FOAM_TEARING: vec2<f32> = vec2<f32>(0.7, 1.3);
// The compression over which a whitecap thins from all foam to none about
// the threshold the sea's wind sets (`water.foam.x`): from its fresh middle
// to its aging edge.
const WATER_FOAM_CAPPING: f32 = 0.12;

// The share of a noise's variance a read through `footprint`
// (`water_footprint`) keeps, its features `size` meters across: about
// 1 / √(1 + (s / size)²) along each of its sides, s the meters it resolves
// across the view and those its long side spans, the rest averaging out
// within it as its levels and anisotropy average it.
fn water_resolved_share(footprint: WaterFootprint, size: f32) -> f32 {
    let ratios = vec2<f32>(footprint.resolved, footprint.along) / size;
    return inverseSqrt((1.0 + ratios.x * ratios.x) * (1.0 + ratios.y * ratios.y));
}

// The share of water's surface its folding crests whiten: where its waves
// pass the breaking slope. Their slopes, resolved and hidden, spread
// `spread` (their variance, σ²), of which the pixel hides `hidden`: a
// Gaussian surface's slopes, their sizes Rayleigh-distributed, pass the
// breaking slope s_b (WATER_STEEPEST) over exp(−s_b²/σ²) of it. Where the
// pixel resolves those waves, that share lies on their crests, the points
// standing nearest their tops (`crest`, taken as the share of the surface
// standing lower); where it hides them, it is the share itself. A share too
// small to show whitens nothing, however near its crest's top a point
// stands.
fn water_folding(spread: f32, hidden: f32, crest: f32) -> f32 {
    let variance = max(spread, 1.0e-6);
    let share = exp(-WATER_STEEPEST * WATER_STEEPEST / variance);
    let on_crests = saturate((crest - 1.0) / max(share, 1.0e-6) + 1.5)
        * saturate(share / WATER_FOAM_LEAST);
    return mix(share, on_crests, saturate(1.0 - hidden / variance));
}

// How strongly water foams against what stands in it behind the pixel
// (Sea of Thieves' depth test), its surface `standing` meters over it,
// changing by `change` across a pixel whose footprint is `footprint`
// (`water_footprint`), where it runs at `speed` meters a second (its
// current or its eddies, whichever is faster): water climbs what it runs
// against by its velocity head U²/2g (Bernoulli), whitest where it meets
// it, and tears air in there as it passes WATER_FOAM_TEARING, so calm water
// foams at nothing it touches. A band rising less than what stands behind
// the pixel rises across a pixel of the display (`material_detail`) spreads
// its foam over that rise, keeping its share, so no line of foam draws
// finer than a pixel. Only solid ground counts: behind blades and twigs
// what the pixel sees jumps from pixel to pixel faster than any bank rises
// along the footprint's long side (WATER_GRADIENT_RANGE a meter).
fn water_contact(standing: f32, change: f32, footprint: WaterFootprint, speed: f32) -> f32 {
    let tearing = smoothstep(WATER_FOAM_TEARING.x, WATER_FOAM_TEARING.y, speed);
    if tearing <= 0.0 {
        return 0.0;
    }
    let head = speed * speed / (2.0 * WATER_GRAVITY);
    let climb = change * material_detail();
    let band = max(head, climb);
    let rise = climb / max(footprint.along, 1.0e-6);
    let solid = 1.0 - smoothstep(WATER_GRADIENT_RANGE, 2.0 * WATER_GRADIENT_RANGE, rise);
    return tearing * solid * saturate(1.0 - standing / band) * (head / band);
}

// A random point of a pattern (in its periods) by which reads at corner
// `corner` of a grid of patches shift it, salted by `salt`, so neighboring
// patches show different parts of it.
fn water_corner_offset(corner: vec2<i32>, salt: u32) -> vec2<f32> {
    let key = (u32(corner.x) * 0x8da6b343u) ^ (u32(corner.y) * 0xd8163841u) ^ (salt * 0xcb1ab31fu);
    return vec2<f32>(terrain_hash_of(key), terrain_hash_of(key ^ 0x68e31da4u));
}

// The triangle of a grid of equilateral triangles, skewed into squares,
// holding `grid` (in its cells' sides): its three corners (in the skewed
// grid) and the weights `grid` takes of each.
struct WaterTriangle {
    corners: array<vec2<i32>, 3>,
    weights: vec3<f32>,
}

fn water_triangle(grid: vec2<f32>) -> WaterTriangle {
    let skewed = vec2<f32>(grid.x, 1.15470054 * grid.y - 0.57735027 * grid.x);
    let cell = floor(skewed);
    let local = skewed - cell;
    let base = vec2<i32>(cell);
    let third = 1.0 - local.x - local.y;
    var triangle: WaterTriangle;
    triangle.weights = vec3<f32>(third, local.y, local.x);
    triangle.corners = array<vec2<i32>, 3>(base, base + vec2<i32>(0, 1), base + vec2<i32>(1, 0));
    if third <= 0.0 {
        triangle.weights = vec3<f32>(-third, 1.0 - local.y, 1.0 - local.x);
        triangle.corners = array<vec2<i32>, 3>(base + vec2<i32>(1, 1), base + vec2<i32>(1, 0),
            base + vec2<i32>(0, 1));
    }
    return triangle;
}

// Foam's grain at `p` (meters along x and z) for a pixel whose footprint is
// `footprint`, read through it (`water_footprint`): falling water's map's
// bubbles' walls and bubbles (`water_fall_map`'s b and a),
// WATER_FOAM_BUBBLE meters across and, if `finer`, WATER_FOAM_FINER times
// finer, drawn `stretch` times as long along `heading` as across it, read
// at the three corners of the triangle of a grid of patches half the map
// across holding it, each read turned to run along `heading` about its
// corner's middle and shifted at random (salted by `salt`), and blended by
// their weights over the root of their squares' sum, which keeps their
// spread (tiling and blending, Heitz and Neyret), so no patch shows the map
// repeat. Turning each read about a middle nearby keeps it from shearing
// where the heading turns. The larger bubbles' walls (x) and bubbles (y)
// and the finer's (zw, none unless `finer`), each as its deviation from
// the map's means `mean`, in its spreads.
fn water_bubbles(p: vec2<f32>, heading: vec2<f32>, stretch: f32, footprint: WaterFootprint,
        mean: vec2<f32>, finer: bool, salt: u32) -> vec4<f32> {
    let side = vec2<f32>(-heading.y, heading.x);
    // The meters across and along the map as foam reads it, drawn out along
    // the heading, and between the corners of the patches' grid.
    let map = vec2<f32>(1.0, FALL_MAP_STRETCH) * (WATER_FOAM_BUBBLE * FALL_MAP_BUBBLES);
    let drawn = map * vec2<f32>(1.0, stretch);
    let spacing = 0.5 * map.x;
    // Across a pixel, turned along the heading.
    let turned_dx = vec2<f32>(dot(footprint.dx, side), dot(footprint.dx, heading)) / drawn;
    let turned_dy = vec2<f32>(dot(footprint.dy, side), dot(footprint.dy, heading)) / drawn;
    let triangle = water_triangle(p / spacing);
    var corners = triangle.corners;
    let weights = triangle.weights;
    var grain = vec4<f32>(0.0);
    for (var corner = 0u; corner < 3u; corner++) {
        let skewed = vec2<f32>(corners[corner]);
        let middle = vec2<f32>(skewed.x, (skewed.y + 0.57735027 * skewed.x) * 0.8660254)
            * spacing;
        let turned = vec2<f32>(dot(p - middle, side), dot(p - middle, heading)) / drawn;
        let larger = textureSampleGrad(water_fall_map, water_fall_sampler,
            turned + water_corner_offset(corners[corner], salt), turned_dx, turned_dy).ba;
        var smaller = mean;
        if finer {
            smaller = textureSampleGrad(water_fall_map, water_fall_sampler,
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

// Foam at `p` (meters along x and z, where what floats on the water lies)
// for a pixel whose footprint is `footprint` (`water_footprint`), its fresh
// white water and the foam the water carries each covering `amounts` of the
// surface (x and y, 0 to 1), where the water flows along `flow` (meters a
// second) shearing `shear` a second, at `time`, the place's clock running
// `stagger` cycles ahead, for a pixel resolving `seen` meters seen from the
// detail eye (`detail_resolved`): one grain (`water_bubbles`) read through
// that footprint, white water's round bubbles and the walls the foam the
// water carries drains into, two octaves of each alike. It moves only as
// the water moves it: carried along the flow in two sets half a life apart
// (`water_set`), each living as long as the flow's shear lets it before it
// shows drawn out (WATER_STRAIN), in whole octaves of lives, the next
// longer blended in over the upper half of the place's own octave as detail
// levels hand over; where the water carries it less than a bubble over a
// life it stands still, one set and no clock. Where the water carries it
// and shears, the carried foam's grain is drawn out along the flow by the
// strain the shear gives it over the place's life, which WATER_STRAIN
// bounds: 1 + shear × life times as long as across, so a line of it reads
// as streaks along the current; white water's fresh foam stays round, read
// apart where both show. The sets blend keeping their spread. Each kind
// covers the share of its grain past the threshold its amount sets, the
// grain taken as logistic of its spread, so bubbles finer than the
// footprint resolves show the amount itself; the threshold's edge spans
// what the footprint averages of the grain, across the view and along it
// (`water_resolved_share`), and how much it changes across a pixel of the
// display (`material_detail`), so foam never draws lace finer than a pixel,
// nor thins out or shimmers as it recedes, and far foam keeps the streaks
// its grain shows across the view. Past its threshold it heaps a layer of
// its finer bubbles for each spread its grain stands over it, white water's
// whole and carried foam's drained, each layer hiding WATER_FOAM_FILM of
// what those below let through. Its relief draws while its bubbles, seen
// from the detail eye, span two pixels across the view (`water_shown`). Its
// derivatives need it called in uniform control flow.
fn water_froth(p: vec2<f32>, footprint: WaterFootprint, amounts: vec2<f32>, flow: vec2<f32>,
        shear: f32, time: f32, stagger: f32, seen: f32) -> WaterFroth {
    // The share of each octave's variance the footprint keeps
    // (`water_resolved_share`), its mipmaps and anisotropy averaging the
    // rest.
    let kept = vec2<f32>(water_resolved_share(footprint, WATER_FOAM_BUBBLE),
        water_resolved_share(footprint, WATER_FOAM_BUBBLE / WATER_FOAM_FINER));
    let finer = kept.y > WATER_FOAM_LEAST;
    // Each kind's grain about its middle, in its spreads (x white water's, y
    // the carried foam's), and the share of its variance the footprint
    // keeps: read nowhere foam is too sparse to show, nor where the
    // footprint keeps too little of its grain to show.
    var grains = vec2<f32>(0.0);
    var resolved = 0.0;
    if max(amounts.x, amounts.y) > WATER_FOAM_LEAST && kept.x > WATER_FOAM_LEAST {
        // The map's channels' means, as its last level holds them.
        let mean = textureSampleLevel(water_fall_map, water_fall_sampler, vec2<f32>(0.5),
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
            sum += water_bubbles(p, vec2<f32>(1.0, 0.0), 1.0, footprint, mean, finer,
                0x1656667bu) * still;
            weights += still * still;
        }
        if moving > 0.0 {
            // The octave of lives at or below the place's own, and the share
            // of the next longer blended in.
            let heading = flow / max(speed, 1.0e-6);
            let octave = log2(life / WATER_FOAM_SHORTEST);
            let below = floor(octave);
            let toward = saturate(2.0 * (octave - below) - 1.0);
            // How many times as long along the flow as across it the carried
            // foam's grain is drawn, by the strain the shear gives it over
            // the place's life; white water's stays round. Each is read only
            // where it shows, and one read serves both where the water does
            // not shear.
            let drawn = 1.0 + shear * life;
            let streaked = amounts.y > WATER_FOAM_LEAST && drawn > 1.0;
            let rounded = amounts.x > WATER_FOAM_LEAST || !streaked;
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
                    let at = p - flow * carried.seconds + carried.offset;
                    var read = vec4<f32>(0.0);
                    if rounded {
                        read = water_bubbles(at, heading, 1.0, footprint, mean, finer, salt);
                    }
                    if streaked {
                        let streaks = water_bubbles(at, heading, drawn, footprint, mean, finer,
                            salt);
                        read = vec4<f32>(streaks.x, read.y, streaks.z, read.w);
                    }
                    sum += read * weight;
                    weights += weight * weight;
                }
            }
        }
        let grain = sum * inverseSqrt(max(weights, 1.0e-12));
        grains = vec2<f32>(grain.y + grain.w, grain.x + grain.z) * inverseSqrt(2.0);
        resolved = 0.5 * (kept.x + select(0.0, kept.y, finer));
    }
    // What the footprint averages of each grain and how much the grain
    // changes across a pixel of the display, in its spreads squared and no
    // more than all of it: the threshold's edge spans both.
    let change = fwidth(grains) * material_detail();
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
    // Its relief, a finer bubble's radius a layer, which draws while its
    // bubbles, seen from the detail eye, span two pixels across the view.
    froth.height = max(over.x, over.y * WATER_FOAM_DRAINED) * WATER_FOAM_BUBBLE
        / (2.0 * WATER_FOAM_FINER) * water_shown(WATER_FOAM_BUBBLE, seen);
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
