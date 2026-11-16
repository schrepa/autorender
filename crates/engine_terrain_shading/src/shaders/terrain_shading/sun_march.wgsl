import package::face::ground_tiles::{
    GroundTerrain, LevelTile, NOT_RESIDENT, PEAK_MIPS, TILE_QUADS, ground_extent, ground_of,
    ground_tile, level_tiles, terrain_info, tile_height, tile_peak, triangle_height, true_height,
};
import package::generated::terrain_shading::{
    SUN_FINISH, SUN_GROUP, SUN_RESERVE, SUN_SLOPE_STEPS, SUN_SPLIT, SUN_STEEPEST, SunJob,
    SunTerrain,
};
import package::terrain_shading::faces::trace_source::{box_span, in_cutout, square_exit};

@group(1) @binding(0) var<storage, read> sun_jobs: array<SunJob>;
@group(1) @binding(1) var<storage, read> sun_terrains: array<SunTerrain>;
@group(1) @binding(2) var sun_out: texture_storage_2d_array<rgba16float, write>;
@group(1) @binding(3) var<storage, read_write> sun_counts: array<atomic<u32>>;

const SUN_PAST: f32 = 1.0e-3;
const SUN_FARTHEST: f32 = 3.4028233e38f;
const SUN_NUDGE: f32 = 1.001;
const SUN_AHEAD: f32 = 1.0e-6;
const SUN_WHOLE: u32 = 0u;
const SUN_SHORT: u32 = 1u;
const SUN_LOST: u32 = 2u;

struct SunCrests {
    slopes: vec2<f32>,
    spans: vec2<f32>,
}

struct SunGround {
    found: bool,
    height: f32,
}

fn sun_width(job: SunJob, span: f32) -> f32 {
    return max(job.place.z, job.heading.z * span);
}

fn sun_asked(ground: GroundTerrain, width: f32) -> u32 {
    return u32(clamp(floor(log2(max(SUN_NUDGE * width / ground.spacing, 1.0))), 0.0,
        f32(ground.levels - 1u)));
}

fn sun_within(width: f32, spacing: f32) -> u32 {
    return u32(clamp(floor(log2(max(SUN_NUDGE * width / spacing, 1.0))) - 1.0, 0.0,
        f32(PEAK_MIPS)));
}

fn sun_covering(width: f32, spacing: f32) -> u32 {
    return u32(clamp(ceil(log2(max(width / spacing, 1.0))) - 1.0, 0.0, f32(PEAK_MIPS)));
}

fn sun_ground(world: vec2<f32>, width: f32, count: u32) -> SunGround {
    var held: SunGround;
    for (var index = 0u; index < count; index++) {
        let terrain = sun_terrains[index];
        let info = terrain_info(terrain.table.x);
        let ground = ground_of(info);
        let local = world - info.origin.xz;
        if any(local < vec2<f32>(0.0)) || any(local > ground_extent(ground))
            || in_cutout(local, terrain.cutout) {
            continue;
        }
        let asked = sun_asked(ground, width);
        var found = ground_tile(ground, local, asked);
        if !found.found && asked > 0u {
            found = ground_tile(ground, local, 0u);
        }
        if found.found {
            held.found = true;
            held.height = info.origin.y + true_height(ground, local, found.level, found.tile);
            return held;
        }
    }
    return held;
}

fn sun_stand(world: vec2<f32>, job: SunJob) -> SunGround {
    var held = sun_ground(world, job.place.z, job.rows.w);
    if !held.found {
        return held;
    }
    for (var corner = 0u; corner < 4u; corner++) {
        let aside = vec2<f32>(f32(corner & 1u), f32(corner >> 1u)) - 0.5;
        let edge = sun_ground(world + aside * job.place.z, job.place.z, job.rows.w);
        if edge.found {
            held.height = max(held.height, edge.height);
        }
    }
    return held;
}

fn sun_block(under: f32, span: f32, tall: f32, crests: ptr<function, SunCrests>) {
    let reach = max(span, SUN_PAST);
    let lifted = vec2<f32>(under, under - tall) / reach;
    let won = lifted > (*crests).slopes;
    (*crests).slopes = select((*crests).slopes, lifted, won);
    (*crests).spans = select((*crests).spans, vec2<f32>(reach), won);
}

fn sun_heights(tile: LevelTile, first: vec2<u32>, z: u32) -> vec3<f32> {
    return vec3<f32>(tile_height(tile, first + vec2<u32>(0u, z)),
        tile_height(tile, first + vec2<u32>(1u, z)), tile_height(tile, first + vec2<u32>(2u, z)));
}

fn sun_crest(tile: LevelTile, first: vec2<u32>, o: vec3<f32>, heading: vec2<f32>,
        span: vec2<f32>, spacing: f32, cutout: vec4<f32>, tall: f32,
        crests: ptr<function, SunCrests>) {
    let near = sun_heights(tile, first, 0u);
    let middle = sun_heights(tile, first, 1u);
    let far = sun_heights(tile, first, 2u);
    let start = o.xz / spacing - vec2<f32>(first);
    let rate = heading / spacing;
    var s = span.x;
    for (var crossed = 0u; crossed < 3u; crossed++) {
        if s >= span.y {
            break;
        }
        let quad = min(vec2<u32>(max(floor(start + rate * s), vec2<f32>(0.0))), vec2<u32>(1u));
        let leave = min(square_exit(start, rate, vec2<f32>(quad), 1.0), span.y);
        let lower = select(near, middle, quad.y == 1u);
        let upper = select(middle, far, quad.y == 1u);
        let right = quad.x == 1u;
        let a = select(lower.x, lower.y, right);
        let b = select(lower.y, lower.z, right);
        let c = select(upper.x, upper.y, right);
        let d = select(upper.y, upper.z, right);
        let f0 = start - vec2<f32>(quad);
        let across = rate.x + rate.y;
        var split = leave;
        if abs(across) > 1.0e-12 {
            split = clamp((1.0 - f0.x - f0.y) / across, s, leave);
        }
        var low = s;
        for (var piece = 0u; piece < 2u; piece++) {
            let high = select(leave, split, piece == 0u);
            if high > low {
                let ends = max(vec2<f32>(low, high), vec2<f32>(SUN_PAST));
                for (var end = 0u; end < 2u; end++) {
                    let f = clamp(f0 + rate * ends[end], vec2<f32>(0.0), vec2<f32>(1.0));
                    let over = triangle_height(a, b, c, d, f) - o.y;
                    let lifted = vec2<f32>(over, over - tall) / ends[end];
                    let won = lifted > (*crests).slopes;
                    if any(won) && !in_cutout((vec2<f32>(first + quad) + f) * spacing, cutout) {
                        (*crests).slopes = select((*crests).slopes, lifted, won);
                        (*crests).spans = select((*crests).spans, vec2<f32>(ends[end]), won);
                    }
                }
            }
            low = max(high, low);
        }
        s = leave;
    }
}

fn sun_walk(index: u32, world: vec2<f32>, base: f32, job: SunJob,
        crests: ptr<function, SunCrests>, steps: ptr<function, u32>) -> u32 {
    let terrain = sun_terrains[index];
    let info = terrain_info(terrain.table.x);
    let ground = ground_of(info);
    let o = vec3<f32>(world.x - info.origin.x, base - info.origin.y, world.y - info.origin.z);
    let heading = job.heading.xy;
    let tall = job.place.w;
    let over = terrain.shape.x - o.y;
    let extent = ground_extent(ground);
    let span = box_span(o.xz, heading, vec2<f32>(0.0), extent, job.place.z, SUN_FARTHEST);
    var hole = vec2<f32>(SUN_FARTHEST);
    if terrain.cutout.z > terrain.cutout.x {
        hole = box_span(o.xz, heading, terrain.cutout.xy, terrain.cutout.zw, span.x, span.y);
    }
    let coarsest = ground.levels - 1u;
    let widest = level_tiles(ground, coarsest);
    let reserve = SUN_RESERVE * (widest.x + widest.y + 2u);
    let limit = *steps + (job.rows.z - min(job.rows.z, *steps)) / (job.rows.w - index);
    let ahead = SUN_AHEAD * max(max(extent.x, extent.y), 1.0);
    var outcome = SUN_WHOLE;
    var s = span.x;
    loop {
        let settled = over <= (*crests).slopes.x * s && over - tall <= (*crests).slopes.y * s;
        if s >= span.y || settled {
            break;
        }
        if *steps >= limit {
            outcome = SUN_LOST;
            break;
        }
        if s >= hole.x && s < hole.y {
            s = hole.y;
            continue;
        }
        *steps += 1u;
        let at = o.xz + heading * (s + ahead);
        var grain = sun_width(job, s);
        if limit - *steps < reserve {
            grain = max(grain, SUN_FINISH * (span.y - s) / f32(max(limit - *steps, 1u)));
            outcome = max(outcome, SUN_SHORT);
        }
        let asked = sun_asked(ground, grain);
        var found = ground_tile(ground, at, asked);
        if found.tile.layer == NOT_RESIDENT && asked > 0u {
            found = ground_tile(ground, at, 0u);
        }
        if found.tile.layer == NOT_RESIDENT {
            let size = f32(TILE_QUADS) * ground.spacing * f32(1u << coarsest);
            s = max(square_exit(o.xz / size, heading / size, floor(at / size), 1.0), s + ahead);
            continue;
        }
        let spacing = ground.spacing * f32(1u << found.level);
        let quads = select(TILE_QUADS, TILE_QUADS / 2u, found.level > 0u);
        let top = select(PEAK_MIPS - 1u, PEAK_MIPS - 2u, found.level > 0u);
        let region = floor(at / spacing / f32(quads)) * f32(quads);
        let offset = vec2<u32>(region) - found.tile.first;
        let start = o.xz / spacing - region;
        let rate = heading / spacing;
        var leaves = min(square_exit(start, rate, vec2<f32>(0.0), f32(quads)), span.y);
        if found.level < coarsest && job.heading.z > 0.0 {
            let coarser = 2.0 * spacing / job.heading.z;
            if coarser > s + ahead {
                leaves = min(leaves, coarser);
            }
        }
        leaves = max(leaves, min(s + ahead, span.y));
        var mip = top;
        loop {
            if s >= leaves || *steps >= limit {
                break;
            }
            *steps += 1u;
            var least = sun_within(sun_width(job, s), spacing);
            if limit - *steps < reserve {
                let needed = SUN_FINISH * (span.y - s) / f32(max(limit - *steps, 1u));
                least = min(max(least, sun_covering(needed, spacing)), top);
                mip = least;
                outcome = max(outcome, SUN_SHORT);
            }
            least = min(least, top);
            mip = max(mip, least);
            let width = f32(2u << mip);
            let cells = quads >> (mip + 1u);
            let cell = min(vec2<u32>(max(floor((start + rate * (s + ahead)) / width),
                vec2<f32>(0.0))), vec2<u32>(cells - 1u));
            let first = vec2<f32>(cell) * width;
            let under = tile_peak(found.tile, offset / (2u << mip) + cell, mip) - o.y;
            let exit = max(min(square_exit(start, rate, first, width), leaves),
                min(s + ahead, leaves));
            let beaten = (*crests).slopes + vec2<f32>(job.heading.w);
            let bars = min(beaten * s, beaten * exit);
            if under <= bars.x && under - tall <= bars.y {
                s = exit;
                mip = min(mip + 1u, top);
                continue;
            }
            if mip > least {
                mip -= 1u;
                continue;
            }
            if mip == 0u {
                sun_crest(found.tile, vec2<u32>(region + first), o, heading, vec2<f32>(s, exit),
                    spacing, terrain.cutout, tall, crests);
            } else {
                sun_block(under, s, tall, crests);
            }
            s = exit;
            mip = min(mip + 1u, top);
        }
    }
    return outcome;
}

fn sun_lean(world: vec2<f32>, job: SunJob, span: f32) -> f32 {
    if span <= 0.0 {
        return 0.0;
    }
    let width = sun_width(job, span);
    let aside = vec2<f32>(-job.heading.y, job.heading.x) * width;
    let crest = world + job.heading.xy * span;
    let left = sun_ground(crest - aside, width, job.rows.w);
    let right = sun_ground(crest + aside, width, job.rows.w);
    if !(left.found && right.found) {
        return 0.0;
    }
    return (right.height - left.height) / (2.0 * width);
}

@compute @workgroup_size(SUN_GROUP, SUN_GROUP, 1)
fn sun_march(@builtin(global_invocation_id) id: vec3<u32>) {
    let job = sun_jobs[id.z];
    let side = textureDimensions(sun_out).x;
    let texel = vec2<u32>(id.x, id.y + job.rows.x);
    if texel.x >= side || texel.y >= side {
        return;
    }
    let world = job.place.xy + (vec2<f32>(texel) + 0.5) * job.place.z;
    let base = sun_stand(world, job);
    var crests: SunCrests;
    var held = vec2<f32>(0.0);
    var lean = 0.0;
    if base.found {
        var steps = 0u;
        var outcome = SUN_WHOLE;
        for (var index = 0u; index < job.rows.w; index++) {
            outcome = max(outcome, sun_walk(index, world, base.height, job, &crests, &steps));
        }
        if outcome != SUN_WHOLE {
            atomicAdd(&sun_counts[outcome - 1u], 1u);
        }
        let coarse = floor(base.height / SUN_SPLIT) * SUN_SPLIT;
        held = vec2<f32>(coarse, base.height - coarse);
        lean = sun_lean(world, job, crests.spans.x);
    }
    let slopes = min(crests.slopes, vec2<f32>(SUN_STEEPEST));
    let near = select(vec2<f32>(0.0), 1.0 / max(crests.spans, vec2<f32>(SUN_PAST)),
        crests.spans > vec2<f32>(0.0));
    let layer = job.rows.y;
    let whole = floor(slopes.x * SUN_SLOPE_STEPS + 0.5) / SUN_SLOPE_STEPS;
    textureStore(sun_out, texel, layer, vec4<f32>(whole, slopes.x - whole, slopes.y, lean));
    textureStore(sun_out, texel, layer + 1u, vec4<f32>(held, near));
}
