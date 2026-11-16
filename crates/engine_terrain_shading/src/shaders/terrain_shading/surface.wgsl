import package::engine::detail::band;
import package::engine::view::{camera, detail_across};
import package::place::heightfield::SurfaceTriangle;
import package::terrain_shading::faces::heightfield::{terrain, terrain_level_height, terrain_surface_leaves, terrain_view};
import package::terrain_shading::tables::{table_floats, terrain_tables};
import package::generated::terrain_shading::{SURFACE_ERROR_MIPS, SURFACE_MESHLET_LEVEL};

const SURFACE_WHOLE: u32 = 0x80000000u;

fn surface_stored(leaf: u32) -> SurfaceTriangle {
    return terrain_surface_leaves[terrain.surface.x + leaf];
}

fn surface_whole(stored: SurfaceTriangle) -> bool {
    return (bitcast<u32>(stored.a.x) & SURFACE_WHOLE) != 0u;
}

fn surface_kept(stored: SurfaceTriangle) -> SurfaceTriangle {
    return SurfaceTriangle(vec2<f32>(abs(stored.a.x), stored.a.y), stored.b, stored.c);
}

fn surface_marked(triangle: SurfaceTriangle, whole: bool) -> SurfaceTriangle {
    let mark = select(0u, SURFACE_WHOLE, whole);
    return SurfaceTriangle(vec2<f32>(bitcast<f32>(bitcast<u32>(triangle.a.x) | mark),
        triangle.a.y), triangle.b, triangle.c);
}

fn surface_bounds() -> vec4<f32> {
    let header = terrain.directory.x + 4u * (terrain.directory.y - 1u);
    return table_floats(terrain_tables[header]
        + terrain_tables[header + 1u] * terrain_tables[header + 2u]);
}

fn surface_cut(triangle: SurfaceTriangle) -> bool {
    let levels = terrain.directory.y;
    if levels == 0u {
        return true;
    }
    let margin = surface_bounds().w;
    if !(margin > 0.0) {
        return true;
    }
    let low = min(min(triangle.a, triangle.b), triangle.c) / terrain.anchor.w;
    let high = max(max(triangle.a, triangle.b), triangle.c) / terrain.anchor.w;
    let size = max(abs(low), abs(high));
    let pad = vec2<f32>(margin * max(size.x, size.y));
    let first = low - pad;
    let last = high + pad;
    var nested = true;
    for (var index = 0u; index < levels; index++) {
        let header = terrain.directory.x + index * 4u;
        let tiles = vec2<u32>(terrain_tables[header + 1u], terrain_tables[header + 2u]);
        let scale = f32(1u << index);
        let lowest = min(vec2<u32>(max(first / scale, vec2<f32>(0.0)) / 256.0), tiles - 1u);
        let highest = min(vec2<u32>(max(last / scale, vec2<f32>(0.0)) / 256.0), tiles - 1u);
        if any(highest > lowest + 1u) {
            nested = false;
            continue;
        }
        let reach = select(highest + 1u, highest, nested);
        if any(reach >= tiles) {
            return true;
        }
        var resident = true;
        for (var z = lowest.y; z <= reach.y; z++) {
            for (var x = lowest.x; x <= reach.x; x++) {
                let layer = terrain_tables[terrain_tables[header] + z * tiles.x + x];
                if layer == 0xffffffffu {
                    resident = false;
                } else if terrain_tables[terrain.surface.y + layer] != 0xffffffffu {
                    return true;
                }
            }
        }
        if resident {
            return false;
        }
    }
    return true;
}

fn surface_half(triangle: SurfaceTriangle, half: u32) -> SurfaceTriangle {
    let middle = (triangle.a + triangle.b) * 0.5;
    if half == 0u {
        return SurfaceTriangle(triangle.c, triangle.a, middle);
    }
    return SurfaceTriangle(triangle.b, triangle.c, middle);
}

fn terrain_true_height(local: vec2<f32>) -> f32 {
    return terrain_level_height(local, 0u).height;
}

fn surface_relative(local: vec2<f32>, height: f32) -> vec3<f32> {
    let top = terrain.anchor.w * f32(1u << (terrain.directory.y - 1u));
    let from_anchor = local - vec2<f32>(bitcast<vec2<i32>>(terrain.surface.zw)) * top;
    return vec3<f32>(from_anchor.x + terrain.anchor.x, height + terrain.anchor.y,
        from_anchor.y + terrain.anchor.z);
}

fn surface_beyond(plane: vec4<f32>, sizes: vec4<f32>, low: vec3<f32>, high: vec3<f32>,
        margin: f32) -> bool {
    let support = select(low, high, plane.xyz >= vec3<f32>(0.0));
    let size = max(abs(low), abs(high));
    return dot(plane.xyz, support) + plane.w < -margin * (dot(sizes.xyz, size) + sizes.w);
}

fn surface_seen(triangle: SurfaceTriangle) -> bool {
    if all(triangle.a == triangle.b) && all(triangle.b == triangle.c) {
        return false;
    }
    let levels = terrain.directory.y;
    if levels == 0u {
        return true;
    }
    let bounds = surface_bounds();
    let margin = bounds.w;
    if !(margin > 0.0) {
        return true;
    }
    let low = min(min(triangle.a, triangle.b), triangle.c);
    let high = max(max(triangle.a, triangle.b), triangle.c);
    let top = terrain.anchor.w * f32(1u << (levels - 1u));
    let shift = abs(vec2<f32>(bitcast<vec2<i32>>(terrain.surface.zw)) * top);
    let offset = terrain_view.eye_offset.xyz;
    let across = max(abs(low), abs(high)) + shift + abs(terrain.anchor.xz) + abs(offset.xz);
    let tall = max(abs(bounds.x), abs(bounds.y)) + abs(terrain.anchor.y) + abs(offset.y);
    let lift = bounds.z * (1.0 + margin);
    let reach = vec3<f32>(lift) + margin * (vec3<f32>(across.x, tall, across.y) + lift);
    let near = surface_relative(low, bounds.x) + offset - reach;
    let far = surface_relative(high, bounds.y) + offset + reach;
    let rows = transpose(terrain_view.relative_view_projection);
    let w = rows[3];
    let sizes = abs(w);
    return !(surface_beyond(w + rows[0], sizes + abs(rows[0]), near, far, margin)
        || surface_beyond(w - rows[0], sizes + abs(rows[0]), near, far, margin)
        || surface_beyond(w + rows[1], sizes + abs(rows[1]), near, far, margin)
        || surface_beyond(w - rows[1], sizes + abs(rows[1]), near, far, margin)
        || surface_beyond(rows[2], abs(rows[2]), near, far, margin)
        || surface_beyond(w - rows[2], sizes + abs(rows[2]), near, far, margin));
}

@group(0) @binding(62) var terrain_errors: texture_2d_array<f32>;
@group(0) @binding(63) var water_errors: texture_2d_array<f32>;
const SURFACE_SPLINE_REACH: f32 = 2.0;

struct SurfaceWater {
    reached: bool,
    error: f32,
}

fn surface_tile(block: vec2<u32>, first: u32) -> vec2<u32> {
    let base = terrain.directory.x;
    for (var index = first; index < terrain.directory.y; index++) {
        let header = base + index * 4u;
        let tiles = vec2<u32>(terrain_tables[header + 1u], terrain_tables[header + 2u]);
        let tile = block >> vec2<u32>(min(index - first, 31u));
        if any(tile >= tiles) {
            continue;
        }
        let layer = terrain_tables[terrain_tables[header] + tile.y * tiles.x + tile.x];
        if layer != 0xffffffffu {
            return vec2<u32>(index, layer);
        }
    }
    return vec2<u32>(0u, 0xffffffffu);
}

fn surface_inside(tail: vec2<f32>, head: vec2<f32>, turn: f32, middle: vec2<f32>,
        half: f32) -> bool {
    let normal = turn * vec2<f32>(tail.y - head.y, head.x - tail.x);
    return dot(normal, middle - tail) + half * (abs(normal.x) + abs(normal.y)) > 0.0;
}

fn surface_overlaps(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>, low: vec2<f32>,
        high: vec2<f32>) -> bool {
    if any(high <= min(min(a, b), c)) || any(low >= max(max(a, b), c)) {
        return false;
    }
    let middle = (low + high) * 0.5;
    let half = (high.x - low.x) * 0.5;
    let turn = sign((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
    return surface_inside(a, b, turn, middle, half) && surface_inside(b, c, turn, middle, half)
        && surface_inside(c, a, turn, middle, half);
}

fn surface_cells(triangle: SurfaceTriangle, errors: texture_2d_array<f32>, reaching: bool)
        -> SurfaceWater {
    var found = SurfaceWater(false, 0.0);
    let low = min(min(triangle.a, triangle.b), triangle.c);
    let high = max(max(triangle.a, triangle.b), triangle.c);
    let side = max(high.x - low.x, high.y - low.y);
    let parts = i32(1u << SURFACE_MESHLET_LEVEL);
    let piece = side / f32(parts);
    let origin = floor(low / side) * side;
    let a = (triangle.a - origin) / piece;
    let b = (triangle.b - origin) / piece;
    let c = (triangle.c - origin) / piece;
    let first = vec2<i32>(origin / piece);
    let octaves = i32(round(log2(piece / (256.0 * terrain.anchor.w))));
    let level = u32(max(octaves, 0));
    let block = u32(i32(level) - octaves);
    var reach = 0;
    if reaching {
        reach = i32(ceil(SURFACE_SPLINE_REACH * exp2(f32(block)) / 256.0));
    }
    let cells_low = first - vec2<i32>(reach);
    let cells_high = first + vec2<i32>(parts - 1 + reach);
    let blocks_low = cells_low >> vec2<u32>(block);
    let blocks_high = cells_high >> vec2<u32>(block);
    for (var bz = blocks_low.y; bz <= blocks_high.y; bz++) {
        for (var bx = blocks_low.x; bx <= blocks_high.x; bx++) {
            if bx < 0 || bz < 0 {
                continue;
            }
            let held = surface_tile(vec2<u32>(vec2<i32>(bx, bz)), level);
            if held.y == 0xffffffffu {
                continue;
            }
            let fit = u32(i32(held.x) - octaves);
            let mip = SURFACE_ERROR_MIPS - 1u - min(fit, SURFACE_ERROR_MIPS - 1u);
            let narrow = min(fit - min(fit, SURFACE_ERROR_MIPS - 1u), 31u);
            let span = min(fit, 31u);
            let spread = select(0.0, SURFACE_SPLINE_REACH * exp2(f32(fit)) / 256.0, reaching);
            let start = max(cells_low, vec2<i32>(bx, bz) << vec2<u32>(block));
            let end = min(cells_high, ((vec2<i32>(bx, bz) + 1) << vec2<u32>(block)) - 1);
            let tile_first = (start >> vec2<u32>(span)) << vec2<u32>(span - narrow);
            let tile_last = tile_first + vec2<i32>((1i << (span - narrow)) - 1i);
            let beside = vec2<i32>(select(0, 1, reaching && narrow > 0u));
            let texels_low = max((start >> vec2<u32>(narrow)) - beside, tile_first);
            let texels_high = min((end >> vec2<u32>(narrow)) + beside, tile_last);
            let width = f32(1u << narrow);
            for (var z = texels_low.y; z <= texels_high.y; z++) {
                for (var x = texels_low.x; x <= texels_high.x; x++) {
                    let corner = vec2<f32>((vec2<i32>(x, z) << vec2<u32>(narrow)) - first);
                    let far = corner + vec2<f32>(width);
                    let over = surface_overlaps(a, b, c, corner, far);
                    if !over && !(reaching && surface_overlaps(a, b, c,
                            corner - vec2<f32>(spread), far + vec2<f32>(spread))) {
                        continue;
                    }
                    let texel = vec2<i32>(x, z) - tile_first;
                    let bound = textureLoad(errors, texel, i32(held.y), i32(mip)).x;
                    found.reached = found.reached || bound >= 0.0;
                    if over && fit < SURFACE_ERROR_MIPS {
                        found.error = max(found.error, bound);
                    }
                }
            }
        }
    }
    return found;
}

fn surface_ground_error(triangle: SurfaceTriangle) -> f32 {
    return surface_cells(triangle, terrain_errors, false).error;
}

fn surface_error_at(local: vec2<f32>, piece: f32) -> f32 {
    let octaves = i32(round(log2(piece / (256.0 * terrain.anchor.w))));
    let level = u32(max(octaves, 0));
    let block = u32(i32(level) - octaves);
    let cell = vec2<i32>(floor(local / piece));
    let held = surface_tile(vec2<u32>(cell >> vec2<u32>(block)), level);
    let fit = u32(i32(held.x) - octaves);
    if held.y == 0xffffffffu || fit >= SURFACE_ERROR_MIPS {
        return 0.0;
    }
    let texel = cell - ((cell >> vec2<u32>(fit)) << vec2<u32>(fit));
    let mip = SURFACE_ERROR_MIPS - 1u - fit;
    return max(textureLoad(terrain_errors, texel, i32(held.y), i32(mip)).x, 0.0);
}

fn surface_height(triangle: SurfaceTriangle, local: vec2<f32>) -> f32 {
    let held = terrain_true_height(local);
    let hypotenuse = abs(triangle.a - triangle.b);
    let unit = (hypotenuse.x + hypotenuse.y) / f32(2u << SURFACE_MESHLET_LEVEL);
    let cells = vec2<i32>(round(local / unit));
    let zeros = countTrailingZeros(vec2<u32>(cells));
    let shift = min(zeros.x, zeros.y);
    if shift + SURFACE_MESHLET_LEVEL + 1u > u32(round(log2(terrain.relief.w / unit))) {
        return held;
    }
    let step = cells >> vec2<u32>(shift);
    let odd = (step & vec2<i32>(1)) == vec2<i32>(1);
    var along = vec2<i32>(select(0, 1, odd.x), select(0, 1, odd.y));
    if all(odd) {
        along = select(vec2<i32>(-1), vec2<i32>(1), ((step + 1) & vec2<i32>(3)) == vec2<i32>(0));
    }
    let size = unit * f32(1u << shift);
    let first = terrain_level_height(vec2<f32>(step + along) * size, 0u);
    let second = terrain_level_height(vec2<f32>(step - along) * size, 0u);
    if !first.found || !second.found {
        return held;
    }
    let world = terrain.origin.xyz + vec3<f32>(local.x, held, local.y);
    let allowed = terrain.pixel_error.y * camera.pixel_error * detail_across(world);
    let error = surface_error_at(local, 2.0 * size);
    let middle = (first.height + second.height) * 0.5;
    if error <= allowed {
        return middle;
    }
    return mix(middle, held, band(log2(error / allowed)));
}

fn surface_water(triangle: SurfaceTriangle) -> SurfaceWater {
    return surface_cells(triangle, water_errors, true);
}
