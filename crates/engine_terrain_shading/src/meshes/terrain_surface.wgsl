import package::face::heightfield::{SURFACE_MESHLET_LEVEL, SurfaceTriangle, terrain,
    terrain_directory, terrain_level_height, terrain_surface_leaves};

fn surface_leaf(leaf: u32) -> SurfaceTriangle {
    return terrain_surface_leaves[terrain.surface.x + leaf];
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

@group(1) @binding(62) var terrain_errors: texture_2d_array<f32>;
@group(1) @binding(63) var water_errors: texture_2d_array<f32>;
const SURFACE_ERROR_MIPS: u32 = 8u;
const SURFACE_SPLINE_REACH: f32 = 2.0;

struct SurfaceWater {
    reached: bool,
    error: f32,
}

fn surface_tile(block: vec2<u32>, first: u32) -> vec2<u32> {
    let base = terrain.directory.x;
    for (var index = first; index < terrain.directory.y; index++) {
        let header = base + index * 4u;
        let tiles = vec2<u32>(terrain_directory[header + 1u], terrain_directory[header + 2u]);
        let tile = block >> vec2<u32>(min(index - first, 31u));
        if any(tile >= tiles) {
            continue;
        }
        let layer = terrain_directory[terrain_directory[header] + tile.y * tiles.x + tile.x];
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

fn surface_water(triangle: SurfaceTriangle) -> SurfaceWater {
    return surface_cells(triangle, water_errors, true);
}
