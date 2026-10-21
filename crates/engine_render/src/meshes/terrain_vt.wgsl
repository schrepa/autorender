// Terrain's runtime virtual texture in the mesh passes, after
// `terrain.wgsl`: pages of its ground looked up where pixels read them,
// their bakes and the feedback asking for them.

// Runtime virtual texture: the ground of each textured terrain baked into
// pages of VT_PAGE² texels, VT_BORDER more around them so filtering never
// reaches a neighbor. Pages lie in sheets, the layers of `terrain_vt_sheets`,
// three a sheet: albedo (its detail and variety applied) and roughness, then
// the cover's canopy color and the ground's vegetated share, then the normal
// maps' turn of the terrain's normal (in `terrain_vt_frame`), occlusion and
// the cover's leaves, linear data the sRGB layer decodes before it filters.
// Mip m's texels span `terrain.vt_scale.x` · 2^m meters. Each terrain keeps,
// per mip, a window of VT_WINDOW² pages around the eye in `terrain_directory`
// from `terrain.vt_layout.x`: its first page, then a table of each page's slot
// plus one (0 while absent), addressed by page modulo VT_WINDOW.
const VT_PAGE: f32 = 128.0;
const VT_BORDER: f32 = 8.0;
const VT_SLOT: f32 = 144.0;
const VT_WINDOW: u32 = 64u;
const VT_MIPS: u32 = 18u;
// Leaves per square meter of ground the leaves texel's 1 stands for.
const VT_LEAVES: f32 = 16.0;
// Ground steeper than this (its normal's y below) stays exact: its texels
// would stretch down cliffs.
const VT_STEEP: f32 = 0.7;
// The texels a pixel whose footprint spans `along` (its two axes) reads:
// its long side's, so a page texel covers the whole pixel and one bilinear
// read filters it (the exact path's maps filter anisotropically up to 16 to
// 1, which here would read many times the texels from memory, unlike the
// materials' maps repeating in cache).
fn terrain_vt_texel(along: vec2<f32>) -> f32 {
    return max(along.x, along.y);
}

// Two unit vectors across `normal`, the frame the texture's normal turns
// lie in.
fn terrain_vt_frame(normal: vec3<f32>) -> mat2x3<f32> {
    let flat_across = vec3<f32>(normal.y, -normal.x, 0.0);
    let length_squared = dot(flat_across, flat_across);
    // Ground facing along z has no such vector across it in the xy plane.
    let across = select(vec3<f32>(1.0, 0.0, 0.0), flat_across * inverseSqrt(length_squared),
        length_squared > 1.0e-8);
    return mat2x3<f32>(across, cross(normal, across));
}

// The table entry (page modulo the window) of mip `mip`'s page at texel
// `texel` of the mip; -1 outside its window.
fn terrain_vt_entry(mip: u32, texel: vec2<f32>) -> i32 {
    let base = terrain.vt_layout.x;
    let page = vec2<i32>(floor(texel / VT_PAGE));
    let first = bitcast<vec2<i32>>(vec2<u32>(terrain_directory[base + 2u * mip],
        terrain_directory[base + 2u * mip + 1u]));
    let inside = page - first;
    if any(inside < vec2<i32>(0)) || any(inside >= vec2<i32>(i32(VT_WINDOW))) {
        return -1;
    }
    let cell = vec2<u32>(page & vec2<i32>(i32(VT_WINDOW) - 1));
    return i32(cell.y * VT_WINDOW + cell.x);
}

// The slot holding mip `mip`'s page at texel `texel` of the mip; -1 where
// it is absent or outside its window.
fn terrain_vt_slot(mip: u32, texel: vec2<f32>) -> i32 {
    let entry = terrain_vt_entry(mip, texel);
    if entry < 0 {
        return -1;
    }
    return i32(terrain_directory[terrain.vt_layout.x + 2u * VT_MIPS
        + mip * VT_WINDOW * VT_WINDOW + u32(entry)]) - 1;
}

// Slot `slot`'s sheet and its corner in it, in texels.
fn terrain_vt_place(slot: u32) -> vec3<u32> {
    let per_row = terrain.vt.z;
    let per_sheet = per_row * per_row;
    let within = slot % per_sheet;
    return vec3<u32>(slot / per_sheet, within % per_row, within / per_row);
}

// The texture's three texels of `slot` at texel `texel` of its mip (base
// color, normal turn, canopy), bilinear in the sheets' first level (they
// have no others).
fn terrain_vt_sample(slot: i32, texel: vec2<f32>) -> array<vec4<f32>, 3> {
    let place = terrain_vt_place(u32(slot));
    let corner = vec2<f32>(place.yz) * VT_SLOT + VT_BORDER;
    let uv = (corner + texel - floor(texel / VT_PAGE) * VT_PAGE) * terrain.vt_scale.y;
    let sheet = 3u * place.x;
    return array<vec4<f32>, 3>(
        textureSampleLevel(terrain_vt_sheets, terrain_material_sampler, uv, sheet, 0.0),
        textureSampleLevel(terrain_vt_sheets, terrain_material_sampler, uv, sheet + 2u, 0.0),
        textureSampleLevel(terrain_vt_sheets, terrain_material_sampler, uv, sheet + 1u, 0.0));
}

// The texture's ground at `site` where its pages hold it, from one of the
// two mips whose texels bracket the pixel's: the coarser where the pixel's
// lies farther toward it than `dither`, a threshold in [0, 1) (0.5 picks
// the nearer; temporally antialiased frames vary it by pixel and frame, so
// the resolve blends the two); `found` false where the ground stays exact:
// no texture, steep ground, texels coarser than the pixel at the finest
// mip, or a page not baked yet.
struct TerrainVtTexels {
    layers: array<vec4<f32>, 3>,
    found: bool,
}

fn terrain_vt_texels(site: TerrainSite, dither: f32) -> TerrainVtTexels {
    var texels: TerrainVtTexels;
    let mips = terrain.vt.y;
    if mips < 2u || site.geometric.y < VT_STEEP {
        return texels;
    }
    let along = vec2<f32>(length(site.world_dx), length(site.world_dy));
    let lod = log2(terrain_vt_texel(along) / terrain.vt_scale.x);
    if !(lod >= 0.0 && lod < f32(mips - 1u)) {
        return texels;
    }
    let fine = u32(lod);
    let blend = lod - f32(fine);
    let size = terrain.vt_scale.x * f32(1u << fine);
    let texel = (site.world.xz - terrain.origin.xz) / size;
    let fine_slot = terrain_vt_slot(fine, texel);
    let coarse_slot = terrain_vt_slot(fine + 1u, texel * 0.5);
    if fine_slot < 0 || coarse_slot < 0 {
        return texels;
    }
    let coarse = blend > dither;
    texels.layers = terrain_vt_sample(select(fine_slot, coarse_slot, coarse),
        select(texel, texel * 0.5, coarse));
    texels.found = true;
    return texels;
}

// The ground at `site` from the texture's `texels`: its normal turned off
// the terrain's in its frame, and the cover's canopy over it as the exact
// path lays each cover's (`terrain_slot_ground`), seen from here.
fn terrain_vt_ground(texels: TerrainVtTexels, site: TerrainSite) -> TerrainGround {
    let albedo_texel = texels.layers[0];
    let turn_texel = texels.layers[1];
    let canopy_texel = texels.layers[2];
    let frame = terrain_vt_frame(site.geometric);
    let turn = turn_texel.xy * 2.0 - 1.0;
    let up = sqrt(max(1.0 - dot(turn, turn), 0.0));
    var ground: TerrainGround;
    ground.albedo = albedo_texel.rgb;
    ground.roughness = albedo_texel.a;
    ground.normal = stable_normalize(frame[0] * turn.x + frame[1] * turn.y + site.geometric * up);
    ground.occlusion = turn_texel.z;
    ground.mapped = 1.0;
    ground.full = 1.0;
    ground.colored = 1.0;
    ground.vegetated = canopy_texel.a;
    if ground.vegetated > 0.0 {
        let canopy = ground.vegetated * terrain_canopy(turn_texel.w * VT_LEAVES, site);
        let shown = terrain_canopy_shown(abs(dot(site.toward, site.geometric)));
        ground.albedo = mix(ground.albedo, canopy_texel.rgb * shown, canopy);
        ground.roughness = mix(ground.roughness, 0.7, canopy);
        ground.occlusion = mix(ground.occlusion, TERRAIN_CANOPY_OCCLUSION * shown, canopy);
        ground.normal = stable_normalize(mix(ground.normal, terrain_blade_normal(site.geometric),
            canopy));
        ground.canopy = canopy;
    }
    return ground;
}

// What the texture's bakes and feedback bind beside the terrain group.
// Pages to bake: each one's slot, mip, and page along x and z.
struct TerrainVtBake {
    slot: u32,
    mip: u32,
    page: vec2<i32>,
}
@group(2) @binding(20) var<storage, read> terrain_vt_bakes: array<TerrainVtBake>;
// Every resident tile's heights, which bakes read in their fragments.
@group(2) @binding(21) var terrain_vt_heights: texture_2d_array<f32>;
// Each textured terrain's pages its pixels want this frame, a bit each by
// mip and table entry, read back for the pages to bake.
@group(2) @binding(19) var<storage, read_write> terrain_vt_wanted: array<atomic<u32>>;

// The terrain's height at terrain-local `local` in its finest resident
// level, bilinear between its samples (heights from the terrain's origin).
fn terrain_vt_height(local: vec2<f32>) -> f32 {
    let base = terrain.directory.x;
    let levels = terrain.directory.y;
    for (var index = 0u; index < levels; index++) {
        let header = base + index * 4u;
        let samples = max(local / (terrain.anchor.w * f32(1u << index)), vec2<f32>(0.0));
        let tiles = vec2<u32>(terrain_directory[header + 1u], terrain_directory[header + 2u]);
        let tile = min(vec2<u32>(samples / 256.0), tiles - vec2<u32>(1u));
        let layer = terrain_directory[terrain_directory[header] + tile.y * tiles.x + tile.x];
        if layer == 0xffffffffu {
            continue;
        }
        // Layers keep a border sample before the tile's first.
        let at = min(samples - vec2<f32>(tile * 256u), vec2<f32>(256.0)) + 1.0;
        let first = vec2<i32>(floor(at));
        let f = at - floor(at);
        let next = min(first + vec2<i32>(1), vec2<i32>(258));
        let low = mix(textureLoad(terrain_vt_heights, first, layer, 0).x,
            textureLoad(terrain_vt_heights, vec2<i32>(next.x, first.y), layer, 0).x, f.x);
        let high = mix(textureLoad(terrain_vt_heights, vec2<i32>(first.x, next.y), layer, 0).x,
            textureLoad(terrain_vt_heights, next, layer, 0).x, f.x);
        return mix(low, high, f.y);
    }
    return 0.0;
}

struct TerrainVtBakeOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) @interpolate(flat) bake: u32,
}

// The slot of bake `instance`, borders and all, in its sheet: two
// triangles.
@vertex
fn vs_terrain_vt_bake(@builtin(vertex_index) vertex: u32,
        @builtin(instance_index) instance: u32) -> TerrainVtBakeOutput {
    let bake = terrain_vt_bakes[instance];
    let place = terrain_vt_place(bake.slot);
    let unit = vec2<f32>(f32((0x32u >> vertex) & 1u), f32((0x2cu >> vertex) & 1u));
    let uv = (vec2<f32>(place.yz) + unit) * VT_SLOT * terrain.vt_scale.y;
    var output: TerrainVtBakeOutput;
    output.clip_position = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    output.bake = instance;
    return output;
}

struct TerrainVtTargets {
    @location(0) albedo: vec4<f32>,
    @location(1) canopy: vec4<f32>,
    @location(2) turn: vec4<f32>,
}

// One texel of a page: the exact path's ground at its center, as seen
// where a pixel's footprint spans the texel, at the shading levels its
// distance from the eye gives (`terrain_shading_level`, as the exact path),
// without the cover's canopy, which pixels lay over it as seen from them.
// A bake draws one sheet's pages.
@fragment
fn fs_terrain_vt_bake(input: TerrainVtBakeOutput) -> TerrainVtTargets {
    let bake = terrain_vt_bakes[input.bake];
    let place = terrain_vt_place(bake.slot);
    let corner = vec2<f32>(place.yz) * VT_SLOT;
    let size = terrain.vt_scale.x * f32(1u << bake.mip);
    let texel = vec2<f32>(bake.page) * VT_PAGE + (input.clip_position.xy - corner) - VT_BORDER;
    let local = texel * size;
    let world = vec3<f32>(local.x + terrain.origin.x, terrain.origin.y + terrain_vt_height(local),
        local.y + terrain.origin.z);
    let distance = length(world - terrain.eye.xyz);
    let shading = terrain_shading_level(distance);
    let geometric = terrain_surface_normal(local, shading);
    let slope = -geometric.xz / max(geometric.y, 0.05);
    var site: TerrainSite;
    site.world = world;
    site.world_dx = vec3<f32>(size, slope.x * size, 0.0);
    site.world_dy = vec3<f32>(0.0, slope.y * size, size);
    let along = vec2<f32>(length(site.world_dx), length(site.world_dy));
    site.footprint = max(along.x, along.y);
    site.texel = max(min(along.x, along.y), site.footprint / 16.0);
    site.geometric = geometric;
    site.distance = distance;
    site.toward = vec3<f32>(0.0, 1.0, 0.0);
    site.thinned = 0.0;
    let ground = terrain_ground(site, shading);
    // The detail and variety pixels apply (`terrain_point_seen`).
    let broad = terrain_detail(world.xz + vec2<f32>(-31.7, 12.3), 60.0, site.footprint);
    let patchy = terrain_detail(world.xz + vec2<f32>(17.3, 41.9), 9.0, site.footprint);
    let fine = terrain_detail(world.xz + vec2<f32>(-5.1, 3.7), 1.6, site.footprint);
    let plain = (0.8 + 0.4 * patchy) * (0.85 + 0.3 * fine);
    let mapped = (0.86 + 0.28 * broad) * (0.92 + 0.16 * patchy);
    let tint = mix(plain, mapped, ground.mapped)
        * ground_variety(terrain_sky_map(local).gb, world.xz, ground.vegetated);
    let frame = terrain_vt_frame(geometric);
    let vegetated = max(ground.vegetated, 1.0e-6);
    var targets: TerrainVtTargets;
    targets.albedo = vec4<f32>(ground.albedo * tint, ground.roughness);
    targets.canopy = vec4<f32>(ground.canopy_color / vegetated * tint, ground.vegetated);
    targets.turn = vec4<f32>(vec2<f32>(dot(ground.normal, frame[0]), dot(ground.normal, frame[1]))
        * 0.5 + 0.5, ground.occlusion, saturate(ground.leaves / vegetated / VT_LEAVES));
    return targets;
}

// The pages the terrain's pixels want: one pixel of each 8x8 block, each of
// its 64 in turn frame by frame, asks for the pages of the mips around its
// footprint (which the resolve measures on the terrain's smooth surface,
// this on its drawn one), from where its depth puts it.
// Pixels a side of the blocks feedback tests one pixel of.
const VT_FEEDBACK_BLOCK: u32 = 8u;
@compute @workgroup_size(8, 8)
fn cs_terrain_vt_feedback(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2<u32>(textureDimensions(terrain_ids_texture));
    let leader = vec2<u32>(terrain.vt.w & 7u, (terrain.vt.w >> 3u) & 7u);
    let pixel = id.xy * VT_FEEDBACK_BLOCK + leader;
    if any(pixel >= size) || terrain.vt.y < 2u {
        return;
    }
    let mine = terrain.directory.x + 1u;
    let here = textureLoad(terrain_ids_texture, pixel, 0);
    if here.x != mine {
        return;
    }
    let screen = vec2<f32>(size);
    let point = terrain_relative_at(vec2<f32>(pixel) + 0.5, bitcast<f32>(here.y), screen);
    // Its derivatives toward neighbors on the same terrain, either side.
    var steps = array<vec3<f32>, 2>();
    for (var axis = 0u; axis < 2u; axis++) {
        let offset = select(vec2<u32>(0u, 1u), vec2<u32>(1u, 0u), axis == 0u);
        var neighbor = pixel + offset;
        var sign = 1.0;
        if any(neighbor >= size) || textureLoad(terrain_ids_texture, neighbor, 0).x != mine {
            if any(pixel < offset) {
                return;
            }
            neighbor = pixel - offset;
            sign = -1.0;
        }
        let other = textureLoad(terrain_ids_texture, neighbor, 0);
        if other.x != mine {
            return;
        }
        steps[axis] = (terrain_relative_at(vec2<f32>(neighbor) + 0.5, bitcast<f32>(other.y), screen)
            - point) * sign;
    }
    let normal = cross(steps[1], steps[0]);
    if !(abs(normal.y) >= VT_STEEP * length(normal)) {
        return;
    }
    let along = vec2<f32>(length(steps[0]), length(steps[1]));
    let lod = log2(terrain_vt_texel(along) / terrain.vt_scale.x);
    let local = point.xz + camera.position.xz - terrain.origin.xz;
    let mips = terrain.vt.y;
    // The resolve's two mips, and one either side for its smoother measure.
    let low = u32(max(floor(lod - 0.25), 0.0));
    let high = min(u32(max(floor(lod + 0.25) + 1.0, 0.0)), mips - 1u);
    if !(lod + 0.25 >= 0.0) {
        return;
    }
    for (var mip = low; mip <= high; mip++) {
        let entry = terrain_vt_entry(mip, local / (terrain.vt_scale.x * f32(1u << mip)));
        if entry >= 0 {
            let word = terrain.vt.x * VT_MIPS * (VT_WINDOW * VT_WINDOW / 32u)
                + mip * (VT_WINDOW * VT_WINDOW / 32u) + u32(entry) / 32u;
            atomicOr(&terrain_vt_wanted[word], 1u << (u32(entry) % 32u));
        }
    }
}
