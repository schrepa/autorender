// Terrain's runtime virtual texture in the mesh passes, after
// `terrain.wgsl`: pages of its ground looked up where pixels read them,
// their bakes and the feedback asking for them.

// Runtime virtual texture: the ground of each textured terrain baked into
// pages of VT_PAGE² texels, VT_BORDER more around them so filtering never
// reaches a neighbor. Pages lie in sheets, the layers of `terrain_vt_sheets`,
// three a sheet: albedo (its detail and macro colors applied) and roughness,
// then the cover's tip color and the ground's vegetated share, then the
// normal maps' turn of the terrain's normal (in `terrain_vt_frame`), occlusion
// and the cover's leaves, linear data the sRGB layer decodes before it
// filters. Mip m's texels span `terrain.vt_scale.x` · 2^m meters, and shade
// at the levels and macro tint their own size gives, wherever the eye
// stands. Each terrain keeps, per mip, a window of VT_WINDOW² pages around the
// detail eye in `terrain_directory` from `terrain.vt_layout.x`: its first
// page, then a table of each page's slot plus one (0 while absent), addressed
// by page modulo VT_WINDOW, and after every table each entry's arrival, the
// time its page entered it (seconds, the bits of a float). Its coarse mips'
// pages are far pages (`terrain_vt.rs`), the same whatever the eye, cooked
// once and resident from its first frames.
const VT_PAGE: f32 = 128.0;
const VT_BORDER: f32 = 8.0;
const VT_SLOT: f32 = 144.0;
const VT_WINDOW: u32 = 64u;
const VT_MIPS: u32 = 18u;
// Leaves per square meter of ground the leaves texel's 1 stands for.
const VT_LEAVES: f32 = 16.0;
// Ground steeper than the first (its normal's y below) stays exact, its
// texels stretching down slopes; from the first to the second the pages
// take over from the ground computed there (`terrain_vt_texels`).
const VT_STEEP: vec2<f32> = vec2<f32>(0.65, 0.75);
// Seconds a page that arrived takes to fade in from the ground computed
// under it to its own: long enough that no arrival pops, as switches do
// over about half a second.
const VT_FADE_SECONDS: f32 = 0.5;
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

// Mip `mip`'s page at texel `texel` of the mip: the slot holding it, -1
// where it is absent or outside its window, and the share of the ground it
// shows, rising from 0 to 1 over VT_FADE_SECONDS since it entered its table
// (whole where the frame's time stands before that).
struct TerrainVtPage {
    slot: i32,
    shown: f32,
}

fn terrain_vt_page(mip: u32, texel: vec2<f32>) -> TerrainVtPage {
    var page = TerrainVtPage(-1, 0.0);
    let entry = terrain_vt_entry(mip, texel);
    if entry < 0 {
        return page;
    }
    let at = terrain.vt_layout.x + 2u * VT_MIPS + mip * VT_WINDOW * VT_WINDOW + u32(entry);
    page.slot = i32(terrain_directory[at]) - 1;
    if page.slot >= 0 {
        let age = terrain.vt_scale.z
            - bitcast<f32>(terrain_directory[at + VT_MIPS * VT_WINDOW * VT_WINDOW]);
        page.shown = select(saturate(age / VT_FADE_SECONDS), 1.0, age < 0.0);
    }
    return page;
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

// The texture's ground at `site` where its pages hold it, from the two mips
// whose texels bracket the pixel's, each its share of the ground by how far
// the pixel's lies toward it, as far as its page has faded in since it
// arrived: the finer page fades in over the coarser one, which pages bake
// first, as textures sharpen when their finer mips arrive; and the share of
// the pixel's ground they show together, the ground computed there showing
// the rest, where the coarser page is still fading in itself or the pages
// are taking over from it. `dither`, a threshold in [0, 1) that temporally
// antialiased frames vary by pixel and frame, shows one of the three
// whole, the coarser page below its share, the finer next and the computed
// ground past both, so the resolve blends them; frames without temporal
// antialiasing pass it negative and blend the pages by their shares.
// `shown` is 0 where the ground stays exact: no texture, steep ground, a
// pixel spanning half the finest texel or less, or a page not baked yet.
struct TerrainVtTexels {
    layers: array<vec4<f32>, 3>,
    shown: f32,
}

fn terrain_vt_texels(site: TerrainSite, dither: f32) -> TerrainVtTexels {
    var texels: TerrainVtTexels;
    let mips = terrain.vt.y;
    if mips < 2u {
        return texels;
    }
    let along = vec2<f32>(length(site.world_dx), length(site.world_dy));
    let lod = log2(terrain_vt_texel(along) / terrain.vt_scale.x);
    // The share of the ground the pages take over: none where a pixel spans
    // half the finest texel or less, or on steep ground; all of it where it
    // spans a whole one on gentler ground; smoothly between, so no edge
    // shows where the computed ground and the pages meet.
    let taking = saturate(lod + 1.0) * smoothstep(VT_STEEP.x, VT_STEEP.y, site.geometric.y);
    if !(taking > 0.0 && lod < f32(mips - 1u)) {
        return texels;
    }
    let fine = u32(max(lod, 0.0));
    let blend = max(lod, 0.0) - f32(fine);
    let size = terrain.vt_scale.x * f32(1u << fine);
    let texel = (site.world.xz - terrain.origin.xz) / size;
    let fine_page = terrain_vt_page(fine, texel);
    let coarse_page = terrain_vt_page(fine + 1u, texel * 0.5);
    if fine_page.slot < 0 || coarse_page.slot < 0 {
        return texels;
    }
    let fine_share = (1.0 - blend) * fine_page.shown * taking;
    // What the finer page does not show yet, the coarser one shows for it.
    let coarse_share = (taking - fine_share) * coarse_page.shown;
    if dither >= 0.0 {
        if dither < coarse_share {
            texels.layers = terrain_vt_sample(coarse_page.slot, texel * 0.5);
            texels.shown = 1.0;
        } else if dither < coarse_share + fine_share {
            texels.layers = terrain_vt_sample(fine_page.slot, texel);
            texels.shown = 1.0;
        }
        return texels;
    }
    // Two reads where both pages show.
    texels.shown = fine_share + coarse_share;
    if texels.shown <= 0.0 {
        return texels;
    }
    if coarse_share <= 0.0 {
        texels.layers = terrain_vt_sample(fine_page.slot, texel);
    } else if fine_share <= 0.0 {
        texels.layers = terrain_vt_sample(coarse_page.slot, texel * 0.5);
    } else {
        let coarser = coarse_share / texels.shown;
        let a = terrain_vt_sample(fine_page.slot, texel);
        let b = terrain_vt_sample(coarse_page.slot, texel * 0.5);
        texels.layers = array<vec4<f32>, 3>(mix(a[0], b[0], coarser), mix(a[1], b[1], coarser),
            mix(a[2], b[2], coarser));
    }
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
        let leaves = turn_texel.w * VT_LEAVES;
        let canopy = ground.vegetated * terrain_canopy(leaves, site);
        let look = terrain_canopy_look(ground.albedo, canopy_texel.rgb, leaves, site);
        ground.albedo = mix(ground.albedo, look.color, canopy);
        ground.roughness = mix(ground.roughness, COVER_ROUGHNESS, canopy);
        ground.occlusion = mix(ground.occlusion, look.occlusion, canopy);
        ground.normal = stable_normalize(mix(ground.normal, terrain_blade_normal(site.geometric),
            canopy));
        ground.canopy = canopy;
    }
    return ground;
}

// The ground where the texture's pages show `shown` of it (`terrain_vt_texels`):
// `exact`, the ground computed there, mixed with `page`, theirs.
fn terrain_vt_shown(exact: TerrainGround, page: TerrainGround, shown: f32) -> TerrainGround {
    var ground = exact;
    ground.albedo = mix(exact.albedo, page.albedo, shown);
    ground.roughness = mix(exact.roughness, page.roughness, shown);
    ground.normal = stable_normalize(mix(exact.normal, page.normal, shown));
    ground.occlusion = mix(exact.occlusion, page.occlusion, shown);
    ground.metalness = mix(exact.metalness, page.metalness, shown);
    ground.mapped = mix(exact.mapped, page.mapped, shown);
    ground.vegetated = mix(exact.vegetated, page.vegetated, shown);
    ground.canopy = mix(exact.canopy, page.canopy, shown);
    ground.full = mix(exact.full, page.full, shown);
    ground.colored = mix(exact.colored, page.colored, shown);
    ground.canopy_color = mix(exact.canopy_color, page.canopy_color, shown);
    ground.leaves = mix(exact.leaves, page.leaves, shown);
    return ground;
}

// What the texture's bakes and feedback bind beside the terrain group.
// Pages to bake: each one's slot, mip (bits 0 to 7; for far pages, the
// level they read plus one from bit 8), and page along x and z.
struct TerrainVtBake {
    slot: u32,
    mip: u32,
    page: vec2<i32>,
}
@group(2) @binding(20) var<storage, read> terrain_vt_bakes: array<TerrainVtBake>;
// Each textured terrain's pages its pixels want this frame, a bit each by
// mip and table entry, read back for the pages to bake.
@group(2) @binding(19) var<storage, read_write> terrain_vt_wanted: array<atomic<u32>>;

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

// The levels shading a texel whose footprint spans `footprint` meters
// (`terrain_shading_level`), no finer than level `first`: far pages read the
// terrain's finest pinned level and coarser ones alone, the levels resident
// everywhere, where finer ones' lookups would fall back to them.
fn terrain_vt_shading(footprint: f32, first: u32) -> TerrainShadingLevel {
    var shading = terrain_shading_level(footprint);
    if shading.level < first {
        shading.level = first;
        shading.blend = 0.0;
    }
    return shading;
}

// Where a texel `size` meters across lies at `world` on ground of smooth
// normal `geometric`, seen from straight above: its sides on the ground's
// slope, its footprint the longer and the footprint its filtering resolves
// the shorter, or a sixteenth of the longer where filtering runs out, as
// for a pixel spanning it.
fn terrain_vt_site(world: vec3<f32>, size: f32, geometric: vec3<f32>) -> TerrainSite {
    let slope = -geometric.xz / max(geometric.y, 0.05);
    var site: TerrainSite;
    site.world = world;
    site.world_dx = vec3<f32>(size, slope.x * size, 0.0);
    site.world_dy = vec3<f32>(0.0, slope.y * size, size);
    let along = vec2<f32>(length(site.world_dx), length(site.world_dy));
    site.footprint = max(along.x, along.y);
    site.texel = max(min(along.x, along.y), site.footprint / 16.0);
    site.geometric = geometric;
    site.toward = vec3<f32>(0.0, 1.0, 0.0);
    site.thinned = 0.0;
    return site;
}

// One texel of a page: the exact path's ground at its center, as seen
// where a pixel's footprint spans the texel, at the shading levels and
// macro tint that footprint gives (`terrain_shading_level`,
// `macro_strength`), measured on the ground's slope at the levels the
// texel's size gives on level ground; no eye enters it. Without the
// cover's canopy, which pixels lay over it as seen from them. A far page's
// texel shades no finer than the level it reads (`terrain_vt_shading`). A
// bake draws one sheet's pages.
@fragment
fn fs_terrain_vt_bake(input: TerrainVtBakeOutput) -> TerrainVtTargets {
    let bake = terrain_vt_bakes[input.bake];
    let mip = bake.mip & 0xffu;
    let far = bake.mip >> 8u;
    let place = terrain_vt_place(bake.slot);
    let corner = vec2<f32>(place.yz) * VT_SLOT;
    let size = terrain.vt_scale.x * f32(1u << mip);
    let texel = vec2<f32>(bake.page) * VT_PAGE + (input.clip_position.xy - corner) - VT_BORDER;
    let local = texel * size;
    let first = select(0u, far - 1u, far > 0u);
    // On the ground's surface, where its materials lie (`terrain_level_height`).
    let held = terrain_level_height(local, first);
    let world = vec3<f32>(local.x + terrain.origin.x,
        terrain.origin.y + select(0.0, held.height, held.found), local.y + terrain.origin.z);
    // Its levels by its footprint on the ground's slope, the slope read at
    // the levels its size gives level ground.
    let normal = terrain_surface_normal(local, terrain_vt_shading(size, first));
    let shading = terrain_vt_shading(terrain_vt_site(world, size, normal).footprint, first);
    let site = terrain_vt_site(world, size, terrain_surface_normal(local, shading));
    let ground = terrain_ground(site, shading);
    // The detail and macro colors pixels apply (`terrain_point_seen`).
    let tint = terrain_tint(world, ground, site.footprint, site.footprint);
    let frame = terrain_vt_frame(site.geometric);
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
    // The pixels of the frame's render size, which it draws at.
    let size = vec2<u32>(camera.region.xy);
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
    if !(abs(normal.y) >= VT_STEEP.x * length(normal)) {
        return;
    }
    // Per display pixel, as the resolves' sites measure it (`terrain_seen`).
    let along = vec2<f32>(length(steps[0]), length(steps[1])) * material_detail();
    let lod = log2(terrain_vt_texel(along) / terrain.vt_scale.x);
    let local = point.xz + camera.position.xz - terrain.origin.xz;
    let mips = terrain.vt.y;
    // The resolve's two mips, and one either side for its smoother measure,
    // from where the pages begin to take over (`terrain_vt_texels`).
    let low = u32(max(floor(lod - 0.25), 0.0));
    let high = min(u32(max(floor(lod + 0.25) + 1.0, 1.0)), mips - 1u);
    if !(lod + 1.25 >= 0.0) {
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
