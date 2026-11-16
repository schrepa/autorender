import package::cover::faces::cover::{COVER_EDGE_ON, COVER_NO_LAYER, COVER_UNBOUND, CoverClump,
    CoverPatchRange, cover_accent_row, cover_bound, cover_bounds, cover_clump,
    cover_clump_point, cover_clump_seed, cover_clump_size, cover_clumped, cover_dense,
    cover_densest, cover_depth, cover_drawn, cover_fall, cover_growth, cover_header,
    cover_layer, cover_leaves, cover_limits, cover_look, cover_patch, cover_patch_point,
    cover_patch_range, cover_patch_reach, cover_row, cover_seen, cover_shape, cover_stand,
    cover_tall, cover_tint, cover_view};
import package::cover::casts::{BLADE_KINDS, blade_bounds, card_bounds, cast_cell, cover_casts,
    hold_casts, mark_casts};
import package::cover::laws::{BLADE_HALF, BLADE_SIZE, BLADE_UPPER, CARD_JOINTS,
    CARD_MIRRORED, CARD_TINT_RANGE, COVER_DRAW_VERTICES, FLAT_JOINTS, ROUNDING, STRAIGHT_JOINTS,
    blade_form, grounded_tilt, pack_color, unorm8};
import package::engine::detail::{ViewDetail, across, fits, handover, seen_share};
import package::engine::drawn::drawn_view_sees;
import package::engine::math::{PI, unit};
import package::face::ground_tiles::{GroundMaterial, GroundTerrain, GroundTile, LevelTile,
    NOT_RESIDENT, TILE_QUADS, ground_colors, ground_extent, ground_tile, level_tile,
    macro_color, material_look, site_shift, terrain_info, tile_height, tile_material, tile_site};
import package::cover::ground::{Between, CellSamples, SITE_HELD, TILE_SPAN, cell_samples,
    ground_at, ground_normal, height_on, held_height, hold_heights, ordered_height,
    ordered_to_height, site_between, site_lush, summed_under, tile_between};
import package::cover::order::{hash, order_tail, ordered_cell, quad_order};
import package::face::view_wind::wind_sway;
import package::cover::spine::{shaped_blade, view_header};
import package::generated::cover::{COVER_CELL_ROWS, CoverBlade, CoverCard, CoverFrame,
    CoverMaterial, Grow, GrowSpot, cover_facing, cover_lean};

struct Counts {
    appended: atomic<u32>,
    whole: atomic<u32>,
    flat: atomic<u32>,
    capacity: u32,
    card_capacity: u32,
    tried: atomic<u32>,
    scanned: atomic<u32>,
    ran: atomic<u32>,
    grew: atomic<u32>,
    carded: array<atomic<u32>>,
}

struct DrawArgs {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
}

@group(1) @binding(0) var<storage, read> grows: array<Grow>;
@group(1) @binding(1) var<storage, read_write> blades: array<CoverBlade>;
@group(1) @binding(2) var<storage, read_write> counts: Counts;
@group(1) @binding(3) var<storage, read_write> args: array<DrawArgs>;
@group(1) @binding(4) var<storage, read_write> cards: array<CoverCard>;
@group(1) @binding(5) var<storage, read_write> grow_spots: array<GrowSpot>;
@group(1) @binding(6) var<storage, read_write> grow_runs: array<atomic<u32>>;
@group(1) @binding(7) var<storage, read> grow_starts: array<u32>;
@group(1) @binding(9) var<uniform> grow_frame: CoverFrame;

const LATTICE_TURN: f32 = 0.618034;
const WIND_SHOWN: f32 = 1.0e9;
const LANE_LEVELS: u32 = 3u;
const REACH_ROUNDING: f32 = 0.01;
const PATCH_POINTS: u32 = 9u;
const CLUMP_REACH: f32 = 1.4143;
const NO_OWNER: vec3<u32> = vec3<u32>(COVER_NO_LAYER, 0u, 0u);
const UNNAMED_MATERIALS: u32 = 31u;
const CELL_LEFT: u32 = 0u;
const CELL_SUMMED: u32 = 1u;
const CELL_UNSUMMED: u32 = 2u;
const CORNER_LANE: u32 = 60u;

fn blade_room(least: vec3<f32>, error: f32, lush: f32) -> f32 {
    let finer = least.z * error;
    return max(0.5 * error * error, max(least.x, finer) * max(least.y, 0.5 * finer / lush));
}

const ROOT_LIFT: f32 = -0.02;
const SOIL_LEAST: f32 = 1.0e-4;
const MIXED: u32 = 0xffffffffu;
const GROUND_HELD: u32 = 64u;
const GROUND_ID: u32 = 0xfffu;

fn hold_corner(terrain: GroundTerrain, corner: vec2<f32>, spacing: f32, middle: f32,
        index: u32) {
    let at = corner + spacing * vec2<f32>(f32(index & 1u), f32(index >> 1u));
    let ground = ground_at(terrain, at);
    cell_corners[index] = select(middle, ground.height, ground.found);
}

fn grow_detail(grow: Grow) -> ViewDetail {
    return ViewDetail(grow.eye.xyz, grow.detail.x, grow.detail.y, grow.detail.z);
}

fn grow_share(grow: Grow, away: f32) -> f32 {
    let view = grow_detail(grow);
    let error = across(view, view.eye + vec3<f32>(away, 0.0, 0.0)) * view.pixel_error;
    return min(grow.terms.x / error, 1.0) * seen_share(grow.terms.y / error) * grow.detail.w;
}

fn cover_gust(root: vec3<f32>, time: f32) -> f32 {
    let still = length(wind_sway(time, 0.0, root, 1.0, vec3<f32>(0.0), 1.0));
    let moving = length(wind_sway(time, WIND_SHOWN, root, 1.0, vec3<f32>(0.0), 1.0));
    return select(0.0, moving / still, still > 0.0);
}

struct CellLooks {
    dense: f32,
    room: f32,
}

fn look_row(grow: Grow, material: u32) -> u32 {
    let row = cover_row(grow.counts.z, material);
    if grow.counts.w > 0u {
        return cover_layer(row, grow.counts.w - 1u);
    }
    return row;
}

fn with_look(looks: CellLooks, grow: Grow, material: u32, closest: f32, growth: vec4<f32>)
        -> CellLooks {
    let row = look_row(grow, material);
    if row == COVER_NO_LAYER {
        return looks;
    }
    let bounds = cover_bounds(row, growth, cover_patch_reach(row), closest);
    if bounds.dense <= 0.0 {
        return looks;
    }
    return CellLooks(max(looks.dense, bounds.dense), min(looks.room, bounds.room));
}

fn grown_on(ground: GroundMaterial) -> vec2<u32> {
    return vec2<u32>(select(ground.first, ground.second, ground.share == 255u),
        select(ground.second, ground.first, ground.share == 0u));
}

fn look_stand(grow: Grow, material: u32) -> f32 {
    let row = look_row(grow, material);
    if row == COVER_NO_LAYER {
        return 0.0;
    }
    return select(0.0, cover_stand(row), cover_densest(row) > 0.0);
}

fn stand_size(grow: Grow, ground: GroundMaterial) -> f32 {
    let major = ground.share > 127u;
    let size = look_stand(grow, select(ground.first, ground.second, major));
    if size > 0.0 || ground.share == 0u || ground.share == 255u {
        return size;
    }
    return look_stand(grow, select(ground.second, ground.first, major));
}

struct CellPlan {
    lie: vec4<f32>,
    growth: vec4<f32>,
    reaches: array<vec4<f32>, 2>,
    slack: f32,
    tiered: f32,
    stacked: f32,
}

var<workgroup> cell_found: vec4<u32>;
var<workgroup> cell_fine: vec4<u32>;
var<workgroup> cell_coarse: vec3<u32>;
var<workgroup> cell_seat: vec4<f32>;
var<workgroup> cell_corners: array<f32, 4>;
var<workgroup> cell_named: atomic<u32>;
var<workgroup> cell_low: atomic<u32>;
var<workgroup> cell_high: atomic<u32>;
var<workgroup> cell_dense: atomic<u32>;
var<workgroup> cell_room: atomic<u32>;
var<workgroup> cell_grew: atomic<u32>;
var<workgroup> cell_driest: atomic<u32>;
var<workgroup> cell_lushest: atomic<u32>;
var<workgroup> cell_driest_near: atomic<u32>;
var<workgroup> cell_lushest_near: atomic<u32>;
var<workgroup> cell_patches: array<atomic<u32>, 16>;
var<workgroup> cell_run: vec4<u32>;
var<workgroup> cell_plan: CellPlan;
var<workgroup> cell_ground: array<u32, GROUND_HELD>;
var<workgroup> cell_site: array<u32, SITE_HELD>;

fn held_ground(tile: LevelTile, sample: vec2<u32>, grounds: CellSamples) -> GroundMaterial {
    let within = sample - grounds.first;
    if all(within < grounds.count) {
        let packed = cell_ground[within.y * grounds.count.x + within.x];
        return GroundMaterial(packed & GROUND_ID, (packed >> 12u) & GROUND_ID, packed >> 24u);
    }
    return tile_material(tile, sample);
}

fn cell_quads(grow: Grow, fine: vec4<u32>, cell: vec2<u32>) -> vec4<u32> {
    let side = u32(grow.tier.y);
    let wide = max(side >> fine.w, 1u) + 1u;
    return vec4<u32>((cell * side) >> vec2<u32>(fine.w), wide,
        select(NOT_RESIDENT, fine.x, wide * wide <= SITE_HELD));
}

fn hold_sites(fine: vec4<u32>, quads: vec4<u32>, lane: u32) {
    if quads.w != NOT_RESIDENT && lane < quads.z * quads.z {
        cell_site[lane] = pack4x8unorm(tile_site(LevelTile(fine.x, fine.yz),
            quads.xy + vec2<u32>(lane % quads.z, lane / quads.z)));
    }
}

fn site_near(tile: LevelTile, at: Between, quads: vec4<u32>) -> vec4<f32> {
    let within = at.corner - quads.xy;
    if tile.layer != quads.w || any(within >= vec2<u32>(quads.z - 1u)) {
        return site_between(tile, at);
    }
    let start = within.y * quads.z + within.x;
    let low = mix(unpack4x8unorm(cell_site[start]), unpack4x8unorm(cell_site[start + 1u]),
        at.f.x);
    let high = mix(unpack4x8unorm(cell_site[start + quads.z]),
        unpack4x8unorm(cell_site[start + quads.z + 1u]), at.f.x);
    return mix(low, high, at.f.y);
}

struct Tussock {
    kept: f32,
    most: f32,
    other: u32,
}

fn tussock_weigh(held: Tussock, ground: GroundMaterial, f: vec2<f32>, index: u32, own: u32)
        -> Tussock {
    var weighed = held;
    let weight = mix(1.0 - f.x, f.x, f32(index & 1u)) * mix(1.0 - f.y, f.y, f32(index >> 1u));
    let second = weight * f32(ground.share) * (1.0 / 255.0);
    let first = weight - second;
    if ground.first == own {
        weighed.kept += first;
    } else if first > weighed.most {
        weighed.most = first;
        weighed.other = ground.first;
    }
    if ground.second == own {
        weighed.kept += second;
    } else if second > weighed.most {
        weighed.most = second;
        weighed.other = ground.second;
    }
    return weighed;
}

fn pick_ground(low: GroundMaterial, high: GroundMaterial, upper: bool) -> GroundMaterial {
    return GroundMaterial(select(low.first, high.first, upper),
        select(low.second, high.second, upper), select(low.share, high.share, upper));
}

fn tussock_material(grow: Grow, tile: LevelTile, samples: vec2<f32>, pick: f32,
        grounds: CellSamples, quads: vec4<u32>) -> u32 {
    let at = tile_between(tile, samples);
    let corner = at.corner;
    let f = at.f;
    let held_0 = held_ground(tile, corner, grounds);
    let held_1 = held_ground(tile, corner + vec2<u32>(1u, 0u), grounds);
    let held_2 = held_ground(tile, corner + vec2<u32>(0u, 1u), grounds);
    let held_3 = held_ground(tile, corner + vec2<u32>(1u, 1u), grounds);
    let right = round(f.x) != 0.0;
    let closest = pick_ground(pick_ground(held_0, held_1, right),
        pick_ground(held_2, held_3, right), round(f.y) != 0.0);
    let own = select(closest.first, closest.second, closest.share > 127u);
    var weighed = Tussock(0.0, 0.0, own);
    weighed = tussock_weigh(weighed, held_0, f, 0u, own);
    weighed = tussock_weigh(weighed, held_1, f, 1u, own);
    weighed = tussock_weigh(weighed, held_2, f, 2u, own);
    weighed = tussock_weigh(weighed, held_3, f, 3u, own);
    var kept = weighed.kept;
    let other = weighed.other;
    if other != own {
        let bare = look_stand(grow, own) <= 0.0;
        if bare != (look_stand(grow, other) <= 0.0) {
            let site = site_near(tile, at, quads);
            if any(site != vec4<f32>(0.0)) {
                let soil = max(site.z, SOIL_LEAST);
                kept = select(1.0 - saturate((1.0 - kept) / soil), saturate(kept / soil), bare);
            }
        }
    }
    return select(other, own, pick < kept);
}

fn unseen_cell(wedge: i32, center: vec2<f32>, spacing: f32, margin: f32) -> bool {
    if wedge < 0 {
        return false;
    }
    let apex = grows[wedge].origin;
    let sides = grows[wedge].eye;
    let away = center - apex.xy;
    let first = apex.zw;
    let last = sides.xy;
    let reach = 0.5 * spacing + margin;
    return first.x * away.y - first.y * away.x + reach * (abs(first.x) + abs(first.y)) < 0.0
        || away.x * last.y - away.y * last.x + reach * (abs(last.x) + abs(last.y)) < 0.0;
}

fn count_grown() {
    if atomicExchange(&cell_grew, 1u) == 0u {
        atomicAdd(&counts.grew, 1u);
    }
}

fn lush_growth(grow: Grow, corner: vec2<f32>, spacing: f32, still: vec2<f32>,
        around: vec2<f32>) -> vec4<f32> {
    var pulled = around;
    let reach = grow.wind.w + REACH_ROUNDING;
    let tile = TILE_SPAN * grow.origin.w;
    if any(floor((corner - reach) / tile) != floor((corner + spacing + reach) / tile)) {
        pulled = vec2<f32>(0.0, 1.0);
    }
    return vec4<f32>(cover_growth(still.x), cover_growth(still.y), cover_growth(pulled.x),
        cover_growth(pulled.y));
}

fn main_grounds(named: u32) -> vec2<u32> {
    let other = named & (named - 1u);
    return min(vec2<u32>(countTrailingZeros(named),
        countTrailingZeros(select(named, other, other != 0u))), vec2<u32>(UNNAMED_MATERIALS));
}

fn hold_grounds(fine: vec4<u32>, grounds: CellSamples, lane: u32) {
    if lane < grounds.count.x * grounds.count.y {
        let ground = tile_material(LevelTile(fine.x, fine.yz),
            grounds.first + vec2<u32>(lane % grounds.count.x, lane / grounds.count.x));
        cell_ground[lane] = min(ground.first, GROUND_ID)
            | (min(ground.second, GROUND_ID) << 12u) | (ground.share << 24u);
    }
}

fn cell_beyond(away: vec2<f32>, half_diagonal: f32, outer: f32, inner: f32) -> bool {
    return max(away.x - half_diagonal, 0.0) > outer || away.y + 3.0 * half_diagonal < inner;
}

fn cell_shown(grow: Grow, corner: vec2<f32>, spacing: f32, heights: vec2<f32>) -> bool {
    let base = grow.origin.xyz + vec3<f32>(corner.x, ROOT_LIFT, corner.y);
    let margin = vec3<f32>(grow.tier.w + grow.least.w);
    return drawn_view_sees(0u, base + vec3<f32>(0.0, heights.x, 0.0) - margin,
        base + vec3<f32>(spacing, heights.y + 0.5 * grow.tier.w, spacing) + margin);
}

struct Found {
    kind: u32,
    fine: vec4<u32>,
    coarse: vec3<u32>,
    seat: vec4<f32>,
    under: vec3<u32>,
    lush: vec4<u32>,
}

fn find_cell(row: u32, group: vec2<u32>) -> Found {
    var cell_out: Found;
    let grid = grows[row].grid;
    let at = grid.xy + vec2<i32>(group);
    if any(at < vec2<i32>(0)) {
        return cell_out;
    }
    let origin = grows[row].origin;
    let tier = grows[row].tier;
    let eye = grows[row].eye;
    let spacing = origin.w * tier.y;
    let corner = vec2<f32>(at) * spacing;
    let middle = corner + vec2<f32>(0.5 * spacing);
    let center = origin.xz + middle;
    let half_diagonal = 0.7072 * spacing;
    let flat = distance(center, eye.xz);
    if flat > eye.w + half_diagonal
            || unseen_cell(grid.z, center, spacing, tier.w + grows[row].least.w) {
        return cell_out;
    }
    let grow = grows[row];
    let info = terrain_info(grow.counts.x);
    let terrain = GroundTerrain(info.levels.x, grow.counts.y, origin.w);
    if any(middle > ground_extent(terrain)) {
        return cell_out;
    }
    let found = ground_tile(terrain, middle, 0u);
    if !found.found {
        return cell_out;
    }
    let cell = vec2<u32>(at);
    let side = u32(tier.y);
    let grown = side <= TILE_QUADS;
    let per = origin.w * f32(1u << found.level);
    var extra = 1u;
    if grown {
        extra += u32(floor((grow.wind.w + REACH_ROUNDING) / per + REACH_ROUNDING));
    }
    var under = summed_under(found.tile, vec2<i32>((cell * side) >> vec2<u32>(found.level))
        - vec2<i32>(found.tile.first), side >> found.level, extra);
    var coarse = LevelTile(NOT_RESIDENT, vec2<u32>(0u));
    let above = found.level + 1u;
    if above < terrain.levels {
        coarse = level_tile(terrain, above, middle / (origin.w * f32(1u << above)));
    }
    if coarse.layer != NOT_RESIDENT {
        let wider = summed_under(coarse, vec2<i32>((cell * side) >> vec2<u32>(above))
            - vec2<i32>(coarse.first), side >> above, 1u);
        under.made = under.made && wider.made;
        under.heights = vec2<f32>(min(under.heights.x, wider.heights.x),
            max(under.heights.y, wider.heights.y));
    }
    let summed = under.made && (!grown || (under.named >> UNNAMED_MATERIALS) == 0u);
    let view = grow_detail(grow);
    var dead = false;
    if summed {
        let heights = origin.y + under.heights - eye.y;
        let away = vec2<f32>(length(vec2<f32>(flat, clamp(0.0, heights.x, heights.y))) - ROUNDING,
            length(vec2<f32>(flat, max(abs(heights.x), abs(heights.y)))) + ROUNDING);
        if cell_beyond(away, half_diagonal, eye.w, tier.x) {
            return cell_out;
        }
        var live = !grown;
        if grown {
            if any(under.low != vec4<f32>(0.0)) {
                let still = site_lush(under.low, under.high);
                cell_out.lush.x = ~bitcast<u32>(still.x);
                cell_out.lush.y = bitcast<u32>(still.y);
            }
            if any(under.wide_low != vec4<f32>(0.0)) {
                let around = site_lush(under.wide_low, under.wide_high);
                cell_out.lush.z = ~bitcast<u32>(around.x);
                cell_out.lush.w = bitcast<u32>(around.y);
            }
            let growth = lush_growth(grow, corner, spacing,
                lush_range(cell_out.lush.x, cell_out.lush.y),
                lush_range(cell_out.lush.z, cell_out.lush.w));
            let closest = across(view, view.eye
                + vec3<f32>(max(away.x - half_diagonal, 0.0), 0.0, 0.0)) * view.pixel_error;
            for (var rest = under.named; rest != 0u && !live; rest &= rest - 1u) {
                live = with_look(CellLooks(0.0, COVER_UNBOUND), grow, countTrailingZeros(rest),
                    closest, growth).dense > 0.0;
            }
        }
        dead = !live || !cell_shown(grow, corner, spacing, under.heights);
        if dead && !cell_beyond(away.yx, half_diagonal, eye.w, tier.x) {
            atomicAdd(&counts.scanned, 1u);
            return cell_out;
        }
    }
    let height = height_on(found.tile, max(middle / per, vec2<f32>(0.0)));
    let center_away = distance(vec3<f32>(center.x, origin.y + height, center.y), eye.xyz);
    if cell_beyond(vec2<f32>(center_away), half_diagonal, eye.w, tier.x) {
        return cell_out;
    }
    atomicAdd(&counts.scanned, 1u);
    if dead {
        return cell_out;
    }
    let closest = across(view, view.eye
        + vec3<f32>(max(center_away - half_diagonal, 0.0), 0.0, 0.0)) * view.pixel_error;
    cell_out.fine = vec4<u32>(found.tile.layer, found.tile.first, found.level);
    cell_out.coarse = vec3<u32>(coarse.layer, coarse.first);
    cell_out.seat = vec4<f32>(height, closest, select(0.0, grow.wind.w, grow.wind.w > closest),
        0.0);
    if !summed {
        cell_out.kind = CELL_UNSUMMED;
        return cell_out;
    }
    cell_out.under = vec3<u32>(~ordered_height(under.heights.x), ordered_height(under.heights.y),
        under.named);
    cell_out.kind = CELL_SUMMED;
    return cell_out;
}

@compute @workgroup_size(64)
fn find_cells(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>,
        @builtin(local_invocation_index) lane: u32) {
    let spot = (group.y * groups.x + group.x) * 64u + lane;
    let sizes = grows[0].cells;
    if spot >= sizes.z {
        return;
    }
    var low = 0u;
    var high = sizes.w - 1u;
    while low < high {
        let middle = (low + high + 1u) / 2u;
        if grows[middle].cells.x <= spot {
            low = middle;
        } else {
            high = middle - 1u;
        }
    }
    let cells = grows[low].cells;
    let local = spot - cells.x;
    let found = find_cell(low, vec2<u32>(local % cells.y, local / cells.y));
    if found.kind == CELL_LEFT {
        return;
    }
    let kept = atomicAdd(&grow_runs[3u * low], 1u);
    grow_spots[cells.x + kept] = GrowSpot(found.fine, vec4<u32>(found.coarse, found.kind),
        found.seat, vec4<u32>(found.under, local), found.lush);
}

fn scan_under(grow: Grow, fine: vec4<u32>, coarse: vec3<u32>, corner: vec2<f32>, closest: f32,
        lane: u32) {
    let spacing = grow.origin.w * grow.tier.y;
    let grown = u32(grow.tier.y) <= TILE_QUADS;
    let tile = LevelTile(fine.x, fine.yz);
    let rect = cell_samples(grow, corner, spacing, fine.w, tile.first,
        select(0.0, grow.wind.w + REACH_ROUNDING, grown));
    let inside = cell_samples(grow, corner, spacing, fine.w, tile.first, 0.0);
    let growth = vec4<f32>(cover_growth(0.0), cover_growth(1.0), cover_growth(0.0),
        cover_growth(1.0));
    var lowest = 0xffffffffu;
    var highest = 0u;
    var lush = vec4<u32>(0u);
    var named = 0u;
    var known = MIXED;
    var looks = CellLooks(0.0, COVER_UNBOUND);
    for (var index = lane; index < rect.count.x * rect.count.y; index += 64u) {
        let sample = rect.first + vec2<u32>(index % rect.count.x, index / rect.count.x);
        let height = ordered_height(tile_height(tile, sample));
        lowest = min(lowest, height);
        highest = max(highest, height);
        if !grown {
            continue;
        }
        let site = tile_site(tile, sample);
        let sited = any(site != vec4<f32>(0.0));
        let wet = 0.5 + 0.25 * (site.x + site.z - site.y - site.w);
        let span = vec2<u32>(~bitcast<u32>(select(0.0, wet, sited)),
            bitcast<u32>(select(1.0, wet, sited)));
        lush = max(lush, vec4<u32>(select(vec2<u32>(0u), span,
            all(sample - inside.first < inside.count)), span));
        let ids = grown_on(tile_material(tile, sample));
        named |= (1u << min(ids.x, UNNAMED_MATERIALS)) | (1u << min(ids.y, UNNAMED_MATERIALS));
        if ids.x >= UNNAMED_MATERIALS && ids.x != known {
            known = ids.x;
            looks = with_look(looks, grow, known, closest, growth);
        }
        if ids.y >= UNNAMED_MATERIALS && ids.y != known {
            known = ids.y;
            looks = with_look(looks, grow, known, closest, growth);
        }
    }
    if coarse.x != NOT_RESIDENT {
        let above = LevelTile(coarse.x, coarse.yz);
        let wider = cell_samples(grow, corner, spacing, fine.w + 1u, above.first, 0.0);
        for (var index = lane; index < wider.count.x * wider.count.y; index += 64u) {
            let height = ordered_height(tile_height(above,
                wider.first + vec2<u32>(index % wider.count.x, index / wider.count.x)));
            lowest = min(lowest, height);
            highest = max(highest, height);
        }
    }
    if highest != 0u {
        atomicMax(&cell_low, ~lowest);
        atomicMax(&cell_high, highest);
    }
    if lush.z != 0u {
        atomicMax(&cell_driest, lush.x);
        atomicMax(&cell_lushest, lush.y);
        atomicMax(&cell_driest_near, lush.z);
        atomicMax(&cell_lushest_near, lush.w);
    }
    if named != 0u {
        atomicOr(&cell_named, named);
    }
    if looks.dense > 0.0 {
        atomicMax(&cell_dense, bitcast<u32>(looks.dense));
        atomicMax(&cell_room, ~bitcast<u32>(looks.room));
    }
}

fn lush_range(driest: u32, lushest: u32) -> vec2<f32> {
    if driest == 0u {
        return vec2<f32>(0.0, 1.0);
    }
    return vec2<f32>(bitcast<f32>(~driest) - ROUNDING, bitcast<f32>(lushest) + ROUNDING);
}

fn grown_range(grow: Grow, corner: vec2<f32>, spacing: f32) -> vec4<f32> {
    return lush_growth(grow, corner, spacing,
        lush_range(atomicLoad(&cell_driest), atomicLoad(&cell_lushest)),
        lush_range(atomicLoad(&cell_driest_near), atomicLoad(&cell_lushest_near)));
}

fn range_patches(grow: Grow, corner: vec2<f32>, lane: u32) {
    let look = lane / (2u * PATCH_POINTS);
    if look >= 2u {
        return;
    }
    let alone = main_grounds(atomicLoad(&cell_named));
    if look == 1u && alone.x == alone.y {
        return;
    }
    let row = look_row(grow, select(alone.x, alone.y, look == 1u));
    if row == COVER_NO_LAYER || cover_dense(row) <= 0.0 {
        return;
    }
    let octave = (lane / PATCH_POINTS) % 2u;
    let low = grow.origin.xz + corner - REACH_ROUNDING;
    let high = low + grow.origin.w * grow.tier.y + 2.0 * REACH_ROUNDING;
    let range = cover_patch_point(row, low, high, 2u * grow.counts.w, octave,
        lane % PATCH_POINTS);
    let at = 8u * look + 4u * octave;
    atomicMax(&cell_patches[at], ~ordered_height(range.x));
    atomicMax(&cell_patches[at + 1u], ~ordered_height(range.y));
    atomicMax(&cell_patches[at + 2u], ordered_height(range.z));
    atomicMax(&cell_patches[at + 3u], ordered_height(range.w));
}

fn patch_span(at: u32) -> vec4<f32> {
    return vec4<f32>(ordered_to_height(~atomicLoad(&cell_patches[at])),
        ordered_to_height(~atomicLoad(&cell_patches[at + 1u])),
        ordered_to_height(atomicLoad(&cell_patches[at + 2u])),
        ordered_to_height(atomicLoad(&cell_patches[at + 3u])));
}

struct PlanLook {
    row: u32,
    reach: vec4<f32>,
    dense: f32,
    room: f32,
    walks: bool,
}

const NO_LOOK: PlanLook = PlanLook(COVER_NO_LAYER, vec4<f32>(0.0), 0.0, COVER_UNBOUND, true);

fn plan_look(grow: Grow, material: u32, at: u32, growth: vec4<f32>, closest: f32) -> PlanLook {
    let row = look_row(grow, material);
    if row == COVER_NO_LAYER || cover_dense(row) <= 0.0 {
        return NO_LOOK;
    }
    let reach = cover_patch_range(row, patch_span(at), patch_span(at + 4u));
    let bounds = cover_bounds(row, growth, reach, closest);
    if bounds.dense <= 0.0 {
        return NO_LOOK;
    }
    let size = cover_clump_size(row);
    let visited = (3.0 * size + grow.origin.w) / size;
    return PlanLook(row, vec4<f32>(reach.density, reach.low, reach.high, 0.0), bounds.dense,
        bounds.room, bounds.base <= 0.0 && bounds.share * visited * visited < 1.0);
}

fn with_plan(looks: CellLooks, look: PlanLook) -> CellLooks {
    if look.row == COVER_NO_LAYER {
        return looks;
    }
    return CellLooks(max(looks.dense, look.dense), min(looks.room, look.room));
}

fn plan_cell(grow: Grow, cell: vec2<u32>, cell_center: vec3<f32>, closest: f32,
        unsummed: bool) {
    let side = u32(grow.tier.y);
    let spacing = grow.origin.w * grow.tier.y;
    let corner = vec2<f32>(cell) * spacing;
    let low = ordered_to_height(~atomicLoad(&cell_low));
    let high = ordered_to_height(atomicLoad(&cell_high));
    let scanned = side <= TILE_QUADS;
    cell_run = vec4<u32>(0u, MIXED, COVER_NO_LAYER, COVER_NO_LAYER);
    if unsummed && !cell_shown(grow, corner, spacing, vec2<f32>(low, high)) {
        return;
    }
    let growth = grown_range(grow, corner, spacing);
    let named = atomicLoad(&cell_named);
    let alone = main_grounds(named);
    let others = named & ~((1u << alone.x) | (1u << alone.y));
    let mixed = others != 0u || (named >> UNNAMED_MATERIALS) != 0u;
    var looks = CellLooks(0.0, COVER_UNBOUND);
    var first = NO_LOOK;
    var second = NO_LOOK;
    var walked = scanned && !mixed;
    if scanned {
        let unnamed = bitcast<f32>(atomicLoad(&cell_dense));
        if unnamed > 0.0 {
            looks = CellLooks(unnamed, bitcast<f32>(~atomicLoad(&cell_room)));
        }
        for (var rest = others; rest != 0u; rest &= rest - 1u) {
            looks = with_look(looks, grow, countTrailingZeros(rest), closest, growth);
        }
        first = plan_look(grow, alone.x, 0u, growth, closest);
        if alone.x != alone.y {
            second = plan_look(grow, alone.y, 8u, growth, closest);
        }
        looks = with_plan(with_plan(looks, first), second);
        walked = walked && first.walks && second.walks;
        if looks.dense <= 0.0 {
            return;
        }
    }
    let heights = vec4<f32>(cell_corners[0], cell_corners[1], cell_corners[2], cell_corners[3]);
    let rise = vec2<f32>(heights.y - heights.x + heights.w - heights.z,
        heights.z - heights.x + heights.w - heights.y) / (2.0 * spacing);
    let normal = normalize(vec3<f32>(-rise.x, 1.0, -rise.y));
    let header = cover_header();
    let area = grow.origin.w * grow.origin.w;
    let thin = grow.detail.w;
    let half_diagonal = 0.7072 * spacing;
    let toward = grow.eye.xyz - cell_center;
    let center_away = length(toward);
    let nearest = max(center_away - half_diagonal, 0.0);
    let close = max(nearest, 1.0e-6);
    let facing = min((cover_view(normal, toward / center_away).x * center_away
        + length(normal.xz) * half_diagonal) / close, 1.0);
    let level = max(sqrt(max(1.0 - toward.y * toward.y / (close * close), 0.0)), COVER_EDGE_ON);
    let lattice = u32(grow.grid.w);
    let tiered = min(grow.tier.z, f32(lattice * lattice) * grow_share(grow, nearest));
    let stacked = (1.0 + ROUNDING) * area * header.layers.x * facing * thin / level;
    var crowd = tiered;
    var room = blade_room(grow.least.xyz, closest, header.lush.w);
    if scanned {
        crowd = min(crowd, (1.0 + ROUNDING) * looks.dense * header.shading.x * area * thin);
        room = looks.room;
    }
    var total = 0u;
    var rows = vec2<u32>(COVER_NO_LAYER);
    if walked {
        cell_plan.reaches = array<vec4<f32>, 2>(first.reach, second.reach);
        rows = vec2<u32>(first.row, second.row);
    } else {
        total = u32(ceil(f32(side * side) * min(crowd, stacked / room)));
    }
    cell_plan.lie = vec4<f32>(normal, order_tail(cell, u32(grow.terms.w), grow.counts.w));
    cell_plan.growth = growth;
    cell_plan.slack = high - low + grow.least.w;
    cell_plan.tiered = tiered;
    cell_plan.stacked = stacked;
    atomicAdd(&counts.tried, total);
    atomicAdd(&counts.ran, select(min(total, 1u), 1u, walked));
    cell_run = vec4<u32>(total, select(MIXED, alone.x, scanned && alone.x == alone.y && !mixed),
        rows.x, rows.y);
}

struct Run {
    grow: Grow,
    view: ViewDetail,
    terrain: GroundTerrain,
    whole: GroundTile,
    grounds: CellSamples,
    quads: vec4<u32>,
    levels: vec4<u32>,
    growth: vec4<f32>,
    steps: vec4<f32>,
    normal: vec3<f32>,
    corner: vec2<f32>,
    extent: vec2<f32>,
    spacing: f32,
    height: f32,
    tail: f32,
    slack: f32,
    ring: f32,
    closest: f32,
    area: f32,
    layers: f32,
    lush: f32,
    density: f32,
    sown: f32,
    crowd: f32,
    tiered: f32,
    stacked: f32,
    grow_in: f32,
    side: u32,
    slot: u32,
    depth: u32,
    finest: u32,
    lattice: u32,
    turn: u32,
    alone: u32,
}

var<private> run: Run;
var<private> lane_material: u32;
var<private> lane_row: u32;
var<private> lane_grew: bool;

fn spec_look(at: u32, density: f32) -> CoverMaterial {
    var spec = cover_look(at);
    spec.shape.x = density;
    return spec;
}

struct Candidate {
    rank: f32,
    seed: u32,
    local: vec2<f32>,
    world_xz: vec2<f32>,
    probe: vec3<f32>,
    error: f32,
    sight: vec2<f32>,
    deep: f32,
    found: GroundTile,
    material: u32,
    turf: f32,
    tussock: CoverClump,
    row: u32,
    kept: bool,
}

fn candidate_at(at: vec2<u32>, order: u32, candidate: u32) -> Candidate {
    var picked: Candidate;
    let slot = run.slot;
    let lattice = run.lattice;
    let held = lattice * lattice;
    let quad = run.grow.origin.w;
    let eye = run.grow.eye.xyz;
    let outer = run.grow.eye.w;
    let inner = run.grow.tier.x;
    let thin = run.grow.detail.w;
    let cell_seed = hash((at.x * 0x8da6b343u) ^ (at.y * 0xd8163841u)
        ^ (slot * 0x68bc21ebu));
    let slot_at = (candidate * (1u + lattice * run.turn) + cell_seed % held) % held;
    let seed = hash(cell_seed + candidate * 0x632be5abu);
    let rank = f32(candidate) + (f32(order) + run.tail) / f32(run.finest);
    let jitter = vec2<f32>(unit(seed), unit(hash(seed ^ 0x5bd1e995u)));
    let offset = (vec2<f32>(f32(slot_at % lattice), f32(slot_at / lattice)) + jitter)
        * (quad / f32(lattice));
    let local = vec2<f32>(at) * quad + offset;
    let world_xz = run.grow.origin.xz + local;
    let probe = vec3<f32>(world_xz.x, run.height, world_xz.y);
    let afar = distance(probe, eye);
    if afar > outer + run.slack || afar + run.slack < inner {
        return picked;
    }
    let error = across(run.view, probe) * run.view.pixel_error;
    let sight = cover_view(run.normal, normalize(eye - probe));
    let deep = run.area * run.layers * sight.x * thin / sight.y;
    if rank >= min(f32(held) * grow_share(run.grow, distance(world_xz, run.grow.eye.xz)),
            deep / blade_room(run.grow.least.xyz, error, run.lush)) {
        return picked;
    }
    if any(local > run.extent) {
        return picked;
    }
    var found = run.whole;
    if run.side > TILE_QUADS {
        found = ground_tile(run.terrain, local, 0u);
        if !found.found {
            return picked;
        }
    }
    var material = run.alone;
    var turf = 0.0;
    var tussock: CoverClump;
    if run.alone == MIXED {
        let per = quad * f32(1u << found.level);
        let ground = held_ground(found.tile,
            vec2<u32>(round(max(local / per, vec2<f32>(0.0)))), run.grounds);
        material = select(ground.first, ground.second,
            unit(hash(seed ^ 0x27d4eb2fu)) < f32(ground.share) * (1.0 / 255.0));
        if run.ring > 0.0 {
            let size = stand_size(run.grow, ground);
            if unit(hash(seed ^ 0x165667b1u)) >= handover(run.view, size, probe) {
                turf = size;
                tussock = cover_clump(world_xz, turf, 2u * slot + 1u);
                material = tussock_material(run.grow, found.tile,
                    (tussock.center - run.grow.origin.xz) / per,
                    unit(hash(tussock.seed ^ 0x27d4eb2fu)), run.grounds, run.quads);
            }
        }
    }
    if material != lane_material {
        lane_material = material;
        lane_row = look_row(run.grow, material);
    }
    let row = lane_row;
    if row == COVER_NO_LAYER {
        return picked;
    }
    let limits = cover_limits(row);
    if limits.x <= 0.5 * error || rank >= min(cover_densest(row) * run.crowd,
            deep / blade_room(limits.yzw, error, run.lush)) {
        return picked;
    }
    picked = Candidate(rank, seed, local, world_xz, probe, error, sight, deep, found, material,
        turf, tussock, row, true);
    return picked;
}

fn grow_candidate(at: vec2<u32>, order: u32, candidate: u32, owner: vec3<u32>) {
    let picked = candidate_at(at, order, candidate);
    if !picked.kept {
        return;
    }
    let slot = run.slot;
    let quad = run.grow.origin.w;
    let eye = run.grow.eye.xyz;
    let outer = run.grow.eye.w;
    let inner = run.grow.tier.x;
    let thin = run.grow.detail.w;
    let rank = picked.rank;
    let seed = picked.seed;
    let local = picked.local;
    let world_xz = picked.world_xz;
    let error = picked.error;
    let sight = picked.sight;
    let found = picked.found;
    let material = picked.material;
    let turf = picked.turf;
    let tussock = picked.tussock;
    let row = picked.row;
    let owned = owner.x != COVER_NO_LAYER;
    if owned && owner.x != row {
        return;
    }
    let density = cover_dense(row);
    let accents = cover_look(row).accents;
    let accented = accents.y > 0u;
    if density <= 0.0 || (owned && !accented) {
        return;
    }
    var clump: CoverClump;
    var at_spec = row;
    if accented {
        clump = cover_clump(world_xz, cover_clump_size(row), 2u * slot);
        if owned && any(bitcast<vec2<u32>>(clump.cell) != owner.yz) {
            return;
        }
        let accent = cover_accent_row(accents, unit(hash(clump.seed ^ 0x9e3779b9u)));
        at_spec = select(accent, row, accent == COVER_NO_LAYER);
    }
    let spec = spec_look(at_spec, density);
    let base_patches = cover_look(row).patches;
    let pulled = spec.tuft.z > 0.0 || spec.fans.y > 0.0;
    let grown = select(run.growth.xy, run.growth.zw, pulled);
    let finer = error * spec.tuft.w;
    let mean = 0.5 * (spec.shape.y + spec.shape.z);
    let reached = (1.0 + base_patches.z) * grown.y;
    if rank >= spec.shape.x * (1.0 + base_patches.y) * run.sown * grown.y
            * min(spec.shape.w * reached / finer, 1.0) * seen_share(mean * reached / finer) {
        return;
    }
    var patched: CoverMaterial;
    patched.patches = base_patches;
    let patches = cover_patch(world_xz, patched, 0.0, 2u * slot);
    let low = patches.height * grown.x;
    let high = patches.height * grown.y;
    let own = spec.shape.x * run.density * run.area * patches.density;
    let full = own * grown.y * min(spec.shape.w * high / finer, 1.0)
        * seen_share(mean * high / finer);
    let layered = run.area * run.layers * sight.x
        / (sight.y * max(spec.shape.w * low, finer) * mean * low);
    if rank >= min(full, layered) * thin {
        return;
    }
    var rooted = local;
    var fan = vec2<f32>(0.0);
    var stand = found;
    if pulled {
        var tuft = tussock;
        if turf != spec.tuft.x {
            tuft = cover_clump(world_xz, spec.tuft.x, 2u * slot + 1u);
        }
        let outward = world_xz - tuft.center;
        let apart = length(outward);
        let pull = mix(spec.tuft.y, spec.tuft.z, unit(hash(seed ^ 0x3c6ef372u)));
        let opened = mix(spec.fans.x, spec.fans.y, saturate(2.0 * apart / spec.tuft.x));
        fan = outward * (opened / max(apart, 1.0e-4));
        rooted = local - outward * pull;
        if any(rooted > run.extent) || any(rooted < vec2<f32>(0.0)) {
            return;
        }
        if run.side > TILE_QUADS || any(rooted < run.corner)
                || any(rooted > run.corner + vec2<f32>(run.spacing)) {
            stand = ground_tile(run.terrain, rooted, 0u);
            if !stand.found {
                return;
            }
        }
    }
    let root = vec3<f32>(run.grow.origin.x + rooted.x, run.grow.origin.y
        + held_height(run.terrain, rooted, stand.level, stand.tile, run.quads) + ROOT_LIFT,
        run.grow.origin.z + rooted.y);
    let away = distance(root, eye);
    if away > outer || away < inner {
        return;
    }
    if !accented {
        clump = cover_clump(world_xz, cover_clump_size(row), 2u * slot);
    }
    let site = site_near(stand.tile, tile_between(stand.tile,
        rooted / (quad * f32(1u << stand.level))), run.quads);
    let sited = any(site != vec4<f32>(0.0));
    var colors: vec4<f32>;
    if !sited {
        colors = ground_colors(run.terrain, run.levels.w, rooted, 0u);
    }
    let growth = cover_growth(select(colors.a,
        0.5 + 0.25 * (site.x + site.z - site.y - site.w), sited));
    let scale = growth * patches.height;
    let dense = own / run.area * growth;
    let drawn = cover_drawn(spec, scale, finer, dense, sight, run.layers);
    let shown = min(drawn.x, drawn.y);
    let keep = own * growth * min(spec.shape.w * scale / finer, 1.0) * shown * thin;
    if rank >= keep {
        return;
    }
    let rise = saturate((keep - rank) / max(run.grow_in * keep, 1.0e-4));
    var tall: CoverMaterial;
    tall.patches = cover_look(row).patches;
    tall.clump = cover_look(row).clump;
    let clumped = cover_clumped(spec, cover_tall(clump, tall, scale, 0.0));
    let width = spec.shape.w * scale * clumped.height;
    let height = mix(spec.shape.y, spec.shape.z, unit(hash(seed ^ 0xc2b2ae35u)))
        * width / spec.shape.w * rise;
    if height <= 0.0 {
        return;
    }
    let span = height + width;
    let middle = root + vec3<f32>(0.0, 0.5 * height, 0.0);
    if !drawn_view_sees(0u, middle - vec3<f32>(span), middle + vec3<f32>(span)) {
        return;
    }
    if sited {
        colors = ground_colors(run.terrain, run.levels.w, rooted, 0u);
    }
    let variation = site_shift((1.0 + spec.extra.y * min(width / finer, 1.0)
        * (unit(hash(seed ^ 0x9e3779b9u)) - 0.5))
        * (macro_color(colors) * cover_tint(colors.a, 1.0)), run.levels.z, material, site);
    let turned = unit(hash(seed ^ 0x85ebca6bu));
    let toward = 2.0 * PI * unit(hash(clump.seed ^ 0x85ebca6bu));
    let lean = spec.clump.z * unit(hash(clump.seed ^ 0xc2b2ae35u))
        * vec2<f32>(cos(toward), sin(toward))
        + spec.clump.w * (world_xz - clump.center) / spec.clump.x + fan;
    let sway = 0.5 * spec.tint.w * run.grow.wind.z;
    var gusts = vec2<f32>(0.0);
    if sway != 0.0 {
        gusts = sway * vec2<f32>(cover_gust(root, run.grow.wind.x),
            cover_gust(root, run.grow.wind.y));
    }
    let leaned = cover_lean(unorm8(lean.x * 0.5 + 0.5), unorm8(lean.y * 0.5 + 0.5),
        unorm8(gusts.x), unorm8(gusts.y));
    let left = select(0.0, 1.0 - shown / drawn.x, drawn.x > 0.0);
    let fall = cover_fall(0.5 * cover_leaves(cover_row(run.grow.counts.z, material))
        * run.density * patches.density * growth
        * clumped.height * clumped.height * scale * scale);
    let seen = cover_seen(cover_depth(dense * (drawn.x - shown) * spec.shape.w * scale
        * mean * scale, sight.x), fall);
    lane_grew = true;
    let tint = spec.tint.rgb * variation * clumped.tint;
    let lit = mix(exp(-fall), seen.occlusion, left);
    let dice = hash(seed ^ 0x1b873593u);
    if spec.cells.y > 0u {
        let fewest = spec.cells.w & 0xffffu;
        let made = fewest + dice % ((spec.cells.w >> 16u) - fewest + 1u);
        let bin = spec.cells.z;
        let placed = atomicAdd(&counts.carded[bin], made);
        let shape = cover_shape();
        var card: CoverCard;
        card.size = pack2x16float(vec2<f32>(height, spec.drift.w));
        card.lean = leaned;
        card.tint = pack_color(CARD_TINT_RANGE * tint, lit);
        for (var leaf = 0u; leaf < made; leaf++) {
            let roll = hash(dice + leaf * 0x9e3779b1u);
            let stands = mix(spec.stand.x, spec.stand.y, unit(roll));
            let lifted = mix(spec.stand.z, spec.stand.w, unit(hash(roll ^ 0x85ebca6bu)));
            let around = fract(turned + f32(leaf) / f32(made) * spec.fans.z);
            card.root = root + vec3<f32>(0.0, lifted, 0.0);
            card.facing = cover_facing(u32(round(around * 65535.0)), unorm8(stands), 0u);
            card.cell = (spec.cells.x + COVER_CELL_ROWS * (roll % spec.cells.y))
                | (hash(roll ^ 0x27d4eb2fu) & CARD_MIRRORED);
            if placed + leaf < run.grow.room.y {
                mark_casts(grows[0].room, card_bounds(card, shape), BLADE_KINDS + bin,
                    placed + leaf, placed + leaf);
                cards[bin * run.grow.room.y + placed + leaf] = card;
            }
        }
        return;
    }
    var tilt = vec2<u32>(0u);
    if run.grow.terms.z > 0.0 {
        tilt = grounded_tilt(ground_normal(run.grow, run.terrain, rooted));
    }
    let facing = cover_facing(u32(round(turned * 65535.0)), tilt.x, tilt.y);
    let calibrated = material_look(run.levels.z, material).far.rgb;
    let ground_color = calibrated * variation;
    let tip_color = calibrated * tint;
    let wide = width * (1.0 + spec.form.w * (unit(hash(dice)) - 0.5));
    var blade: CoverBlade;
    blade.root = root;
    blade.size = pack2x16float(vec2<f32>(height, max(wide * rise, finer))) & BLADE_SIZE;
    blade.facing = facing;
    blade.lean = leaned;
    blade.root_color = (pack_color(mix(ground_color, mix(ground_color, tip_color, seen.tip),
        left), 0.0) & 0x00ffffffu) | (blade_form(spec.form, dice) << 24u);
    blade.tip_color = pack_color(tip_color, lit);
    let flattened = fits(run.view, height / run.steps.y, root);
    let pieces = select(2u, 1u, flattened || fits(run.view, height / run.steps.x, root));
    if atomicAdd(&counts.appended, pieces) + pieces > run.grow.room.x {
        return;
    }
    blade = shaped_blade(blade, cover_shape(),
        vec3<f32>(grow_frame.wind.x, 0.0, grow_frame.wind.y));
    if flattened {
        let flat = atomicAdd(&counts.flat, 1u);
        hold_casts(blade_bounds(blade, cover_shape()), 1u, flat, flat);
        blades[run.grow.room.x - 1u - flat] = blade;
        return;
    }
    let first = atomicAdd(&counts.whole, pieces);
    hold_casts(blade_bounds(blade, cover_shape()), 0u, first, first + pieces - 1u);
    if pieces == 2u {
        blade.size |= BLADE_HALF;
        var upper = blade;
        upper.size |= BLADE_UPPER;
        blades[first + 1u] = upper;
    }
    blades[first] = blade;
}

struct Pick {
    at: vec2<u32>,
    order: u32,
    candidate: u32,
    owner: vec3<u32>,
}

const WALK_LOOK: u32 = 0u;
const WALK_CLUMP: u32 = 1u;
const WALK_QUAD: u32 = 2u;
const WALK_TRY: u32 = 3u;

struct Walk {
    stage: u32,
    look: u32,
    clump: u32,
    quad: u32,
    order: u32,
    next: u32,
    tried: u32,
}

struct WalkLook {
    row: u32,
    accents: vec4<u32>,
    density: f32,
    size: f32,
    patches: CoverPatchRange,
    first: vec2<i32>,
    wide: u32,
    count: u32,
}

fn walk_look(planned: vec4<u32>, at: u32) -> WalkLook {
    var walked: WalkLook;
    walked.row = select(planned.z, planned.w, at == 1u);
    if walked.row == COVER_NO_LAYER {
        return walked;
    }
    let look = cover_look(walked.row);
    let size = look.clump.x;
    let reach = cell_plan.reaches[at];
    let low = run.grow.origin.xz + run.corner - REACH_ROUNDING;
    let high = low + run.spacing + 2.0 * REACH_ROUNDING;
    let first = vec2<i32>(floor(low / size)) - vec2<i32>(1);
    let last = vec2<i32>(floor(high / size)) + vec2<i32>(1);
    walked.accents = look.accents;
    walked.density = look.shape.x;
    walked.size = size;
    walked.patches = CoverPatchRange(reach.x, reach.y, reach.z);
    walked.first = first;
    walked.wide = u32(last.x - first.x + 1);
    walked.count = walked.wide * u32(last.y - first.y + 1);
    return walked;
}

struct WalkClump {
    kept: bool,
    id: vec2<i32>,
    limit: u32,
    point: vec2<f32>,
    lowest: vec2<i32>,
    highest: vec2<i32>,
}

fn walk_clump(look: WalkLook, index: u32, cell: vec2<u32>) -> WalkClump {
    var clump: WalkClump;
    let size = look.size;
    let quad = run.grow.origin.w;
    let id = look.first + vec2<i32>(vec2<u32>(index % look.wide, index / look.wide));
    let kind = cover_accent_row(look.accents,
        unit(hash(cover_clump_seed(id, 2u * run.slot) ^ 0x9e3779b9u)));
    if kind == COVER_NO_LAYER {
        return clump;
    }
    let bound = cover_bound(kind, run.growth, look.patches, run.closest);
    if bound.x <= 0.0 {
        return clump;
    }
    let start = vec2<i32>(cell * run.side);
    let end = start + vec2<i32>(i32(run.side) - 1);
    let near = (vec2<f32>(id - vec2<i32>(1)) * size - run.grow.origin.xz - REACH_ROUNDING)
        / quad;
    let far = (vec2<f32>(id + vec2<i32>(2)) * size - run.grow.origin.xz + REACH_ROUNDING)
        / quad;
    clump.lowest = max(vec2<i32>(floor(near)), start);
    clump.highest = min(vec2<i32>(floor(far)), end);
    clump.kept = all(clump.lowest <= clump.highest);
    clump.id = id;
    clump.limit = u32(ceil(f32(run.finest) * min(run.tiered, min((1.0 + ROUNDING)
        * look.density * bound.x * run.sown, run.stacked / bound.y))));
    clump.point = cover_clump_point(id, size, 2u * run.slot) - run.grow.origin.xz;
    return clump;
}

fn walk_next(walk: ptr<function, Walk>, planned: vec4<u32>, cell: vec2<u32>, lane: u32,
        pick: ptr<function, Pick>) -> bool {
    var made = false;
    loop {
        if (*walk).look >= 2u {
            break;
        }
        let look = walk_look(planned, (*walk).look);
        if (*walk).stage == WALK_LOOK {
            (*walk).clump = lane;
            (*walk).stage = WALK_CLUMP;
        }
        if (*walk).clump >= look.count {
            (*walk).look += 1u;
            (*walk).stage = WALK_LOOK;
            continue;
        }
        let clump = walk_clump(look, (*walk).clump, cell);
        if !clump.kept {
            (*walk).clump += 64u;
            (*walk).stage = WALK_CLUMP;
            continue;
        }
        let span = clump.highest - clump.lowest + vec2<i32>(1);
        if (*walk).stage == WALK_CLUMP {
            (*walk).quad = 0u;
            (*walk).stage = WALK_QUAD;
        }
        if (*walk).quad >= u32(span.x * span.y) {
            (*walk).clump += 64u;
            (*walk).stage = WALK_CLUMP;
            continue;
        }
        let quad = (*walk).quad;
        let at = vec2<u32>(clump.lowest
            + vec2<i32>(vec2<u32>(quad % u32(span.x), quad / u32(span.x))));
        if (*walk).stage == WALK_QUAD {
            let reach = run.grow.origin.w;
            let furthest = CLUMP_REACH * look.size + REACH_ROUNDING;
            let gap = max(abs((vec2<f32>(at) + 0.5) * reach - clump.point) - 0.5 * reach,
                vec2<f32>(0.0));
            if dot(gap, gap) > furthest * furthest {
                (*walk).quad += 1u;
                continue;
            }
            (*walk).order = quad_order(at, run.depth, run.slot, clump.limit);
            (*walk).next = (*walk).order;
            (*walk).stage = WALK_TRY;
        }
        if (*walk).next >= clump.limit {
            (*walk).quad += 1u;
            (*walk).stage = WALK_QUAD;
            continue;
        }
        *pick = Pick(at, (*walk).order, (*walk).next >> countTrailingZeros(run.finest),
            vec3<u32>(look.row, bitcast<vec2<u32>>(clump.id)));
        (*walk).next += run.finest;
        (*walk).tried += 1u;
        made = true;
        break;
    }
    return made;
}

@compute @workgroup_size(1)
fn gather_cells() {
    let rows = grows[0].cells.w;
    let widest = atomicLoad(&grow_runs[3u * rows]);
    var total = 0u;
    for (var row = 0u; row < rows; row++) {
        atomicStore(&grow_runs[3u * row + 1u], total);
        total += atomicLoad(&grow_runs[3u * row]);
    }
    atomicStore(&grow_runs[3u * rows], min(total, widest));
    atomicStore(&grow_runs[3u * rows + 1u], (total + widest - 1u) / widest);
    atomicStore(&grow_runs[3u * rows + 2u], 1u);
}

fn take_spot(index: u32) {
    var low = 0u;
    var high = grows[0].cells.w - 1u;
    while low < high {
        let middle = (low + high + 1u) / 2u;
        if grow_starts[3u * middle + 1u] <= index {
            low = middle;
        } else {
            high = middle - 1u;
        }
    }
    let at = index - grow_starts[3u * low + 1u];
    if at >= grow_starts[3u * low] {
        cell_found = vec4<u32>(low, CELL_LEFT, 0u, 0u);
        return;
    }
    let cells = grows[low].cells;
    let held = grow_spots[cells.x + at];
    cell_fine = held.fine;
    cell_coarse = held.coarse.xyz;
    cell_seat = held.seat;
    atomicStore(&cell_low, held.under.x);
    atomicStore(&cell_high, held.under.y);
    atomicStore(&cell_named, held.under.z);
    atomicStore(&cell_driest, held.lush.x);
    atomicStore(&cell_lushest, held.lush.y);
    atomicStore(&cell_driest_near, held.lush.z);
    atomicStore(&cell_lushest_near, held.lush.w);
    let local = held.under.w;
    let cell = vec2<u32>(grows[low].grid.xy + vec2<i32>(vec2<u32>(local % cells.y,
        local / cells.y)));
    cell_found = vec4<u32>(low, held.coarse.w, cell);
}

@compute @workgroup_size(64)
fn grow_cover(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>,
        @builtin(local_invocation_index) lane: u32) {
    if lane == 0u {
        take_spot(group.y * groups.x + group.x);
    }
    let found = workgroupUniformLoad(&cell_found);
    if found.y == CELL_LEFT {
        return;
    }
    let grow = grows[found.x];
    let scale = grow.tier.y;
    let side = u32(scale);
    let depth = u32(grow.terms.w);
    let slot = grow.counts.w;
    let spacing = grow.origin.w * scale;
    let cell = found.zw;
    let corner = vec2<f32>(cell) * spacing;
    let middle = corner + vec2<f32>(0.5 * spacing);
    let center = grow.origin.xz + middle;
    let fine = cell_fine;
    let seat = cell_seat;
    let cell_center = vec3<f32>(center.x, grow.origin.y + seat.x, center.y);
    let closest = seat.y;
    let ring = seat.z;
    let info = terrain_info(grow.counts.x);
    let terrain = GroundTerrain(info.levels.x, grow.counts.y, grow.origin.w);
    if found.y == CELL_UNSUMMED {
        scan_under(grow, fine, cell_coarse, corner, closest, lane);
        workgroupBarrier();
    }
    if side <= TILE_QUADS {
        range_patches(grow, corner, lane);
    }
    if lane >= CORNER_LANE {
        hold_corner(terrain, corner, spacing, cell_center.y - grow.origin.y, lane - CORNER_LANE);
    }
    workgroupBarrier();
    if lane == 0u {
        plan_cell(grow, cell, cell_center, closest, found.y == CELL_UNSUMMED);
    }
    let planned = workgroupUniformLoad(&cell_run);
    let total = planned.x;
    if total == 0u && all(planned.zw == vec2<u32>(COVER_NO_LAYER)) {
        return;
    }
    let quads = cell_quads(grow, fine, cell);
    hold_sites(fine, quads, lane);
    hold_heights(LevelTile(fine.x, fine.yz), quads, lane);
    var grounds = cell_samples(grow, corner, spacing, fine.w, fine.yz, ring);
    if planned.y != MIXED || side > TILE_QUADS
            || grounds.count.x * grounds.count.y > GROUND_HELD {
        grounds.count = vec2<u32>(0u);
    }
    hold_grounds(fine, grounds, lane);
    workgroupBarrier();
    let header = cover_header();
    let area = grow.origin.w * grow.origin.w;
    let thin = grow.detail.w;
    let lattice = u32(grow.grid.w);
    let finest = side * side;
    run.grow = grow;
    run.view = grow_detail(grow);
    run.terrain = terrain;
    run.whole = GroundTile(true, fine.w, LevelTile(fine.x, fine.yz));
    run.grounds = grounds;
    run.quads = quads;
    run.levels = info.levels;
    run.growth = cell_plan.growth;
    run.steps = cover_shape().steps;
    run.normal = cell_plan.lie.xyz;
    run.corner = corner;
    run.extent = ground_extent(terrain);
    run.spacing = spacing;
    run.height = cell_center.y;
    run.tail = cell_plan.lie.w;
    run.slack = cell_plan.slack;
    run.ring = ring;
    run.closest = closest;
    run.area = area;
    run.layers = header.layers.x;
    run.lush = header.lush.w;
    run.density = header.shading.x;
    run.sown = header.shading.x * area * thin;
    run.crowd = header.shading.x * area * cover_growth(1.0) * thin;
    run.tiered = cell_plan.tiered;
    run.stacked = cell_plan.stacked;
    run.grow_in = header.layers.y;
    run.side = side;
    run.slot = slot;
    run.depth = depth;
    run.finest = finest;
    run.lattice = lattice;
    run.turn = clamp(u32(round(f32(lattice) * LATTICE_TURN)), 1u, max(lattice, 2u) - 1u);
    run.alone = planned.y;
    lane_material = MIXED;
    lane_grew = false;
    let lane_levels = min(depth, LANE_LEVELS);
    let below = depth - lane_levels;
    let home = ordered_cell(cell, depth, lane_levels, lane, slot);
    let walking = any(planned.zw != vec2<u32>(COVER_NO_LAYER));
    var walk: Walk;
    var plain = lane;
    let finest_bits = countTrailingZeros(finest);
    loop {
        var pick: Pick;
        if walking {
            if !walk_next(&walk, planned, cell, lane, &pick) {
                break;
            }
        } else {
            if plain >= total {
                break;
            }
            let order = plain & (finest - 1u);
            pick = Pick(ordered_cell(home, below, below, order >> (2u * lane_levels), slot), order,
                plain >> finest_bits, NO_OWNER);
            plain += 64u;
        }
        grow_candidate(pick.at, pick.order, pick.candidate, pick.owner);
    }
    if walk.tried > 0u {
        atomicAdd(&counts.tried, walk.tried);
    }
    if lane_grew {
        count_grown();
    }
    workgroupBarrier();
    cast_cell(grows[0].room, lane);
}

fn cast_args(at: u32, vertices: u32, local: u32, locals: u32) -> DrawArgs {
    let end = atomicLoad(&cover_casts[at + 1u]);
    if end == 0u {
        return DrawArgs(vertices, 0u, local * COVER_DRAW_VERTICES, 0u);
    }
    var first = ~atomicLoad(&cover_casts[at]);
    if first > (0xffffffffu / COVER_DRAW_VERTICES) / locals - 1u {
        first = 0u;
    }
    return DrawArgs(vertices, end - first, (first * locals + local) * COVER_DRAW_VERTICES, 0u);
}

@compute @workgroup_size(1)
fn finish_cover() {
    let room = grows[0].room;
    blades[room.x] = view_header();
    args[0] = DrawArgs(2u * STRAIGHT_JOINTS, atomicLoad(&counts.whole), 0u, 0u);
    args[1] = DrawArgs(2u * FLAT_JOINTS, atomicLoad(&counts.flat), COVER_DRAW_VERTICES, 0u);
    for (var bin = 0u; bin < room.z; bin++) {
        args[2u + bin] = DrawArgs(2u * CARD_JOINTS, min(atomicLoad(&counts.carded[bin]), room.y),
            bin * COVER_DRAW_VERTICES, 0u);
    }
    let kinds = BLADE_KINDS + room.z;
    for (var view = 0u; view < room.w; view++) {
        let at = kinds * (view + 1u);
        let ranges = 2u * kinds * view;
        args[at] = cast_args(ranges, 2u * STRAIGHT_JOINTS, 0u, BLADE_KINDS);
        args[at + 1u] = cast_args(ranges + 2u, 2u * FLAT_JOINTS, 1u, BLADE_KINDS);
        for (var bin = 0u; bin < room.z; bin++) {
            args[at + 2u + bin] = cast_args(ranges + 2u * (2u + bin), 2u * CARD_JOINTS, bin,
                room.z);
        }
    }
    counts.capacity = room.x;
    counts.card_capacity = room.y;
}
