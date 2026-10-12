//! What a frame wants of each terrain, decided without changing anything:
//! from the view, the tile budget and the focus points, the tiles it loads
//! in order (the urgent first) and those it keeps, collision's tiles, what
//! is missing with its jobs' priorities and what waits, and the tiles
//! workers may generate ahead of need, in the order the terrain's
//! documentation gives.

use super::*;

/// What a frame plans one terrain's tiles from.
pub(super) struct PlanInput<'a> {
    pub(super) settings: &'a TerrainSettings,
    pub(super) pyramid: Pyramid,
    /// Where the terrain lies.
    pub(super) origin: Vec3,
    pub(super) ranges: &'a LodRanges,
    /// The eye and where it is headed, in the world.
    pub(super) eye: Vec3,
    pub(super) ahead: Vec3,
    /// The view's horizontal direction and its planes.
    pub(super) forward: Vec2,
    pub(super) planes: &'a ViewPlanes,
    /// The ground surroundings leave to the terrain they surround.
    pub(super) cutout: Option<(Vec2, Vec2)>,
    /// Points collision's tiles load around (the eye first, then focus
    /// entities), and places games name with their reach.
    pub(super) focus: &'a [Vec3],
    pub(super) places: &'a [(Vec3, f32)],
    /// Whether the terrain collides, and a list collision's tiles fill
    /// (the last frame's, to reuse).
    pub(super) collides: bool,
    pub(super) collision: Vec<TileKey>,
    /// Whether nothing was missing nor generating at the last frame, with
    /// workers: tiles within the idle reach generate ahead of need.
    pub(super) quiet: bool,
    /// Tiles the frame's budget leaves the terrain.
    pub(super) remaining: usize,
    /// The finest levels waiting at the eye's speed, and whether it has yet
    /// to settle.
    pub(super) waiting: u32,
    pub(super) unsettled: bool,
    /// Its tiles: the ground they hold, and which still load.
    pub(super) store: &'a TileStore,
}

/// What a frame wants of one terrain.
pub(super) struct Plan {
    /// The level-0 tiles collision keeps.
    pub(super) collision: Vec<TileKey>,
    /// How many tiles the view wants, over the budget or not.
    pub(super) ordered: usize,
    /// Every tile it loads, and every tile it keeps.
    pub(super) target: HashSet<TileKey>,
    pub(super) keep: HashSet<TileKey>,
    /// What still loads, most important first, with its jobs' priorities,
    /// and how many tiles wait for the eye to slow down.
    pub(super) missing: Vec<(TileKey, u32)>,
    pub(super) waiting: u32,
    /// The tiles within the idle reach, and those places games name want.
    pub(super) around: Vec<WantedTile>,
    pub(super) named: Vec<WantedTile>,
}

/// The tiles each reach of the view wants.
struct Wanted {
    /// The camera's within the loading reach, where it is headed, within
    /// the keeping reach and the idle reach, and what places games name
    /// want.
    load: Vec<WantedTile>,
    headed: Vec<WantedTile>,
    held: Vec<WantedTile>,
    around: Vec<WantedTile>,
    named: Vec<WantedTile>,
}

/// What a frame wants of the terrain `input` describes.
pub(super) fn plan(mut input: PlanInput<'_>) -> Plan {
    let mut collision = std::mem::take(&mut input.collision);
    let (loading, under) = collision_tiles(&input, &mut collision);
    let input = &input;
    let pyramid = &input.pyramid;
    let wanted = wanted(input);
    let tier = |pick: fn(&WantedTile) -> bool| {
        wanted
            .load
            .iter()
            .filter(move |tile| pick(tile))
            .map(|tile| tile.key)
    };
    // What to load, most important first, each tile after its ancestors:
    // the pinned levels; the tiles the view draws and shows, then those
    // under focus entities, which jobs take up first; the tiles it draws
    // out of sight, then collision's; those only shading what it shows;
    // where the eye is headed; the rest.
    let mut order: Vec<TileKey> = Vec::with_capacity(wanted.load.len() + loading.len());
    let mut pushed: HashSet<TileKey> =
        HashSet::with_capacity_and_hasher(order.capacity(), Default::default());
    for key in tier(|tile| tile.score < 0.0)
        .chain(tier(|tile| tile.score >= 0.0 && tile.drawn && tile.seen))
        .chain(under.iter().copied())
    {
        push_with_ancestors(pyramid, key, &mut order, &mut pushed);
    }
    let urgent = order.len();
    for key in tier(|tile| tile.score >= 0.0 && tile.drawn && !tile.seen)
        .chain(loading.iter().copied())
        .chain(tier(|tile| tile.score >= 0.0 && !tile.drawn && tile.seen))
        .chain(wanted.headed.iter().map(|tile| tile.key))
        .chain(tier(|tile| tile.score >= 0.0 && !tile.drawn && !tile.seen))
    {
        push_with_ancestors(pyramid, key, &mut order, &mut pushed);
    }
    // Over a tile budget, the pinned levels and collision's tiles stay
    // first, then the rest in order.
    let target: HashSet<TileKey> = if order.len() <= input.remaining {
        order.iter().copied().collect()
    } else {
        let mut first = Vec::new();
        let mut held = HashSet::default();
        for key in tier(|tile| tile.score < 0.0).chain(loading.iter().copied()) {
            push_with_ancestors(pyramid, key, &mut first, &mut held);
        }
        first
            .iter()
            .copied()
            .chain(order.iter().copied().filter(|key| !held.contains(key)))
            .take(input.remaining)
            .collect()
    };
    // What to keep: everything within a wider reach, unless over budget.
    let keep: HashSet<TileKey> = wanted
        .held
        .iter()
        .map(|tile| tile.key)
        .chain(collision.iter().copied())
        .chain(target.iter().copied())
        .collect();
    // While the eye moves fast, the finest levels' tiles wait, and until it
    // settles those it does not show; the pinned levels, collision's tiles
    // and those under focus entities, with their ancestors, never wait.
    let (waiting, unsettled) = (input.waiting, input.unsettled);
    let mut never = HashSet::default();
    let mut shown = HashSet::default();
    if waiting > 0 || unsettled {
        let mut chain = Vec::new();
        for key in tier(|tile| tile.score < 0.0)
            .chain(loading.iter().copied())
            .chain(under.iter().copied())
        {
            push_with_ancestors(pyramid, key, &mut chain, &mut never);
        }
        if unsettled {
            chain.clear();
            for key in tier(|tile| tile.seen) {
                push_with_ancestors(pyramid, key, &mut chain, &mut shown);
            }
        }
    }
    let waits = |key: &TileKey| {
        (waiting > 0 || unsettled)
            && !never.contains(key)
            && (u32::from(key.level) < waiting || (unsettled && !shown.contains(key)))
    };
    // What still loads, most important first; disabled terrains load
    // nothing. Tiles under edits since they loaded regenerate in their turn
    // and draw until they do.
    let store = input.store;
    let waiting_tiles = order
        .iter()
        .filter(|key| target.contains(key) && store.loads(key) && waits(key))
        .count() as u32;
    let missing: Vec<(TileKey, u32)> = order
        .iter()
        .enumerate()
        .filter(|(_, key)| {
            input.settings.enabled && target.contains(key) && store.loads(key) && !waits(key)
        })
        .map(|(index, &key)| {
            let class = if index < urgent {
                URGENT_JOBS
            } else {
                VIEW_JOBS
            };
            (key, job_priority(class, index as u32))
        })
        .collect();
    Plan {
        collision,
        ordered: order.len(),
        target,
        keep,
        missing,
        waiting: waiting_tiles,
        around: wanted.around,
        named: wanted.named,
    }
}

/// Collision's tiles around `input`'s focus points: those it keeps, in
/// `collision`, then those it loads and the level-0 tiles under the
/// points. They load within its radius of the camera and focus entities
/// and stay, colliding, a quarter farther, as the camera's tiles do: focus
/// moving to and fro across a tile's edge must not regenerate it each time.
fn collision_tiles(
    input: &PlanInput<'_>,
    collision: &mut Vec<TileKey>,
) -> (Vec<TileKey>, Vec<TileKey>) {
    collision.clear();
    let (mut loading, mut under) = (Vec::new(), Vec::new());
    let radius = input.settings.collision_radius;
    if input.collides && radius > 0.0 {
        let held = radius * KEEP_REACH / LOAD_REACH;
        for point in input.focus {
            let local = *point - input.origin;
            collision_keys(&input.pyramid, local, radius, &mut loading);
            collision_keys(&input.pyramid, local, held, collision);
            under.extend(input.pyramid.tile_at(0, Vec2::new(local.x, local.z)));
        }
        loading.sort_unstable();
        loading.dedup();
        collision.sort_unstable();
        collision.dedup();
    }
    (loading, under)
}

/// What the view wants: the camera's tiles within their reach of the eye
/// and where it is headed, what it keeps within a wider reach, ahead of
/// need what lies within the idle reach while the terrain is quiet, and
/// what places games name want; distances run to each tile's ground.
fn wanted(input: &PlanInput<'_>) -> Wanted {
    let (pyramid, origin, store) = (&input.pyramid, input.origin, input.store);
    let ground = |key: TileKey| ground_bounds(store, pyramid, key);
    let seen = |low: Vec3, high: Vec3| input.planes.sees(low + origin, high + origin);
    let view = |eye: Vec3| WantedView {
        eye,
        forward: input.forward,
        ground: &ground,
        seen: &seen,
    };
    let want = |eye: Vec3, reach: f32| {
        let mut out = Vec::new();
        engine_terrain::wanted_tiles(
            pyramid,
            input.ranges,
            &view(eye),
            reach,
            input.settings.pinned_levels,
            input.cutout,
            &mut out,
        );
        out
    };
    let local_eye = input.eye - origin;
    Wanted {
        load: want(local_eye, LOAD_REACH),
        headed: if input.ahead == input.eye {
            Vec::new()
        } else {
            want(input.ahead - origin, LOAD_REACH)
        },
        held: want(local_eye, KEEP_REACH),
        around: if input.quiet && input.settings.idle_reach > KEEP_REACH {
            want(local_eye, input.settings.idle_reach)
        } else {
            Vec::new()
        },
        // What places games name want, each frame.
        named: input
            .places
            .iter()
            .flat_map(|&(place, reach)| want(place - origin, reach))
            .collect(),
    }
}

/// What workers generate ahead of need: the nearest tiles within the idle
/// reach and that places games name want, neither kept (`keep`) nor
/// resident nor kept in memory, as many as `fits`, so none memory holds
/// goes to make room for another.
pub(super) fn ahead_of_need(
    (around, named): (&[WantedTile], &[WantedTile]),
    keep: &HashSet<TileKey>,
    fits: usize,
    store: &TileStore,
) -> Vec<TileKey> {
    let mut nearest: Vec<&WantedTile> = around.iter().chain(named).collect();
    nearest.sort_by(|a, b| {
        a.score
            .total_cmp(&b.score)
            .then(b.key.level.cmp(&a.key.level))
    });
    let mut listed = HashSet::default();
    nearest
        .into_iter()
        .map(|tile| tile.key)
        .filter(|key| {
            listed.insert(*key)
                && !keep.contains(key)
                && !store.contains(key)
                && !store.in_memory(key)
        })
        .take(fits)
        .collect()
}

/// The lowest and highest ground tile `key` can hold,
/// terrain-local, as distances to it take them: a resident tile's own,
/// widened by how far its level measured from the next coarser (its
/// patches morph toward that); else its nearest resident ancestor's over
/// its ground, widened by how far the levels between measured apart and by
/// four of the ancestor's spacings, as finer levels add detail. `None`
/// without either.
fn ground_bounds(store: &TileStore, pyramid: &Pyramid, key: TileKey) -> Option<[f32; 2]> {
    let errors = store.errors();
    let error = |level: u8| errors.get(usize::from(level)).copied().unwrap_or(0.0);
    if let Some(tile) = store.get(&key) {
        let [low, high] = tile.bounds();
        return Some([low - error(key.level), high + error(key.level)]);
    }
    let size = pyramid.tile_size(u32::from(key.level));
    let low = Vec2::new(f32::from(key.x), f32::from(key.z)) * size;
    let high = low + size;
    let mut ancestor = key;
    let mut margin = 0.0;
    while let Some(parent) = pyramid.parent(ancestor) {
        margin += error(ancestor.level);
        ancestor = parent;
        let Some(tile) = store.get(&ancestor) else {
            continue;
        };
        let level = u32::from(ancestor.level);
        let spacing = pyramid.spacing(level);
        let patch = PATCH_QUADS as f32 * spacing;
        let corner =
            Vec2::new(f32::from(ancestor.x), f32::from(ancestor.z)) * pyramid.tile_size(level);
        let last = (PATCHES_PER_TILE - 1) as f32;
        let first = ((low - corner) / patch)
            .floor()
            .clamp(Vec2::ZERO, Vec2::splat(last));
        let end = ((high - corner) / patch - 1.0)
            .ceil()
            .clamp(Vec2::ZERO, Vec2::splat(last));
        let (mut bottom, mut top) = (f32::INFINITY, f32::NEG_INFINITY);
        for z in first.y as u32..=end.y as u32 {
            for x in first.x as u32..=end.x as u32 {
                let [patch_low, patch_high] = tile.patch_bounds(x, z);
                bottom = crate::math::min(bottom, patch_low);
                top = crate::math::max(top, patch_high);
            }
        }
        let margin = margin + 4.0 * spacing;
        return Some([bottom - margin, top + margin]);
    }
    None
}

/// The view's side planes and near plane, inward, in world space:
/// which boxes show in it, however far away.
pub(super) struct ViewPlanes(Option<[Vec4; 5]>);

impl ViewPlanes {
    /// `camera`'s view from `eye` (its own, unless frozen) on `viewport`.
    pub(super) fn new(camera: &Camera, eye: Vec3, viewport: Viewport) -> Self {
        let aspect = viewport.width as f32 / viewport.height.max(1) as f32;
        let from = Camera {
            position: eye,
            target: eye + (camera.target - camera.eye()),
            ..*camera
        };
        let rows = from.view_projection(aspect).transpose();
        let planes = [
            rows.w_axis + rows.x_axis,
            rows.w_axis - rows.x_axis,
            rows.w_axis + rows.y_axis,
            rows.w_axis - rows.y_axis,
            // Reversed depth: the near plane is depth 1.
            rows.w_axis - rows.z_axis,
        ];
        Self(
            planes
                .iter()
                .all(|plane| plane.is_finite())
                .then_some(planes),
        )
    }

    /// Whether any of the box from `low` to `high` lies inside every plane;
    /// always without planes.
    pub(super) fn sees(&self, low: Vec3, high: Vec3) -> bool {
        let Some(planes) = &self.0 else {
            return true;
        };
        planes.iter().all(|plane| {
            let corner = Vec3::new(
                if plane.x >= 0.0 { high.x } else { low.x },
                if plane.y >= 0.0 { high.y } else { low.y },
                if plane.z >= 0.0 { high.z } else { low.z },
            );
            plane.truncate().dot(corner) + plane.w >= 0.0
        })
    }
}

/// Appends the level-0 tiles within `radius` of terrain-local `point` to
/// `out`.
fn collision_keys(pyramid: &Pyramid, point: Vec3, radius: f32, out: &mut Vec<TileKey>) {
    let size = pyramid.tile_size(0);
    let [tiles_x, tiles_z] = pyramid.tiles(0);
    let low = ((Vec2::new(point.x, point.z) - radius) / size).floor();
    let high = ((Vec2::new(point.x, point.z) + radius) / size).floor();
    if high.x < 0.0 || high.y < 0.0 || low.x >= tiles_x as f32 || low.y >= tiles_z as f32 {
        return;
    }
    let first = [
        crate::math::max(low.x, 0.0) as u32,
        crate::math::max(low.y, 0.0) as u32,
    ];
    let last = [
        (high.x as u32).min(tiles_x - 1),
        (high.y as u32).min(tiles_z - 1),
    ];
    for z in first[1]..=last[1] {
        for x in first[0]..=last[0] {
            out.push(TileKey::new(0, x as u16, z as u16));
        }
    }
}

/// Pushes `key` onto `order` after those of its ancestors not yet there,
/// coarsest first: a tile draws only within its parent, so a
/// finer one loaded first would wait unused, while what stands on the
/// ground (blades, placed objects) already followed its finer heights.
pub(super) fn push_with_ancestors(
    pyramid: &engine_terrain::Pyramid,
    key: TileKey,
    order: &mut Vec<TileKey>,
    seen: &mut HashSet<TileKey>,
) {
    if seen.contains(&key) {
        return;
    }
    let mut chain = vec![key];
    let mut child = key;
    while let Some(parent) = pyramid.parent(child) {
        if seen.contains(&parent) {
            break;
        }
        chain.push(parent);
        child = parent;
    }
    for key in chain.into_iter().rev() {
        seen.insert(key);
        order.push(key);
    }
}
