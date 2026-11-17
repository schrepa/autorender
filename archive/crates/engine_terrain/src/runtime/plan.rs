use super::*;

pub(super) struct PlanInput<'a> {
    pub(super) settings: &'a TerrainSettings,
    pub(super) pyramid: Pyramid,
    pub(super) origin: Vec3,
    pub(super) ranges: &'a LodRanges,
    pub(super) eye: Vec3,
    pub(super) ahead: Vec3,
    pub(super) forward: Vec2,
    pub(super) planes: &'a ViewPlanes,
    pub(super) cutout: Option<(Vec2, Vec2)>,
    pub(super) focus: &'a [Vec3],
    pub(super) places: &'a [(Vec3, f32)],
    pub(super) collides: bool,
    pub(super) collision: Vec<TileKey>,
    pub(super) remaining: usize,
    pub(super) store: &'a TileStore,
}

pub(super) struct Plan {
    pub(super) collision: Vec<TileKey>,
    pub(super) ordered: usize,
    pub(super) target: HashSet<TileKey>,
    pub(super) keep: HashSet<TileKey>,
    pub(super) missing: Vec<(TileKey, u32)>,
    pub(super) fallback: u32,
    pub(super) named: Vec<WantedTile>,
}

struct Wanted {
    load: Vec<WantedTile>,
    headed: Vec<WantedTile>,
    held: Vec<WantedTile>,
    named: Vec<WantedTile>,
}

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
    let keep: HashSet<TileKey> = wanted
        .held
        .iter()
        .map(|tile| tile.key)
        .chain(collision.iter().copied())
        .chain(target.iter().copied())
        .collect();
    let store = input.store;
    let missing: Vec<(TileKey, u32)> = order
        .iter()
        .enumerate()
        .filter(|(_, key)| target.contains(key) && store.loads(key))
        .map(|(index, &key)| {
            let class = if index < urgent {
                URGENT_JOBS
            } else {
                VIEW_JOBS
            };
            (key, job_priority(class, index as u32))
        })
        .collect();
    let fallback = tier(|tile| tile.score < 0.0 || tile.drawn)
        .filter(|key| target.contains(key) && store.get(key).is_none())
        .count() as u32;
    Plan {
        collision,
        ordered: order.len(),
        target,
        keep,
        missing,
        fallback,
        named: wanted.named,
    }
}

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
        crate::wanted_tiles(
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
        named: input
            .places
            .iter()
            .flat_map(|&(place, reach)| want(place - origin, reach))
            .collect(),
    }
}

fn ground_bounds(store: &TileStore, pyramid: &Pyramid, key: TileKey) -> Option<[f32; 2]> {
    let errors = store.errors();
    let error = |level: u8| errors.get(usize::from(level)).copied().unwrap_or(0.0);
    if let Some(tile) = store.known(&key) {
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
                bottom = engine_math::min(bottom, patch_low);
                top = engine_math::max(top, patch_high);
            }
        }
        let margin = margin + 4.0 * spacing;
        return Some([bottom - margin, top + margin]);
    }
    None
}

pub(super) struct ViewPlanes(Option<[Vec4; 6]>);

impl ViewPlanes {
    pub(super) fn new(camera: &Camera, (eye, forward): (Vec3, Vec3), size: glam::UVec2) -> Self {
        let aspect = size.x as f32 / size.y.max(1) as f32;
        let planes = engine_core::frustum_planes(camera.view_projection(eye, forward, aspect));
        Self(
            planes[..5]
                .iter()
                .all(|plane| plane.is_finite())
                .then_some(planes),
        )
    }

    pub(super) fn sees(&self, low: Vec3, high: Vec3) -> bool {
        self.0
            .as_ref()
            .is_none_or(|planes| engine_core::frustum_sees(&planes[..5], low, high))
    }
}

fn collision_keys(pyramid: &Pyramid, point: Vec3, radius: f32, out: &mut Vec<TileKey>) {
    let size = pyramid.tile_size(0);
    let [tiles_x, tiles_z] = pyramid.tiles(0);
    let low = ((Vec2::new(point.x, point.z) - radius) / size).floor();
    let high = ((Vec2::new(point.x, point.z) + radius) / size).floor();
    if high.x < 0.0 || high.y < 0.0 || low.x >= tiles_x as f32 || low.y >= tiles_z as f32 {
        return;
    }
    let first = [
        engine_math::max(low.x, 0.0) as u32,
        engine_math::max(low.y, 0.0) as u32,
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

pub(super) fn push_with_ancestors(
    pyramid: &crate::Pyramid,
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
