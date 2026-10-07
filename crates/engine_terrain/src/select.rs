//! Which tiles a view needs and which patches it draws: continuous
//! distance-dependent level of detail (Strugar's CDLOD) over resident
//! tiles. Each level's patches morph into the next coarser level's over the
//! last quarter of the level's range, so neighbors of different levels meet
//! without cracks; where a finer tile is not resident yet, its coarser
//! ancestor draws instead.

use engine_math::{max, min, sin_cos};
use glam::{Vec2, Vec3};

use crate::{PATCH_QUADS, PATCHES_PER_TILE, Pyramid, Tile, TileKey};

/// How far each level draws: level `L` up to `range(L)` meters from the
/// eye, finer levels nearer. Ranges double per level, as the spacing does.
#[derive(Clone, Debug, PartialEq)]
pub struct LodRanges {
    ranges: Vec<f32>,
}

/// Where morphing starts, as a share of a level's range.
const MORPH_START: f32 = 0.75;
/// A morph range no vertex reaches: the coarsest level never morphs.
const NEVER: [f32; 2] = [1.0e30, 2.0e30];

impl LodRanges {
    /// The shortest range of a level in its sample spacings. Patches of a
    /// level meet the next coarser level's within a patch's diagonal past
    /// the finer level's range, and must have finished morphing there: with
    /// 16-quad patches and morphing over the last quarter, ranges must span
    /// over 91 spacings.
    pub const MIN_SPACINGS: f32 = 96.0;

    /// Ranges at which a level's samples lie `pixel_error` pixels apart on
    /// a viewport `viewport_height` pixels tall with vertical field of view
    /// `vertical_fov` radians: its triangles span `pixel_error` to twice as
    /// many pixels. Ranges never go below [`Self::MIN_SPACINGS`].
    pub fn new(
        pyramid: &Pyramid,
        pixel_error: f32,
        viewport_height: f32,
        vertical_fov: f32,
    ) -> Self {
        let (sin, cos) = sin_cos(0.5 * vertical_fov);
        let tangent = sin / max(cos, 1.0e-6);
        let spacings = max(
            viewport_height / max(2.0 * tangent * pixel_error, 1.0e-6),
            Self::MIN_SPACINGS,
        );
        let spacings = if spacings.is_finite() {
            spacings
        } else {
            Self::MIN_SPACINGS
        };
        Self {
            ranges: (0..pyramid.levels())
                .map(|level| spacings * pyramid.spacing(level))
                .collect(),
        }
    }

    /// Meters from the eye up to which `level` draws.
    pub fn range(&self, level: u32) -> f32 {
        self.ranges[level as usize]
    }

    pub fn levels(&self) -> u32 {
        self.ranges.len() as u32
    }

    /// The distances over which `level`'s vertices morph into the next
    /// coarser level's; never for the coarsest.
    pub fn morph(&self, level: u32) -> [f32; 2] {
        if level + 1 >= self.levels() {
            return NEVER;
        }
        let range = self.range(level);
        [range * MORPH_START, range]
    }
}

/// A patch a view draws: the tile holding it, its column and row there, the
/// distances over which its vertices morph and its terrain-local bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Patch {
    pub tile: TileKey,
    pub x: u8,
    pub z: u8,
    pub morph: [f32; 2],
    pub min: Vec3,
    pub max: Vec3,
    /// Whether any of its cells is a hole.
    pub holes: bool,
}

/// The patches a view draws, and how many of them stand in for finer
/// levels whose tiles were not resident.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Selection {
    pub patches: Vec<Patch>,
    pub fallbacks: u32,
}

/// How far below and above the eye ground of unknown height is taken to
/// reach (decision 112): any ground at all.
const UNKNOWN_GROUND: f32 = 1.0e6;

/// The squared distance from `point` to the box from `min` to `max`.
fn distance_squared(point: Vec3, min: Vec3, max: Vec3) -> f32 {
    let nearest = point.clamp(min, max);
    nearest.distance_squared(point)
}

/// Whether the square from `low` to `high` lies within `cutout`, the
/// ground a terrain's surroundings (decision 094) leave to it.
fn cut_out(cutout: Option<(Vec2, Vec2)>, low: Vec2, high: Vec2) -> bool {
    cutout.is_some_and(|(first, last)| {
        low.x >= first.x && low.y >= first.y && high.x <= last.x && high.y <= last.y
    })
}

/// The patch of `level` holding patch-grid position `(x, z)`: its tile and
/// column and row there.
fn patch_tile(level: u32, x: u32, z: u32) -> (TileKey, u32, u32) {
    (
        TileKey::new(
            level as u8,
            (x / PATCHES_PER_TILE) as u16,
            (z / PATCHES_PER_TILE) as u16,
        ),
        x % PATCHES_PER_TILE,
        z % PATCHES_PER_TILE,
    )
}

/// Selects the patches a view from terrain-local `eye` draws into
/// `selection`, reading resident tiles through `tile`. Patches start at the
/// coarsest level and split into their four children while the eye is
/// within the children's level's range and their tile is resident. They
/// come out nearest the eye first, so nearer ground hides what lies behind
/// it before it is shaded. Patches within `cutout` (terrain-local) never
/// draw: surroundings (decision 094) leave that ground to the terrain they
/// surround, whose edges every level's patches align with.
pub fn select_patches<'a>(
    pyramid: &Pyramid,
    ranges: &LodRanges,
    eye: Vec3,
    tile: impl Fn(TileKey) -> Option<&'a Tile>,
    cutout: Option<(Vec2, Vec2)>,
    selection: &mut Selection,
) {
    selection.patches.clear();
    selection.fallbacks = 0;
    let extent = pyramid.extent();
    let top = pyramid.levels() - 1;
    let [tiles_x, tiles_z] = pyramid.tiles(top);
    // Patches by level and patch-grid position, last in first out: the
    // coarsest level's farthest first, so its nearest pops first.
    let top_size = PATCH_QUADS as f32 * pyramid.spacing(top);
    let eye_xz = Vec2::new(eye.x, eye.z);
    let mut coarsest: Vec<(f32, u32, u32)> = (0..tiles_z * PATCHES_PER_TILE)
        .flat_map(|z| (0..tiles_x * PATCHES_PER_TILE).map(move |x| (x, z)))
        .map(|(x, z)| {
            let center = (Vec2::new(x as f32, z as f32) + 0.5) * top_size;
            (center.distance_squared(eye_xz), x, z)
        })
        .collect();
    coarsest.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then((b.2, b.1).cmp(&(a.2, a.1))));
    let mut stack: Vec<(u32, u32, u32)> = Vec::with_capacity(coarsest.len() + 64);
    stack.extend(coarsest.into_iter().map(|(_, x, z)| (top, x, z)));
    while let Some((level, x, z)) = stack.pop() {
        let spacing = pyramid.spacing(level);
        let size = PATCH_QUADS as f32 * spacing;
        let corner = Vec2::new(x as f32, z as f32) * size;
        if corner.x >= extent.x || corner.y >= extent.y || cut_out(cutout, corner, corner + size) {
            continue;
        }
        let (key, px, pz) = patch_tile(level, x, z);
        let Some(owner) = tile(key) else {
            continue;
        };
        let [mut low, mut high] = owner.patch_bounds(px, pz);
        // Morphing vertices move toward the parent level's heights.
        if level < top {
            let (parent, ppx, ppz) = patch_tile(level + 1, x / 2, z / 2);
            if let Some(parent) = tile(parent) {
                let [parent_low, parent_high] = parent.patch_bounds(ppx, ppz);
                low = min(low, parent_low);
                high = max(high, parent_high);
            }
        }
        let min_corner = Vec3::new(corner.x, low, corner.y);
        let max_corner = Vec3::new(corner.x + size, high, corner.y + size);
        if level > 0 {
            let reach = ranges.range(level - 1);
            if distance_squared(eye, min_corner, max_corner) < reach * reach {
                let (child, _, _) = patch_tile(level - 1, 2 * x, 2 * z);
                if tile(child).is_some() {
                    // The child nearest the eye pops first, then the other
                    // child of the nearer half along z, then the farther
                    // half's, nearer first.
                    let center = corner + size * 0.5;
                    let near_x = u32::from(eye.x >= center.x);
                    let near_z = u32::from(eye.z >= center.y);
                    let (far_x, far_z) = (1 - near_x, 1 - near_z);
                    stack.push((level - 1, 2 * x + far_x, 2 * z + far_z));
                    stack.push((level - 1, 2 * x + near_x, 2 * z + far_z));
                    stack.push((level - 1, 2 * x + far_x, 2 * z + near_z));
                    stack.push((level - 1, 2 * x + near_x, 2 * z + near_z));
                    continue;
                }
                selection.fallbacks += 1;
            }
        }
        selection.patches.push(Patch {
            tile: key,
            x: px as u8,
            z: pz as u8,
            morph: ranges.morph(level),
            min: min_corner,
            max: max_corner,
            holes: owner.patch_has_holes(px, pz),
        });
    }
}

/// A tile a view wants resident, with its importance (decision 108): the
/// pinned levels' first, coarser first; then by the distance to the eye in
/// units of its level's range, ahead of the eye before behind, so the
/// tiles that show the most of the view's detail load first whatever their
/// level, a tile's parent (twice as far in range, half as far in its
/// units) before it. And whether the view draws it, within its level's
/// range, and whether its ground shows in the view (decision 112).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WantedTile {
    pub key: TileKey,
    pub score: f32,
    pub drawn: bool,
    pub seen: bool,
}

/// How a view sees a terrain (decision 112): its eye (terrain-local) and
/// horizontal view direction, the lowest and highest ground a tile can
/// hold (`None` where unknown), and whether a terrain-local box shows in
/// the view.
pub struct WantedView<'a> {
    pub eye: Vec3,
    pub forward: Vec2,
    pub ground: &'a dyn Fn(TileKey) -> Option<[f32; 2]>,
    pub seen: &'a dyn Fn(Vec3, Vec3) -> bool,
}

/// Appends the tiles of every level within `reach` times its range of
/// `view`'s eye, and every tile of the coarsest `pinned` levels, to `out`,
/// most important first; none wholly within `cutout` (decision 094).
/// Distances run to a tile's ground (decision 112): a high eye wants fine
/// tiles only as far as it is from their ground, as patches draw and
/// ground shades by those distances.
pub fn wanted_tiles(
    pyramid: &Pyramid,
    ranges: &LodRanges,
    view: &WantedView<'_>,
    reach: f32,
    pinned: u32,
    cutout: Option<(Vec2, Vec2)>,
    out: &mut Vec<WantedTile>,
) {
    let start = out.len();
    let eye = view.eye;
    let eye2 = Vec2::new(eye.x, eye.z);
    let forward = view.forward.try_normalize().unwrap_or(Vec2::ZERO);
    for level in (0..pyramid.levels()).rev() {
        let size = pyramid.tile_size(level);
        let [tiles_x, tiles_z] = pyramid.tiles(level);
        let everywhere = level + pinned >= pyramid.levels();
        let radius = ranges.range(level) * reach;
        let (first, last) = if everywhere {
            ([0, 0], [tiles_x - 1, tiles_z - 1])
        } else {
            let low = ((eye2 - radius) / size).floor();
            let high = ((eye2 + radius) / size).floor();
            if high.x < 0.0 || high.y < 0.0 || low.x >= tiles_x as f32 || low.y >= tiles_z as f32 {
                continue;
            }
            (
                [max(low.x, 0.0) as u32, max(low.y, 0.0) as u32],
                [
                    (high.x as u32).min(tiles_x - 1),
                    (high.y as u32).min(tiles_z - 1),
                ],
            )
        };
        for z in first[1]..=last[1] {
            for x in first[0]..=last[0] {
                let low = Vec2::new(x as f32, z as f32) * size;
                let high = low + size;
                if cut_out(cutout, low, high) {
                    continue;
                }
                let key = TileKey::new(level as u8, x as u16, z as u16);
                let nearest = eye2.clamp(low, high);
                let ground = (view.ground)(key);
                // Above or below its ground, the eye is that much farther.
                let rise = ground.map_or(0.0, |[bottom, top]| {
                    max(max(bottom - eye.y, eye.y - top), 0.0)
                });
                let across = nearest.distance(eye2);
                let distance = (across * across + rise * rise).sqrt();
                if !everywhere && distance > radius {
                    continue;
                }
                let toward = ((low + high) * 0.5 - eye2)
                    .try_normalize()
                    .unwrap_or(forward);
                let behind = 1.0 - forward.dot(toward);
                // Pinned tiles come first, whatever their distance.
                let score = if everywhere {
                    -1.0
                } else {
                    distance / max(ranges.range(level), 1.0e-3) * (1.0 + behind)
                };
                let [bottom, top] = ground.unwrap_or([-UNKNOWN_GROUND, UNKNOWN_GROUND]);
                out.push(WantedTile {
                    key,
                    score,
                    drawn: distance <= ranges.range(level),
                    seen: (view.seen)(
                        Vec3::new(low.x, bottom, low.y),
                        Vec3::new(high.x, top, high.y),
                    ),
                });
            }
        }
    }
    out[start..].sort_by(|a, b| {
        a.score
            .total_cmp(&b.score)
            .then(b.key.level.cmp(&a.key.level))
    });
}
