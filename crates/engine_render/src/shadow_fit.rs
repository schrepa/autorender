//! Placement of the sun's shadow maps, independent of GPU
//! objects: the light-space basis, the regions each map covers and their
//! orthographic projections.

use engine_core::{
    glam::{Mat4, Vec3, Vec4},
    math,
};

use crate::gpu_data::{FAR_DEPTH, MAX_LEVELS, NEAR_DEPTH, frustum_uniform};
pub(crate) use crate::gpu_layouts::constants::SHADOW_TILE_WORDS as TILE_WORDS;
pub(crate) use crate::gpu_layouts::constants::SHADOW_TILES as TILES;

/// World to light space: x and y across the sun's view, z away from the sun.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LightBasis {
    pub rows: [Vec3; 3],
}

impl LightBasis {
    /// The basis of a sun in (nonzero, finite) direction `toward_sun`.
    pub(crate) fn new(toward_sun: Vec3) -> Self {
        let away = -toward_sun.normalize();
        let reference = if away.y.abs() > 0.99 {
            Vec3::Z
        } else {
            Vec3::Y
        };
        let right = away.cross(reference).normalize();
        let up = right.cross(away);
        Self {
            rows: [right, up, away],
        }
    }

    pub(crate) fn apply(&self, point: Vec3) -> Vec3 {
        Vec3::new(
            self.rows[0].dot(point),
            self.rows[1].dot(point),
            self.rows[2].dot(point),
        )
    }

    /// Light-space bounds of the box `min`..`max`.
    pub(crate) fn bounds(&self, min: Vec3, max: Vec3) -> (Vec3, Vec3) {
        let center = self.apply((min + max) * 0.5);
        let half = (max - min) * 0.5;
        let extent = Vec3::new(
            self.rows[0].abs().dot(half),
            self.rows[1].abs().dot(half),
            self.rows[2].abs().dot(half),
        );
        (center - extent, center + extent)
    }
}

/// A light-space region a shadow map covers: a rectangle across the sun's
/// view and the depths of its receivers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MapRegion {
    /// Minimum x and y, maximum x and y.
    pub rect: [f32; 4],
    /// Nearest and farthest receiver depth from the sun.
    pub depth: [f32; 2],
}

impl MapRegion {
    /// World to this map's clip space: x and y span [-1, 1] across the
    /// rectangle and z spans [0, 1] over the receiver depths.
    pub(crate) fn clip(&self, basis: &LightBasis) -> Mat4 {
        let [x0, y0, x1, y1] = self.rect;
        let [z0, z1] = self.depth;
        let sx = 2.0 / (x1 - x0);
        let sy = 2.0 / (y1 - y0);
        let sz = 1.0 / (z1 - z0);
        let row = |axis: Vec3, scale: f32, offset: f32| (axis * scale).extend(-offset * scale);
        Mat4::from_cols(
            row(basis.rows[0], sx, (x0 + x1) * 0.5),
            row(basis.rows[1], sy, (y0 + y1) * 0.5),
            row(basis.rows[2], sz, z0),
            Vec4::W,
        )
        .transpose()
    }

    /// World size of one of `texels` texels across the larger side.
    pub(crate) fn texel(&self, texels: u32) -> f32 {
        let [x0, y0, x1, y1] = self.rect;
        math::max(x1 - x0, y1 - y0) / texels as f32
    }

    /// The region of the box `min`..`max`, widened by `margin` texels of a
    /// `texels`-wide map so filtering never reads past its casters.
    pub(crate) fn of_box(
        basis: &LightBasis,
        min: Vec3,
        max: Vec3,
        texels: u32,
        margin: f32,
    ) -> Self {
        let (low, high) = basis.bounds(min, max);
        let pad = math::max(high.x - low.x, high.y - low.y) / texels as f32 * margin;
        Self {
            rect: [low.x - pad, low.y - pad, high.x + pad, high.y + pad],
            depth: [low.z - pad, high.z + pad],
        }
    }

    /// The region of the sphere at `center` with `radius`, its rectangle
    /// moved in whole texels of a `texels`-wide map so the map's texels keep
    /// their world positions as the sphere moves.
    pub(crate) fn of_sphere(basis: &LightBasis, center: Vec3, radius: f32, texels: u32) -> Self {
        let center = basis.apply(center);
        let texel = 2.0 * radius / texels as f32;
        let x = (center.x / texel).round() * texel;
        let y = (center.y / texel).round() * texel;
        Self {
            rect: [x - radius, y - radius, x + radius, y + radius],
            depth: [center.z - radius, center.z + radius],
        }
    }
}

/// Light-space bounds of the part of the view frustum of `view_projection`
/// (looking from `eye`) within view depth `depth`: where the receivers of a
/// map that shadows that far can be. `None` for an unusable projection.
pub(crate) fn frustum_footprint(
    basis: &LightBasis,
    view_projection: Mat4,
    eye: Vec3,
    depth: f32,
) -> Option<(Vec3, Vec3)> {
    let inverse = view_projection.inverse();
    let unproject = |x: f32, y: f32, z: f32| {
        let point = inverse * Vec4::new(x, y, z, 1.0);
        point.truncate() / point.w
    };
    let far_center = unproject(0.0, 0.0, FAR_DEPTH);
    let direction = (far_center - unproject(0.0, 0.0, NEAR_DEPTH)).try_normalize()?;
    let far_depth = (far_center - eye).dot(direction);
    if !(far_depth.is_finite() && far_depth > 0.0 && depth > 0.0) {
        return None;
    }
    // Corners at `depth` lie along the rays through the far plane's.
    let scale = math::min(depth, far_depth) / far_depth;
    let mut low = Vec3::splat(f32::MAX);
    let mut high = Vec3::splat(f32::MIN);
    for (x, y) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
        let far = eye + (unproject(x, y, FAR_DEPTH) - eye) * scale;
        for point in [unproject(x, y, NEAR_DEPTH), far] {
            let light = basis.apply(point);
            low = low.min(light);
            high = high.max(light);
        }
    }
    (low.is_finite() && high.is_finite()).then_some((low, high))
}

/// The smallest sphere around the part of the view frustum of
/// `view_projection` (looking from `eye`) within view depth `reach`: the
/// region the near shadow map covers. `None` for an unusable projection.
pub(crate) fn view_sphere(view_projection: Mat4, eye: Vec3, reach: f32) -> Option<(Vec3, f32)> {
    let inverse = view_projection.inverse();
    let unproject = |x: f32, y: f32, z: f32| {
        let point = inverse * Vec4::new(x, y, z, 1.0);
        point.truncate() / point.w
    };
    let near_center = unproject(0.0, 0.0, NEAR_DEPTH);
    let far_center = unproject(0.0, 0.0, FAR_DEPTH);
    let direction = (far_center - near_center).try_normalize()?;
    let far_depth = (far_center - eye).dot(direction);
    let near_depth = math::max((near_center - eye).dot(direction), 0.0);
    // Tangent of the half-diagonal angle of the view.
    let slope = (unproject(1.0, 1.0, FAR_DEPTH) - far_center).length() / far_depth;
    let far = math::min(reach, far_depth);
    if !(slope.is_finite() && far > near_depth) {
        return None;
    }
    // Equidistant from the slice's near and far corners, or at its far
    // plane when wider views put that center beyond it.
    let offset = math::min((far + near_depth) * (1.0 + slope * slope) * 0.5, far);
    let radius = ((far - offset) * (far - offset) + far * far * slope * slope).sqrt();
    let center = eye + direction * offset;
    (center.is_finite() && radius.is_finite()).then_some((center, radius))
}

/// Inward planes, normalized, of the view frustum's slice up to view depth
/// `reach` that sweeping the slice toward the sun (`toward_sun`) never
/// crosses. A caster only darkens what lies away from the sun behind it, so
/// one outside any of these planes darkens nothing the view sees within
/// `reach`.
pub(crate) fn caster_planes(
    view_projection: Mat4,
    eye: Vec3,
    reach: f32,
    toward_sun: Vec3,
) -> Vec<Vec4> {
    let (planes, _) = frustum_uniform(view_projection);
    let inverse = view_projection.inverse();
    let unproject = |z: f32| {
        let point = inverse * Vec4::new(0.0, 0.0, z, 1.0);
        point.truncate() / point.w
    };
    // Depth along the view at most `reach`.
    let slice = (unproject(FAR_DEPTH) - unproject(NEAR_DEPTH))
        .try_normalize()
        .map(|direction| (-direction).extend(direction.dot(eye) + reach));
    planes
        .iter()
        .map(|plane| Vec4::from_array(*plane))
        .chain(slice)
        .filter_map(|plane| {
            let length = plane.truncate().length();
            (length > 0.0 && plane.is_finite()).then(|| plane / length)
        })
        .filter(|plane| plane.truncate().dot(toward_sun) >= 0.0)
        .collect()
}

/// The plane past which receivers at least `distance` from `eye` lie in the
/// view frustum of `view_projection`: view depth beyond `distance` times
/// the cosine of the frustum's widest ray. `None` unless it faces
/// `toward_sun`, for casters behind it can shadow past it, or for an
/// unusable projection.
pub(crate) fn receivers_from(
    view_projection: Mat4,
    eye: Vec3,
    distance: f32,
    toward_sun: Vec3,
) -> Option<Vec4> {
    let inverse = view_projection.inverse();
    let unproject = |x: f32, y: f32, z: f32| {
        let point = inverse * Vec4::new(x, y, z, 1.0);
        point.truncate() / point.w
    };
    let direction =
        (unproject(0.0, 0.0, FAR_DEPTH) - unproject(0.0, 0.0, NEAR_DEPTH)).try_normalize()?;
    if direction.dot(toward_sun) < 0.0 {
        return None;
    }
    let mut cosine = 1.0_f32;
    for (x, y) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
        let ray = (unproject(x, y, FAR_DEPTH) - eye).try_normalize()?;
        cosine = math::min(cosine, ray.dot(direction));
    }
    let plane = direction.extend(-direction.dot(eye) - distance * cosine);
    (cosine > 0.0 && plane.is_finite()).then_some(plane)
}

/// Tiles one refresh of the kept map redraws at most: a row's worth.
pub(crate) const REFRESH_TILES: u32 = TILES;

/// Dirty tiles of the kept far map: bit `column` of row `row`, rows counting
/// up from the map's light-space bottom, 16 bits a row.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DirtyTiles(pub [u32; TILE_WORDS]);

/// A rectangle of kept-map tiles: its first and last column and row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TileRegion {
    pub columns: [u32; 2],
    pub rows: [u32; 2],
}

/// The bits of columns `columns[0]` to `columns[1]` of a row.
fn column_mask(columns: [u32; 2]) -> u32 {
    (0xffff >> (15 - columns[1])) & (0xffff << columns[0])
}

impl DirtyTiles {
    pub(crate) const ALL: Self = Self([u32::MAX; TILE_WORDS]);

    fn row(&self, row: u32) -> u32 {
        (self.0[(row / 2) as usize] >> (row % 2 * TILES)) & 0xffff
    }

    pub(crate) fn merge(&mut self, words: [u32; TILE_WORDS]) {
        for (word, dirty) in self.0.iter_mut().zip(words) {
            *word |= dirty;
        }
    }

    fn any(&self) -> bool {
        self.0.iter().any(|&word| word != 0)
    }

    fn count(&self) -> u32 {
        self.0.iter().map(|word| word.count_ones()).sum()
    }

    /// Sets the tiles of `region`.
    fn set(&mut self, region: TileRegion) {
        let mask = column_mask(region.columns);
        for row in region.rows[0]..=region.rows[1] {
            self.0[(row / 2) as usize] |= mask << (row % 2 * TILES);
        }
    }

    /// Clears the tiles of `region`.
    fn clear(&mut self, region: TileRegion) {
        let mask = column_mask(region.columns);
        for row in region.rows[0]..=region.rows[1] {
            self.0[(row / 2) as usize] &= !(mask << (row % 2 * TILES));
        }
    }

    /// The first tile set (rows first), by column and row.
    fn first(&self) -> Option<(u32, u32)> {
        let row = (0..TILES).find(|&row| self.row(row) != 0)?;
        Some((self.row(row).trailing_zeros(), row))
    }

    /// Takes the next region to redraw: the dirty tiles of the first row
    /// holding any, and those of the rows after it while the region's
    /// rectangle stays within [`REFRESH_TILES`] tiles.
    pub(crate) fn take_region(&mut self) -> Option<TileRegion> {
        self.take_within([0, TILES - 1], [0, TILES - 1], REFRESH_TILES)
    }

    /// [`Self::take_region`] among the tiles of `columns` and `rows`
    /// alone, within `budget` tiles.
    fn take_within(
        &mut self,
        columns: [u32; 2],
        rows: [u32; 2],
        budget: u32,
    ) -> Option<TileRegion> {
        let within = column_mask(columns);
        let row_bits = |tiles: &Self, row: u32| tiles.row(row) & within;
        let first = (rows[0]..=rows[1]).find(|&row| row_bits(self, row) != 0)?;
        let span = |bits: u32| [bits.trailing_zeros(), 31 - bits.leading_zeros()];
        let mut columns = span(row_bits(self, first));
        let mut last = first;
        while last < rows[1] && row_bits(self, last + 1) != 0 {
            let [low, high] = span(row_bits(self, last + 1));
            let widened = [columns[0].min(low), columns[1].max(high)];
            if (widened[1] - widened[0] + 1) * (last + 2 - first) > budget {
                break;
            }
            columns = widened;
            last += 1;
        }
        let region = TileRegion {
            columns,
            rows: [first, last],
        };
        self.clear(region);
        Some(region)
    }

    /// The smallest region holding every tile set among `columns` and
    /// `rows`.
    fn bounds_within(&self, columns: [u32; 2], rows: [u32; 2]) -> Option<TileRegion> {
        let within = column_mask(columns);
        let mut bits = 0;
        let mut found: Option<[u32; 2]> = None;
        for row in rows[0]..=rows[1] {
            let set = self.row(row) & within;
            if set != 0 {
                bits |= set;
                found = Some(found.map_or([row, row], |[first, _]| [first, row]));
            }
        }
        found.map(|rows| TileRegion {
            columns: [bits.trailing_zeros(), 31 - bits.leading_zeros()],
            rows,
        })
    }
}

/// A redraw of the kept far map: whole, or a rectangle of its tiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Refresh {
    Whole,
    Tiles(TileRegion),
}

/// Texels of margin around the far region.
const MARGIN_TEXELS: f32 = 2.0;

/// What the kept far map holds: the light basis and region
/// it was drawn for, the anchor its region follows the camera with when no
/// bounds are set, whether it holds the current casters there, and the
/// tiles that casters moving since dirtied.
#[derive(Debug, Default)]
pub(crate) struct KeptMap {
    basis: Option<LightBasis>,
    region: Option<MapRegion>,
    anchor: Option<Vec3>,
    valid: bool,
    dirty: DirtyTiles,
}

impl KeptMap {
    /// Adds tiles the selection pass marked dirty.
    pub(crate) fn merge(&mut self, tiles: [u32; TILE_WORDS]) {
        self.dirty.merge(tiles);
    }

    /// The map no longer holds the current casters: the next refresh
    /// redraws it whole.
    pub(crate) fn invalidate(&mut self) {
        self.valid = false;
    }

    /// Keeps the map's casters only while shadows stay `enabled`.
    pub(crate) fn keep_while(&mut self, enabled: bool) {
        self.valid &= enabled;
    }

    /// Every tile needs redrawing, as after casters were removed.
    pub(crate) fn dirty_all(&mut self) {
        self.dirty = DirtyTiles::ALL;
    }

    /// Marks dirty the tiles casters within the world box `min`..`max`
    /// draw into, as when they appeared or left; nothing before the map
    /// has a region.
    pub(crate) fn dirty_box(&mut self, min: Vec3, max: Vec3) {
        let (Some(basis), Some(region)) = (self.basis, self.region) else {
            return;
        };
        let (low, high) = basis.bounds(min, max);
        let [x0, y0, x1, y1] = region.rect;
        let tile = |value: f32, start: f32, end: f32| {
            ((value - start) / (end - start) * TILES as f32).floor()
        };
        let columns = [tile(low.x, x0, x1), tile(high.x, x0, x1)];
        let rows = [tile(low.y, y0, y1), tile(high.y, y0, y1)];
        let last = (TILES - 1) as f32;
        if columns[1] < 0.0 || rows[1] < 0.0 || columns[0] > last || rows[0] > last {
            return;
        }
        let clamped = |value: f32| math::clamp(value, 0.0, last) as u32;
        self.dirty.set(TileRegion {
            columns: columns.map(clamped),
            rows: rows.map(clamped),
        });
    }

    /// The far maps' region in light `basis`, `texels` wide: the box
    /// `bounds`, or a cube `distance` around an anchor that follows `eye` a
    /// quarter of the distance at a time. A new basis or region invalidates
    /// the map.
    pub(crate) fn place(
        &mut self,
        basis: LightBasis,
        bounds: Option<(Vec3, Vec3)>,
        distance: f32,
        eye: Vec3,
        texels: u32,
    ) -> MapRegion {
        if self.basis != Some(basis) {
            self.basis = Some(basis);
            self.valid = false;
        }
        let (min, max) = match bounds {
            Some(bounds) => {
                self.anchor = None;
                bounds
            }
            None => {
                let anchor = match self.anchor {
                    Some(anchor) if anchor.distance(eye) <= distance * 0.25 => anchor,
                    _ => {
                        let grid = distance * 0.125;
                        (eye / grid).round() * grid
                    }
                };
                self.anchor = Some(anchor);
                (
                    anchor - Vec3::splat(distance),
                    anchor + Vec3::splat(distance),
                )
            }
        };
        let region = MapRegion::of_box(&basis, min, max, texels, MARGIN_TEXELS);
        if self.region != Some(region) {
            self.region = Some(region);
            self.valid = false;
        }
        region
    }

    /// This frame's redraw: the whole map once invalidated (dirty tiles go
    /// with it), otherwise the next rectangle of dirty tiles.
    pub(crate) fn refresh(&mut self) -> Option<Refresh> {
        if self.valid {
            self.dirty.take_region().map(Refresh::Tiles)
        } else {
            self.valid = true;
            self.dirty = DirtyTiles::default();
            Some(Refresh::Whole)
        }
    }
}

impl TileRegion {
    /// The region's light-space rectangle in the kept map's rectangle
    /// `rect`, widened by `margin`.
    pub(crate) fn rect(&self, rect: [f32; 4], margin: f32) -> [f32; 4] {
        let width = (rect[2] - rect[0]) / TILES as f32;
        let height = (rect[3] - rect[1]) / TILES as f32;
        [
            rect[0] + self.columns[0] as f32 * width - margin,
            rect[1] + self.rows[0] as f32 * height - margin,
            rect[0] + (self.columns[1] + 1) as f32 * width + margin,
            rect[1] + (self.rows[1] + 1) as f32 * height + margin,
        ]
    }

    /// The region's texels in a map `resolution` texels a side, whose rows
    /// run downward from its light-space top: x, y, width and height.
    pub(crate) fn scissor(&self, resolution: u32) -> [u32; 4] {
        let size = resolution / TILES;
        [
            self.columns[0] * size,
            resolution - (self.rows[1] + 1) * size,
            (self.columns[1] - self.columns[0] + 1) * size,
            (self.rows[1] - self.rows[0] + 1) * size,
        ]
    }
}

/// A kept level: [`TILES`] by [`TILES`] tiles `tile` wide
/// following the camera in light space. Its map holds the absolute tile
/// `(a, b)`, whose light-space corner is `(a, b) * tile`, at slot
/// `(a mod TILES, b mod TILES)`: the window moves a tile at a time and only
/// the tiles it uncovers redraw. Its depths span a band around the camera,
/// which follows it (redrawing the level) once it leaves the band's middle:
/// receivers deeper or shallower read coarser maps.
#[derive(Debug)]
struct KeptLevel {
    tile: f32,
    // The absolute tile of the window's first corner.
    origin: [i32; 2],
    // Light-space depth of the band's middle, and half its depth.
    depth: f32,
    half_depth: f32,
    // Slots whose texels do not hold their tiles' casters, which lookups
    // skip, and slots to redraw.
    invalid: DirtyTiles,
    dirty: DirtyTiles,
}

/// A redraw of a kept level: a region of its slots within one page, the
/// absolute tiles from `page * TILES` on, whose slots are their offsets
/// there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LevelRefresh {
    pub level: usize,
    pub region: TileRegion,
    pub page: [i32; 2],
}

/// The kept levels, finest first, each four times wider
/// than the one before it, and the light basis and settings they were
/// placed for: other ones start them over.
#[derive(Debug, Default)]
pub(crate) struct KeptLevels {
    basis: Option<LightBasis>,
    key: Option<(u32, u32)>,
    levels: Vec<KeptLevel>,
}

/// Tiles across a level, as a signed count.
const SIDE: i32 = TILES as i32;

impl KeptLevel {
    /// A level with its window's first corner at `origin` and its band
    /// around `depth`, holding nothing yet.
    fn new(tile: f32, origin: [i32; 2], depth: f32, half_depth: f32) -> Self {
        Self {
            tile,
            origin,
            depth,
            half_depth,
            invalid: DirtyTiles::ALL,
            dirty: DirtyTiles::ALL,
        }
    }

    /// The window's width (world units).
    fn width(&self) -> f32 {
        self.tile * TILES as f32
    }

    /// Moves the window's first corner to `target` once the camera is more
    /// than a tile from the middle, invalidating the slots it uncovers.
    fn scroll(&mut self, target: [i32; 2]) {
        let shift = [target[0] - self.origin[0], target[1] - self.origin[1]];
        if shift.iter().all(|step| step.abs() <= 1) {
            return;
        }
        let mut uncovered = DirtyTiles::default();
        let entering = |origin: i32, shift: i32| {
            if shift >= 0 {
                origin + SIDE..origin + SIDE + shift.min(SIDE)
            } else {
                origin + shift.max(-SIDE)..origin
            }
        };
        for column in entering(self.origin[0], shift[0]) {
            let slot = column.rem_euclid(SIDE) as u32;
            uncovered.set(TileRegion {
                columns: [slot, slot],
                rows: [0, TILES - 1],
            });
        }
        for row in entering(self.origin[1], shift[1]) {
            let slot = row.rem_euclid(SIDE) as u32;
            uncovered.set(TileRegion {
                columns: [0, TILES - 1],
                rows: [slot, slot],
            });
        }
        self.invalid.merge(uncovered.0);
        self.dirty.merge(uncovered.0);
        self.origin = target;
    }

    /// The part of the window holding slot `slot` along an axis whose
    /// first corner is the absolute tile `origin`: its slots, on one page,
    /// and that page. Slots from the corner's on are on its page, those
    /// before it on the next.
    fn part(origin: i32, slot: u32) -> ([u32; 2], i32) {
        let first = origin.rem_euclid(SIDE) as u32;
        let page = origin.div_euclid(SIDE);
        if slot >= first {
            ([first, TILES - 1], page)
        } else {
            ([0, first - 1], page + 1)
        }
    }

    /// Marks dirty the slots of the window's tiles within the light-space
    /// rectangle `low`..`high`.
    fn dirty_rect(&mut self, low: Vec3, high: Vec3) {
        let tile = |value: f32| (value / self.tile).floor() as i32;
        let columns = tile(low.x).max(self.origin[0])..=tile(high.x).min(self.origin[0] + SIDE - 1);
        let rows = tile(low.y).max(self.origin[1])..=tile(high.y).min(self.origin[1] + SIDE - 1);
        for row in rows {
            for column in columns.clone() {
                let [column, row] = [column, row].map(|tile| tile.rem_euclid(SIDE) as u32);
                self.dirty.set(TileRegion {
                    columns: [column, column],
                    rows: [row, row],
                });
            }
        }
    }
}

impl KeptLevels {
    /// Places `levels` levels, the finest `extent` wide, in light `basis`
    /// around the camera at `eye`: each window follows it, uncovering
    /// slots, and each band moves (invalidating its level) once the camera
    /// leaves the band's middle. A new basis or other settings start every
    /// level over.
    pub(crate) fn place(&mut self, basis: LightBasis, eye: Vec3, levels: u32, extent: f32) {
        let key = (levels, extent.to_bits());
        if self.basis != Some(basis) || self.key != Some(key) {
            self.basis = Some(basis);
            self.key = Some(key);
            self.levels.clear();
        }
        let light = basis.apply(eye);
        // Receivers within a window lie across its width, and those it
        // shadows within half its width above or below the camera; the band
        // reaches them along the sun, deeper the lower it stands, and half a
        // width more each way, which the camera crosses before the band
        // follows it. Others read coarser maps.
        let rise = basis.rows[1].y.abs();
        let sine = math::max(-basis.rows[2].y, 0.1);
        for index in 0..levels as usize {
            let width = extent * 4.0_f32.powi(index as i32);
            let tile = width / TILES as f32;
            let target = [
                (light.x / tile).floor() as i32 - SIDE / 2,
                (light.y / tile).floor() as i32 - SIDE / 2,
            ];
            let half_depth = 0.5 * width * ((1.0 + rise) / sine + 1.0);
            match self.levels.get_mut(index) {
                Some(level) if (light.z - level.depth).abs() <= 0.5 * width => {
                    level.scroll(target);
                }
                Some(level) => *level = KeptLevel::new(tile, target, light.z, half_depth),
                None => {
                    self.levels
                        .push(KeptLevel::new(tile, target, light.z, half_depth));
                }
            }
        }
    }

    /// Levels in use.
    pub(crate) fn len(&self) -> usize {
        self.levels.len()
    }

    /// Every level holds nothing current: each redraws whole.
    pub(crate) fn invalidate(&mut self) {
        for level in &mut self.levels {
            level.invalid = DirtyTiles::ALL;
            level.dirty = DirtyTiles::ALL;
        }
    }

    /// Every tile needs redrawing, as after casters were removed.
    pub(crate) fn dirty_all(&mut self) {
        for level in &mut self.levels {
            level.dirty = DirtyTiles::ALL;
        }
    }

    /// Marks dirty each level's tiles under the world box `min`..`max`,
    /// where casters appeared or left.
    pub(crate) fn dirty_box(&mut self, min: Vec3, max: Vec3) {
        let Some(basis) = self.basis else {
            return;
        };
        let (low, high) = basis.bounds(min, max);
        for level in &mut self.levels {
            level.dirty_rect(low, high);
        }
    }

    /// Adds each level's slots the selection pass marked dirty.
    pub(crate) fn merge(&mut self, words: [[u32; TILE_WORDS]; MAX_LEVELS]) {
        for (level, words) in self.levels.iter_mut().zip(words) {
            level.dirty.merge(words);
        }
    }

    /// Tiles whose casters no level holds yet.
    pub(crate) fn pending(&self) -> u32 {
        self.levels.iter().map(|level| level.invalid.count()).sum()
    }

    /// This frame's redraw, finest level first: every invalid tile within
    /// one page of a level, or else, once no level has any, a level's next
    /// region of dirty tiles there within `budget` tiles. Those tiles then
    /// hold their casters.
    pub(crate) fn refresh(&mut self, budget: u32) -> Option<LevelRefresh> {
        let invalid = self.levels.iter().any(|level| level.invalid.any());
        for (index, level) in self.levels.iter_mut().enumerate() {
            let tiles = if invalid {
                &level.invalid
            } else {
                &level.dirty
            };
            let Some((column, row)) = tiles.first() else {
                continue;
            };
            let (columns, page_x) = KeptLevel::part(level.origin[0], column);
            let (rows, page_y) = KeptLevel::part(level.origin[1], row);
            let region = if invalid {
                let region = level
                    .invalid
                    .bounds_within(columns, rows)
                    .expect("holds its first tile");
                level.invalid.clear(region);
                level.dirty.clear(region);
                region
            } else {
                level
                    .dirty
                    .take_within(columns, rows, budget)
                    .expect("holds its first tile")
            };
            return Some(LevelRefresh {
                level: index,
                region,
                page: [page_x, page_y],
            });
        }
        None
    }

    /// Light space to level `index`'s lookups: one over its width (x and
    /// y in level widths), and its band's nearest depth and one over its
    /// depth (z in [0, 1] over the band).
    pub(crate) fn scales(&self, index: usize) -> [f32; 4] {
        self.levels.get(index).map_or([0.0; 4], |level| {
            [
                1.0 / level.width(),
                level.depth - level.half_depth,
                0.5 / level.half_depth,
                0.0,
            ]
        })
    }

    /// Whether level `index` has tiles not yet drawn.
    pub(crate) fn has_pending(&self, index: usize) -> bool {
        self.levels
            .get(index)
            .is_some_and(|level| level.invalid.any())
    }

    /// Level `index`'s window: its first corner in level widths.
    pub(crate) fn window(&self, index: usize) -> [f32; 2] {
        self.levels.get(index).map_or([0.0; 2], |level| {
            level.origin.map(|tile| tile as f32 / TILES as f32)
        })
    }

    /// Level `index`'s slots holding their tiles' casters, a bit each.
    pub(crate) fn valid(&self, index: usize) -> [u32; TILE_WORDS] {
        self.levels
            .get(index)
            .map_or([0; TILE_WORDS], |level| level.invalid.0.map(|word| !word))
    }

    /// The world size of a texel of level `index` in a map `texels` wide.
    pub(crate) fn texel(&self, index: usize, texels: u32) -> f32 {
        self.levels
            .get(index)
            .map_or(0.0, |level| level.width() / texels as f32)
    }

    /// Light-space x and y of level `index`'s first tile corner, and its
    /// tile width; w unused.
    pub(crate) fn tiles(&self, index: usize) -> [f32; 4] {
        self.levels.get(index).map_or([0.0; 4], |level| {
            [
                level.origin[0] as f32 * level.tile,
                level.origin[1] as f32 * level.tile,
                level.tile,
                0.0,
            ]
        })
    }

    /// The nearest and farthest depths of level `index`'s band.
    pub(crate) fn band(&self, index: usize) -> [f32; 2] {
        self.levels.get(index).map_or([0.0; 2], |level| {
            [
                level.depth - level.half_depth,
                level.depth + level.half_depth,
            ]
        })
    }

    /// What `refresh` draws: the light-space rectangle of its tiles,
    /// widened by `margin`, and the region of its page, whose clip maps the
    /// page onto the level's whole map.
    pub(crate) fn refresh_regions(
        &self,
        refresh: LevelRefresh,
        margin: f32,
    ) -> ([f32; 4], MapRegion) {
        let width = self.levels[refresh.level].width();
        let corner = refresh.page.map(|page| page as f32 * width);
        let page = [corner[0], corner[1], corner[0] + width, corner[1] + width];
        (
            refresh.region.rect(page, margin),
            MapRegion {
                rect: page,
                depth: self.band(refresh.level),
            },
        )
    }
}
