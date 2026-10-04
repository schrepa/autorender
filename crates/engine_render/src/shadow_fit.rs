//! Placement of the sun's shadow maps (decision 020), independent of GPU
//! objects: the light-space basis, the regions each map covers and their
//! orthographic projections.

use engine_core::{
    glam::{Mat4, Vec3, Vec4},
    math,
};

use crate::gpu_data::{FAR_DEPTH, NEAR_DEPTH, frustum_uniform};

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

/// Tiles across each side of the kept far map (decision 020): kept casters
/// that move mark the tiles under them dirty, and frames redraw them.
pub(crate) const TILES: u32 = 16;
/// Words of the dirty-tile mask, each holding two rows of tiles, as the
/// selection pass marks them.
pub(crate) const TILE_WORDS: usize = (TILES * TILES / 32) as usize;
/// Tiles one refresh of the kept map redraws at most: a row's worth.
const REFRESH_TILES: u32 = TILES;

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

    /// Takes the next region to redraw: the dirty tiles of the first row
    /// holding any, and those of the rows after it while the region's
    /// rectangle stays within [`REFRESH_TILES`] tiles.
    pub(crate) fn take_region(&mut self) -> Option<TileRegion> {
        let first = (0..TILES).find(|&row| self.row(row) != 0)?;
        let span = |bits: u32| [bits.trailing_zeros(), 31 - bits.leading_zeros()];
        let mut columns = span(self.row(first));
        let mut last = first;
        while last + 1 < TILES && self.row(last + 1) != 0 {
            let [low, high] = span(self.row(last + 1));
            let widened = [columns[0].min(low), columns[1].max(high)];
            if (widened[1] - widened[0] + 1) * (last + 2 - first) > REFRESH_TILES {
                break;
            }
            columns = widened;
            last += 1;
        }
        let mask = (0xffff >> (15 - columns[1])) & (0xffff << columns[0]);
        for row in first..=last {
            self.0[(row / 2) as usize] &= !(mask << (row % 2 * TILES));
        }
        Some(TileRegion {
            columns,
            rows: [first, last],
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

/// What the kept far map holds (decision 020): the light basis and region
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
