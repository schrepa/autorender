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

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::Camera;

    // Tiles as the selection pass marks them, read back.
    fn tiles(marked: &[(u32, u32)]) -> DirtyTiles {
        let mut words = [0; TILE_WORDS];
        for &(column, row) in marked {
            words[(row / 2) as usize] |= 1 << (column + row % 2 * TILES);
        }
        let mut dirty = DirtyTiles::default();
        dirty.merge(words);
        dirty
    }

    #[test]
    fn refreshes_take_dirty_tiles_a_bounded_rectangle_at_a_time() {
        // One dirty tile: that tile alone, then nothing.
        let mut dirty = tiles(&[(5, 3)]);
        let region = dirty.take_region().unwrap();
        assert_eq!((region.columns, region.rows), ([5, 5], [3, 3]));
        assert_eq!(dirty.take_region(), None);
        // A caster over tiles of three rows: one rectangle.
        let mut dirty = tiles(&[(4, 7), (5, 7), (4, 8), (5, 8), (4, 9), (5, 9)]);
        let region = dirty.take_region().unwrap();
        assert_eq!((region.columns, region.rows), ([4, 5], [7, 9]));
        assert_eq!(dirty, DirtyTiles::default());
        // Rows too wide together are redrawn a frame apart, and the tiles
        // between a row's dirty ones with them.
        let mut dirty = tiles(&[(0, 2), (9, 2), (15, 3)]);
        let region = dirty.take_region().unwrap();
        assert_eq!((region.columns, region.rows), ([0, 9], [2, 2]));
        let region = dirty.take_region().unwrap();
        assert_eq!((region.columns, region.rows), ([15, 15], [3, 3]));
        assert_eq!(dirty.take_region(), None);
        // Everything dirty: a row per refresh, bottom to top.
        let mut dirty = DirtyTiles::ALL;
        for row in 0..TILES {
            let region = dirty.take_region().unwrap();
            assert_eq!((region.columns, region.rows), ([0, TILES - 1], [row, row]));
        }
        assert_eq!(dirty.take_region(), None);
    }

    #[test]
    fn tile_regions_cover_their_rectangle_and_texels() {
        let region = TileRegion {
            columns: [2, 3],
            rows: [0, 1],
        };
        let rect = region.rect([-8.0, 0.0, 8.0, 32.0], 0.25);
        assert_eq!(rect, [-6.25, -0.25, -3.75, 4.25]);
        // Rows count up from the bottom texel row.
        assert_eq!(region.scissor(4096), [512, 3584, 512, 512]);
        let whole = TileRegion {
            columns: [0, TILES - 1],
            rows: [0, TILES - 1],
        };
        assert_eq!(whole.scissor(2048), [0, 0, 2048, 2048]);
    }

    #[test]
    fn caster_planes_keep_what_can_shadow_the_view() {
        let camera = Camera {
            position: Vec3::new(0.0, 2.0, 0.0),
            target: Vec3::new(0.0, 2.0, -10.0),
            ..Camera::default()
        };
        let view_projection = camera.view_projection(1.5);
        let inside = |planes: &[Vec4], point: Vec3| {
            planes
                .iter()
                .all(|plane| plane.truncate().dot(point) + plane.w >= 0.0)
        };
        // The sun overhead: anything above the view slice can shadow it,
        // nothing beside or below it can.
        let planes = caster_planes(view_projection, camera.position, 30.0, Vec3::Y);
        assert!(inside(&planes, Vec3::new(0.0, 500.0, -15.0)));
        assert!(inside(&planes, Vec3::new(0.0, 2.0, -15.0)));
        assert!(!inside(&planes, Vec3::new(-60.0, 2.0, -15.0)));
        assert!(!inside(&planes, Vec3::new(0.0, 2.0, -45.0)));
        assert!(!inside(&planes, Vec3::new(0.0, 2.0, 5.0)));
        // A low sun on the right: casters to the right still shadow into it.
        let planes = caster_planes(view_projection, camera.position, 30.0, Vec3::X);
        assert!(inside(&planes, Vec3::new(200.0, 2.0, -15.0)));
        assert!(!inside(&planes, Vec3::new(-200.0, 2.0, -15.0)));
    }

    #[test]
    fn light_space_is_orthonormal_and_looks_away_from_the_sun() {
        for sun in [
            Vec3::new(0.82, 0.3, 0.1),
            Vec3::Y,
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(-0.35, 0.6, -0.45),
        ] {
            let basis = LightBasis::new(sun);
            let [x, y, z] = basis.rows;
            for (a, b) in [(x, y), (y, z), (z, x)] {
                assert!(a.dot(b).abs() < 1.0e-6);
            }
            for axis in basis.rows {
                assert!((axis.length() - 1.0).abs() < 1.0e-6);
            }
            assert!(z.dot(sun.normalize()) < -0.999);
        }
    }

    #[test]
    fn map_clip_spans_the_region_and_receiver_depths() {
        let basis = LightBasis::new(Vec3::new(0.5, 0.8, 0.2));
        let region = MapRegion::of_box(&basis, Vec3::splat(-20.0), Vec3::splat(20.0), 1024, 0.0);
        let clip = region.clip(&basis);
        let [x0, y0, x1, y1] = region.rect;
        let world =
            |x: f32, y: f32, z: f32| basis.rows[0] * x + basis.rows[1] * y + basis.rows[2] * z;
        let low = clip * world(x0, y0, region.depth[0]).extend(1.0);
        let high = clip * world(x1, y1, region.depth[1]).extend(1.0);
        assert!(
            low.abs_diff_eq(Vec4::new(-1.0, -1.0, 0.0, 1.0), 1.0e-4),
            "{low}"
        );
        assert!(
            high.abs_diff_eq(Vec4::new(1.0, 1.0, 1.0, 1.0), 1.0e-4),
            "{high}"
        );
        // Every corner of the box lands inside the map.
        for corner in 0..8 {
            let point = Vec3::new(
                if corner & 1 == 0 { -20.0 } else { 20.0 },
                if corner & 2 == 0 { -20.0 } else { 20.0 },
                if corner & 4 == 0 { -20.0 } else { 20.0 },
            );
            let p = clip * point.extend(1.0);
            assert!(p.x.abs() <= 1.0001 && p.y.abs() <= 1.0001, "{p}");
            assert!((-1.0e-4..=1.0001).contains(&p.z), "{p}");
        }
    }

    #[test]
    fn sphere_regions_keep_texels_in_place_as_they_move() {
        let basis = LightBasis::new(Vec3::new(0.3, 0.9, 0.1));
        let a = MapRegion::of_sphere(&basis, Vec3::new(1.0, 2.0, 3.0), 50.0, 1000);
        let b = MapRegion::of_sphere(&basis, Vec3::new(1.37, 2.0, 3.21), 50.0, 1000);
        let texel = 0.1;
        for (a, b) in a.rect.iter().zip(b.rect) {
            let shift = (b - a) / texel;
            assert!((shift - shift.round()).abs() < 1.0e-2, "{shift}");
        }
    }

    #[test]
    fn view_sphere_holds_the_near_view_slice() {
        let camera = Camera {
            position: Vec3::new(4.0, 30.0, 9.0),
            target: Vec3::new(40.0, 0.0, -60.0),
            ..Camera::default()
        };
        let aspect = 2262.0 / 987.0;
        let view_projection = camera.view_projection(aspect);
        let (center, radius) = view_sphere(view_projection, camera.position, 100.0).unwrap();
        let inverse = view_projection.inverse();
        let direction = (camera.target - camera.position).normalize();
        for corner in 0..4 {
            let x = if corner & 1 == 0 { -1.0 } else { 1.0 };
            let y = if corner & 2 == 0 { -1.0 } else { 1.0 };
            let far = inverse * Vec4::new(x, y, FAR_DEPTH, 1.0);
            let ray = (far.truncate() / far.w - camera.position).normalize();
            for depth in [camera.near, 100.0] {
                let point = camera.position + ray * (depth / ray.dot(direction));
                assert!(point.distance(center) <= radius * 1.0001);
            }
        }
        // A wide view's slice spans its far corners: 1.3 times the reach.
        assert!(radius < 100.0 * 1.35 && radius > 50.0, "{radius}");
        assert!(view_sphere(Mat4::ZERO, Vec3::ZERO, 10.0).is_none());
        // A texel of a 1024-wide map of this sphere.
        let basis = LightBasis::new(Vec3::Y);
        let region = MapRegion::of_sphere(&basis, center, radius, 1024);
        assert!((region.texel(1024) - 2.0 * radius / 1024.0).abs() < 1.0e-4);
    }

    #[test]
    fn frustum_footprint_holds_every_point_of_the_view_within_its_depth() {
        let camera = Camera {
            position: Vec3::new(4.0, 30.0, 9.0),
            target: Vec3::new(40.0, 0.0, -60.0),
            ..Camera::default()
        };
        let view_projection = camera.view_projection(2262.0 / 987.0);
        let basis = LightBasis::new(Vec3::new(0.6, 0.35, -0.4));
        let (low, high) =
            frustum_footprint(&basis, view_projection, camera.position, 100.0).unwrap();
        let inverse = view_projection.inverse();
        let direction = (camera.target - camera.position).normalize();
        // Points across the view at depths up to the footprint's lie inside.
        for step in 0..=24 {
            let x = step as f32 / 12.0 - 1.0;
            for y in [-1.0, -0.3, 0.4, 1.0] {
                let far = inverse * Vec4::new(x, y, FAR_DEPTH, 1.0);
                let ray = (far.truncate() / far.w - camera.position).normalize();
                for depth in [camera.near, 20.0, 60.0, 100.0] {
                    let point = basis.apply(camera.position + ray * (depth / ray.dot(direction)));
                    // Both sides unproject through the far plane in f32.
                    let slack = Vec3::splat(1.0e-2);
                    assert!(
                        point.cmpge(low - slack).all() && point.cmple(high + slack).all(),
                        "x {x} y {y} depth {depth}: {point} not in {low}..{high}"
                    );
                }
            }
        }
        // A shallower footprint is no larger.
        let (near_low, near_high) =
            frustum_footprint(&basis, view_projection, camera.position, 10.0).unwrap();
        assert!(near_low.cmpge(low).all() && near_high.cmple(high).all());
        assert!(frustum_footprint(&basis, Mat4::ZERO, Vec3::ZERO, 10.0).is_none());
    }
}
