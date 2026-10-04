use std::cell::Cell;

use engine_core::math;
use glam::{DMat3, DVec3, Mat3, Mat4, Vec3};
use parry3d::{
    math::Pose,
    query::{
        details,
        epa::EPA,
        gjk::{self, CsoPoint, GJKResult, VoronoiSimplex},
    },
    shape::SupportMap,
};

use crate::{
    ColliderShape, CollisionError, CollisionRay, ShapeCastOptions, validate_collider_transform,
};

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Bounds {
    pub min: Vec3,
    pub max: Vec3,
}
impl Bounds {
    pub fn union(self, other: Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }
    pub fn overlaps(self, other: Self) -> bool {
        self.min.cmple(other.max).all() && other.min.cmple(self.max).all()
    }
    pub fn expanded(self, margin: f32) -> Self {
        Self {
            min: self.min - Vec3::splat(margin),
            max: self.max + Vec3::splat(margin),
        }
    }
    pub fn swept(self, displacement: Vec3) -> Self {
        self.union(Self {
            min: self.min + displacement,
            max: self.max + displacement,
        })
    }
    pub fn center(self) -> Vec3 {
        (self.min + self.max) * 0.5
    }
    pub fn surface_area(self) -> f32 {
        let extent = self.max - self.min;
        2.0 * (extent.x * extent.y + extent.y * extent.z + extent.z * extent.x)
    }
    pub fn ray(self, ray: CollisionRay) -> bool {
        let mut near = 0.0_f64;
        let mut far = f64::from(ray.max_distance);
        for axis in 0..3 {
            let origin = f64::from(ray.origin[axis]);
            let direction = f64::from(ray.direction[axis]);
            if direction == 0.0 {
                if origin < f64::from(self.min[axis]) || origin > f64::from(self.max[axis]) {
                    return false;
                }
            } else {
                let a = (f64::from(self.min[axis]) - origin) / direction;
                let b = (f64::from(self.max[axis]) - origin) / direction;
                near = near.max(a.min(b));
                far = far.min(a.max(b));
                if near > far {
                    return false;
                }
            }
        }
        true
    }
}

/// Dimension-scaled linear map of a unit box or unit ball. Translation remains
/// separate so narrow-phase works in pair-relative coordinates.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AffineShape {
    pub shape: ColliderShape,
    pub center: Vec3,
    pub basis: Mat3,
    pub bounds: Bounds,
    /// Radius of a ball around `center` inside the shape (a lower bound).
    pub inner: f64,
    /// Radius of a ball around `center` containing the shape (an upper bound).
    pub outer: f64,
    /// Inverse of `basis` in f64: maps offsets from `center` into the unit shape.
    pub inverse: DMat3,
}
impl AffineShape {
    pub fn new(shape: ColliderShape, matrix: Mat4) -> Result<Self, CollisionError> {
        validate_collider_transform(shape, matrix)?;
        let basis = Mat3::from_mat4(matrix) * Mat3::from_diagonal(shape.dimensions());
        let center = matrix.w_axis.truncate();
        let half = match shape {
            ColliderShape::Cuboid { .. } => {
                basis.x_axis.abs() + basis.y_axis.abs() + basis.z_axis.abs()
            }
            ColliderShape::Sphere { .. } => {
                let squared = basis.x_axis * basis.x_axis
                    + basis.y_axis * basis.y_axis
                    + basis.z_axis * basis.z_axis;
                Vec3::new(squared.x.sqrt(), squared.y.sqrt(), squared.z.sqrt())
            }
        };
        // Round broadphase bounds outward at the scene's current magnitude.
        let margin = 16.0 * f32::EPSILON * math::max((center.abs() + half).max_element(), 1.0);
        let bounds = Bounds {
            min: center - half,
            max: center + half,
        }
        .expanded(margin);
        let (inner, outer) = ball_radii(shape, basis);
        Ok(Self {
            shape,
            center,
            basis,
            bounds,
            inner,
            outer,
            inverse: basis.as_dmat3().inverse(),
        })
    }

    pub fn ray(self, ray: CollisionRay) -> Option<(f32, Vec3, bool)> {
        let inverse = self.basis.as_dmat3().inverse();
        let origin = inverse * (ray.origin.as_dvec3() - self.center.as_dvec3());
        let direction = inverse * ray.direction.as_dvec3();
        let (distance, local_normal, inside) = match self.shape {
            ColliderShape::Cuboid { .. } => {
                let inside = origin.abs().cmple(DVec3::ONE).all();
                if inside {
                    return Some((0.0, Vec3::ZERO, true));
                }
                let mut near = 0.0_f64;
                let mut far = f64::from(ray.max_distance);
                let mut normal = DVec3::ZERO;
                for axis in 0..3 {
                    if direction[axis] == 0.0 {
                        if origin[axis].abs() > 1.0 {
                            return None;
                        }
                    } else {
                        let a = (-1.0 - origin[axis]) / direction[axis];
                        let b = (1.0 - origin[axis]) / direction[axis];
                        if a.min(b) > near {
                            near = a.min(b);
                            normal = DVec3::ZERO;
                            normal[axis] = -direction[axis].signum();
                        }
                        far = far.min(a.max(b));
                        if near > far {
                            return None;
                        }
                    }
                }
                (near, normal, false)
            }
            ColliderShape::Sphere { .. } => {
                if origin.length_squared() <= 1.0 {
                    return Some((0.0, Vec3::ZERO, true));
                }
                let a = direction.length_squared();
                let middle = -origin.dot(direction) / a;
                let closest = origin + direction * middle;
                let discriminant = 1.0 - closest.length_squared();
                if discriminant < 0.0 {
                    return None;
                }
                let near = middle - (discriminant / a).sqrt();
                (near, origin + direction * near, false)
            }
        };
        if distance < 0.0 || distance > f64::from(ray.max_distance) {
            return None;
        }
        let normal = (inverse.transpose() * local_normal).normalize().as_vec3();
        Some((distance as f32, normal, inside))
    }
}

/// An affine shape's support map relative to its center, counting calls.
pub(crate) struct Support<'a> {
    affine: &'a AffineShape,
    calls: &'a Cell<usize>,
}

impl<'a> Support<'a> {
    pub(crate) fn new(affine: &'a AffineShape, calls: &'a Cell<usize>) -> Self {
        Self { affine, calls }
    }
}
impl SupportMap for Support<'_> {
    fn local_support_point(&self, direction: Vec3) -> Vec3 {
        self.calls.set(self.calls.get() + 1);
        let local = self.affine.basis.transpose() * direction;
        let point = match self.affine.shape {
            ColliderShape::Cuboid { .. } => Vec3::new(
                if local.x < 0.0 { -1.0 } else { 1.0 },
                if local.y < 0.0 { -1.0 } else { 1.0 },
                if local.z < 0.0 { -1.0 } else { 1.0 },
            ),
            ColliderShape::Sphere { .. } => local.normalize_or_zero(),
        };
        self.affine.basis * point
    }
}

/// Conservative radii of balls around the center inside and around the unit box
/// or unit ball mapped by `basis`, in f64: the inner radius is at most, and the
/// outer at least, the exact value up to f64 rounding, which the pair tests'
/// relative margin exceeds by orders of magnitude.
fn ball_radii(shape: ColliderShape, basis: Mat3) -> (f64, f64) {
    let (x, y, z) = (
        basis.x_axis.as_dvec3(),
        basis.y_axis.as_dvec3(),
        basis.z_axis.as_dvec3(),
    );
    let determinant = x.dot(y.cross(z)).abs();
    match shape {
        ColliderShape::Cuboid { .. } => {
            // The nearest face plane is |det| over the largest face area away;
            // the farthest corner bounds the rest.
            let face = y
                .cross(z)
                .length()
                .max(z.cross(x).length())
                .max(x.cross(y).length());
            let corner = (x + y + z)
                .length()
                .max((x + y - z).length())
                .max((x - y + z).length())
                .max((-x + y + z).length());
            (if face > 0.0 { determinant / face } else { 0.0 }, corner)
        }
        ColliderShape::Sphere { .. } => {
            // Gershgorin bounds on the Gram matrix's eigenvalues, the squared
            // singular values of `basis` (exact for orthogonal columns), and
            // σ_min >= |det| / σ_max² when shear loosens them.
            let columns = [x, y, z];
            let (mut largest, mut smallest) = (0.0_f64, f64::INFINITY);
            for (row, column) in columns.iter().enumerate() {
                let off: f64 = (0..3)
                    .filter(|&other| other != row)
                    .map(|other| column.dot(columns[other]).abs())
                    .sum();
                let diagonal = column.length_squared();
                largest = largest.max(diagonal + off);
                smallest = smallest.min(diagonal - off);
            }
            let sheared = if largest > 0.0 {
                determinant / largest
            } else {
                0.0
            };
            (smallest.max(0.0).sqrt().max(sheared), largest.sqrt())
        }
    }
}

/// Relative widening of pre-test radii: far beyond f64 radius rounding,
/// the box test's 64-epsilon tolerance and parry's f32 GJK tolerances.
const PRETEST_MARGIN: f64 = 1.0 / 4096.0;

/// Whether two affine shapes overlap (touching counts). Pre-tests settle pairs
/// far from contact exactly as the tests below would: pairs sharing an interior
/// point (overlapping inscribed balls, or reaches along the line between the
/// centers that together cover it) are accepted by every exact test; for GJK
/// pairs, disjoint containing balls or a separating axis along that line leave a
/// gap GJK always finds. Two boxes that are not settled early run the exact
/// separating-axis test, and other remaining pairs GJK, with unchanged decisions.
#[cfg_attr(feature = "attribution", inline(never))]
pub(crate) fn intersects(a: &AffineShape, b: &AffineShape) -> Result<bool, CollisionError> {
    let offset = b.center.as_dvec3() - a.center.as_dvec3();
    let squared = offset.length_squared();
    let inner = (a.inner + b.inner) * (1.0 - PRETEST_MARGIN);
    if squared < inner * inner {
        return Ok(true);
    }
    // Each shape reaches 1/n of the way along the center line, n being the
    // offset's norm in its unit shape (max norm for boxes). Reaches summing
    // past the whole line meet at a point inside both shapes.
    let (from_a, from_b) = (a.unit_norm(offset), b.unit_norm(offset));
    if from_a + from_b > (1.0 + PRETEST_MARGIN) * from_a * from_b {
        return Ok(true);
    }
    let boxes = matches!(a.shape, ColliderShape::Cuboid { .. })
        && matches!(b.shape, ColliderShape::Cuboid { .. });
    if boxes {
        return Ok(boxes_overlap(a, b));
    }
    let outer = (a.outer + b.outer) * (1.0 + PRETEST_MARGIN);
    // Along the center line both projections are offset-scaled half-widths.
    if squared > outer * outer
        || squared > (a.half_width(offset) + b.half_width(offset)) * (1.0 + PRETEST_MARGIN)
    {
        return Ok(false);
    }
    intersects_support_maps(a, b)
}

impl AffineShape {
    /// Norm of `offset` mapped into the unit shape: the shape's surface lies
    /// 1/norm of `offset` away from its center in that direction.
    #[inline]
    fn unit_norm(&self, offset: DVec3) -> f64 {
        let local = self.inverse * offset;
        match self.shape {
            ColliderShape::Cuboid { .. } => local.abs().max_element(),
            ColliderShape::Sphere { .. } => local.length(),
        }
    }

    /// Half-width of the shape along `direction`, times `direction`'s length:
    /// the largest `direction · (x - center)` over the shape.
    #[inline]
    fn half_width(&self, direction: DVec3) -> f64 {
        let projected = self.basis.as_dmat3().transpose() * direction;
        match self.shape {
            ColliderShape::Cuboid { .. } => projected.abs().element_sum(),
            ColliderShape::Sphere { .. } => projected.length(),
        }
    }
}

/// `cast_boxes(a, b, Vec3::ZERO, ShapeCastOptions::default()).is_some()`: the
/// same 15 axes and f64 operations, stopping at the first separating axis
/// instead of first forming every axis.
#[cfg_attr(feature = "attribution", inline(never))]
fn boxes_overlap(a: &AffineShape, b: &AffineShape) -> bool {
    let a_axes = [
        a.basis.x_axis.as_dvec3(),
        a.basis.y_axis.as_dvec3(),
        a.basis.z_axis.as_dvec3(),
    ];
    let b_axes = [
        b.basis.x_axis.as_dvec3(),
        b.basis.y_axis.as_dvec3(),
        b.basis.z_axis.as_dvec3(),
    ];
    let offset = a.center.as_dvec3() - b.center.as_dvec3();
    let separates = |axis: DVec3| {
        // The unnormalized gap is the normalized gap times the axis length. A
        // nonpositive one stays below the positive tolerance after
        // normalization, whose rounding is far smaller than that tolerance,
        // so only positive gaps pay for the exact normalized test.
        let spread = a_axes
            .iter()
            .chain(&b_axes)
            .map(|basis| axis.dot(*basis).abs())
            .sum::<f64>();
        if offset.dot(axis).abs() <= spread {
            return false;
        }
        let Some(axis) = axis.try_normalize() else {
            return false;
        };
        let radius = a_axes
            .iter()
            .chain(&b_axes)
            .map(|basis| axis.dot(*basis).abs())
            .sum::<f64>();
        let distance = offset.dot(axis);
        let gap = distance.abs() - radius;
        gap > 64.0 * f64::EPSILON * (distance.abs() + radius).max(1.0)
    };
    for index in 0..3 {
        if separates(a_axes[(index + 1) % 3].cross(a_axes[(index + 2) % 3])) {
            return false;
        }
    }
    for index in 0..3 {
        if separates(b_axes[(index + 1) % 3].cross(b_axes[(index + 2) % 3])) {
            return false;
        }
    }
    for a_axis in a_axes {
        for b_axis in b_axes {
            if separates(a_axis.cross(b_axis)) {
                return false;
            }
        }
    }
    true
}

/// GJK intersection of two support maps with parry's iteration bound exposed.
#[cfg_attr(feature = "attribution", inline(never))]
fn intersects_support_maps(a: &AffineShape, b: &AffineShape) -> Result<bool, CollisionError> {
    let calls = Cell::new(0);
    let hit = details::intersection_test_support_map_support_map(
        &Pose::from_translation(b.center - a.center),
        &Support {
            affine: a,
            calls: &calls,
        },
        &Support {
            affine: b,
            calls: &calls,
        },
    );
    // Parry's 100-iteration exhaustion maps to a miss. Expose an error instead
    // when its support-call bound is reached, including marginal convergence.
    if calls.get() >= 200 {
        Err(CollisionError::NumericalFailure)
    } else {
        Ok(hit)
    }
}

#[derive(Default)]
pub(crate) struct NarrowScratch {
    epa: EPA,
}

impl NarrowScratch {
    pub fn cast(
        &mut self,
        a: &AffineShape,
        b: &AffineShape,
        displacement: Vec3,
        options: ShapeCastOptions,
    ) -> Result<Option<(f32, Vec3, Vec3, bool)>, CollisionError> {
        if matches!(a.shape, ColliderShape::Cuboid { .. })
            && matches!(b.shape, ColliderShape::Cuboid { .. })
        {
            return Ok(cast_boxes(a, b, displacement, options));
        }
        let calls = Cell::new(0);
        let a_support = Support {
            affine: a,
            calls: &calls,
        };
        let b_support = Support {
            affine: b,
            calls: &calls,
        };
        let pose = Pose::from_translation(b.center - a.center);
        let mut simplex = VoronoiSimplex::new();
        let direction = (b.center - a.center).try_normalize().unwrap_or(Vec3::X);
        simplex.reset(CsoPoint::from_shapes(
            &pose, &a_support, &b_support, direction,
        ));
        let proximity = gjk::closest_points(
            &pose,
            &a_support,
            &b_support,
            options.target_distance,
            true,
            &mut simplex,
        );
        if calls.get() >= 200 {
            return Err(CollisionError::NumericalFailure);
        }
        let contact = match proximity {
            GJKResult::Intersection => self
                .epa
                .closest_points(&pose, &a_support, &b_support, &simplex),
            GJKResult::ClosestPoints(p1, p2, normal) => Some((p1, p2, normal)),
            _ => None,
        };
        if matches!(proximity, GJKResult::Intersection) && contact.is_none() {
            return Err(CollisionError::NumericalFailure);
        }
        if let Some((_p1, p2, outward)) = contact {
            let normal = (-outward)
                .try_normalize()
                .ok_or(CollisionError::NumericalFailure)?;
            if !normal.is_finite() || !p2.is_finite() {
                return Err(CollisionError::NumericalFailure);
            }
            if !options.stop_at_penetration
                && displacement.dot(normal) >= -1.0e-6 * math::max(displacement.length(), 1.0)
            {
                return Ok(None);
            }
            return Ok(Some((0.0, a.center + p2, normal, true)));
        }
        if displacement == Vec3::ZERO {
            return Ok(None);
        }
        calls.set(0);
        // Penetrating cases above use our retained EPA instead of Parry's
        // allocating convenience contact path.
        let hit = details::cast_shapes_support_map_support_map(
            &pose,
            -displacement,
            &a_support,
            &b_support,
            parry3d::query::ShapeCastOptions {
                max_time_of_impact: 1.0,
                target_distance: options.target_distance,
                stop_at_penetration: true,
                compute_impact_geometry_on_penetration: false,
            },
        );
        if calls.get() >= 200 {
            return Err(CollisionError::NumericalFailure);
        }
        let Some(hit) = hit else {
            return Ok(None);
        };
        if hit.status != parry3d::query::ShapeCastStatus::Converged
            || !hit.time_of_impact.is_finite()
            || !hit.normal2.is_finite()
            || !hit.witness2.is_finite()
            || hit.normal2.length_squared() < 0.5
            || !(0.0..=1.0).contains(&hit.time_of_impact)
        {
            return Err(CollisionError::NumericalFailure);
        }
        Ok(Some((
            hit.time_of_impact,
            b.center + hit.witness2,
            hit.normal2.normalize(),
            false,
        )))
    }
}

/// Fixed-orientation affine boxes have exactly 15 separating-axis families:
/// three face normals each, and nine edge cross products. f64 projection avoids
/// f32 GJK witness cancellation at exact contact with wide floors.
fn cast_boxes(
    a: &AffineShape,
    b: &AffineShape,
    displacement: Vec3,
    options: ShapeCastOptions,
) -> Option<(f32, Vec3, Vec3, bool)> {
    let a_axes = [
        a.basis.x_axis.as_dvec3(),
        a.basis.y_axis.as_dvec3(),
        a.basis.z_axis.as_dvec3(),
    ];
    let b_axes = [
        b.basis.x_axis.as_dvec3(),
        b.basis.y_axis.as_dvec3(),
        b.basis.z_axis.as_dvec3(),
    ];
    let mut axes = [DVec3::ZERO; 15];
    for index in 0..3 {
        axes[index] = a_axes[(index + 1) % 3].cross(a_axes[(index + 2) % 3]);
        axes[index + 3] = b_axes[(index + 1) % 3].cross(b_axes[(index + 2) % 3]);
        for other in 0..3 {
            axes[6 + index * 3 + other] = a_axes[index].cross(b_axes[other]);
        }
    }
    let offset = a.center.as_dvec3() - b.center.as_dvec3();
    let movement = displacement.as_dvec3();
    let mut enter = f64::NEG_INFINITY;
    let mut exit = f64::INFINITY;
    let mut enter_normal = DVec3::ZERO;
    let mut initial_normal = DVec3::ZERO;
    let mut largest_gap = f64::NEG_INFINITY;
    let mut initially_touching = true;
    for axis in axes {
        let Some(axis) = axis.try_normalize() else {
            continue;
        };
        let radius = a_axes
            .iter()
            .chain(&b_axes)
            .map(|basis| axis.dot(*basis).abs())
            .sum::<f64>()
            + f64::from(options.target_distance);
        let distance = offset.dot(axis);
        let velocity = movement.dot(axis);
        let gap = distance.abs() - radius;
        let epsilon = 64.0 * f64::EPSILON * (distance.abs() + radius).max(1.0);
        let normal = if distance >= 0.0 { axis } else { -axis };
        if gap > largest_gap {
            largest_gap = gap;
            initial_normal = normal;
        }
        if gap > epsilon {
            initially_touching = false;
        }
        if velocity == 0.0 {
            if gap > epsilon {
                return None;
            }
        } else {
            let first = (-radius - distance) / velocity;
            let last = (radius - distance) / velocity;
            let near = first.min(last);
            if near > enter {
                enter = near;
                enter_normal = if velocity > 0.0 { -axis } else { axis };
            }
            exit = exit.min(first.max(last));
            if enter > exit + 64.0 * f64::EPSILON {
                return None;
            }
        }
    }
    let (fraction, normal) = if initially_touching {
        if !options.stop_at_penetration
            && (exit <= 0.0
                || movement.dot(initial_normal) >= -1.0e-12 * movement.length().max(1.0))
        {
            return None;
        }
        (0.0, initial_normal)
    } else {
        if enter < 0.0 || enter > 1.0 || exit < 0.0 {
            return None;
        }
        (enter, enter_normal)
    };
    // A representative surface witness on the target. At a face/edge contact
    // the closest point to the moving center is nonunique under affine shear.
    let moving_center = a.center.as_dvec3() + movement * fraction;
    let mut target_local = b.basis.as_dmat3().inverse() * (moving_center - b.center.as_dvec3());
    if target_local.abs().max_element() < 1.0 {
        let direction = b.basis.as_dmat3().transpose() * normal;
        target_local = DVec3::new(
            direction.x.signum(),
            direction.y.signum(),
            direction.z.signum(),
        );
    } else {
        target_local = target_local.clamp(-DVec3::ONE, DVec3::ONE);
    }
    let point = b.center.as_dvec3() + b.basis.as_dmat3() * target_local;
    Some((
        fraction as f32,
        point.as_vec3(),
        normal.as_vec3(),
        initially_touching,
    ))
}
