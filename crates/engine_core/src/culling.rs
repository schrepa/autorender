use engine_assets::Aabb;
use glam::{Mat4, Vec3, Vec3A, Vec4};

// Outside this broad range, finite assets remain renderable but culling fails
// open. It bounds amplification of f32 subnormal flushing below the absolute
// margin, without double-precision arithmetic in the extraction loop.
const MAX_COEFFICIENT: f32 = 1.0e15;
const ROUNDING_MARGIN: f32 = 32.0 * f32::EPSILON;
const MAX_PRODUCT_BOUND: f32 = f32::MAX / 8.0;
// The interior test evaluates doubled distances, so it doubles this margin too.
const INTERIOR_MARGIN: f32 = 2.0 * ROUNDING_MARGIN;

/// Result of one conservative visibility test.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Classification {
    /// Proven strictly inside every plane by the cheap world-box test.
    Interior,
    /// Kept by the transformed-plane support test: intersecting or unproven.
    Retained,
    /// Proven outside one plane by the transformed-plane support test.
    Outside,
}

impl Classification {
    pub(crate) fn outside(self) -> bool {
        self == Self::Outside
    }
}

/// Four planes in component-major lanes for the interior test. Absolute
/// coefficients weight the product bound; `doubled_w` matches doubled centers.
#[derive(Clone, Copy)]
struct PlaneLanes {
    x: Vec4,
    y: Vec4,
    z: Vec4,
    doubled_w: Vec4,
    abs_x: Vec4,
    abs_y: Vec4,
    abs_z: Vec4,
    abs_w: Vec4,
}

impl PlaneLanes {
    fn new(planes: [Vec4; 4]) -> Self {
        let lane = |axis: usize| Vec4::from_array(planes.map(|plane| plane[axis]));
        Self {
            x: lane(0),
            y: lane(1),
            z: lane(2),
            // Plane coefficients are bounded by 2e15, so doubling is exact.
            doubled_w: lane(3) * 2.0,
            abs_x: lane(0).abs(),
            abs_y: lane(1).abs(),
            abs_z: lane(2).abs(),
            abs_w: lane(3).abs(),
        }
    }
}

/// Inward clip planes with WebGPU's -w <= x,y <= w and 0 <= z <= w.
/// No normalization is needed because the support test preserves plane scale.
pub(crate) struct Frustum {
    planes: [Vec4; 6],
    absolute_view_projection: Mat4,
    // The six planes plus two neutral (0, 0, 0, 1) lanes that always pass.
    interior: [PlaneLanes; 2],
}

impl Frustum {
    pub(crate) fn new(view_projection: Mat4) -> Option<Self> {
        let absolute = absolute_matrix(view_projection);
        if !bounded_matrix(absolute) {
            return None;
        }
        let rows = view_projection.transpose();
        let planes = [
            rows.w_axis + rows.x_axis,
            rows.w_axis - rows.x_axis,
            rows.w_axis + rows.y_axis,
            rows.w_axis - rows.y_axis,
            rows.z_axis,
            rows.w_axis - rows.z_axis,
        ];
        if planes.iter().any(|plane| !plane.is_finite()) {
            return None;
        }
        Some(Self {
            planes,
            absolute_view_projection: absolute,
            interior: [
                PlaneLanes::new([planes[0], planes[1], planes[2], planes[3]]),
                PlaneLanes::new([planes[4], planes[5], Vec4::W, Vec4::W]),
            ],
        })
    }

    /// Most boxes in a wide view are wholly inside. Accepting those with the
    /// cheap interior test is sound: it proves every point of the box strictly
    /// inside every plane, while the support test below rejects only boxes
    /// proven outside one plane. Everything else takes the support test, so
    /// the result always equals `support_outside` alone.
    #[inline]
    pub(crate) fn classify(&self, model: Mat4, bounds: Aabb) -> Classification {
        if self.contains(model, bounds) {
            Classification::Interior
        } else if self.support_outside(model, bounds) {
            Classification::Outside
        } else {
            Classification::Retained
        }
    }

    #[cfg(test)]
    pub(crate) fn outside(&self, model: Mat4, bounds: Aabb) -> bool {
        self.classify(model, bounds).outside()
    }

    /// Returns true only when the whole box is proven strictly inside all six
    /// planes (exact arithmetic on these f32 inputs). False means "not proven";
    /// it never means outside. Non-affine, extreme and nonfinite inputs return
    /// false. One world-space box around the transformed local box replaces
    /// transforming each plane into local space.
    ///
    /// Proof sketch, with u = EPSILON / 2 and gamma(k) = k*u / (1 - k*u).
    /// Mesh bounds satisfy lo <= hi. Doubled center c = fl(lo + hi) and extent
    /// e = fl(hi - lo) need no inexact halving; m = max(|lo|, |hi|) is exact,
    /// and |c| + e <= 2(1 + u)m. For affine M = [L | t], plane (n, w) and box
    /// point x, 2f(x) = n.(Lc + 2t) + 2w + n.L(2x - c), with
    /// |2x - c| <= e + 4u*m, so 2*min f >= A - 4u*B, where
    /// A = n.(Lc + 2t) + 2w - |n|.|L|e and B = |n|.(|L|m + |t|) + |w|.
    /// The computed world center/extent are 4- and 3-term sums of products,
    /// followed by 4- and 3-term plane dot products and one subtraction, so
    /// |D - A| <= (2 gamma(4) + u + O(u^2)) * 2(1 + u)B < 18.1u*B. The computed
    /// bound (nonnegative terms) is at least (1 - gamma(4))^2 * B. Acceptance
    /// requires D > 2 * ROUNDING_MARGIN * max(bound, 1) = 128u * max(bound, 1),
    /// leaving 2*min f > 105u * max(bound, 1) > 0. The floor absorbs absolute
    /// subnormal product errors (below 1e-28 here). The coefficient and product
    /// guards keep every intermediate below f32::MAX; any overflow or NaN makes
    /// a comparison fail. Unlike rejection, acceptance needs no shader-rounding
    /// margin: a false acceptance could only draw extra geometry.
    #[inline]
    fn contains(&self, model: Mat4, bounds: Aabb) -> bool {
        if model.x_axis.w != 0.0
            || model.y_axis.w != 0.0
            || model.z_axis.w != 0.0
            || model.w_axis.w != 1.0
        {
            return false;
        }
        let min = Vec3A::from(bounds.min());
        let max = Vec3A::from(bounds.max());
        let maximum = min.abs().max(max.abs());
        if !maximum.cmple(Vec3A::splat(MAX_COEFFICIENT)).all() {
            return false;
        }
        let center = min + max;
        let extent = max - min;
        let x = Vec3A::from_vec4(model.x_axis);
        let y = Vec3A::from_vec4(model.y_axis);
        let z = Vec3A::from_vec4(model.z_axis);
        let translation = Vec3A::from_vec4(model.w_axis);
        let (abs_x, abs_y, abs_z) = (x.abs(), y.abs(), z.abs());
        let world_bound =
            abs_x * maximum.x + abs_y * maximum.y + abs_z * maximum.z + translation.abs();
        if !world_bound.cmple(Vec3A::splat(MAX_PRODUCT_BOUND)).all() {
            return false;
        }
        let world_center = x * center.x + y * center.y + z * center.z + (translation + translation);
        let world_extent = abs_x * extent.x + abs_y * extent.y + abs_z * extent.z;
        let splat = |value: Vec3A| {
            (
                Vec4::splat(value.x),
                Vec4::splat(value.y),
                Vec4::splat(value.z),
            )
        };
        let (center_x, center_y, center_z) = splat(world_center);
        let (extent_x, extent_y, extent_z) = splat(world_extent);
        let (bound_x, bound_y, bound_z) = splat(world_bound);
        self.interior.iter().all(|lanes| {
            let distance =
                lanes.x * center_x + lanes.y * center_y + lanes.z * center_z + lanes.doubled_w;
            let radius = lanes.abs_x * extent_x + lanes.abs_y * extent_y + lanes.abs_z * extent_z;
            let bound =
                lanes.abs_x * bound_x + lanes.abs_y * bound_y + lanes.abs_z * bound_z + lanes.abs_w;
            let proven = (distance - radius).cmpgt(bound.max(Vec4::ONE) * INTERIOR_MARGIN);
            (proven & bound.cmple(Vec4::splat(MAX_PRODUCT_BOUND))).all()
        })
    }

    /// Reject only boxes proven outside one plane. Transforming the plane into
    /// local space tests the transformed box directly, including shear and
    /// reflections. Vertices, strings and heap allocations are not involved.
    fn support_outside(&self, model: Mat4, bounds: Aabb) -> bool {
        let min = bounds.min();
        let max = bounds.max();
        let maximum = min.abs().max(max.abs());
        let absolute_model = absolute_matrix(model);
        if maximum.max_element() > MAX_COEFFICIENT || !bounded_matrix(absolute_model) {
            return false;
        }
        // Bounds on sums of absolute products, including cancellation, provide
        // the scale for a conservative floating-point margin. They also avoid
        // declaring visibility from intermediate overflow or ambiguous infinities.
        let world_bound = absolute_model * maximum.extend(1.0);
        let clip_bound = self.absolute_view_projection * world_bound;
        if !world_bound.is_finite()
            || !clip_bound.is_finite()
            || world_bound.max_element() > MAX_PRODUCT_BOUND
            || clip_bound.max_element() > MAX_PRODUCT_BOUND
        {
            return false;
        }
        let plane_bounds = [
            clip_bound.x + clip_bound.w,
            clip_bound.x + clip_bound.w,
            clip_bound.y + clip_bound.w,
            clip_bound.y + clip_bound.w,
            clip_bound.z,
            clip_bound.z + clip_bound.w,
        ];
        let transpose = model.transpose();
        for (plane, bound) in self.planes.iter().zip(plane_bounds) {
            let local = transpose * *plane;
            let support = Vec3::new(
                if local.x >= 0.0 { max.x } else { min.x },
                if local.y >= 0.0 { max.y } else { min.y },
                if local.z >= 0.0 { max.z } else { min.z },
            );
            let distance = local.truncate().dot(support) + local.w;
            if !local.is_finite() || !distance.is_finite() {
                return false;
            }
            // Covers CPU plane/support rounding and shader model/VP rounding.
            // The floor also covers subnormal flushing within MAX_COEFFICIENT.
            if distance < -ROUNDING_MARGIN * crate::math::max(bound, 1.0) {
                return true;
            }
        }
        false
    }
}

fn absolute_matrix(matrix: Mat4) -> Mat4 {
    Mat4::from_cols(
        matrix.x_axis.abs(),
        matrix.y_axis.abs(),
        matrix.z_axis.abs(),
        matrix.w_axis.abs(),
    )
}

fn bounded_matrix(absolute: Mat4) -> bool {
    absolute.is_finite()
        && absolute
            .x_axis
            .max(absolute.y_axis)
            .max(absolute.z_axis)
            .max(absolute.w_axis)
            .max_element()
            <= MAX_COEFFICIENT
}

#[cfg(test)]
mod tests;
