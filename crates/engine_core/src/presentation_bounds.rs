//! The bounds GPU presentation's shaders share with the CPU reference
//! (`presentation.wgsl`): culling's coefficient and rounding limits, the
//! hierarchy validator's common admission path and interpolation's linear
//! rotation threshold.

// Outside this broad range, finite assets remain renderable but culling fails
// open. It bounds amplification of f32 subnormal flushing below the absolute
// margin, without double-precision arithmetic in the extraction loop.
/// The largest matrix coefficient culling tests.
pub const MAX_COEFFICIENT: f32 = 1.0e15;
/// The relative rounding margin of culling's plane tests.
pub const ROUNDING_MARGIN: f32 = 32.0 * f32::EPSILON;
/// The largest product culling's plane tests form.
pub const MAX_PRODUCT_BOUND: f32 = f32::MAX / 8.0;

/// The largest coefficient the common admission path proves (2^20).
pub const COEFFICIENT_LIMIT: f32 = 1_048_576.0;
/// The smallest linear scale the common admission path proves (2^-10).
pub const MIN_LINEAR_SCALE: f32 = 1.0 / 1024.0;
/// The determinant margin of the common admission path (2^-9).
pub const COMMON_DETERMINANT_MARGIN: f32 = 1.0 / 512.0;

/// The rotation dot product above which interpolation blends linearly:
/// about 3.624 degrees of physical rotation. Normalized lerp removes the three
/// sine evaluations on ordinary small fixed-step motion. The ideal angular
/// deviation is below 1.1e-6 rad. This is an explicit interpolation policy,
/// not bit-equivalent SLERP.
pub const LINEAR_ROTATION_DOT_THRESHOLD: f32 = 0.9995;
