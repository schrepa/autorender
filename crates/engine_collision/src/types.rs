use engine_core::{Error, ErrorCode, settings, stats};
use engine_places::physics::{
    ColliderShape, CollisionRay, MAX_COLLIDER_EXTENT, MAX_COLLISION_COORDINATE,
    MIN_COLLIDER_EXTENT, ShapeCastOptions,
};
use glam::{Mat3, Mat4, Vec3};

pub const MAX_COLLISION_NODES: usize = 100_000;
pub const MAX_QUERY_HITS: usize = 4_096;
pub const MAX_CANDIDATE_PAIRS: usize = 1_000_000;
pub const MAX_SENSOR_PAIRS: usize = 65_536;

pub(crate) fn check_shape(shape: ColliderShape) -> Result<(), Fault> {
    let dimensions = shape.dimensions();
    if dimensions.is_finite()
        && dimensions.cmpge(Vec3::splat(MIN_COLLIDER_EXTENT)).all()
        && dimensions.cmple(Vec3::splat(MAX_COLLIDER_EXTENT)).all()
    {
        Ok(())
    } else {
        Err(Fault::InvalidShape)
    }
}

pub(crate) fn validate_shape(shape: ColliderShape) -> Result<(), Error> {
    check_shape(shape).map_err(Fault::error)
}

pub(crate) fn validate_collider_transform(shape: ColliderShape, matrix: Mat4) -> Result<(), Error> {
    check_collider_transform(shape, matrix).map_err(Fault::error)
}

pub(crate) fn check_collider_transform(shape: ColliderShape, matrix: Mat4) -> Result<(), Fault> {
    check_shape(shape)?;
    if !engine_core::valid_global_matrix(matrix) || !valid_position(matrix.w_axis.truncate()) {
        return Err(Fault::InvalidTransform);
    }
    let basis = Mat3::from_mat4(matrix) * Mat3::from_diagonal(shape.dimensions());
    let lengths = Vec3::new(
        basis.x_axis.length(),
        basis.y_axis.length(),
        basis.z_axis.length(),
    );
    if !lengths.is_finite()
        || lengths.min_element() < MIN_COLLIDER_EXTENT
        || lengths.max_element() > MAX_COLLIDER_EXTENT
    {
        return Err(Fault::InvalidTransform);
    }
    let normalized = basis / lengths.max_element();
    let inverse = normalized.inverse();
    let row_sums = inverse.x_axis.abs() + inverse.y_axis.abs() + inverse.z_axis.abs();
    if !inverse.is_finite() || row_sums.max_element() > 1000.0 {
        return Err(Fault::InvalidTransform);
    }
    Ok(())
}

pub(crate) fn valid_position(position: Vec3) -> bool {
    position.is_finite() && position.abs().max_element() <= MAX_COLLISION_COORDINATE
}

settings! {
    #[section("collision")]
    #[derive(Copy, Eq)]
    pub struct CollisionConfig {
        pub enabled: bool = true => any,
        pub max_colliders: Option<u32> = None => at_least(1),
        pub max_candidate_pairs: Option<u32> = None => at_least(1),
        pub max_sensor_pairs: Option<u32> = None => at_least(1),
        pub max_query_candidates: Option<u32> = None => at_least(1),
        pub max_query_hits: Option<u32> = None => at_least(1),
    }
}

impl CollisionConfig {
    pub fn colliders(&self) -> usize {
        within(self.max_colliders, MAX_COLLISION_NODES)
    }

    pub fn candidate_pairs(&self) -> usize {
        within(self.max_candidate_pairs, MAX_CANDIDATE_PAIRS)
    }

    pub fn sensor_pairs(&self) -> usize {
        within(self.max_sensor_pairs, MAX_SENSOR_PAIRS)
    }

    pub fn query_candidates(&self) -> usize {
        within(self.max_query_candidates, MAX_COLLISION_NODES)
    }

    pub fn query_hits(&self) -> usize {
        within(self.max_query_hits, MAX_QUERY_HITS)
    }
}

fn within(budget: Option<u32>, limit: usize) -> usize {
    budget.map_or(limit, |budget| (budget as usize).min(limit))
}

stats! {
    #[section("collision_stats")]
    #[derive(Copy)]
    pub struct CollisionStats {
        pub revision: u64,
        pub frame: u64,
        pub tick: u64,
        pub colliders: usize,
        pub active_colliders: usize,
        pub invalid_colliders: usize,
        pub overflow_colliders: usize,
        pub bvh_nodes: usize,
        pub cache_bytes: usize,
        pub complete: bool,
        pub terrain_tiles: usize,
        pub placed_colliders: usize,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Fault {
    InvalidShape,
    InvalidTransform,
    InvalidRay,
    InvalidCast,
    InvalidConfig,
    IncompleteWorld,
    CandidateBudget,
    PairBudget,
    ResultBudget,
    NumericalFailure,
}

impl Fault {
    pub(crate) fn error(self) -> Error {
        const PATH: &str = "$.collision";
        let code = match self {
            Self::InvalidShape => {
                return Error::new(ErrorCode::InvalidShape, PATH)
                    .range(bound(MIN_COLLIDER_EXTENT), bound(MAX_COLLIDER_EXTENT));
            }
            Self::InvalidTransform => ErrorCode::InvalidWorldTransform,
            Self::InvalidRay => ErrorCode::InvalidRay,
            Self::InvalidCast => ErrorCode::InvalidArguments,
            Self::InvalidConfig | Self::IncompleteWorld => ErrorCode::InvalidValue,
            Self::CandidateBudget | Self::PairBudget | Self::ResultBudget => {
                ErrorCode::OutOfRange
            }
            Self::NumericalFailure => ErrorCode::InvalidNumber,
        };
        Error::new(code, PATH)
    }
}

pub(crate) fn bound(value: f32) -> f64 {
    value.to_string().parse().unwrap_or(f64::from(value))
}

pub(crate) fn validate_shape_cast(
    shape: ColliderShape,
    matrix: Mat4,
    displacement: Vec3,
    options: ShapeCastOptions,
) -> Result<(), Error> {
    check_shape_cast(shape, matrix, displacement, options).map_err(Fault::error)
}

pub(crate) fn check_shape_cast(
    shape: ColliderShape,
    matrix: Mat4,
    displacement: Vec3,
    options: ShapeCastOptions,
) -> Result<(), Fault> {
    check_collider_transform(shape, matrix)?;
    check_options(options)?;
    if !displacement.is_finite() || !valid_position(matrix.w_axis.truncate() + displacement) {
        return Err(Fault::InvalidCast);
    }
    Ok(())
}

pub(crate) fn check_options(options: ShapeCastOptions) -> Result<(), Fault> {
    if options.target_distance.is_finite() && (0.0..=1.0).contains(&options.target_distance) {
        Ok(())
    } else {
        Err(Fault::InvalidCast)
    }
}

pub(crate) fn validate_ray(ray: CollisionRay) -> Result<(), Error> {
    normalized(ray).map(|_| ()).map_err(Fault::error)
}

pub(crate) fn normalized(mut ray: CollisionRay) -> Result<CollisionRay, Fault> {
    if !valid_position(ray.origin)
        || !ray.direction.is_finite()
        || !ray.max_distance.is_finite()
        || !(0.0..=2.0 * MAX_COLLISION_COORDINATE).contains(&ray.max_distance)
    {
        return Err(Fault::InvalidRay);
    }
    let max = ray.direction.abs().max_element();
    if max == 0.0 {
        return Err(Fault::InvalidRay);
    }
    ray.direction = (ray.direction / max).normalize();
    if !valid_position(ray.origin + ray.direction * ray.max_distance) {
        return Err(Fault::InvalidRay);
    }
    Ok(ray)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ColliderStatus {
    Active,
    Invalid,
}
