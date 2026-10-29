use bevy_ecs::prelude::*;
use engine_core::{Error, ErrorCode, missing_brick};
use engine_settings::rule::Bound;
use glam::{Mat3, Mat4, Vec3};

pub use engine_terrain::{MAX_COLLIDER_EXTENT, MIN_COLLIDER_EXTENT};
pub const MAX_COLLISION_COORDINATE: f32 = 1.0e6;
pub const MAX_COLLISION_NODES: usize = 100_000;
pub const MAX_QUERY_HITS: usize = 4_096;
pub const MAX_CANDIDATE_PAIRS: usize = 1_000_000;
pub const MAX_SENSOR_PAIRS: usize = 65_536;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ColliderShape {
    Cuboid { half_extents: Vec3 },
    Sphere { radius: f32 },
}

impl ColliderShape {
    pub fn validate(self) -> Result<(), Error> {
        self.check().map_err(Fault::error)
    }

    pub(crate) fn check(self) -> Result<(), Fault> {
        let dimensions = match self {
            Self::Cuboid { half_extents } => half_extents,
            Self::Sphere { radius } => Vec3::splat(radius),
        };
        if dimensions.is_finite()
            && dimensions.cmpge(Vec3::splat(MIN_COLLIDER_EXTENT)).all()
            && dimensions.cmple(Vec3::splat(MAX_COLLIDER_EXTENT)).all()
        {
            Ok(())
        } else {
            Err(Fault::InvalidShape)
        }
    }

    pub(crate) fn dimensions(self) -> Vec3 {
        match self {
            Self::Cuboid { half_extents } => half_extents,
            Self::Sphere { radius } => Vec3::splat(radius),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CollisionLayers {
    pub memberships: u32,
    pub filter: u32,
}
impl Default for CollisionLayers {
    fn default() -> Self {
        Self {
            memberships: 1,
            filter: u32::MAX,
        }
    }
}
impl CollisionLayers {
    pub const fn new(memberships: u32, filter: u32) -> Self {
        Self {
            memberships,
            filter,
        }
    }
    pub const fn interacts(self, other: Self) -> bool {
        self.memberships & other.filter != 0 && other.memberships & self.filter != 0
    }
}

#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct Collider {
    shape: ColliderShape,
    layers: CollisionLayers,
    sensor: bool,
}
impl Collider {
    pub fn new(shape: ColliderShape) -> Result<Self, Error> {
        shape.validate()?;
        Ok(Self {
            shape,
            layers: CollisionLayers::default(),
            sensor: false,
        })
    }
    pub fn cuboid(half_extents: Vec3) -> Result<Self, Error> {
        Self::new(ColliderShape::Cuboid { half_extents })
    }
    pub fn sphere(radius: f32) -> Result<Self, Error> {
        Self::new(ColliderShape::Sphere { radius })
    }
    #[must_use]
    pub fn with_layers(mut self, layers: CollisionLayers) -> Self {
        self.layers = layers;
        self
    }
    #[must_use]
    pub fn with_sensor(mut self, sensor: bool) -> Self {
        self.sensor = sensor;
        self
    }
    pub fn shape(self) -> ColliderShape {
        self.shape
    }
    pub fn layers(self) -> CollisionLayers {
        self.layers
    }
    pub fn is_sensor(self) -> bool {
        self.sensor
    }
}

pub fn validate_collider_transform(shape: ColliderShape, matrix: Mat4) -> Result<(), Error> {
    check_collider_transform(shape, matrix).map_err(Fault::error)
}

pub(crate) fn check_collider_transform(shape: ColliderShape, matrix: Mat4) -> Result<(), Fault> {
    shape.check()?;
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

engine_settings::settings! {
    unauthored;
    #[derive(Resource, Debug, Clone, Copy, PartialEq, Eq)]
    pub struct CollisionConfig {
        pub max_colliders: Option<u32> = None => budget,
        pub max_candidate_pairs: Option<u32> = None => budget,
        pub max_sensor_pairs: Option<u32> = None => budget,
        pub max_query_candidates: Option<u32> = None => budget,
        pub max_query_hits: Option<u32> = None => budget,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Fault {
    MissingPlugin,
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
            Self::MissingPlugin => return missing_brick("collision", PATH),
            Self::InvalidShape => {
                return Error::new(ErrorCode::CollisionInvalidShape, PATH)
                    .range(MIN_COLLIDER_EXTENT.bound(), MAX_COLLIDER_EXTENT.bound());
            }
            Self::InvalidTransform => ErrorCode::CollisionInvalidTransform,
            Self::InvalidRay => ErrorCode::CollisionInvalidRay,
            Self::InvalidCast => ErrorCode::CollisionInvalidCast,
            Self::InvalidConfig => ErrorCode::CollisionInvalidConfig,
            Self::IncompleteWorld => ErrorCode::CollisionIncompleteWorld,
            Self::CandidateBudget => ErrorCode::CollisionCandidateBudget,
            Self::PairBudget => ErrorCode::CollisionPairBudget,
            Self::ResultBudget => ErrorCode::CollisionResultBudget,
            Self::NumericalFailure => ErrorCode::CollisionNumericalFailure,
        };
        Error::new(code, PATH)
    }
}

pub fn validate_shape_cast(
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
    options.check()?;
    if !displacement.is_finite() || !valid_position(matrix.w_axis.truncate() + displacement) {
        return Err(Fault::InvalidCast);
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColliderStatus {
    Active,
    Invalid,
    Overflow,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SensorFilter {
    #[default]
    Include,
    Exclude,
    Only,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct QueryFilter {
    pub layers: CollisionLayers,
    pub sensors: SensorFilter,
    pub exclude: Option<Entity>,
}
impl QueryFilter {
    pub(crate) fn accepts(self, entity: Entity, collider: Collider) -> bool {
        self.exclude != Some(entity)
            && self.layers.interacts(collider.layers)
            && match self.sensors {
                SensorFilter::Include => true,
                SensorFilter::Exclude => !collider.sensor,
                SensorFilter::Only => collider.sensor,
            }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CollisionRay {
    pub origin: Vec3,
    pub direction: Vec3,
    pub max_distance: f32,
}
impl CollisionRay {
    pub fn validate(self) -> Result<(), Error> {
        self.normalized().map(|_| ()).map_err(Fault::error)
    }
    pub(crate) fn normalized(mut self) -> Result<Self, Fault> {
        if !valid_position(self.origin)
            || !self.direction.is_finite()
            || !self.max_distance.is_finite()
            || !(0.0..=2.0 * MAX_COLLISION_COORDINATE).contains(&self.max_distance)
        {
            return Err(Fault::InvalidRay);
        }
        let max = self.direction.abs().max_element();
        if max == 0.0 {
            return Err(Fault::InvalidRay);
        }
        self.direction = (self.direction / max).normalize();
        if !valid_position(self.origin + self.direction * self.max_distance) {
            return Err(Fault::InvalidRay);
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RayHit {
    pub entity: Entity,
    pub distance: f32,
    pub point: Vec3,
    pub normal: Vec3,
    pub started_inside: bool,
    pub placed: Option<PlacedPart>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlacedPart {
    pub layer: u16,
    pub asset: u16,
}

#[derive(Debug, Clone, Copy)]
pub struct ShapeCastOptions {
    pub stop_at_penetration: bool,
    pub target_distance: f32,
}
impl Default for ShapeCastOptions {
    fn default() -> Self {
        Self {
            stop_at_penetration: true,
            target_distance: 0.0,
        }
    }
}
impl ShapeCastOptions {
    pub fn validate(self) -> Result<(), Error> {
        self.check().map_err(Fault::error)
    }

    pub(crate) fn check(self) -> Result<(), Fault> {
        if self.target_distance.is_finite() && (0.0..=1.0).contains(&self.target_distance) {
            Ok(())
        } else {
            Err(Fault::InvalidCast)
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ShapeHit {
    pub entity: Entity,
    pub fraction: f32,
    pub point: Vec3,
    pub normal: Vec3,
    pub started_overlapping: bool,
    pub placed: Option<PlacedPart>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CollisionReport {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SensorPair {
    pub first: Entity,
    pub second: Entity,
}
impl SensorPair {
    pub(crate) fn new(a: Entity, b: Entity) -> Self {
        if a < b {
            Self {
                first: a,
                second: b,
            }
        } else {
            Self {
                first: b,
                second: a,
            }
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SensorEventKind {
    Enter,
    Exit,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SensorEvent {
    pub pair: SensorPair,
    pub kind: SensorEventKind,
}
