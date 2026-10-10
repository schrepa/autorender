use bevy_ecs::prelude::*;
use engine_core::ErrorCode;
use glam::{Mat3, Mat4, Vec3};

pub use engine_core::{MAX_COLLIDER_EXTENT, MIN_COLLIDER_EXTENT};
pub const MAX_COLLISION_COORDINATE: f32 = 1.0e6;
pub const MAX_COLLISION_NODES: usize = 100_000;
pub const MAX_QUERY_HITS: usize = 4_096;
/// The most sensor candidate pairs one detection examines.
pub const MAX_CANDIDATE_PAIRS: usize = 1_000_000;
/// The most sensor pairs one detection keeps.
pub const MAX_SENSOR_PAIRS: usize = 65_536;

/// A local convex primitive. Affine world transforms preserve shear/stretch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ColliderShape {
    Cuboid { half_extents: Vec3 },
    Sphere { radius: f32 },
}

impl ColliderShape {
    pub fn validate(self) -> Result<(), CollisionError> {
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
            Err(CollisionError::InvalidShape)
        }
    }

    pub(crate) fn dimensions(self) -> Vec3 {
        match self {
            Self::Cuboid { half_extents } => half_extents,
            Self::Sphere { radius } => Vec3::splat(radius),
        }
    }
}

/// Both participants must admit each other's membership bits.
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

/// Checked immutable collision data; attach to an entity with a local Transform.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct Collider {
    shape: ColliderShape,
    layers: CollisionLayers,
    sensor: bool,
}
impl Collider {
    pub fn new(shape: ColliderShape) -> Result<Self, CollisionError> {
        shape.validate()?;
        Ok(Self {
            shape,
            layers: CollisionLayers::default(),
            sensor: false,
        })
    }
    pub fn cuboid(half_extents: Vec3) -> Result<Self, CollisionError> {
        Self::new(ColliderShape::Cuboid { half_extents })
    }
    pub fn sphere(radius: f32) -> Result<Self, CollisionError> {
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

/// Numerical admission shared by authored validation and runtime synchronization.
///
/// Shape-scaled affine basis column lengths must be in [1e-4,1e4], center
/// coordinates in [-1e6,1e6], and normalized basis inverse infinity norm <=1000.
/// Render transform validity remains required. These bounds do not claim exact
/// arithmetic at contact boundaries; narrow-phase has finite tolerances.
pub fn validate_collider_transform(
    shape: ColliderShape,
    matrix: Mat4,
) -> Result<(), CollisionError> {
    shape.validate()?;
    if !engine_core::valid_global_matrix(matrix) || !valid_position(matrix.w_axis.truncate()) {
        return Err(CollisionError::InvalidTransform);
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
        return Err(CollisionError::InvalidTransform);
    }
    let normalized = basis / lengths.max_element();
    let inverse = normalized.inverse();
    let row_sums = inverse.x_axis.abs() + inverse.y_axis.abs() + inverse.z_axis.abs();
    if !inverse.is_finite() || row_sums.max_element() > 1000.0 {
        return Err(CollisionError::InvalidTransform);
    }
    Ok(())
}

pub(crate) fn valid_position(position: Vec3) -> bool {
    position.is_finite() && position.abs().max_element() <= MAX_COLLISION_COORDINATE
}

engine_settings::settings! {
    /// Collision budgets, none by default: the collision world's own
    /// limits bound colliders, sensor pairs and query results. Past a budget
    /// the world reports itself incomplete or a query fails with its
    /// budget's code. Changes apply at the next refresh.
    #[derive(Resource, Debug, Clone, Copy, PartialEq, Eq)]
    pub struct CollisionConfig {
        /// Colliders the world admits; the rest leave it incomplete.
        pub max_colliders: Option<u32> = None => budget,
        /// Sensor candidate pairs one detection examines.
        pub max_candidate_pairs: Option<u32> = None => budget,
        /// Sensor pairs one detection keeps.
        pub max_sensor_pairs: Option<u32> = None => budget,
        /// Candidates one query examines.
        pub max_query_candidates: Option<u32> = None => budget,
        /// Hits one query returns.
        pub max_query_hits: Option<u32> = None => budget,
    }
}

impl CollisionConfig {
    /// Colliders the world admits: the budget within the world's limit.
    pub fn colliders(&self) -> usize {
        within(self.max_colliders, MAX_COLLISION_NODES)
    }

    /// Sensor candidate pairs one detection examines.
    pub fn candidate_pairs(&self) -> usize {
        within(self.max_candidate_pairs, MAX_CANDIDATE_PAIRS)
    }

    /// Sensor pairs one detection keeps.
    pub fn sensor_pairs(&self) -> usize {
        within(self.max_sensor_pairs, MAX_SENSOR_PAIRS)
    }

    /// Candidates one query examines.
    pub fn query_candidates(&self) -> usize {
        within(self.max_query_candidates, MAX_COLLISION_NODES)
    }

    /// Hits one query returns.
    pub fn query_hits(&self) -> usize {
        within(self.max_query_hits, MAX_QUERY_HITS)
    }
}

/// `budget` within `limit`, or `limit` without one.
fn within(budget: Option<u32>, limit: usize) -> usize {
    budget.map_or(limit, |budget| (budget as usize).min(limit))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollisionError {
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
impl std::fmt::Display for CollisionError {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.write_str(match self {
            Self::MissingPlugin => "CollisionPlugin is not installed",
            Self::InvalidShape => "collider dimensions must be finite and within [1e-4,1e4]",
            Self::InvalidTransform => "collider requires valid affine world transform, center within +/-1e6, effective basis lengths [1e-4,1e4], and normalized inverse infinity norm <=1000",
            Self::InvalidRay => "ray requires finite bounded origin/end, nonzero finite direction and distance in [0,2e6]",
            Self::InvalidCast => "shape cast requires finite bounded start/end and target distance in [0,1]",
            Self::InvalidConfig => "collision budgets exceed their documented hard limits",
            Self::IncompleteWorld => "collision snapshot contains invalid or capacity-overflow colliders",
            Self::CandidateBudget => "collision candidate budget exhausted; result is incomplete",
            Self::PairBudget => "sensor pair budget exhausted; previous complete pairs retained",
            Self::ResultBudget => "query result limit exceeded; output cleared",
            Self::NumericalFailure => "collision narrow-phase failed to produce a reliable finite result",
        })
    }
}
impl std::error::Error for CollisionError {}
impl CollisionError {
    pub fn code(self) -> ErrorCode {
        match self {
            Self::MissingPlugin => ErrorCode::MissingPlugin,
            Self::InvalidShape => ErrorCode::CollisionInvalidShape,
            Self::InvalidTransform => ErrorCode::CollisionInvalidTransform,
            Self::InvalidRay => ErrorCode::CollisionInvalidRay,
            Self::InvalidCast => ErrorCode::CollisionInvalidCast,
            Self::InvalidConfig => ErrorCode::CollisionInvalidConfig,
            Self::IncompleteWorld => ErrorCode::CollisionIncompleteWorld,
            Self::CandidateBudget => ErrorCode::CollisionCandidateBudget,
            Self::PairBudget => ErrorCode::CollisionPairBudget,
            Self::ResultBudget => ErrorCode::CollisionResultBudget,
            Self::NumericalFailure => ErrorCode::CollisionNumericalFailure,
        }
    }
}

/// Validation available before any on-demand world refresh.
pub fn validate_shape_cast(
    shape: ColliderShape,
    matrix: Mat4,
    displacement: Vec3,
    options: ShapeCastOptions,
) -> Result<(), CollisionError> {
    validate_collider_transform(shape, matrix)?;
    options.validate()?;
    if !displacement.is_finite() || !valid_position(matrix.w_axis.truncate() + displacement) {
        return Err(CollisionError::InvalidCast);
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

/// Direction is normalized internally; max_distance and returned distance use world units.
#[derive(Debug, Clone, Copy)]
pub struct CollisionRay {
    pub origin: Vec3,
    pub direction: Vec3,
    pub max_distance: f32,
}
impl CollisionRay {
    pub fn validate(self) -> Result<(), CollisionError> {
        self.normalized().map(|_| ())
    }
    pub(crate) fn normalized(mut self) -> Result<Self, CollisionError> {
        if !valid_position(self.origin)
            || !self.direction.is_finite()
            || !self.max_distance.is_finite()
            || !(0.0..=2.0 * MAX_COLLISION_COORDINATE).contains(&self.max_distance)
        {
            return Err(CollisionError::InvalidRay);
        }
        let max = self.direction.abs().max_element();
        if max == 0.0 {
            return Err(CollisionError::InvalidRay);
        }
        self.direction = (self.direction / max).normalize();
        if !valid_position(self.origin + self.direction * self.max_distance) {
            return Err(CollisionError::InvalidRay);
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
    /// The placed object hit, whose terrain is `entity`.
    pub placed: Option<PlacedPart>,
}

/// A placed object: its placement layer and asset, by index
/// in its terrain's surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlacedPart {
    pub layer: u16,
    pub asset: u16,
}

#[derive(Debug, Clone, Copy)]
pub struct ShapeCastOptions {
    /// False permits initial contact when motion is tangential or separating.
    pub stop_at_penetration: bool,
    /// World-space skin, in [0,1]. For two cuboids this expands the 15 SAT
    /// intervals conservatively; corner distance is not an exact rounded hull.
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
    pub fn validate(self) -> Result<(), CollisionError> {
        if self.target_distance.is_finite() && (0.0..=1.0).contains(&self.target_distance) {
            Ok(())
        } else {
            Err(CollisionError::InvalidCast)
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ShapeHit {
    pub entity: Entity,
    pub fraction: f32,
    /// Representative target surface point, not a contact manifold or a
    /// guarantee of the closest witness under shear or target-distance skin.
    pub point: Vec3,
    /// Unit world normal pointing outward from the target.
    pub normal: Vec3,
    /// Includes initial touching or being within the configured skin/tolerance.
    pub started_overlapping: bool,
    /// The placed object hit, whose terrain is `entity`.
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
    /// Terrain tiles colliding as heightfields.
    pub terrain_tiles: usize,
    /// Placed objects colliding within terrains' collision radius.
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
