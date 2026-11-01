use std::any::Any;

use bevy_ecs::prelude::{Component, Entity, Resource};
use engine_place::{Error, ErrorCode, InterpolateTransform, Slot};
use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};

pub const MIN_COLLIDER_EXTENT: f32 = 1.0e-4;
pub const MAX_COLLIDER_EXTENT: f32 = 1.0e4;
pub const MAX_COLLISION_COORDINATE: f32 = 1.0e6;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ColliderShape {
    Cuboid { half_extents: Vec3 },
    Sphere { radius: f32 },
}

impl ColliderShape {
    pub fn dimensions(self) -> Vec3 {
        match self {
            Self::Cuboid { half_extents } => half_extents,
            Self::Sphere { radius } => Vec3::splat(radius),
        }
    }

    pub fn validate(self) -> Result<(), Error> {
        let dimensions = self.dimensions();
        if dimensions.is_finite()
            && dimensions.cmpge(Vec3::splat(MIN_COLLIDER_EXTENT)).all()
            && dimensions.cmple(Vec3::splat(MAX_COLLIDER_EXTENT)).all()
        {
            Ok(())
        } else {
            Err(Error::new(ErrorCode::CollisionInvalidShape, "$.collision").range(
                engine_heightfield::bound(MIN_COLLIDER_EXTENT),
                engine_heightfield::bound(MAX_COLLIDER_EXTENT),
            ))
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

#[derive(Component, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(from = "AuthoredCollider", into = "AuthoredCollider")]
pub struct Collider {
    shape: ColliderShape,
    layers: CollisionLayers,
    sensor: bool,
}

#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum AuthoredShape {
    Cuboid { half_extents: [f32; 3] },
    Sphere { radius: f32 },
}

impl From<AuthoredShape> for ColliderShape {
    fn from(authored: AuthoredShape) -> Self {
        match authored {
            AuthoredShape::Cuboid { half_extents } => Self::Cuboid {
                half_extents: Vec3::from_array(half_extents),
            },
            AuthoredShape::Sphere { radius } => Self::Sphere { radius },
        }
    }
}

impl From<ColliderShape> for AuthoredShape {
    fn from(shape: ColliderShape) -> Self {
        match shape {
            ColliderShape::Cuboid { half_extents } => Self::Cuboid {
                half_extents: half_extents.to_array(),
            },
            ColliderShape::Sphere { radius } => Self::Sphere { radius },
        }
    }
}

#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AuthoredCollider {
    shape: AuthoredShape,
    #[serde(default = "one_membership")]
    memberships: u32,
    #[serde(default = "every_filter")]
    filter: u32,
    #[serde(default)]
    sensor: bool,
}

fn one_membership() -> u32 {
    1
}

fn every_filter() -> u32 {
    u32::MAX
}

impl From<AuthoredCollider> for Collider {
    fn from(authored: AuthoredCollider) -> Self {
        Self {
            shape: authored.shape.into(),
            layers: CollisionLayers {
                memberships: authored.memberships,
                filter: authored.filter,
            },
            sensor: authored.sensor,
        }
    }
}

impl From<Collider> for AuthoredCollider {
    fn from(collider: Collider) -> Self {
        Self {
            shape: collider.shape.into(),
            memberships: collider.layers.memberships,
            filter: collider.layers.filter,
            sensor: collider.sensor,
        }
    }
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum BodyMotion {
    #[default]
    Dynamic,
    Kinematic,
}

#[derive(Component, Clone, Copy, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[require(InterpolateTransform)]
#[schemars(rename = "SceneRigidBody")]
#[serde(default, deny_unknown_fields)]
pub struct RigidBody {
    pub motion: BodyMotion,
    pub density: f32,
    pub friction: f32,
    pub restitution: f32,
    pub drag: f32,
    pub velocity: [f32; 3],
    pub spin: [f32; 3],
    pub fast: bool,
}

impl Default for RigidBody {
    fn default() -> Self {
        Self {
            motion: BodyMotion::Dynamic,
            density: 600.0,
            friction: 0.6,
            restitution: 0.05,
            drag: 1.0,
            velocity: [0.0; 3],
            spin: [0.0; 3],
            fast: false,
        }
    }
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
    pub fn accepts(self, entity: Entity, collider: Collider) -> bool {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlacedPart {
    pub layer: u16,
    pub asset: u16,
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

#[derive(Debug, Clone, Copy)]
pub struct ShapeHit {
    pub entity: Entity,
    pub fraction: f32,
    pub point: Vec3,
    pub normal: Vec3,
    pub started_overlapping: bool,
    pub placed: Option<PlacedPart>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SensorPair {
    pub first: Entity,
    pub second: Entity,
}

impl SensorPair {
    pub fn new(a: Entity, b: Entity) -> Self {
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

pub trait Collide: Any + Send + Sync + 'static {
    fn raycast(&self, ray: CollisionRay, filter: QueryFilter) -> Result<Option<RayHit>, Error>;

    fn raycast_all(
        &self,
        ray: CollisionRay,
        filter: QueryFilter,
        output: &mut Vec<RayHit>,
        limit: usize,
    ) -> Result<(), Error>;

    fn overlap(
        &self,
        shape: ColliderShape,
        matrix: Mat4,
        filter: QueryFilter,
        output: &mut Vec<Entity>,
        limit: usize,
    ) -> Result<(), Error>;

    fn cast_shape(
        &self,
        shape: ColliderShape,
        matrix: Mat4,
        displacement: Vec3,
        filter: QueryFilter,
        options: ShapeCastOptions,
    ) -> Result<Option<ShapeHit>, Error>;

    fn events(&self) -> &[SensorEvent];

    fn pairs(&self) -> &[SensorPair];

    fn colliders(&self) -> Option<usize>;
}

struct NoCollision;

impl Collide for NoCollision {
    fn raycast(&self, _: CollisionRay, _: QueryFilter) -> Result<Option<RayHit>, Error> {
        Ok(None)
    }

    fn raycast_all(
        &self,
        _: CollisionRay,
        _: QueryFilter,
        _: &mut Vec<RayHit>,
        _: usize,
    ) -> Result<(), Error> {
        Ok(())
    }

    fn overlap(
        &self,
        _: ColliderShape,
        _: Mat4,
        _: QueryFilter,
        _: &mut Vec<Entity>,
        _: usize,
    ) -> Result<(), Error> {
        Ok(())
    }

    fn cast_shape(
        &self,
        _: ColliderShape,
        _: Mat4,
        _: Vec3,
        _: QueryFilter,
        _: ShapeCastOptions,
    ) -> Result<Option<ShapeHit>, Error> {
        Ok(None)
    }

    fn events(&self) -> &[SensorEvent] {
        &[]
    }

    fn pairs(&self) -> &[SensorPair] {
        &[]
    }

    fn colliders(&self) -> Option<usize> {
        None
    }
}

#[derive(Resource)]
pub struct Collision(pub Box<dyn Collide>);

impl Default for Collision {
    fn default() -> Self {
        Self(Box::new(NoCollision))
    }
}

impl Slot for Collision {
    const NAME: &'static str = "collision";
}
