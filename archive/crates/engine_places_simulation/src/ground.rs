use std::sync::Arc;

use bevy_ecs::prelude::{Component, Entity, Resource};
use engine_place::Slot;
use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum GroundSurface {
    Collider,
    Field,
    Placed,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroundPoint {
    pub height: f32,
    pub normal: Vec3,
    pub surface: GroundSurface,
    pub owner: Entity,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroundProbe {
    pub at: Vec2,
    pub below: Option<f32>,
    pub exclude: Option<Entity>,
    pub layers: u32,
}

impl GroundProbe {
    pub const fn at(at: Vec2) -> Self {
        Self {
            at,
            below: None,
            exclude: None,
            layers: u32::MAX,
        }
    }

    #[must_use]
    pub const fn below(mut self, height: f32) -> Self {
        self.below = Some(height);
        self
    }

    #[must_use]
    pub const fn excluding(mut self, entity: Entity) -> Self {
        self.exclude = Some(entity);
        self
    }

    #[must_use]
    pub const fn on_layers(mut self, layers: u32) -> Self {
        self.layers = layers;
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GroundRead<T> {
    Ready(T),
    NotLoaded,
    Off,
}

impl<T> GroundRead<T> {
    pub fn ready(self) -> Option<T> {
        match self {
            Self::Ready(value) => Some(value),
            Self::NotLoaded | Self::Off => None,
        }
    }
}

pub trait GroundQuery: Send + Sync + 'static {
    fn down(&self, probe: GroundProbe) -> GroundRead<Option<GroundPoint>>;

    fn revision(&self) -> u64;

    fn layout(&self) -> u64;
}

struct NoGround;

impl GroundQuery for NoGround {
    fn down(&self, _: GroundProbe) -> GroundRead<Option<GroundPoint>> {
        GroundRead::Off
    }

    fn revision(&self) -> u64 {
        0
    }

    fn layout(&self) -> u64 {
        0
    }
}

#[derive(Resource, Clone)]
pub struct Ground(pub Arc<dyn GroundQuery>);

impl Ground {
    pub fn down(&self, probe: GroundProbe) -> GroundRead<Option<GroundPoint>> {
        self.0.down(probe)
    }

    pub fn point(&self, at: Vec2) -> GroundRead<Option<GroundPoint>> {
        self.0.down(GroundProbe::at(at))
    }

    pub fn revision(&self) -> u64 {
        self.0.revision()
    }

    pub fn layout(&self) -> u64 {
        self.0.layout()
    }
}

impl Default for Ground {
    fn default() -> Self {
        Self(Arc::new(NoGround))
    }
}

impl Slot for Ground {
    const NAME: &'static str = "ground";
}

#[derive(
    Component, Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema,
)]
#[require(GroundedAt)]
#[serde(default, deny_unknown_fields)]
pub struct Grounded {
    pub offset: f32,
}

#[derive(Component, Clone, Copy, Debug, Default, PartialEq)]
pub enum GroundedAt {
    #[default]
    Unsettled,
    Stood {
        position: Vec2,
        offset: f32,
        surface: Option<GroundSurface>,
        revision: u64,
    },
    Waiting,
    Unsupported {
        position: Vec2,
        offset: f32,
        revision: u64,
    },
}

impl GroundedAt {
    pub fn surface(self) -> Option<GroundSurface> {
        match self {
            Self::Stood { surface, .. } => surface,
            Self::Unsettled | Self::Waiting | Self::Unsupported { .. } => None,
        }
    }
}
