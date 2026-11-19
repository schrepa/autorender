use std::sync::{Mutex, PoisonError};

use bevy_ecs::component::Component;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::gpu_bits;

#[gpu_bits(crate = crate)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[derive(schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Clock {
    #[default]
    Game = 0,
    Real = 1,
}

#[gpu_bits(crate = crate)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[derive(schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Mobility {
    #[default]
    Static = 0,
    Moving = 1,
}

#[gpu_bits(crate = crate)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[derive(schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Pace {
    #[default]
    Tick = 0,
    Frame = 1,
}

#[derive(Component, Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[derive(schemars::JsonSchema)]
#[component(immutable)]
#[serde(default, deny_unknown_fields)]
pub struct Node {
    pub active: bool,
    pub visible: bool,
    pub clock: Option<Clock>,
    pub mobility: Option<Mobility>,
}

impl Default for Node {
    fn default() -> Self {
        Self {
            active: true,
            visible: true,
            clock: None,
            mobility: None,
        }
    }
}

#[gpu_bits(crate = crate)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "dev", derive(schemars::JsonSchema))]
pub struct NodeWord {
    pub active: bool,
    pub visible: bool,
    #[bits(1)]
    pub clock: Clock,
    #[bits(1)]
    pub mobility: Mobility,
}

impl NodeWord {
    pub const ROOT: Self = Self {
        active: true,
        visible: true,
        clock: Clock::Game,
        mobility: Mobility::Static,
    };

    pub const fn drawn(self) -> bool {
        self.active && self.visible
    }

    pub fn of(node: Node, owned: Mobility, parent: Option<Self>) -> Self {
        let above = parent.unwrap_or(Self::ROOT);
        let mobility = match (above.mobility, node.mobility.unwrap_or(owned)) {
            (Mobility::Moving, _) | (_, Mobility::Moving) => Mobility::Moving,
            (Mobility::Static, Mobility::Static) => Mobility::Static,
        };
        Self {
            active: above.active && node.active,
            visible: above.visible && node.visible,
            clock: node.clock.unwrap_or(above.clock),
            mobility,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MoverId {
    Game,
    Brick(&'static str),
}

impl MoverId {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Game => "game",
            Self::Brick(name) => name,
        }
    }
}

impl Serialize for MoverId {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.name())
    }
}

impl<'de> Deserialize<'de> for MoverId {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let name = String::deserialize(deserializer)?;
        Ok(match name.as_str() {
            "game" => Self::Game,
            _ => Self::Brick(named(&name)),
        })
    }
}

fn named(name: &str) -> &'static str {
    static NAMES: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
    let mut names = NAMES.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(&known) = names.iter().find(|&&known| known == name) {
        return known;
    }
    let held: &'static str = Box::leak(name.to_owned().into_boxed_str());
    names.push(held);
    held
}

impl schemars::JsonSchema for MoverId {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "MoverId".into()
    }

    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({ "type": "string" })
    }
}

#[derive(Component, Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[derive(schemars::JsonSchema)]
#[component(immutable)]
#[serde(deny_unknown_fields)]
pub struct Mover {
    pub by: MoverId,
    pub pace: Pace,
}
