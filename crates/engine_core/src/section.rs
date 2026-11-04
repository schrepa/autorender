use bevy_ecs::component::Mutable;
use bevy_ecs::resource::Resource;
use bevy_ecs::system::{SystemParam, SystemParamItem};
use engine_registry::{ExplainAsk, ExplainRecord};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;

use crate::{Error, SceneContext};

pub trait Section:
    Resource + Clone + Default + PartialEq + Serialize + DeserializeOwned + schemars::JsonSchema
{
    const KEY: &'static str;

    fn check(&self) -> Result<(), Error> {
        Ok(())
    }

    fn check_scene(&self, _: &SceneContext) -> Result<(), Error> {
        Ok(())
    }

    fn pending(&self) -> u32 {
        0
    }

    fn schema() -> schemars::Schema {
        schemars::schema_for!(Self)
    }
}

#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    Serialize,
    serde::Deserialize,
    schemars::JsonSchema,
)]
pub struct Cut {
    pub asked: u32,
    pub kept: u32,
    pub cut: u32,
}

impl Cut {
    pub fn of(asked: u32, kept: u32) -> Self {
        let kept = kept.min(asked);
        Self {
            asked,
            kept,
            cut: asked - kept,
        }
    }
}

pub const RESERVED_REPLY_KEYS: [&str; 4] = ["observation", "paused", "source", "rendering"];

pub trait Operation: Send + Sync + 'static {
    const NAME: &'static str;
    const CONTROL: bool = true;
    type Params: SystemParam + 'static;
    type Request: DeserializeOwned + 'static;
    fn run(
        params: SystemParamItem<'_, '_, Self::Params>,
        request: Self::Request,
    ) -> Result<Value, Error>;

    fn schema() -> Value {
        serde_json::json!({ "type": "object" })
    }

    fn result_schema() -> Option<Value> {
        None
    }
}

pub trait Explain: Send + Sync + 'static {
    type Params: SystemParam + 'static;
    fn explain(
        params: SystemParamItem<'_, '_, Self::Params>,
        ask: &ExplainAsk<'_>,
    ) -> Option<ExplainRecord>;
}

pub trait Readback: Resource<Mutability = Mutable> + Default {
    fn arrived(&mut self, frame: u64, tag: u32, bytes: &[u8]);
}
