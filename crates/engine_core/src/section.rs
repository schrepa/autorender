use bevy_ecs::resource::Resource;
use bevy_ecs::system::SystemParamItem;
use bevy_ecs::world::World;
use engine_place::{EventPage, Operation};
use engine_registry::Registry;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};

use crate::{Error, ErrorCode, SceneContext};

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

    fn schema() -> schemars::Schema {
        schemars::schema_for!(Self)
    }
}

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, serde::Deserialize, schemars::JsonSchema,
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

#[derive(Debug, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EventsRequest {
    pub name: String,
    #[serde(default)]
    pub after: u64,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct EventsReply {
    pub name: String,
    #[serde(flatten)]
    pub page: EventPage,
}

pub struct EventsQuery;

impl Operation for EventsQuery {
    const NAME: &'static str = "events";
    const CONTROL: bool = false;
    type Params = &'static World;
    type Request = EventsRequest;
    type Reply = EventsReply;

    fn run(
        world: SystemParamItem<'_, '_, Self::Params>,
        request: EventsRequest,
    ) -> Result<EventsReply, Error> {
        let registry = world
            .get_resource::<Registry>()
            .ok_or_else(|| Error::new(ErrorCode::MissingResource, "$.name"))?;
        let event = registry.event(&request.name).ok_or_else(|| {
            Error::new(ErrorCode::UnknownVariant, "$.name")
                .got(&request.name)
                .variants(registry.events().map(|event| event.name))
        })?;
        let page = event.read(world, request.after)?;

        Ok(EventsReply {
            name: request.name,
            page,
        })
    }

    fn reply_schema(world: &World) -> Value {
        let mut definitions = serde_json::Map::new();
        let reply = crate::embedded(crate::schema_of::<EventsReply>(), &mut definitions);
        let page = |name: Value, rows: Value| {
            let mut page = reply.clone();
            if let Some(properties) = page.get_mut("properties").and_then(Value::as_object_mut) {
                properties.insert("name".to_owned(), name);
                properties.insert("rows".to_owned(), rows);
            }
            page
        };
        let mut pages = Vec::new();
        for event in world
            .get_resource::<Registry>()
            .into_iter()
            .flat_map(Registry::events)
        {
            let row = crate::embedded(event.row_schema(), &mut definitions);
            pages.push(page(
                json!({ "const": event.name, "type": "string" }),
                json!({ "type": "array", "items": row }),
            ));
        }
        if pages.is_empty() {
            pages.push(page(
                json!({ "type": "string" }),
                json!({ "type": "array", "maxItems": 0 }),
            ));
        }
        crate::rooted(
            json!({ "title": "EventsReply", "oneOf": pages }),
            definitions,
        )
    }
}
