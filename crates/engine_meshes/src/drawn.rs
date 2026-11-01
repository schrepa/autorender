use bevy_ecs::prelude::*;
use engine_core::{Error, ErrorCode, Expected, FrameTime, Operation, SceneIds};
use engine_places::meshes::Instances;
use serde_json::{Value, json};

use crate::layouts::LOD_STATE;
use crate::presentation::PresentationTotals;

#[derive(Resource, Default)]
pub(crate) struct WatchedSlot {
    pub slot: Option<u32>,
}

#[derive(engine_core::__private::serde::Deserialize)]
#[serde(crate = "engine_core::__private::serde", deny_unknown_fields)]
pub(crate) struct DrawnRequest {
    entity: String,
}

pub(crate) struct MeshDrawn;

impl Operation for MeshDrawn {
    const NAME: &'static str = "mesh_drawn";
    const CONTROL: bool = false;
    type Params = (
        SceneIds<'static, 'static>,
        Res<'static, Instances>,
        Res<'static, PresentationTotals>,
        Res<'static, FrameTime>,
        ResMut<'static, WatchedSlot>,
    );
    type Request = DrawnRequest;

    fn run(
        (ids, instances, totals, time, mut watched): SystemParamItem<'_, '_, Self::Params>,
        request: DrawnRequest,
    ) -> Result<Value, Error> {
        let id = request.entity;
        let entity = ids.entity(&id).ok_or_else(|| {
            Error::new(ErrorCode::UnknownEntity, "$.entity")
                .expected(Expected::Entity)
                .got(&id)
        })?;
        let Some(slot) = instances.drawn(entity) else {
            return Ok(json!({ "slot": null, "drawn_level": null, "read_back": null }));
        };
        watched.slot = Some(slot);
        let read_back = totals
            .last()
            .and_then(|last| last.watched)
            .filter(|&(seen, _)| seen == slot)
            .map(|(_, state)| (state & LOD_STATE != 0).then_some(state & 3));
        let age = totals.age(time.frame);
        Ok(json!({
            "slot": slot,
            "drawn_level": read_back.flatten(),
            "read_back": read_back
                .zip(age)
                .map(|(level, age)| json!({ "level": level, "age": age })),
        }))
    }
}
