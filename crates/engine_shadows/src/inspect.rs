use bevy_ecs::prelude::*;
use engine_core::glam::Vec3;
use engine_core::{Error, Operation};
use serde_json::{Value, json};

use crate::plan::ShadowPlan;

pub(crate) struct InspectShadowTile;

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TileRequest {
    at: [f32; 3],
}

impl Operation for InspectShadowTile {
    const NAME: &'static str = "inspect_shadow_tile";
    type Params = Res<'static, ShadowPlan>;
    type Request = TileRequest;

    fn run(
        plan: SystemParamItem<'_, '_, Self::Params>,
        request: TileRequest,
    ) -> Result<Value, Error> {
        let clipmap = plan.clipmap();
        Ok(json!({
            "at": request.at,
            "levels": clipmap.len(),
            "tiles": clipmap.inspect(Vec3::from_array(request.at)),
        }))
    }
}
