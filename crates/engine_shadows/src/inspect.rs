use bevy_ecs::prelude::*;
use engine_core::glam::Vec3;
use engine_core::{Error, Operation};
use crate::plan::ShadowPlan;
use crate::shadow_fit::TileState;

pub(crate) struct InspectShadowTile;

#[derive(serde::Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct TileRequest {
    at: [f32; 3],
}

#[derive(serde::Serialize, schemars::JsonSchema)]
pub(crate) struct TileReply {
    at: [f32; 3],
    levels: usize,
    tiles: Vec<TileState>,
}

impl Operation for InspectShadowTile {
    const NAME: &'static str = "inspect_shadow_tile";
    type Params = Res<'static, ShadowPlan>;
    type Request = TileRequest;
    type Reply = TileReply;

    fn run(
        plan: SystemParamItem<'_, '_, Self::Params>,
        request: TileRequest,
    ) -> Result<TileReply, Error> {
        let clipmap = plan.clipmap();
        Ok(TileReply {
            at: request.at,
            levels: clipmap.len(),
            tiles: clipmap.inspect(Vec3::from_array(request.at)),
        })
    }
}
