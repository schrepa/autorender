use bevy_ecs::prelude::*;
use engine_core::{Error, Operation};
use glam::Vec3;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::RiverFlow;
use crate::state::WaterState;

#[derive(Serialize)]
pub struct PoolView {
    pub at: [f32; 3],
    pub rise: f32,
    pub inflow: f32,
    pub outflow: f32,
    pub blocked: f32,
    pub crest: Option<f32>,
    pub area: f32,
}

#[derive(Serialize)]
pub(crate) struct FlowView {
    pub blocks: usize,
    pub pools: Vec<PoolView>,
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct WaterFlowQuery {}

impl Operation for WaterFlowQuery {
    const NAME: &'static str = "water_pools";
    type Params = Res<'static, WaterState>;
    type Request = Self;

    fn run(state: SystemParamItem<'_, '_, Self::Params>, _: Self) -> Result<Value, Error> {
        let flows: Vec<FlowView> = state
            .terrains
            .iter()
            .filter(|terrain| !terrain.surroundings)
            .map(|terrain| {
                let held = terrain.flow.as_ref();
                FlowView {
                    blocks: held.map_or(0, |flow| flow.blocks().len()),
                    pools: held.map_or_else(Vec::new, |flow| pools(flow, terrain.origin)),
                }
            })
            .collect();
        serde_json::to_value(flows)
            .map_err(|error| Error::new(engine_core::ErrorCode::InvalidValue, "$").got(error.to_string()))
    }
}

pub(crate) fn pools(flow: &RiverFlow, origin: Vec3) -> Vec<PoolView> {
    flow.pools()
        .iter()
        .map(|pool| {
            let (point, rise) = flow.pool_place(pool);
            let (blocked, crest, area) = flow.holding(pool);
            PoolView {
                at: [
                    point[0] + origin.x,
                    pool.level + origin.y,
                    point[1] + origin.z,
                ],
                rise,
                inflow: pool.inflow,
                outflow: pool.outflow,
                blocked,
                crest: crest.map(|crest| crest + origin.y),
                area,
            }
        })
        .collect()
}
