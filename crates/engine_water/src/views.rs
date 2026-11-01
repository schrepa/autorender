use super::*;
use crate::RiverFlow;

#[derive(Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct PoolView {
    pub at: [f32; 3],
    pub rise: f32,
    pub inflow: f32,
    pub outflow: f32,
    pub blocked: f32,
    pub crest: Option<f32>,
    pub area: f32,
}

fn pools(flow: &RiverFlow, origin: Vec3) -> Vec<PoolView> {
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
