use super::*;
use crate::{Hydrology, RiverFlow};
use engine_core::TerrainDebug;

pub const FALLS_SHOWN: usize = 16;
const LAKES_SHOWN: usize = 8;

#[derive(Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct TerrainsView {
    pub terrains: u32,
    pub resident_tiles: u32,
    pub loading_tiles: u32,
    pub waiting_tiles: u32,
    pub cached_tiles: u32,
    pub cached_bytes: u64,
    pub restored_tiles: u64,
    pub prefetching_tiles: u32,
    pub dropped_tiles: u32,
    pub fallback_tiles: u32,
    pub baking: u32,
    pub bridges: u32,
    pub tunnels: u32,
    pub falls: u32,
    pub surroundings_tiles: u32,
    pub water: Option<TerrainWaterView>,
    pub failed_jobs: u64,
    pub latest_job_failure: Option<String>,
}

#[derive(Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct TerrainWaterView {
    pub flow: RiverFlowView,
    pub falls: Vec<FallView>,
    pub lakes: Vec<LakeView>,
    pub sights: Vec<RiverSightView>,
    pub blocks: usize,
    pub pools: Vec<PoolView>,
}

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

#[derive(Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct LakeView {
    pub level: f32,
    pub area: f32,
    pub middle: [f32; 3],
    pub shore: [f32; 3],
}

#[derive(Serialize)]
#[cfg_attr(
    feature = "schema",
    derive(schemars::JsonSchema),
    schemars(rename = "RiverSight")
)]
pub struct RiverSightView {
    pub kind: &'static str,
    pub point: [f32; 3],
    pub direction: [f32; 2],
    pub half: f32,
    pub speed: f32,
    pub discharge: f32,
}

#[derive(Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct RiverFlowView {
    pub rivers: usize,
    pub lakes: usize,
    pub points: usize,
    pub max_discharge: f32,
    pub fastest: f32,
    pub rapid_share: f32,
}

#[derive(Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct FallView {
    pub top: [f32; 3],
    pub bottom: [f32; 3],
    pub width: f32,
    pub speed: f32,
    pub discharge: f32,
}

#[derive(Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct TerrainDebugView {
    pub freeze: Option<[f32; 3]>,
}

impl TerrainDebugView {
    pub fn of(debug: &TerrainDebug) -> Self {
        Self {
            freeze: debug.freeze.map(|point| point.to_array()),
        }
    }
}

impl RiverFlowView {
    pub fn of(found: &Hydrology) -> Self {
        let flow = found.flow_summary();
        Self {
            rivers: flow.rivers,
            lakes: flow.lakes,
            points: flow.points,
            max_discharge: flow.max_discharge,
            fastest: flow.fastest,
            rapid_share: flow.rapid_share,
        }
    }
}

impl TerrainsView {
    pub fn of(terrains: &Terrains) -> Self {
        let stats = terrains.stats();
        let (failed_jobs, latest_job_failure) = terrains.job_failures();
        Self {
            terrains: stats.terrains,
            resident_tiles: stats.resident_tiles,
            loading_tiles: stats.loading_tiles,
            waiting_tiles: stats.waiting_tiles,
            cached_tiles: stats.cached_tiles,
            cached_bytes: stats.cached_bytes,
            restored_tiles: stats.restored_tiles,
            prefetching_tiles: stats.prefetching_tiles,
            dropped_tiles: stats.dropped_tiles,
            fallback_tiles: stats.fallback_tiles,
            baking: stats.baking,
            bridges: stats.bridges,
            tunnels: stats.tunnels,
            falls: stats.falls,
            surroundings_tiles: stats.surroundings_tiles,
            water: terrains
                .waters()
                .next()
                .and_then(|(entity, origin, layers)| {
                    let found = layers.hydrology()?;
                    let held = terrains.river_flow(entity);
                    let world = |point: [f32; 3]| {
                        [
                            point[0] + origin.x,
                            point[1] + origin.y,
                            point[2] + origin.z,
                        ]
                    };
                    Some(TerrainWaterView {
                        flow: RiverFlowView::of(found),
                        lakes: lakes(found, origin),
                        sights: sights(found, origin),
                        blocks: held.map_or(0, |flow| flow.blocks().len()),
                        pools: held.map_or_else(Vec::new, |flow| pools(flow, origin)),
                        falls: found
                            .falls
                            .iter()
                            .take(FALLS_SHOWN)
                            .map(|fall| FallView {
                                top: world(fall.top),
                                bottom: world(fall.bottom),
                                width: fall.width,
                                speed: fall.speed,
                                discharge: fall.discharge,
                            })
                            .collect(),
                    })
                }),
            failed_jobs,
            latest_job_failure: latest_job_failure.map(str::to_owned),
        }
    }
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

fn lakes(found: &Hydrology, origin: Vec3) -> Vec<LakeView> {
    found
        .lake_sights(LAKES_SHOWN)
        .into_iter()
        .map(|lake| {
            let at = |[x, z]: [f32; 2]| [x + origin.x, lake.level + origin.y, z + origin.z];
            LakeView {
                level: lake.level + origin.y,
                area: lake.area,
                middle: at(lake.middle),
                shore: at(lake.shore),
            }
        })
        .collect()
}

fn sights(found: &Hydrology, origin: Vec3) -> Vec<RiverSightView> {
    found
        .river_sights()
        .into_iter()
        .map(|sight| RiverSightView {
            kind: sight.kind,
            point: [
                sight.point[0] + origin.x,
                sight.point[1] + origin.y,
                sight.point[2] + origin.z,
            ],
            direction: sight.direction,
            half: sight.half,
            speed: sight.speed,
            discharge: sight.discharge,
        })
        .collect()
}
