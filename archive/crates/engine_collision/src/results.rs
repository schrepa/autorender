use serde::Serialize;

use crate::types::CollisionStats;

#[derive(Serialize, schemars::JsonSchema)]
pub(crate) struct HitEntity {
    pub id: Option<String>,
    pub scene: Option<String>,
    pub name: Option<String>,
    pub in_active_scene: bool,
    pub alive: bool,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(crate) struct PlacedHit {
    pub layer: u16,
    pub asset: u16,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(crate) struct RaycastResult {
    pub snapshot: CollisionStats,
    pub complete: bool,
    pub hit: Option<RayHit>,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(crate) struct RayHit {
    pub entity: HitEntity,
    pub distance: f32,
    pub point: [f32; 3],
    pub normal: [f32; 3],
    pub started_inside: bool,
    pub placed: Option<PlacedHit>,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(crate) struct OverlapResult {
    pub snapshot: CollisionStats,
    pub complete: bool,
    pub total: usize,
    pub hits: Vec<HitEntity>,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(crate) struct ShapeCastResult {
    pub snapshot: CollisionStats,
    pub complete: bool,
    pub hit: Option<CastHit>,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(crate) struct CastHit {
    pub entity: HitEntity,
    pub fraction: f32,
    pub distance: f32,
    pub point: [f32; 3],
    pub normal: [f32; 3],
    pub started_overlapping: bool,
    pub placed: Option<PlacedHit>,
}
