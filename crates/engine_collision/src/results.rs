use engine_core::Error;
use serde::Serialize;

use crate::types::CollisionStats;

#[derive(Serialize)]
pub(crate) struct HitEntity {
    pub id: Option<String>,
    pub scene: Option<String>,
    pub name: Option<String>,
    pub in_active_scene: bool,
    pub alive: bool,
}

#[derive(Serialize)]
pub(crate) struct PlacedHit {
    pub layer: u16,
    pub asset: u16,
}

#[derive(Serialize)]
pub(crate) struct RaycastResult {
    pub snapshot: CollisionStats,
    pub complete: bool,
    pub hit: Option<RayHit>,
}

#[derive(Serialize)]
pub(crate) struct RayHit {
    pub entity: HitEntity,
    pub distance: f32,
    pub point: [f32; 3],
    pub normal: [f32; 3],
    pub started_inside: bool,
    pub placed: Option<PlacedHit>,
}

#[derive(Serialize)]
pub(crate) struct OverlapResult {
    pub snapshot: CollisionStats,
    pub complete: bool,
    pub total: usize,
    pub hits: Vec<HitEntity>,
}

#[derive(Serialize)]
pub(crate) struct ShapeCastResult {
    pub snapshot: CollisionStats,
    pub complete: bool,
    pub hit: Option<CastHit>,
}

#[derive(Serialize)]
pub(crate) struct CastHit {
    pub entity: HitEntity,
    pub fraction: f32,
    pub distance: f32,
    pub point: [f32; 3],
    pub normal: [f32; 3],
    pub started_overlapping: bool,
    pub placed: Option<PlacedHit>,
}

#[derive(Serialize)]
pub(crate) struct CollisionEventsResult {
    pub sensor_only: bool,
    pub sequence: String,
    pub tick: Option<String>,
    pub committed_tick: Option<String>,
    pub complete: bool,
    pub candidate_pairs: usize,
    pub active_pairs: usize,
    pub error: Option<Error>,
    pub total: usize,
    pub events: Vec<SensorEventRow>,
    pub next_offset: Option<usize>,
}

#[derive(Serialize)]
pub(crate) struct SensorEventRow {
    pub kind: SensorEventKindView,
    pub first: HitEntity,
    pub second: HitEntity,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SensorEventKindView {
    Enter,
    Exit,
}
