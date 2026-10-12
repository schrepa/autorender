//! What collision's queries answer: the hits of rays, overlaps and shape
//! casts, and pages of sensor events, each with the collision world they
//! were answered from.

use serde::Serialize;

use crate::DevError;
use crate::views::CollisionWorldView;

/// An entity a hit or event names, by its authored identity.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct HitEntity {
    /// Its authored ID, where it has one.
    pub id: Option<String>,
    /// The scene it was authored in.
    pub scene: Option<String>,
    /// Its display name.
    pub name: Option<String>,
    /// Whether it belongs to the active scene, streamed in or not.
    pub in_active_scene: bool,
    /// Whether it still exists.
    pub alive: bool,
}

/// A placed object a hit names: its terrain's placement layer and asset.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct PlacedHit {
    /// The placement layer's index.
    pub layer: u16,
    /// The asset's index.
    pub asset: u16,
}

/// What `raycast` answers: the closest complete hit, if any.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct RaycastResult {
    /// The collision world queried.
    pub snapshot: CollisionWorldView,
    /// Whether the query saw every candidate: always, or it fails.
    pub complete: bool,
    /// The closest hit.
    pub hit: Option<RayHit>,
}

/// Where a ray hit.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct RayHit {
    /// What it hit.
    pub entity: HitEntity,
    /// How far along the ray.
    pub distance: f32,
    /// Where, in world space.
    pub point: [f32; 3],
    /// The surface's normal there.
    pub normal: [f32; 3],
    /// Whether the ray started inside it.
    pub started_inside: bool,
    /// The placed object hit, where a terrain's placement was.
    pub placed: Option<PlacedHit>,
}

/// What `overlap` answers: every entity the shape overlaps, by scene and ID.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct OverlapResult {
    /// The collision world queried.
    pub snapshot: CollisionWorldView,
    /// Whether the query saw every candidate: always, or it fails.
    pub complete: bool,
    /// How many it overlaps.
    pub total: usize,
    /// Each, sorted by scene and ID.
    pub hits: Vec<HitEntity>,
}

/// What `shape_cast` answers: the closest complete hit, if any.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct ShapeCastResult {
    /// The collision world queried.
    pub snapshot: CollisionWorldView,
    /// Whether the query saw every candidate: always, or it fails.
    pub complete: bool,
    /// The closest hit.
    pub hit: Option<CastHit>,
}

/// Where a cast shape hit.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct CastHit {
    /// What it hit.
    pub entity: HitEntity,
    /// The fraction of the displacement travelled.
    pub fraction: f32,
    /// How far it travelled.
    pub distance: f32,
    /// Where it touched, in world space.
    pub point: [f32; 3],
    /// The surface's normal there.
    pub normal: [f32; 3],
    /// Whether the shape started overlapping it.
    pub started_overlapping: bool,
    /// The placed object hit, where a terrain's placement was.
    pub placed: Option<PlacedHit>,
}

/// What `collision_events` answers: a page of the latest fixed tick's
/// sensor transitions, which reading does not consume.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct CollisionEventsResult {
    /// Sensors alone make events: always true.
    pub sensor_only: bool,
    /// The events' sequence number, a decimal string.
    pub sequence: String,
    /// The tick that attempted them.
    pub tick: Option<String>,
    /// The tick that committed them.
    pub committed_tick: Option<String>,
    /// Whether every candidate pair was checked.
    pub complete: bool,
    /// Pairs the sensors' broad phase found.
    pub candidate_pairs: usize,
    /// Pairs overlapping now.
    pub active_pairs: usize,
    /// Why the tick's events are incomplete, where they are.
    pub error: Option<DevError>,
    /// The events selected.
    pub total: usize,
    /// The page's events.
    pub events: Vec<SensorEventRow>,
    /// Where the next page starts, where one does.
    pub next_offset: Option<usize>,
}

/// One sensor transition: a pair began or stopped overlapping.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct SensorEventRow {
    /// Whether the pair began or stopped overlapping.
    pub kind: SensorEventKindView,
    /// The pair's first entity.
    pub first: HitEntity,
    /// Its second.
    pub second: HitEntity,
}

/// Whether a pair began or stopped overlapping.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum SensorEventKindView {
    Enter,
    Exit,
}
