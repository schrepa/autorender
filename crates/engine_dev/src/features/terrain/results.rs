//! What terrain's queries and edits answer: the ground under points, a
//! sweep comparing what queries, rays and layers say the ground is, a route,
//! and what an edit, undo or redo did.

use serde::Serialize;

use super::queries::TerrainKey;

/// What `terrain_height` answers: each point's ground, null where no tile
/// is resident, and the terrains' streaming counters.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct TerrainHeightResult {
    /// Each point's, in the order asked.
    pub samples: Vec<Option<HeightSample>>,
    /// The terrains' streaming counters.
    pub stats: engine_core::TerrainStats,
}

/// The ground under one point.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct HeightSample {
    /// The terrain answering.
    pub terrain: TerrainKey,
    /// The ground's world height.
    pub height: f32,
    /// Its normal.
    pub normal: [f32; 3],
    /// Whether a hole cuts it there.
    pub hole: bool,
    /// The level of the resident tile answering: 0, the finest.
    pub level: u32,
    /// What the terrain's hydrology bake found there, where it has one.
    pub water: Option<engine_core::terrain_data::HydrologySample>,
    /// The water standing there, where any does.
    pub standing_water: Option<StandingWater>,
    /// Meters from the nearest road's edge (negative on roads); null
    /// beyond every path's masks' reach.
    pub path: Option<f32>,
}

/// Water standing over the ground.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct StandingWater {
    /// Its surface's world height.
    pub surface: f32,
    /// Its depth over the ground.
    pub depth: f32,
}

/// What `terrain_sweep` answers: how far what terrain queries, collision
/// rays and generated terrains' layers say of the ground differ.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct TerrainSweepResult {
    /// Points swept.
    pub points: u32,
    /// Those under a resident tile.
    pub resident: usize,
    /// Those whose tile is of level 0.
    pub level_zero: usize,
    /// Those in holes.
    pub holes: usize,
    /// Collision rays against the tiles, where the engine has collision.
    pub collision: Option<SweepRays>,
    /// Generated terrains' layers against the tiles.
    pub layers: SweepLayers,
    /// A few points swept and their heights.
    pub examples: Vec<SweepExample>,
}

/// Collision rays against the tiles.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct SweepRays {
    /// Rays that met the terrain.
    pub compared: usize,
    /// Rays that met another entity or nothing.
    pub missed: usize,
    /// Rays that met a placed object first.
    pub placed_first: usize,
    /// The largest difference and where.
    pub worst: Worst,
}

/// Generated terrains' layers against the tiles.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct SweepLayers {
    /// Points compared.
    pub compared: usize,
    /// The largest difference and where.
    pub worst: Worst,
}

/// The largest difference so far and where.
#[derive(Default, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct Worst {
    /// Meters.
    #[serde(rename = "max_difference")]
    pub difference: f32,
    /// Where, world x and z; null before any comparison.
    pub at: Option<[f32; 2]>,
}

impl Worst {
    /// Counts `difference` at `point`.
    pub(super) fn add(&mut self, difference: f32, point: engine_core::glam::Vec2) {
        if difference > self.difference || self.at.is_none() {
            self.difference = engine_core::math::max(self.difference, difference);
            self.at = Some(point.to_array());
        }
    }
}

/// A point swept and its height.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct SweepExample {
    /// World x and z.
    pub at: [f32; 2],
    /// The tile's height there.
    pub height: f32,
}

/// What `terrain_route` answers: the route found, if one reaches.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct TerrainRouteResult {
    /// The terrain routed over.
    pub terrain: TerrainKey,
    /// How the route weighed the ground.
    pub options: engine_core::RouteOptions,
    /// The terrain's origin, world x and z, which the route's points are
    /// relative to.
    pub origin: [f32; 2],
    /// The route; null where none reaches.
    pub route: Option<RouteFound>,
}

/// A route over the terrain.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct RouteFound {
    /// Its points, terrain-local x and z, rounded to centimeters.
    pub points: Vec<[f64; 2]>,
    /// Its length, meters.
    pub length: f32,
    /// Its steepest step, rise per meter.
    pub steepest: f32,
    /// Steps of its hairpin turns.
    pub hairpin_steps: usize,
}

/// What `terrain_edit`, `terrain_undo` and `terrain_redo` answer.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(untagged)]
pub enum TerrainEdited {
    /// An edit, applied or standing already.
    Edit {
        /// Controls change the session's world, not the scene's source.
        temporary: bool,
        /// Whether it applied: false where an edit of its ID stands.
        applied: bool,
        /// Its ID.
        id: String,
        /// Edits standing now.
        edits: usize,
        /// Where tiles regenerate.
        dirty: Reach,
        /// The most the ground moved, meters.
        largest_change: f32,
    },
    /// An undo.
    Undo {
        /// Controls change the session's world, not the scene's source.
        temporary: bool,
        /// The ID of the edit taken away; null where none stood.
        undone: Option<String>,
        /// Edits standing now.
        edits: usize,
        /// Where tiles regenerate, where an edit was taken away.
        #[serde(skip_serializing_if = "Option::is_none")]
        dirty: Option<Reach>,
    },
    /// A redo.
    Redo {
        /// Controls change the session's world, not the scene's source.
        temporary: bool,
        /// The ID of the edit put back; null where none waited.
        redone: Option<String>,
        /// Edits standing now.
        edits: usize,
        /// Where tiles regenerate, where an edit was put back.
        #[serde(skip_serializing_if = "Option::is_none")]
        dirty: Option<Reach>,
    },
}

/// Where an edit reaches, world x and z.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct Reach {
    /// Its least corner.
    pub min: [f32; 2],
    /// Its greatest.
    pub max: [f32; 2],
}
