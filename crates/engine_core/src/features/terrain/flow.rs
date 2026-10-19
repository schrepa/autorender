//! Terrains' rivers as the solids standing in them hold them back
//! ([`crate::WaterObstacles`]), each fixed tick: a terrain whose rivers a
//! solid rests in, or whose pools have yet to drain, steps its rivers'
//! [`RiverFlow`], whose water the world's water reads and renderers draw.
//! Terrains nothing stands in keep their bake's rivers and do no work.

use std::sync::Arc;

use bevy_ecs::prelude::{DetectChangesMut, Res, ResMut};
use engine_terrain::{Block, Layer, RiverFlow};
use glam::Vec2;

use super::Terrains;
use crate::{FixedTime, WaterObstacles};

/// Meters beyond its rivers' reach an obstacle's center may stand and still
/// be looked for in them.
const NEAR: f32 = 16.0;
/// Fixed ticks between the rivers' steps: water rises and falls by
/// centimeters a second, so they follow at ten a second at the engine's
/// sixty ticks.
const EVERY: u64 = 6;

/// Steps each terrain's rivers' flow every [`EVERY`] ticks, over those
/// ticks, with the tick's obstacles.
pub(crate) fn step(
    terrains: Option<ResMut<Terrains>>,
    obstacles: Option<Res<WaterObstacles>>,
    time: Res<FixedTime>,
) {
    let Some(mut terrains) = terrains else {
        return;
    };
    let obstacles = obstacles
        .as_deref()
        .map_or(&[][..], |held| &held.obstacles[..]);
    let still = |terrains: &Terrains| {
        terrains
            .entries
            .iter()
            .all(|state| state.hydrology.flow.as_ref().is_none_or(RiverFlow::still))
    };
    if obstacles.is_empty() && still(&terrains) || !time.tick.is_multiple_of(EVERY) {
        return;
    }
    let _profile = engine_profile::span("terrain", "river_flow");
    // Rivers' water is read, not watched: stepping it changes nothing
    // terrains' observers wait on.
    let terrains = terrains.bypass_change_detection();
    for state in &mut terrains.entries {
        if state.around.is_some() {
            continue;
        }
        let Some(hydrology) = state.baked.as_deref().and_then(|layers| layers.hydrology()) else {
            state.hydrology.flow = None;
            continue;
        };
        if hydrology.rivers().is_empty() {
            state.hydrology.flow = None;
            continue;
        }
        // Edited layers hold the same rivers as baked; others flow anew.
        if state
            .hydrology
            .flow
            .as_ref()
            .is_some_and(|flow| !Arc::ptr_eq(flow.base(), hydrology.rivers()))
        {
            state.hydrology.flow = None;
        }
        let origin = state.terrain.origin;
        let (low, high) = hydrology.rivers().bounds();
        let near: Vec<_> = obstacles
            .iter()
            .filter(|obstacle| {
                let at = Vec2::new(obstacle.center.x - origin.x, obstacle.center.z - origin.z);
                at.cmpge(low - Vec2::splat(NEAR)).all() && at.cmple(high + Vec2::splat(NEAR)).all()
            })
            .collect();
        if near.is_empty() && state.hydrology.flow.as_ref().is_none_or(RiverFlow::still) {
            continue;
        }
        let roughness = state
            .baked
            .as_deref()
            .and_then(|layers| {
                layers.layers().iter().find_map(|layer| {
                    if let Layer::Hydrology(stage) = layer {
                        Some(stage.layer.roughness)
                    } else {
                        None
                    }
                })
            })
            .unwrap_or(engine_terrain::ROUGHNESS);
        let flow = state
            .hydrology
            .flow
            .get_or_insert_with(|| RiverFlow::new(Arc::clone(hydrology.rivers()), roughness));
        let blocks: Vec<Block> = near
            .iter()
            .filter_map(|obstacle| {
                flow.block(obstacle.center - origin, obstacle.axes, obstacle.round)
            })
            .collect();
        flow.step(&blocks, time.delta_seconds * EVERY as f32);
    }
}
