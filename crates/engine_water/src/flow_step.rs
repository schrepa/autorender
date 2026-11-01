use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::FixedTime;
use engine_places::water::WaterObstacles;
use glam::Vec2;

use crate::state::WaterState;
use crate::{Block, RiverFlow};

const NEAR: f32 = 16.0;
const EVERY: u64 = 6;

pub(crate) fn step(
    mut state: ResMut<WaterState>,
    obstacles: Res<WaterObstacles>,
    time: Res<FixedTime>,
) {
    let obstacles = &obstacles.obstacles[..];
    let still = |state: &WaterState| {
        state
            .terrains
            .iter()
            .all(|terrain| terrain.flow.as_ref().is_none_or(RiverFlow::still))
    };
    if obstacles.is_empty() && still(&state) || !time.tick.is_multiple_of(EVERY) {
        return;
    }
    let _profile = engine_core::profile::span("terrain", "river_flow");
    let state = state.bypass_change_detection();
    for terrain in &mut state.terrains {
        if terrain.surroundings {
            continue;
        }
        let Some(rivers) = terrain.rivers().cloned() else {
            terrain.flow = None;
            continue;
        };
        if rivers.is_empty() {
            terrain.flow = None;
            continue;
        }
        if terrain
            .flow
            .as_ref()
            .is_some_and(|flow| !Arc::ptr_eq(flow.base(), &rivers))
        {
            terrain.flow = None;
        }
        let origin = terrain.origin;
        let (low, high) = rivers.bounds();
        let near: Vec<_> = obstacles
            .iter()
            .filter(|obstacle| {
                let at = Vec2::new(obstacle.center.x - origin.x, obstacle.center.z - origin.z);
                at.cmpge(low - Vec2::splat(NEAR)).all() && at.cmple(high + Vec2::splat(NEAR)).all()
            })
            .collect();
        if near.is_empty() && terrain.flow.as_ref().is_none_or(RiverFlow::still) {
            continue;
        }
        let flow = terrain
            .flow
            .get_or_insert_with(|| RiverFlow::new(Arc::clone(&rivers)));
        let blocks: Vec<Block> = near
            .iter()
            .filter_map(|obstacle| {
                flow.block(obstacle.center - origin, obstacle.axes, obstacle.round)
            })
            .collect();
        flow.step(&blocks, time.delta_seconds * EVERY as f32);
    }
}
