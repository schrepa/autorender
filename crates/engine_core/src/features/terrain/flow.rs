use std::sync::Arc;

use bevy_ecs::prelude::{DetectChangesMut, Res, ResMut};
use engine_terrain::{Block, RiverFlow};
use glam::Vec2;

use super::Terrains;
use crate::{FixedTime, WaterObstacles};

const NEAR: f32 = 16.0;
const EVERY: u64 = 6;

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
        let flow = state
            .hydrology
            .flow
            .get_or_insert_with(|| RiverFlow::new(Arc::clone(hydrology.rivers())));
        let blocks: Vec<Block> = near
            .iter()
            .filter_map(|obstacle| {
                flow.block(obstacle.center - origin, obstacle.axes, obstacle.round)
            })
            .collect();
        flow.step(&blocks, time.delta_seconds * EVERY as f32);
    }
}
