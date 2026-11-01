use bevy_ecs::prelude::*;
use engine_core::FixedTime;
use engine_places::navigation::{NavAgent, NavStatus, NavWalk};
use serde::{Deserialize, Serialize};

#[derive(Component, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Hiker {
    ends: [[f32; 2]; 2],
    toward: usize,
    rest: f32,
}

const HIKER_REST: f32 = 10.0;

pub(crate) fn hike(
    time: Res<FixedTime>,
    mut hikers: Query<(&mut NavAgent, &NavWalk, &mut Hiker)>,
) {
    for (mut agent, walk, mut hiker) in &mut hikers {
        if walk.destination != agent.destination {
            continue;
        }
        match walk.status {
            NavStatus::Unreachable if hiker.rest < HIKER_REST => {
                hiker.rest += time.delta_seconds;
                continue;
            }
            NavStatus::Arrived | NavStatus::Unreachable => {}
            NavStatus::Idle | NavStatus::Searching | NavStatus::Walking => continue,
        }
        hiker.rest = 0.0;
        hiker.toward = 1 - hiker.toward;
        agent.destination = Some(hiker.ends[hiker.toward]);
    }
}
