use super::*;
use bevy_ecs::world::World;
use engine_core::CookAsks;

pub(crate) fn cook(world: &World, asks: &mut CookAsks<'_>) {
    let Some(ground) = world.get_resource::<Ground>() else {
        return;
    };
    let Some(mut agents) =
        world.try_query_filtered::<(&NavAgent, &NavWalk, &Transform), Without<Parent>>()
    else {
        return;
    };
    let mut graphs: Vec<(Walked, Gait)> = Vec::new();
    for (agent, walk, transform) in agents.iter(world) {
        if walk.status == NavStatus::Refused {
            continue;
        }
        let at = Vec2::new(transform.translation.x, transform.translation.z);
        match ground_source(ground, at, at) {
            GroundRead::Ready(walked) => {
                let gait = options(agent).gait(&walked.source.pyramid);
                if !graphs.iter().any(|(known, known_gait)| {
                    known.source.identity == walked.source.identity
                        && known.grid == walked.grid
                        && *known_gait == gait
                }) {
                    graphs.push((walked, gait));
                }
            }
            GroundRead::NotLoaded => asks.left(1),
            GroundRead::Off => {}
        }
    }
    for (walked, gait) in &graphs {
        for key in walked.source.pyramid.level_tiles(gait.level) {
            asks.ask(&PORTALS, portals_input(walked, key, gait));
        }
    }
}
