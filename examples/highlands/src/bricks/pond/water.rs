use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::Spawner;
use engine_places::ground::{Ground, GroundRead};
use engine_places::water::{WaterBody, WaterKind, WaterQuery, WaterSample};
use glam::{Vec2, Vec3};

use crate::bricks::pond::{Pond, PondBrick};

const PRECEDENCE: i32 = -1;

#[derive(Component)]
pub(crate) struct PondBody;

struct PondWater {
    ponds: Vec<Pond>,
    ground: Ground,
}

impl WaterQuery for PondWater {
    fn water(&self, point: Vec3) -> GroundRead<Option<WaterSample>> {
        let surface = self
            .ponds
            .iter()
            .filter(|pond| pond.holds(point.x, point.z))
            .map(|pond| pond.surface)
            .reduce(engine_core::math::max);
        let Some(surface) = surface else {
            return GroundRead::Ready(None);
        };
        let ground = match self.ground.point(Vec2::new(point.x, point.z)) {
            GroundRead::Ready(ground) => ground,
            GroundRead::NotLoaded => return GroundRead::NotLoaded,
            GroundRead::Off => return GroundRead::Off,
        };
        let depth = surface - ground.height;
        GroundRead::Ready((depth > 0.0 && !ground.hole).then_some(WaterSample {
            kind: WaterKind::Lake,
            surface,
            depth,
            velocity: [0.0; 3],
            foam: 0.0,
        }))
    }
}

pub(crate) fn publish_ponds(
    ponds: Query<Ref<Pond>>,
    ground: Res<Ground>,
    bodies: Query<Entity, With<PondBody>>,
    mut held: Local<usize>,
    mut spawner: Spawner<PondBrick>,
) {
    let count = ponds.iter().count();
    let changed =
        ground.is_changed() || count != *held || ponds.iter().any(|pond| pond.is_changed());
    if !changed {
        return;
    }
    *held = count;
    if count == 0 {
        for body in &bodies {
            spawner.despawn(body);
        }
        return;
    }
    let body = WaterBody {
        precedence: PRECEDENCE,
        shape: Arc::new(PondWater {
            ponds: ponds.iter().map(|pond| *pond).collect(),
            ground: Ground::clone(&ground),
        }),
    };
    match bodies.single() {
        Ok(entity) => spawner.insert(entity, body),
        Err(_) => {
            spawner.spawn((PondBody, body));
        }
    }
}
