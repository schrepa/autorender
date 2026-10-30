use glam::{Vec2, Vec3};

use super::Terrains;
use crate::{Fall, Layers};
use engine_core::math::{GRAVITY, max, min};
use engine_core::{WaterKind, WaterSample};

const SHEET: f32 = 1.0;

impl Terrains {
    pub fn waters(&self) -> impl Iterator<Item = (bevy_ecs::entity::Entity, Vec3, &Layers)> {
        self.entries
            .iter()
            .filter(|state| state.around.is_none())
            .filter_map(|state| {
                let layers = state.baked.as_deref()?;
                layers
                    .hydrology()
                    .map(|_| (state.entity, state.terrain.origin, layers))
            })
    }

    pub fn river_flow(&self, entity: bevy_ecs::entity::Entity) -> Option<&crate::RiverFlow> {
        self.entries
            .iter()
            .find(|state| state.entity == entity)?
            .hydrology
            .flow
            .as_ref()
    }

    pub fn water_at(&self, point: Vec3) -> Option<WaterSample> {
        self.entries
            .iter()
            .filter(|state| state.around.is_none())
            .find_map(|state| {
                let origin = state.terrain.origin;
                let local = point - origin;
                let held = state
                    .water(Vec2::new(local.x, local.z))
                    .map(|water| WaterSample {
                        kind: if water.lake {
                            WaterKind::Lake
                        } else {
                            WaterKind::River
                        },
                        surface: water.surface,
                        depth: water.depth,
                        velocity: [water.flow.x, 0.0, water.flow.y],
                        foam: water.state.breaking,
                    });
                [held, fall(&state.falls, local)]
                    .into_iter()
                    .flatten()
                    .max_by(|a, b| a.surface.total_cmp(&b.surface))
                    .map(|sample| WaterSample {
                        surface: sample.surface + origin.y,
                        ..sample
                    })
            })
    }
}

fn fall(falls: &[Fall], point: Vec3) -> Option<WaterSample> {
    let at = Vec2::new(point.x, point.z);
    falls.iter().find_map(|fall| {
        let top = Vec3::from_array(fall.top);
        fall.path.windows(2).find_map(|pair| {
            let (a, b) = (Vec3::from_array(pair[0]), Vec3::from_array(pair[1]));
            let run = Vec2::new(b.x - a.x, b.z - a.z);
            let length = run.length();
            let t = if length > 0.0 {
                min(
                    max((at - Vec2::new(a.x, a.z)).dot(run) / (length * length), 0.0),
                    1.0,
                )
            } else {
                0.0
            };
            let on = a.lerp(b, t);
            let height = on.y;
            if at.distance(Vec2::new(on.x, on.z)) >= 0.5 * fall.width
                || (point.y - height).abs() > SHEET
            {
                return None;
            }
            let across = Vec2::new(fall.bottom[0] - fall.top[0], fall.bottom[2] - fall.top[2]);
            let drop = max(top.y - fall.bottom[1], 0.1);
            let horizontal = across.length() * (GRAVITY / (2.0 * drop)).sqrt();
            let falling = (2.0 * GRAVITY * max(top.y - height, 0.0)).sqrt();
            let heading = across.normalize_or(Vec2::X) * horizontal;
            Some(WaterSample {
                kind: WaterKind::Fall,
                surface: height,
                depth: SHEET,
                velocity: [heading.x, -falling, heading.y],
                foam: 1.0,
            })
        })
    })
}
